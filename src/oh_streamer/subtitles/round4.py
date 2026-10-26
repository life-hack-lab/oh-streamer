"""Replay fixed round-three Japanese cues, then prepare Sakura blind review.

Models must already exist locally. This command never downloads or runs ASR.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import shutil
import subprocess

from .cli import mux_subtitles, translate_file
from .core import Segment, parse_srt
from .runtime import atomic_write_json, atomic_write_text, resolve_ffmpeg


def read_json(path):
    return json.loads(Path(path).read_text('utf-8-sig'))


def _blind_seed(out, explicit_seed=None):
    """Keep an experiment's random mapping stable across report-only retries."""
    path = Path(out) / 'experiment.json'
    manifest = read_json(path) if path.exists() else {}
    if not isinstance(manifest, dict):
        raise ValueError('Experiment manifest must be a JSON object')
    seed = explicit_seed if explicit_seed is not None else manifest.get('blind_seed')
    if seed is None:
        seed = secrets.randbits(64)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError('Blind seed must be an integer')
    if manifest.get('blind_seed') != seed:
        manifest['blind_seed'] = seed
        atomic_write_json(path, manifest)
    return seed


def alignment(rows):
    return [(r['start'], r['end'], r.get('text_ja', r.get('text'))) for r in rows]


def preflight_fixed_input(source, baseline):
    """Refuse input/recipe drift before launching any translation server.

    Historical batch/context sizes are documented as 30/8192 but were not
    persisted in early result JSON. Validate them if present and explicitly
    record that provenance limit in the returned experiment metadata.
    """
    source_rows, baseline_rows = source['segments'], baseline['segments']
    if not source_rows or alignment(source_rows) != alignment(baseline_rows):
        raise ValueError('Fixed Japanese input differs from the historical baseline')
    identities = alignment(source_rows)
    for start, end, text in identities:
        if not isinstance(text, str) or not text.strip():
            raise ValueError('Fixed Japanese input contains invalid text')
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in (start, end)):
            raise ValueError('Fixed Japanese input contains invalid timestamps')
        Segment(start, end, text)
    if len(set(identities)) != len(identities):
        raise ValueError('Fixed Japanese input contains duplicate cue identities')
    expected = dict(prompt_mode='official-numbered', context_mode='none',
                    secondary_asr=None, suspicion_threshold=.45,
                    sentence_recipe='punctuation-proportional-v1',
                    short_threshold=1., short_policy='whitelist')
    historical = baseline.get('recipe_options', {})
    source_recipe = source.get('options', {})
    for key, value in expected.items():
        for label, recipe in (('historical', historical), ('source', source_recipe)):
            if key not in recipe or recipe[key] != value:
                raise ValueError(f'{label} recipe {key} differs from the fixed experiment')
    if not isinstance(historical.get('glossary'), list) or source_recipe.get('glossary') != historical['glossary']:
        raise ValueError('Source and historical glossaries differ')
    for field, expected_value in (('batch_size', 30), ('context_size', 8192), ('translate_model', 'sakura-v1')):
        if field in historical and historical[field] != expected_value:
            raise ValueError(f'Historical recipe {field} differs from the fixed experiment')
    return dict(fixed_input_cues=len(source_rows), prompt_mode=expected['prompt_mode'],
                context_mode=expected['context_mode'], fixed_cues=True, on_fail='hide',
                historical_unrecorded_settings=[name for name in ('batch_size', 'context_size') if name not in historical],
                historical_setting_note='Batch 30 and context 8192 are documented historical settings; '
                                        'missing JSON fields cannot be independently verified.')


def compare_reproduction(baseline, replay):
    old, new = baseline['segments'], replay['segments']
    if alignment(old) != alignment(new):
        raise ValueError('Replay Japanese cues differ from the historical baseline')
    changes = [dict(cue=i + 1, start=a['start'], end=a['end'], text_ja=a['text_ja'],
                    baseline=a['text'], replay=b['text'])
               for i, (a, b) in enumerate(zip(old, new)) if a['text'] != b['text']]
    return dict(total_cues=len(old), identical_cues=len(old) - len(changes),
                changed_cues=len(changes), changes=changes,
                note='Sampling is not seeded; text equality is measured, not guaranteed.')


def measurement(result):
    m = result['metrics']
    nvml = m.get('nvml', {})
    peak = nvml.get('peak_bytes')
    return dict(seconds=m.get('elapsed_seconds_with_load'),
                tokens_per_second=m.get('tokens_per_second'),
                generation_tokens_per_second=m.get('generation_tokens_per_second'),
                peak_mib=peak / 1048576 if peak is not None else None,
                memory_scope=nvml.get('measurement'),
                mismatch_batches=m.get('mismatch_batches'),
                fallback_sentences=m.get('fallback_sentences'),
                preserved_original=sum(bool(r.get('preserved_original')) for r in result['segments']),
                untranslated=m.get('untranslated'),
                translate_model=result.get('translate_model', 'sakura-v1'))


def checkpoint_markdown(translations):
    from .postprocess import taiwan_text
    def cell(value):
        return str(value).replace('|', '\\|').replace('\n', '<br>')
    models = list(translations)
    lines = ['# 第四輪人名與稱謂核對', '',
             '文字對照僅供人工判讀，沒有依字面命中宣稱翻譯正確。', '',
             '| 秒數 | 日文 | ' + ' | '.join(models) + ' |',
             '| --- | --- | ' + ' | '.join('---' for _ in models) + ' |']
    first = translations[models[0]]['segments']
    for i, row in enumerate(first):
        ja = row['text_ja']
        if any(term in ja for term in ('酒井', 'さかい', '坂井', '堺', '部長', '社長', 'お会計', 'ほら')):
            lines.append('| ' + ' | '.join([f"{row['start']:.2f}–{row['end']:.2f}", cell(ja)] +
                         [cell(taiwan_text(translations[name]['segments'][i]['text'])) for name in models]) + ' |')
    return '\n'.join(lines) + '\n'


def _media_command(command):
    """Decode machine-readable UTF-8 output without opening Windows consoles."""
    return subprocess.run(command, check=True, capture_output=True, text=True,
                          encoding='utf-8-sig', timeout=60,
                          creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0) if os.name == 'nt' else 0).stdout


def _probe_streams(path, ffprobe):
    data = json.loads(_media_command([ffprobe, '-v', 'error', '-show_streams',
                                     '-show_format', '-of', 'json', str(path)]))
    if not isinstance(data, dict) or not isinstance(data.get('streams'), list):
        raise ValueError(f'ffprobe returned no stream list: {path}')
    return data['streams']


def _finite_time(value):
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def verify_preview(video, output, subtitles):
    """Inspect copied A/V and extract every subtitle track for full-cue checks.

    ``subtitles`` is the ordered list of expected SRT paths. Return a JSON-safe
    report with ``passed`` and ``errors``; failures never become successes by
    checking only the first/last cue. Allow one common mux offset of at most
    100 ms, measured from A/V start times, with 2 ms cue rounding tolerance.
    Missing A/V start times permit only an unshifted subtitle timeline.
    """
    video, output = Path(video).resolve(), Path(output).resolve()
    report = dict(passed=False, source=str(video), output=str(output), errors=[],
                  timestamp_tolerance_seconds=.002, max_mux_offset_seconds=.1,
                  av_streams=[], subtitle_tracks=[])
    errors = report['errors']
    try:
        ffmpeg = resolve_ffmpeg()
        sibling = Path(ffmpeg).with_name('ffprobe.exe' if Path(ffmpeg).suffix.lower() == '.exe' else 'ffprobe')
        ffprobe = str(sibling) if sibling.is_file() else (shutil.which('ffprobe') or shutil.which('ffprobe.exe'))
        if not ffprobe:
            raise FileNotFoundError('ffprobe is required to verify the preview')
        source_streams = _probe_streams(video, ffprobe)
        output_streams = _probe_streams(output, ffprobe)
        report['output_bytes'] = output.stat().st_size
        offsets = []
        for kind, properties in (
            ('video', ('codec_name', 'width', 'height')),
            ('audio', ('codec_name', 'sample_rate', 'channels')),
        ):
            originals = [s for s in source_streams if s.get('codec_type') == kind]
            copied = [s for s in output_streams if s.get('codec_type') == kind]
            expected_count = 1 if originals else 0
            if kind == 'video' and not originals:
                errors.append('Source has no video stream')
            if len(copied) != expected_count:
                errors.append(f'Preview {kind} stream count: expected {expected_count}, found {len(copied)}')
            if originals and copied:
                original, actual = originals[0], copied[0]
                entry = dict(kind=kind, source_codec=original.get('codec_name'),
                             output_codec=actual.get('codec_name'))
                for prop in properties:
                    if prop == 'codec_name' and not original.get(prop):
                        errors.append(f'Source {kind} codec is unknown')
                    if original.get(prop) != actual.get(prop):
                        errors.append(f'Preview {kind} {prop} differs: {original.get(prop)} -> {actual.get(prop)}')
                before, after = _finite_time(original.get('start_time')), _finite_time(actual.get('start_time'))
                entry.update(source_start=before, output_start=after)
                if before is not None and after is not None:
                    entry['mux_offset_seconds'] = after - before
                    offsets.append(after - before)
                report['av_streams'].append(entry)
        offset = sum(offsets) / len(offsets) if offsets else 0.
        report['mux_offset_seconds'] = offset
        if abs(offset) > .1 + .002:
            errors.append(f'Mux offset exceeds 100 ms: {offset:.6f} s')
        if any(abs(value - offset) > .002 + 1e-9 for value in offsets):
            errors.append('Audio/video timeline offsets disagree')
        tracks = [s for s in output_streams if s.get('codec_type') == 'subtitle']
        if len(tracks) != len(subtitles):
            errors.append(f'Preview subtitle track count: expected {len(subtitles)}, found {len(tracks)}')
        if any(s.get('codec_type') not in ('video', 'audio', 'subtitle') for s in output_streams):
            errors.append('Preview contains unexpected non-A/V/subtitle streams')
        for index, path in enumerate(subtitles):
            path = Path(path)
            expected_bytes = path.read_bytes()
            expected = parse_srt(expected_bytes.decode('utf-8-sig'))
            track_report = dict(track=index, expected_path=str(path.resolve()),
                                expected_sha256=hashlib.sha256(expected_bytes).hexdigest(),
                                expected_cues=len(expected), extracted_cues=0,
                                mismatched_cues=[], passed=False)
            report['subtitle_tracks'].append(track_report)
            if index >= len(tracks):
                continue
            stream = tracks[index]
            track_report['stream_index'] = stream['index']
            track_report['codec'] = stream.get('codec_name')
            if stream.get('codec_name') != 'subrip':
                errors.append(f'Subtitle track {index} codec is not copied SubRip')
            extracted_text = _media_command([ffmpeg, '-nostdin', '-v', 'error', '-copyts', '-i', str(output),
                                             '-map', f"0:{stream['index']}", '-c:s', 'srt', '-f', 'srt', '-'])
            extracted = parse_srt(extracted_text)
            track_report['extracted_cues'] = len(extracted)
            if len(expected) != len(extracted):
                errors.append(f'Subtitle track {index} cue count: expected {len(expected)}, found {len(extracted)}')
            maximum_error = 0.
            for cue_number, (wanted, actual) in enumerate(zip(expected, extracted), 1):
                timing_error = max(abs(actual.start - wanted.start - offset),
                                   abs(actual.end - wanted.end - offset))
                maximum_error = max(maximum_error, timing_error)
                reasons = []
                if actual.text != wanted.text:
                    reasons.append('text')
                if timing_error > .002 + 1e-9:
                    reasons.append('timestamps')
                if reasons:
                    track_report['mismatched_cues'].append(dict(cue=cue_number, reasons=reasons,
                                                                timing_error_seconds=timing_error))
            track_report['max_timing_error_seconds'] = maximum_error
            if track_report['mismatched_cues']:
                errors.append(f'Subtitle track {index} differs in {len(track_report["mismatched_cues"])} cues')
            track_report['passed'] = (len(expected) == len(extracted)
                                      and not track_report['mismatched_cues']
                                      and stream.get('codec_name') == 'subrip')
        report['passed'] = not errors
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
        errors.append(f'Artifact verification failed: {exc}')
    return report


def prepare_report(round3, out, results, *, video=None, seed=None):
    from .blind_ab import write_blind_review
    baseline = read_json(round3 / 'sakura.zh-TW.json')
    expected = alignment(baseline['segments'])
    for name, result in results.items():
        if alignment(result['segments']) != expected:
            raise ValueError(f'Fixed input alignment mismatch: {name}')
    reproduction = compare_reproduction(baseline, results['sakura-v1'])
    atomic_write_json(out / 'reproduction.json', reproduction)
    # The approved historical v1 translation, rather than a newly sampled v1,
    # is the blind-test baseline. Replay is reported separately.
    blind = dict(results)
    blind['sakura-v1'] = baseline
    source = {'segments': [dict(start=r['start'], end=r['end'], text=r['text_ja'])
                           for r in baseline['segments']]}
    write_blind_review(source, blind, out, seed=_blind_seed(out, seed))
    atomic_write_text(out / 'checkpoints.md', checkpoint_markdown(blind))
    atomic_write_json(out / 'measurements.json', {name: measurement(r) for name, r in results.items()})
    if video:
        tracks = [(name, (round3 if name == 'sakura-v1' else out / name) / 'sakura.zh-TW.srt', 'chi')
                  for name in results]
        preview = out / 'quality_review.mkv'
        # A report-only replay may reuse an existing preview only after the
        # exact same full-cue verification. Never silently overwrite it.
        if not preview.exists():
            mux_subtitles(video, tracks, preview, seconds=600)
        verification = verify_preview(video, preview, [path for _, path, _ in tracks])
        atomic_write_json(out / 'artifact_verification.json', verification)
        if not verification['passed']:
            raise ValueError('Preview artifact verification failed; see artifact_verification.json')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--round3', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--models', type=Path, required=True, help='Directory containing local Sakura GGUF files')
    p.add_argument('--llama-server', required=True)
    p.add_argument('--video', type=Path)
    p.add_argument('--only-7b', action='store_true')
    p.add_argument('--report-only', action='store_true', help='No inference; require saved results')
    p.add_argument('--force', action='store_true')
    p.add_argument('--seed', type=int)
    p.add_argument('--layers-14b', type=int, default=28)
    a = p.parse_args(argv)
    if a.layers_14b < 0:
        p.error('--layers-14b must be nonnegative')
    a.out.mkdir(parents=True, exist_ok=True)
    # Capture exactly the original glossary, including duplicate entries and
    # their order. translate_file adds the generic prefix itself.
    from .glossary import load_glossary
    baseline = read_json(a.round3 / 'sakura.zh-TW.json')
    source = a.round3 / 'sentence_cues.json'
    preflight = preflight_fixed_input(read_json(source), baseline)
    terms = baseline['recipe_options']['glossary']
    generic = load_glossary(Path(__file__).with_name('glossary_default.json'))
    if terms[:len(generic)] != generic:
        raise ValueError('Historical generic glossary differs; refuse silent recipe drift')
    glossary = a.out / 'glossary.json'
    atomic_write_json(glossary, terms[len(generic):])
    specs = [('sakura-v1', 'sakura-7b-qwen2.5-v1.0-iq4xs.gguf', 'auto'),
             ('sakura-galtransl-7b-v3.7', 'Sakura-Galtransl-7B-v3.7-IQ4_XS.gguf', 'auto')]
    if not a.only_7b:
        specs.append(('sakura-galtransl-14b-v3.8', 'Sakura-Galtransl-14B-v3.8-IQ4_XS.gguf', a.layers_14b))
    manifest = dict(input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    historical_sha256=hashlib.sha256((a.round3 / 'sakura.zh-TW.json').read_bytes()).hexdigest(),
                    batch_size=30, context_size=8192, asr_rerun=False,
                    models=[dict(profile=n, filename=f, gpu_layers=l) for n, f, l in specs])
    blind_seed = _blind_seed(a.out, a.seed)
    manifest['blind_seed'] = blind_seed
    manifest.update(preflight)
    atomic_write_json(a.out / 'experiment.json', manifest)
    results = {}
    for name, filename, layers in specs:
        if a.report_only:
            results[name] = read_json(a.out / name / 'sakura.zh-TW.json')
        else:
            print(f'Translating {name}', flush=True)
            results[name] = translate_file(source, a.out / name, model='sakura',
                model_path=a.models / filename, llama_server=a.llama_server,
                translate_model=name, glossary=glossary, fixed_cues=True,
                batch_size=30, context_size=8192, prompt_mode='official-numbered',
                context_mode='none', on_fail='hide',
                gpu_layers=layers, force=a.force)
        # Write partial measurements after every model for interrupted runs.
        atomic_write_json(a.out / 'measurements.json', {n: measurement(r) for n, r in results.items()})
    prepare_report(a.round3, a.out, results, video=a.video, seed=blind_seed)


if __name__ == '__main__':
    main()
