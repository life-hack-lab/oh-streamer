"""Independent subtitle commands and sequential PoC benchmark orchestration."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .core import (
    Segment,
    clean_segments,
    format_timestamp,
    has_japanese_residue,
    inspect_hallucinations,
    parse_numbered_translations,
    render_srt,
    translate_segments,
)
from .runtime import (
    NvmlMemorySampler,
    assert_gpu_safe,
    atomic_write_json,
    atomic_write_text,
    default_model_path,
    resolve_ffmpeg,
)


def load_segments(data):
    rows = data if isinstance(data, list) else data['segments']
    return [Segment(float(r['start']), float(r['end']), r.get('text', r.get('text_ja', '')),
                    r.get('no_speech_prob'), r.get('avg_logprob'), tuple(r.get('flags', ())),
                    compression_ratio=r.get('compression_ratio'), gap=bool(r.get('gap')),
                    low_confidence=bool(r.get('low_confidence')), alt_text=r.get('alt_text'),
                    asr_agreement=r.get('asr_agreement')) for r in rows]


def translation_key(
    source,
    model,
    model_path,
    show_interjections,
    batch_size,
    recipe_options=None,
    model_identity=None,
):
    """Content/settings identity prevents reusing another ASR or LLM's result."""
    stat = Path(model_path).stat() if model_identity is None else None
    identity = {
        'source_sha256': hashlib.sha256(Path(source).read_bytes()).hexdigest(),
        'model': model,
        'model_path': str(Path(model_path).resolve()) if stat else None,
        'model_bytes': stat.st_size if stat else None,
        'model_mtime_ns': stat.st_mtime_ns if stat else None,
        'registered_model': model_identity,
        'show_interjections': show_interjections,
        'batch_size': batch_size,
        'recipe': 5,
        'options': recipe_options,
    }
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


_COUNTER_METRICS = (
    'requests', 'failures', 'tokens', 'prompt_tokens', 'completion_tokens',
    'total_tokens', 'elapsed_seconds', 'generation_seconds',
)


def _metrics_delta(after, before):
    """Per-call metrics from a cumulative client snapshot (shared llama-server)."""
    delta = {key: after.get(key, 0) - before.get(key, 0) for key in _COUNTER_METRICS}
    elapsed, generation = delta['elapsed_seconds'], delta['generation_seconds']
    delta['tokens_per_second'] = delta['completion_tokens'] / elapsed if elapsed else 0.0
    delta['generation_tokens_per_second'] = (
        delta['completion_tokens'] / generation if generation else delta['tokens_per_second']
    )
    return delta


class _CallMemo:
    """Call-level checkpoint so a stopped translation resumes without re-calling the server."""

    def __init__(self, path, cache_key):
        self.path = Path(path)
        self.cache_key = cache_key
        self.calls = {}
        try:
            data = json.loads(self.path.read_text('utf-8'))
            if (isinstance(data, dict) and data.get('cache_key') == cache_key
                    and isinstance(data.get('calls'), dict)):
                self.calls = data['calls']
        except (OSError, ValueError):
            pass

    def digest(self, kind, payload):
        raw = json.dumps([self.cache_key, kind, payload], ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(raw.encode('utf-8')).hexdigest()

    def wrap(self, kind, call, payload_of):
        def wrapped(arg):
            digest = self.digest(kind, payload_of(arg))
            if digest in self.calls:
                return self.calls[digest]
            value = call(arg)  # raises on invalid output; only successes are stored
            self.calls[digest] = value
            atomic_write_json(self.path, {'cache_key': self.cache_key, 'calls': self.calls})
            return value
        return wrapped

    def clear(self):
        try:
            self.path.unlink()
        except OSError:
            pass


def translate_file(
    source,
    out_dir,
    *,
    model,
    model_path,
    llama_server,
    batch_size=30,
    show_interjections=False,
    force=False,
    prompt_mode='official-numbered',
    glossary=None,
    context_mode='none',
    on_fail='hide',
    secondary_asr=None,
    suspicious='keep',
    suspicion_threshold=0.45,
    sentence_mode=False,
    short_threshold=1.0,
    short_policy='whitelist',
    generic_glossary_only=False,
    on_progress=None,
    gpu_managed=False,
    on_phase=None,
    translate_model='sakura-v1',
    gpu_layers='auto',
    context_size=8192,
    fixed_cues=False,
    shared_client=None,
    model_identity=None,
    sampling=None,
    dual_tracks=True,
    translation_lookup=None,
    device=None,
    backend='cuda',
    on_batch=None,
):
    """Translate subtitles, optionally publishing cumulative finalized row snapshots.

    ``on_batch(rows)`` receives the same row schema as ``result['segments']``;
    callers apply their usual display policy. Each row's translation and timing
    are final. Completed duplicate strings include every source occurrence.
    Cache/lookup results publish once; callback errors propagate to the caller.
    """
    from .tracks import (deterministic_sampling, unique_segments, render_tracks,
                         wording_identity, TRANSLATION_RECIPE)
    sampling = deterministic_sampling(sampling)
    phase_percent = {}
    phase_none_detail = {}
    phase_detail = {}

    def report_phase(phase, percent, detail, *, force=False):
        if not callable(on_phase):
            return
        if percent is None:
            detail = str(detail)
            retained_percent = phase_percent.get(phase)
            if retained_percent is None:
                if not force and phase_none_detail.get(phase) == detail:
                    return
                phase_none_detail[phase] = detail
                on_phase(phase, None, detail)
            else:
                if not force and phase_detail.get(phase) == detail:
                    return
                phase_detail[phase] = detail
                on_phase(phase, retained_percent, detail)
            return
        try:
            value = max(0, min(100, int(round(float(percent)))))
        except (TypeError, ValueError):
            return
        value = max(phase_percent.get(phase, 0), value)
        if (
            phase_percent.get(phase) == value
            and (not force or phase_detail.get(phase) == str(detail))
        ):
            return
        phase_percent[phase] = value
        phase_detail[phase] = str(detail)
        on_phase(phase, value, str(detail))

    report_phase(
        "prepare_translate",
        None,
        "正在準備字幕片段並檢查本機翻譯模型",
    )
    from .llm import LlamaServerClient, LlamaServerError
    from .postprocess import display_segments
    from .sentences import filter_short, render_sentence_srt, sentence_cues
    if sentence_mode and secondary_asr:
        raise ValueError('sentence mode requires original-span suspicion review separately')
    if fixed_cues and sentence_mode:
        raise ValueError('fixed cues and sentence mode are mutually exclusive')
    filter_short([], short_threshold, short_policy)
    render = render_sentence_srt if sentence_mode or fixed_cues or dual_tracks else render_srt
    if on_fail not in ('hide', 'japanese'):
        raise ValueError('on_fail must be hide or japanese')
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    from .core import estimate_translation_max_tokens
    from .glossary import (
        DEFAULT_TRANSLATE_MODEL,
        filter_relevant_terms,
        general_llm_messages,
        general_llm_single_messages,
        get_sakura_profile,
        load_glossary,
        parse_lines,
        sakura_messages,
    )
    profile = get_sakura_profile(translate_model)
    selected_profile = profile.name if model == 'sakura' else 'qwen3'
    if model != 'sakura' and profile.name != DEFAULT_TRANSLATE_MODEL:
        raise ValueError('--translate-model Sakura profiles require --backend sakura')
    if model == 'sakura' and profile.name != DEFAULT_TRANSLATE_MODEL and prompt_mode == 'legacy':
        raise ValueError('GalTransl profiles require an official prompt mode')
    if (
        prompt_mode not in ('legacy', 'official-lines', 'official-numbered', 'general-llm')
        or context_mode not in ('none', 'user')
    ):
        raise ValueError('invalid prompt or context mode')
    terms = load_glossary(Path(__file__).with_name('glossary_default.json'))
    glossary_path = Path(glossary) if glossary else out_dir / 'glossary.txt'
    if not glossary and not glossary_path.exists():
        glossary_path = out_dir / 'glossary.json'
    if not generic_glossary_only and glossary_path.exists():
        terms.extend(load_glossary(glossary_path))
    elif not generic_glossary_only and glossary:
        raise FileNotFoundError(glossary_path)
    if suspicious not in ('keep', 'mark', 'hide'):
        raise ValueError('invalid suspicious policy')
    if not 0 <= suspicion_threshold <= 1:
        raise ValueError('suspicion threshold must be between zero and one')
    if suspicious != 'keep' and not secondary_asr:
        raise ValueError(
            'suspicious policy requires --secondary-asr JSON '
            'with identical VAD spans'
        )
    secondary_data = (
        json.loads(Path(secondary_asr).read_text('utf-8')) if secondary_asr else None
    )
    options = dict(
        prompt_mode=prompt_mode,
        glossary=terms,
        context_mode=context_mode,
        secondary_asr=secondary_data,
        suspicion_threshold=suspicion_threshold,
        translate_model=selected_profile,
        gpu_layers=gpu_layers,
        context_size=context_size,
        translation_recipe=TRANSLATION_RECIPE,
        taiwan_terms=wording_identity(),
        dual_tracks=dual_tracks,
    )
    if device is not None or backend != 'cuda':
        server_file = Path(llama_server).resolve()
        stamp = server_file.stat()
        options['execution_backend'] = dict(backend=backend, device=device,
            executable=str(server_file), size=stamp.st_size, mtime_ns=stamp.st_mtime_ns)
    if sentence_mode:
        options.update(
            sentence_recipe='punctuation-proportional-v1',
            short_threshold=short_threshold,
            short_policy=short_policy,
        )
    if fixed_cues:
        options['fixed_cues'] = True
    options['sampling'] = dict(sorted(sampling.items()))
    from .quality import quality_fingerprint
    options['quality'] = quality_fingerprint()
    if translation_lookup is not None:
        options['translation_table'] = translation_lookup
    key = translation_key(
        source,
        model,
        model_path,
        show_interjections,
        batch_size,
        options,
        model_identity=model_identity,
    )
    result_path = out_dir / f'{model}.zh-TW.json'
    srt_path = out_dir / f'{model}.zh-TW.srt'

    def write_outputs(result):
        if dual_tracks:
            concise, full = render_tracks(result['segments'], render, on_fail=on_fail, suspicious=suspicious)
            atomic_write_text(srt_path, concise)
            atomic_write_text(out_dir / f'{model}.zh-TW.full.srt', full)
        else:
            atomic_write_text(srt_path, render(display_segments(result['segments'], on_fail, suspicious)))
    memo = _CallMemo(out_dir / f'{model}.translate.partial.json', key)
    if result_path.exists() and not force:
        result = json.loads(result_path.read_text('utf-8'))
        if result.get('cache_key') == key:
            write_outputs(result)
            if sentence_mode or fixed_cues:
                atomic_write_text(
                    out_dir / 'ja.srt',
                    render_sentence_srt(
                        [
                            Segment(r['start'], r['end'], r['text_ja'])
                            for r in result['segments']
                        ]
                    ),
                )
            report_phase("prepare_translate", 100, "翻譯結果已由快取準備完成", force=True)
            report_phase("translate", 100, "已從快取還原翻譯字幕", force=True)
            if callable(on_batch):
                on_batch([dict(row) for row in result['segments']])
            return result
    data = json.loads(Path(source).read_text('utf-8'))
    suspect_lookup = {}
    if secondary_data is not None:
        from .suspicion import mark_disagreement
        rows = mark_disagreement(
            data['segments'], secondary_data['segments'], suspicion_threshold
        )
        suspect_lookup = {(r['start'], r['end']): r for r in rows}
        if not any(r['asr_similarity'] is not None for r in rows):
            raise ValueError(
                'secondary ASR has no identical utterance spans; '
                'cannot compare grouped windows'
            )
    intervals = data.get('speech_intervals') if isinstance(data, dict) else None
    if intervals is not None:
        intervals = [
            (r['start'], r['end']) if isinstance(r, dict) else tuple(r)
            for r in intervals
        ]
    # ASR no-VAD runs retain their own raw output; cleanup still uses the shared VAD reference.
    original_segments = load_segments(data)
    rejected = []
    if sentence_mode and not dual_tracks:
        original_segments, rejected = filter_short(
            original_segments,
            short_threshold,
            short_policy,
        )
    if fixed_cues:
        segments = original_segments
        atomic_write_text(out_dir / 'ja.srt', render_sentence_srt(segments))
    elif sentence_mode:
        parents = original_segments
        segments = clean_segments(
            [cue for parent in parents for cue in sentence_cues(parent)],
            vad_intervals=intervals,
            show_interjections=True if dual_tracks else show_interjections,
            include_low_confidence=dual_tracks,
        )
        atomic_write_json(
            out_dir / 'sentence_cues.json',
            {
                'source': str(source),
                'options': options,
                'parents': [dict(start=s.start, end=s.end, text=s.text) for s in parents],
                'rejected_short': [
                    dict(start=s.start, end=s.end, text=s.text) for s in rejected
                ],
                'segments': [
                    asdict(s)
                    for s in segments
                ],
            },
        )
        atomic_write_text(out_dir / 'ja.srt', render_sentence_srt(segments))
    else:
        segments = clean_segments(
            original_segments,
            vad_intervals=intervals,
            show_interjections=True if dual_tracks else show_interjections,
            include_low_confidence=dual_tracks,
        )
    if dual_tracks and not fixed_cues:
        atomic_write_text(out_dir / 'ja.srt', render(segments))
    all_segments = segments
    segments = unique_segments(all_segments)

    def translated_rows(completed):
        by_text = {r.source.text: r for r in completed}
        rows = []
        for source in all_segments:
            translated = by_text.get(source.text)
            if translated is None:
                continue
            row = dict(
                start=source.start, end=source.end, text=translated.text,
                text_ja=source.text, preserved_original=translated.preserved_original,
                low_confidence=source.low_confidence, gap=source.gap,
                avg_logprob=source.avg_logprob, no_speech_prob=source.no_speech_prob,
                compression_ratio=source.compression_ratio,
            )
            paired = suspect_lookup.get((source.start, source.end))
            if paired:
                row.update({name: paired[name] for name in
                            ('suspected_asr', 'asr_similarity', 'suspicion_status')})
            rows.append(row)
        return rows

    def publish_batch(completed):
        on_batch(translated_rows(completed))

    if translation_lookup is not None:
        # Reattach an already translated video-wide table directly. Never send
        # these values through the model, conversion, or fallback machinery again.
        result = {
            'cache_key': key, 'model': model, 'translate_model': selected_profile,
            'recipe_options': options,
            'segments': [dict(
                start=s.start, end=s.end, text_ja=s.text,
                text=translation_lookup[s.text]['text'],
                preserved_original=bool(translation_lookup[s.text].get('preserved_original')),
                low_confidence=s.low_confidence, gap=s.gap, avg_logprob=s.avg_logprob,
                no_speech_prob=s.no_speech_prob, compression_ratio=s.compression_ratio,
            ) for s in all_segments],
            'metrics': {'input_sentences': len(all_segments), 'unique_sentences': len(segments),
                        'output_sentences': len(all_segments), 'shared_translation_table': True},
        }
        report_phase('prepare_translate', 100, '共用翻譯表已準備完成', force=True)
        report_phase('translate', 100, '已套用共用翻譯表', force=True)
    elif not segments:
        result = {'cache_key': key, 'model': model, 'translate_model': selected_profile,
                  'recipe_options': options, 'segments': [],
                  'metrics': {'input_sentences': 0, 'output_sentences': 0, 'status': 'no speech'}}
        report_phase(
            "prepare_translate",
            100,
            "翻譯準備完成；沒有需要模型處理的字幕片段",
            force=True,
        )
        report_phase("translate", 100, "沒有字幕片段需要翻譯", force=True)
    else:
        safety = (
            {'required': False, 'status': 'queue worker GPU admission'}
            if gpu_managed
            else assert_gpu_safe()
        )
        started = time.perf_counter()
        with (nullcontext(None) if backend == 'vulkan' else NvmlMemorySampler()) as memory:
            client_options = {
                "server_log_path": out_dir / f'{model}.server.log',
            }
            if device is not None or backend != 'cuda':
                client_options.update(device=device, backend=backend)
            if model == 'sakura' and profile.name != DEFAULT_TRANSLATE_MODEL:
                client_options['translate_model'] = profile.name
            if gpu_layers != 'auto':
                client_options['gpu_layers'] = gpu_layers
            if context_size != 8192:
                client_options['context_size'] = context_size
            if callable(on_phase):
                client_options["on_phase"] = report_phase
            if sampling:
                client_options.update(sampling)
            if shared_client is not None:
                # One llama-server serves several translations of the same model
                # (subtitle combos); the caller owns its lifecycle.
                expected_profile = profile.name if model == 'sakura' else 'qwen3'
                if getattr(shared_client, 'provider', None) != 'openai-compatible':
                    if (getattr(shared_client, 'model', model) != model
                            or getattr(shared_client, 'translate_model', expected_profile) != expected_profile):
                        raise ValueError('shared llama-server client uses a different model/profile')
                shared_before = dict(shared_client.metrics)
                client_context = nullcontext(shared_client)
            else:
                shared_before = None
                client_context = LlamaServerClient(
                    model,
                    model_path,
                    llama_server,
                    **client_options,
                )
            with client_context as client:
                report_phase(
                    "prepare_translate",
                    100,
                    "本機翻譯模型與伺服器已就緒",
                )
                report_phase(
                    "translate",
                    0,
                    f"開始翻譯 {len(segments)} 個字幕片段",
                    force=True,
                )
                responses = []
                history = []

                def history_text():
                    return '\n'.join(history[-5:]) if context_mode == 'user' else ''

                def batch_call(batch):
                    if prompt_mode == 'general-llm':
                        relevant_terms = filter_relevant_terms(
                            terms,
                            batch.numbered_text + ('\n' + batch.context_text if batch.context_text else ''),
                        )
                        prompt = general_llm_messages(
                            batch.numbered_text,
                            relevant_terms,
                            (batch.context_text if context_mode == 'user' else ''),
                        )
                        max_tokens = estimate_translation_max_tokens([cue for _, cue in batch.items])
                        raw = client.translate(prompt, max_tokens=max_tokens)
                        responses.append(
                            dict(ids=batch.expected_ids, messages=prompt, raw=raw)
                        )
                        atomic_write_json(
                            out_dir / f'{model}.responses.json',
                            responses,
                        )
                        values = parse_numbered_translations(raw, batch.expected_ids, strict=False)
                        if any(len(v) > 120 for v in values):
                            raise ValueError('Translation exceeds per-cue output limit')
                        return raw
                    if model == 'sakura' and prompt_mode != 'legacy':
                        japanese = (
                            '\n'.join(s.text for _, s in batch.items)
                            if prompt_mode == 'official-lines'
                            else batch.numbered_text
                        )
                        prompt = sakura_messages(
                            japanese,
                            terms,
                            (batch.context_text if profile.name == DEFAULT_TRANSLATE_MODEL
                             else history_text()) if context_mode == 'user' else '',
                            translate_model=profile.name,
                        )
                        raw = client.translate(prompt, max_tokens=len(batch.items) * 128)
                        responses.append(
                            dict(ids=batch.expected_ids, messages=prompt, raw=raw)
                        )
                        # Persist before validation so mismatched raw output remains inspectable.
                        atomic_write_json(
                            out_dir / f'{model}.responses.json',
                            responses,
                        )
                        text = (
                            parse_lines(raw, batch.expected_ids)
                            if prompt_mode == 'official-lines'
                            else raw
                        )
                        values = parse_numbered_translations(text, batch.expected_ids)
                        if any(len(v) > 120 for v in values):
                            raise ValueError('Translation exceeds per-cue output limit')
                        return text
                    instructions = (
                        '逐句翻成簡體中文。原樣保留編號，一句一行，只輸出本批譯文。'
                        '請翻譯完整意思，不保留任何日文假名。不要輸出指令或上下文。'
                    )
                    if batch.context_text:
                        instructions += '\n' + batch.context_text
                    text = client.translate(
                        [
                            {'role': 'system', 'content': instructions},
                            {'role': 'user', 'content': batch.numbered_text},
                        ],
                        max_tokens=len(batch.items) * 128,
                    )
                    values = parse_numbered_translations(text, batch.expected_ids)
                    if any(len(v) > 120 for v in values):
                        raise ValueError('Translation exceeds per-cue output limit')
                    return text

                def one_call(text):
                    if prompt_mode == 'general-llm':
                        relevant_terms = filter_relevant_terms(terms, text)
                        prompt = general_llm_single_messages(text, relevant_terms)
                        max_tokens = estimate_translation_max_tokens([text])
                        raw = client.translate(prompt, max_tokens=max_tokens)
                        responses.append(
                            dict(single_source=text, messages=prompt, raw=raw)
                        )
                        atomic_write_json(
                            out_dir / f'{model}.responses.json',
                            responses,
                        )
                        value = parse_numbered_translations(raw, (1,), strict=False)[0]
                        if len(value) > 120 or has_japanese_residue(value):
                            raise ValueError('Invalid single-cue translation')
                        return value
                    if model == 'sakura' and prompt_mode != 'legacy':
                        prompt = sakura_messages(
                            text if prompt_mode == 'official-lines' else '1. ' + text,
                            terms,
                            history_text() if profile.name != DEFAULT_TRANSLATE_MODEL else '',
                            translate_model=profile.name,
                        )
                        raw = client.translate(prompt, max_tokens=128)
                        responses.append(
                            dict(single_source=text, messages=prompt, raw=raw)
                        )
                        atomic_write_json(
                            out_dir / f'{model}.responses.json',
                            responses,
                        )
                        value = (
                            parse_numbered_translations(parse_lines(raw, (1,)), 1)[0]
                            if prompt_mode == 'official-lines'
                            else parse_numbered_translations(raw, 1)[0]
                        )
                        if len(value) > 120 or has_japanese_residue(value):
                            raise ValueError('Invalid single-cue translation')
                        return value
                    prompt = [
                        {
                            'role': 'system',
                            'content': (
                                '將日文完整翻成簡體中文，禁止保留日文假名。'
                                '只輸出 1. 譯文，不輸出指令或解釋。'
                            ),
                        },
                        {'role': 'user', 'content': '1. ' + text},
                    ]
                    value = parse_numbered_translations(
                        client.translate(prompt, max_tokens=128),
                        1,
                    )[0]
                    if len(value) > 120 or has_japanese_residue(value):
                        raise ValueError('Translation exceeds per-cue output limit')
                    return value

                def report_translation_progress(completed, total):
                    if on_progress:
                        on_progress(completed, total)
                    report_phase(
                        "translate",
                        100.0 * completed / max(1, total),
                        f"已翻譯 {completed}/{total} 個字幕片段",
                    )

                batch_call = memo.wrap(
                    'batch', batch_call,
                    lambda b: [b.numbered_text, b.context_text, list(b.expected_ids)] + (
                        [history_text()] if profile.name != DEFAULT_TRANSLATE_MODEL else []
                    ),
                )
                one_call = memo.wrap('one', one_call, lambda text: (
                    [text, history_text()] if profile.name != DEFAULT_TRANSLATE_MODEL else text
                ))
                if model == 'sakura' and profile.name != DEFAULT_TRANSLATE_MODEL:
                    # Restore context on memo hits as well as fresh calls. Context
                    # must contain translations, never the Japanese source rows.
                    cached_batch, cached_one = batch_call, one_call

                    def batch_call(batch):
                        value = cached_batch(batch)
                        history.extend(parse_numbered_translations(value, batch.expected_ids))
                        return value

                    def one_call(text):
                        value = cached_one(text)
                        history.append(value)
                        return value
                from .providers.http import ProviderError
                propagate = [ProviderError]
                if backend == 'vulkan':
                    propagate.extend([LlamaServerError, OSError, TimeoutError])
                is_cloud_llm = (
                    prompt_mode == 'general-llm' or
                    (hasattr(client, 'provider') and getattr(client, 'provider', None) == 'openai-compatible' and getattr(client, '_spec', {}).get('api_family') != 'azure-translator')
                )
                effective_batch_size = 12 if (batch_size == 30 and is_cloud_llm) else batch_size
                run = translate_segments(
                    segments,
                    batch_call,
                    one_call,
                    batch_size=effective_batch_size,
                    context_size=0 if context_mode == 'none' else 5,
                    on_progress=report_translation_progress,
                    propagate_errors=tuple(propagate),
                    tiered_retry=(prompt_mode == 'general-llm' or model_identity is not None),
                    refusal_threshold=0.20,
                    **({'on_batch': publish_batch} if callable(on_batch) else {}),
                )
                # Translation happens once per exact Japanese string. Reattach
                # every occurrence's timing and confidence to the shared result.
                by_text = {r.source.text: r for r in run}
                run = replace(run, results=tuple(
                    replace(by_text[s.text], source=s) for s in all_segments
                ))
                report_phase(
                    "translate",
                    100,
                    f"已完成 {len(run)} 個字幕片段的翻譯",
                )
                client_metrics = (
                    _metrics_delta(client.metrics, shared_before)
                    if shared_before is not None
                    else client.metrics
                )
                metrics = {**client_metrics, **run.metrics}
                if shared_before is not None:
                    metrics['shared_server'] = True
        metrics.update(
            elapsed_seconds_with_load=time.perf_counter() - started,
            input_sentences=len(segments),
            output_sentences=len(run),
            nvml=(memory.to_dict() if memory is not None else {
                'available': False, 'reason': 'NVML does not measure the Vulkan integrated GPU'}),
            softening_count=None,
            softening_note='Requires human comparison; cannot infer from numbered output',
        )
        metrics['suspected_japanese_residue'] = sum(
            has_japanese_residue(r.text) for r in run if not r.preserved_original
        )
        result = {
            'cache_key': key,
            'model': model,
            'translate_model': selected_profile,
            'gpu_safety': safety,
            'recipe_options': options,
            'segments': translated_rows(run),
            'metrics': metrics,
        }
    for row in result['segments']:
        paired = suspect_lookup.get((row['start'], row['end']))
        if paired:
            row.update(
                {
                    key: paired[key]
                    for key in ('suspected_asr', 'asr_similarity', 'suspicion_status')
                }
            )
    if model_identity is not None:
        total_segments = len(result['segments'])
        untranslated_count = sum(1 for row in result['segments'] if row.get('preserved_original'))
        if total_segments > 0 and (untranslated_count / total_segments) > 0.5:
            raise RuntimeError(
                f"Translation failed: {untranslated_count}/{total_segments} sentences remained untranslated ([原文]). "
                "Output not cached to prevent bad cache solidification."
            )

    atomic_write_json(result_path, result)
    memo.clear()
    write_outputs(result)
    if callable(on_batch) and (translation_lookup is not None or not segments):
        on_batch([dict(row) for row in result['segments']])
    return result


def compare_markdown(source, translations):
    def cell(value):
        return str(value).replace('|', '\\|').replace('\n', '<br>')
    names = list(translations)
    lines = ['# 字幕品質比較', '', '機器疑似幻覺與拒翻計數僅供篩選；正確性、淡化與台詞語氣需聽音確認。', '',
             '| 時間碼 | 日文 | ' + ' | '.join(names) + ' |',
             '| --- | --- | ' + ' | '.join('---' for _ in names) + ' |']
    # Match source timestamps/text rather than assuming cleanup preserves all row indices.
    lookups = {
        name: {
            (row['start'], row['end'], row.get('text_ja')): row['text']
            for row in data['segments']
        }
        for name, data in translations.items()
    }
    span_lookups = {
        name: {
            (row['start'], row['end']): row['text']
            for row in data['segments']
        }
        for name, data in translations.items()
    }
    for segment in load_segments(source):
        key = (segment.start, segment.end, segment.text)
        values = [
            lookups[name].get(
                key,
                span_lookups[name].get(
                    (segment.start, segment.end),
                    '（過濾／未翻譯）',
                ),
            )
            for name in names
        ]
        lines.append(
            '| '
            + ' | '.join(
                map(cell, [format_timestamp(segment.start), segment.text, *values])
            )
            + ' |'
        )
    return '\n'.join(lines) + '\n'


def mux_subtitles(video, subtitles, output, seconds=120):
    """Stream copy a short local preview with named soft subtitle tracks."""
    video, output = Path(video).resolve(), Path(output).resolve()
    if video == output:
        raise ValueError('Preview output must differ from input')
    if output.exists():
        raise FileExistsError(f'Preview already exists: {output}')
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [resolve_ffmpeg(), '-nostdin', '-hide_banner', '-loglevel', 'error', '-i', str(video)]
    for _, path, _ in subtitles:
        command += ['-i', str(Path(path).resolve())]
    command += ['-map', '0:v:0', '-map', '0:a:0?']
    for index in range(len(subtitles)):
        command += ['-map', f'{index + 1}:0']
    command += ['-t', str(seconds), '-c', 'copy']
    for index, (title, _, language) in enumerate(subtitles):
        command += [f'-metadata:s:s:{index}', f'language={language}',
                    f'-metadata:s:s:{index}', f'title={title}']
        command += [
            f'-disposition:s:{index}',
            'default' if index == 1 else '0',
        ]
    command += [str(output)]
    subprocess.run(command, check=True)
    return output


def run_matrix(args):
    """Use subprocesses so each ASR releases its complete CUDA context before LLMs."""
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    records = []
    for model in ('anime-whisper', 'kotoba', 'turbo'):
        for use_vad in (True, False):
            name = model + ('-vad' if use_vad else '-no-vad')
            folder = root / name
            command = [sys.executable, '-m', 'oh_streamer.subtitles', 'poc', args.video,
                       '--asr', model, '--out', str(folder), '--models', args.models,
                       '--cache', str(root / 'cache')]
            if not use_vad:
                command += ['--no-vad']
            if args.force:
                command += ['--force']
            environment = os.environ.copy()
            environment['PYTHONPATH'] = (
                str(Path(__file__).resolve().parents[2])
                + os.pathsep
                + environment.get('PYTHONPATH', '')
            )
            subprocess.run(command, check=True, env=environment)
            records.append((name, folder, json.loads((folder / 'ja.json').read_text('utf-8'))))
    # Complete all six outputs for both translation backends, without choosing a quality winner.
    translation_metrics = {}
    for name, folder, data in records:
        translations = {}
        for model, filename in (('sakura', 'sakura-7b-qwen2.5-v1.0-iq4xs.gguf'),
                                ('qwen3', 'Qwen3-8B-Q4_K_M.gguf')):
            translations[model] = translate_file(
                folder / 'ja.json',
                folder,
                model=model,
                model_path=Path(args.models) / model / filename,
                llama_server=args.llama_server,
                force=args.force or getattr(args, 'translate_force', False),
            )
        atomic_write_text(folder / 'compare.md', compare_markdown(data, translations))
        translation_metrics[name] = {
            model: data['metrics'] for model, data in translations.items()
        }
        preview = folder / 'preview.mkv'
        if not preview.exists():
            mux_subtitles(
                args.video,
                [
                    ('日本語（AI）', folder / 'ja.srt', 'jpn'),
                    ('繁體中文（AI Sakura）', folder / 'sakura.zh-TW.srt', 'chi'),
                    ('繁體中文（AI Qwen3）', folder / 'qwen3.zh-TW.srt', 'chi'),
                ],
                preview,
            )
    refresh_reports(root)


def refresh_reports(root):
    """Recompute CPU-only quality flags and comparisons, preserving measured inference times."""
    root = Path(root)
    asr_metrics, translation_metrics = {}, {}
    for source in sorted(root.glob('*/ja.json')):
        data = json.loads(source.read_text('utf-8'))
        intervals = [tuple(r) for r in data['speech_intervals']]
        data['metrics']['hallucinations'] = inspect_hallucinations(
            load_segments(data),
            vad_intervals=intervals,
        )
        data['metrics']['max_raw_span_seconds'] = max(
            (segment.end - segment.start for segment in load_segments(data)),
            default=0,
        )
        atomic_write_json(source, data)
        asr_metrics[source.parent.name] = data['metrics']
        translations = {}
        for target in sorted(source.parent.glob('*.zh-TW.json')):
            translated = json.loads(target.read_text('utf-8'))
            translated['metrics']['suspected_japanese_residue'] = sum(
                has_japanese_residue(row['text'])
                for row in translated['segments']
                if not row.get('preserved_original')
            )
            for row in translated['segments']:
                if not row.get('preserved_original') and has_japanese_residue(row['text']):
                    row['review_flags'] = ['japanese_residue']
            atomic_write_json(target, translated)
            translations[translated['model']] = translated
        if translations:
            atomic_write_text(
                source.parent / 'compare.md',
                compare_markdown(data, translations),
            )
            translation_metrics[source.parent.name] = {
                model: data['metrics'] for model, data in translations.items()
            }
    atomic_write_json(
        root / 'matrix.metrics.json',
        {'asr': asr_metrics, 'translation': translation_metrics},
    )
    atomic_write_text(
        root / 'measurements.md',
        measurements_markdown(asr_metrics, translation_metrics),
    )


def measurements_markdown(asr_metrics, translation_metrics):
    def value(v):
        return (
            '未量測'
            if v is None
            else str(round(v, 2) if isinstance(v, float) else v)
        )
    def peak(metrics):
        n = metrics.get('nvml', {}).get('peak_bytes')
        return None if n is None else round(n / (1024 * 1024), 1)
    lines = [
        '# PoC 實測',
        '',
        '峰值為裝置整體 NVML 每 0.1 秒取樣；非程序專屬，可能漏掉瞬間尖峰。',
        '',
        '| ASR/VAD | 秒（含準備/VAD） | 即時倍率 | 峰值 MiB | 原始片段數 | 疑似幻覺 | '
        '黑名單 | 連續重複 | ≥3 次 | 非語音 | n-gram重複 |',
        '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |',
    ]
    for name, metrics in asr_metrics.items():
        h = metrics.get('hallucinations', {})
        row = [
            name,
            metrics.get('wall_seconds'),
            metrics.get('realtime_factor_wall'),
            peak(metrics),
            metrics.get('raw_segment_count'),
            *[
                h.get(key)
                for key in (
                    'unique_suspect',
                    'blacklist',
                    'consecutive_repeat',
                    'frequent_repeat',
                    'outside_vad',
                    'repeated_ngram',
                )
            ],
        ]
        lines.append('| ' + ' | '.join(map(value, row)) + ' |')
    lines += [
        '',
        '| ASR/VAD + LLM | 秒（含載入） | tokens/s（請求） | 峰值 MiB | '
        '編號/長度不符批次 | 退回逐句 | 保留原文 | 疑似日文殘留 | 疑似拒翻 | 淡化 |',
        '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |',
    ]
    for name, models in translation_metrics.items():
        for model, m in models.items():
            row = [
                name + ' + ' + model,
                m.get('elapsed_seconds_with_load'),
                m.get('tokens_per_second'),
                peak(m),
                m.get('mismatch_batches'),
                m.get('fallback_sentences'),
                m.get('untranslated'),
                m.get('suspected_japanese_residue'),
                m.get('suspected_refusals'),
                '待人工聽音',
            ]
            lines.append('| ' + ' | '.join(map(value, row)) + ' |')
    lines += [
        '',
        '建議組合：等待使用者對 compare.md 與 preview.mkv 做品質確認；'
        '疑似幻覺少不代表辨識正確。',
    ]
    return '\n'.join(lines) + '\n'


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    poc = sub.add_parser('poc', help='Japanese ASR only; releases GPU at process exit')
    poc.add_argument('video')
    poc.add_argument('--asr', choices=['anime-whisper', 'kotoba', 'turbo', 'whisper-ja-760M',
                                       'whisper-ja-1.5B', 'whisper-large-v3', 'qwen3-asr-1.7b'],
                     default=None, help='Explicit diagnostic single-model run')
    poc.add_argument('--asr-mode', choices=['dual', 'single'], default='dual',
                     help='Production ASR: A1+A2 (default), or A1 only')
    poc.add_argument('--no-vad', action='store_true')
    poc.add_argument(
        '--segmentation',
        choices=['grouped', 'utterance'],
        default='grouped',
    )
    poc.add_argument('--vad-threshold', type=float, default=0.3)
    poc.add_argument(
        '--vad-rescue-chunk-seconds',
        type=float,
        default=60.0,
        help='Independent chunk VAD adds only missing intervals; 0 disables rescue',
    )
    poc.add_argument('--models', default=str(default_model_path('anime-whisper').parent))
    poc.add_argument('--out', required=True)
    poc.add_argument('--cache', help='Audio/progress cache; default is inside --out')
    poc.add_argument('--force', action='store_true')
    translate = sub.add_parser('translate', help='Translate a cached Japanese JSON, without rerunning ASR')
    translate.add_argument('source')
    translate.add_argument('--model', required=True, help='Local GGUF file')
    translate.add_argument('--backend', choices=['sakura', 'qwen3'], default='sakura')
    from .glossary import DEFAULT_TRANSLATE_MODEL, SAKURA_PROFILES
    translate.add_argument('--translate-model', choices=[*SAKURA_PROFILES, 'v1.0'],
                           default=None, help='Sakura profile (default: GalTransl 7B v3.7)')
    translate.add_argument('--reproducible', action='store_true',
                           help='fixed sampling seed and llama.cpp prompt cache off (identical reruns)')
    translate.add_argument('--gpu-layers', default='auto',
                           type=lambda value: value if value in ('auto', 'all') else int(value))
    translate.add_argument('--context-size', type=int, default=8192)
    translate.add_argument('--fixed-cues', action='store_true',
                           help='Translate exact source cue rows without cleanup or segmentation')
    translate.add_argument('--llama-server', required=True)
    translate.add_argument('--out', required=True)
    translate.add_argument('--batch-size', type=int, choices=range(20, 41), default=30)
    translate.add_argument('--show-interjections', action='store_true')
    translate.add_argument('--force', action='store_true')
    translate.add_argument(
        '--prompt-mode',
        choices=['legacy', 'official-lines', 'official-numbered'],
        default='official-numbered',
    )
    translate.add_argument('--context-mode', choices=['none', 'user'], default='none')
    translate.add_argument('--glossary')
    translate.add_argument('--on-fail', choices=['hide', 'japanese'], default='hide')
    translate.add_argument(
        '--secondary-asr',
        help='Optional cached second ASR; identical VAD boundaries required',
    )
    translate.add_argument('--suspicious', choices=['keep', 'mark', 'hide'], default='keep')
    translate.add_argument('--suspicion-threshold', type=float, default=.45)
    translate.add_argument(
        '--sentence-cues',
        action='store_true',
        help='Split each utterance BEFORE translation',
    )
    translate.add_argument(
        '--short-threshold',
        type=float,
        default=1.0,
        help='Original VAD span duration in seconds',
    )
    translate.add_argument(
        '--short-policy',
        choices=['keep', 'whitelist', 'drop'],
        default='whitelist',
    )
    matrix = sub.add_parser('matrix', help='All 6 ASR runs, both LLMs, comparisons and 2-minute mux previews')
    matrix.add_argument('video')
    matrix.add_argument('--models', required=True)
    matrix.add_argument('--llama-server', required=True)
    matrix.add_argument('--out', required=True)
    matrix.add_argument('--force', action='store_true')
    matrix.add_argument(
        '--translate-force',
        action='store_true',
        help='Rerun translation while retaining ASR caches',
    )
    mux = sub.add_parser('mux')
    mux.add_argument('video')
    mux.add_argument('--ja', required=True)
    mux.add_argument('--zh', required=True)
    mux.add_argument('--out', required=True)
    mux.add_argument('--seconds', type=float, default=120)
    args = parser.parse_args(arguments)
    try:
        if args.command == 'poc':
            from .asr import run_asr
            if args.asr is None:
                from .pipeline import run_production_asr
                data = run_production_asr(args.video, args.out, models_dir=args.models,
                    asr_mode=args.asr_mode, force=args.force, vad_threshold=args.vad_threshold,
                    vad_rescue_chunk_seconds=args.vad_rescue_chunk_seconds)
                print(json.dumps(data['metrics'], ensure_ascii=False, indent=2))
                return 0
            data = run_asr(
                args.video,
                args.out,
                model=args.asr,
                use_vad=not args.no_vad,
                model_path=Path(args.models) / args.asr,
                force=args.force,
                segmentation=args.segmentation,
                vad_threshold=args.vad_threshold,
                vad_rescue_chunk_seconds=args.vad_rescue_chunk_seconds,
                cache_dir=args.cache or Path(args.out) / 'cache',
            )
            print(json.dumps(data['metrics'], ensure_ascii=False, indent=2))
        elif args.command == 'translate':
            result = translate_file(
                args.source,
                args.out,
                model=args.backend,
                translate_model=args.translate_model or (
                    'sakura-galtransl-7b-v3.7' if args.backend == 'sakura' else DEFAULT_TRANSLATE_MODEL),
                sampling=({'seed': 42, 'cache_prompt': False} if args.reproducible else None),
                gpu_layers=args.gpu_layers,
                context_size=args.context_size,
                fixed_cues=args.fixed_cues,
                model_path=args.model,
                llama_server=args.llama_server,
                batch_size=args.batch_size,
                show_interjections=args.show_interjections,
                force=args.force,
                prompt_mode=args.prompt_mode,
                glossary=args.glossary,
                context_mode=args.context_mode,
                on_fail=args.on_fail,
                secondary_asr=args.secondary_asr,
                suspicious=args.suspicious,
                suspicion_threshold=args.suspicion_threshold,
                sentence_mode=args.sentence_cues,
                short_threshold=args.short_threshold,
                short_policy=args.short_policy,
            )
            print(json.dumps(result['metrics'], ensure_ascii=False, indent=2))
        elif args.command == 'matrix':
            run_matrix(args)
        elif args.command == 'mux':
            mux_subtitles(args.video, [('日本語（AI）', args.ja, 'jpn'),
                          ('繁體中文（AI）', args.zh, 'chi')], args.out, args.seconds)
        return 0
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as exc:
        print(f'Subtitle PoC: {exc}', file=sys.stderr)
        return 1
