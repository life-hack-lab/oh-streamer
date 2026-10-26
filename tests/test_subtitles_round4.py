import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock
from copy import deepcopy
import json
import shutil
import subprocess

from oh_streamer.subtitles import round4
from oh_streamer.subtitles.round4 import compare_reproduction, measurement, checkpoint_markdown, verify_preview


class Round4Tests(unittest.TestCase):
    def result(self, text='部長'):
        return {'segments': [dict(start=1., end=2., text_ja='部長', text=text,
                                  preserved_original=False)], 'metrics': {}}

    def test_reproduction_reports_changes_without_quality_claim(self):
        result = compare_reproduction(self.result(), self.result('社長'))
        self.assertEqual(result['changed_cues'], 1)
        self.assertEqual(result['changes'][0]['baseline'], '部長')

    def test_misaligned_replay_rejected(self):
        replay = self.result()
        replay['segments'][0]['end'] = 3.
        with self.assertRaises(ValueError):
            compare_reproduction(self.result(), replay)

    def test_unknown_vram_is_not_zero(self):
        self.assertIsNone(measurement(self.result())['peak_mib'])
        result = self.result()
        result['metrics']['nvml'] = {'peak_bytes': 1048576}
        self.assertEqual(measurement(result)['peak_mib'], 1)

    def test_checkpoint_table_escapes_text(self):
        table = checkpoint_markdown({'a': self.result('部|長\n你好'), 'b': self.result()})
        self.assertIn('部\\|長<br>你好', table)

    def inputs(self):
        recipe = dict(prompt_mode='official-numbered', context_mode='none', secondary_asr=None,
                      suspicion_threshold=.45, sentence_recipe='punctuation-proportional-v1',
                      short_threshold=1., short_policy='whitelist', glossary=[])
        baseline = self.result()
        baseline['recipe_options'] = deepcopy(recipe)
        source = dict(segments=[dict(start=1., end=2., text='部長')], options=recipe)
        return source, baseline

    def test_preflight_validates_input_and_reports_historical_provenance_limit(self):
        source, baseline = self.inputs()
        result = round4.preflight_fixed_input(source, baseline)
        self.assertEqual(result['fixed_input_cues'], 1)
        self.assertEqual(result['historical_unrecorded_settings'], ['batch_size', 'context_size'])
        baseline['recipe_options'].update(batch_size=30, context_size=8192)
        self.assertEqual(round4.preflight_fixed_input(source, baseline)['historical_unrecorded_settings'], [])

    def test_preflight_rejects_input_recipe_and_recorded_settings_drift(self):
        for mutation in (
            lambda s, b: s['segments'][0].update(text='課長'),
            lambda s, b: s['segments'][0].update(start=.5),
            lambda s, b: b['recipe_options'].update(prompt_mode='legacy'),
            lambda s, b: b['recipe_options'].update(context_mode='user'),
            lambda s, b: b['recipe_options'].update(context_size=4096),
            lambda s, b: b['recipe_options'].update(batch_size=20),
            lambda s, b: s['options'].update(glossary=[dict(src='a', dst='b')]),
        ):
            with self.subTest(mutation=mutation):
                source, baseline = self.inputs()
                mutation(source, baseline)
                with self.assertRaises(ValueError):
                    round4.preflight_fixed_input(source, baseline)

    def test_main_input_drift_stops_before_translation(self):
        source, baseline = self.inputs()
        source['segments'][0]['text'] = 'changed'
        with TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as folder:
            root = Path(folder)
            (root / 'sakura.zh-TW.json').write_text(json.dumps(baseline), 'utf-8')
            (root / 'sentence_cues.json').write_text(json.dumps(source), 'utf-8')
            with mock.patch.object(round4, 'translate_file') as translate:
                with self.assertRaisesRegex(ValueError, 'Fixed Japanese input'):
                    round4.main(['--round3', str(root), '--out', str(root / 'out'),
                                 '--models', str(root), '--llama-server', 'unused'])
                translate.assert_not_called()

    def test_main_passes_explicit_fixed_recipe(self):
        source, baseline = self.inputs()
        with TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as folder:
            root = Path(folder)
            (root / 'sakura.zh-TW.json').write_text(json.dumps(baseline), 'utf-8')
            (root / 'sentence_cues.json').write_text(json.dumps(source), 'utf-8')
            with mock.patch('oh_streamer.subtitles.glossary.load_glossary', return_value=[]), \
                    mock.patch.object(round4, 'translate_file', return_value=self.result()) as translate, \
                    mock.patch.object(round4, 'prepare_report'):
                round4.main(['--round3', str(root), '--out', str(root / 'out'),
                             '--models', str(root), '--llama-server', 'unused', '--only-7b'])
                self.assertEqual(translate.call_count, 2)
                for call in translate.call_args_list:
                    self.assertTrue(call.kwargs['fixed_cues'])
                    self.assertEqual(call.kwargs['batch_size'], 30)
                    self.assertEqual(call.kwargs['context_size'], 8192)
                    self.assertEqual(call.kwargs['context_mode'], 'none')
                    self.assertEqual(call.kwargs['prompt_mode'], 'official-numbered')
                manifest = json.loads((root / 'out' / 'experiment.json').read_text('utf-8'))
                self.assertEqual(manifest['fixed_input_cues'], 1)
                self.assertIsInstance(manifest['blind_seed'], int)


class PreviewVerificationTests(unittest.TestCase):
    srt = ('1\n00:00:00,200 --> 00:00:00,400\n第一句\n\n'
           '2\n00:00:00,500 --> 00:00:00,700\n中間句\n\n'
           '3\n00:00:00,800 --> 00:00:01,100\n最後句\n')

    def setUp(self):
        self.folder = TemporaryDirectory(dir=Path(__file__).resolve().parents[1])
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.source, self.preview, self.subtitle = [self.root / name for name in ('source.mkv', 'preview.mkv', 'expected.srt')]
        self.source.write_bytes(b'source')
        self.preview.write_bytes(b'preview')
        self.subtitle.write_text(self.srt, 'utf-8')
        self.source_streams = [
            dict(index=0, codec_type='video', codec_name='h264', width=1280, height=720, start_time='0'),
            dict(index=1, codec_type='audio', codec_name='aac', sample_rate='48000', channels=2, start_time='0'),
        ]
        self.output_streams = [dict(row) for row in self.source_streams]
        self.output_streams.append(dict(index=2, codec_type='subtitle', codec_name='subrip'))
        self.extracted = self.srt

    def command(self, arguments):
        if '-show_streams' in arguments:
            streams = self.source_streams if arguments[-1] == str(self.source.resolve()) else self.output_streams
            return json.dumps(dict(streams=streams))
        self.assertIn('-copyts', arguments)
        self.assertEqual(arguments[arguments.index('-map') + 1], '0:2')
        return self.extracted

    def verify(self, subtitles=None):
        with mock.patch.object(round4, 'resolve_ffmpeg', return_value='ffmpeg'), \
                mock.patch.object(round4.shutil, 'which', return_value='ffprobe'), \
                mock.patch.object(round4, '_media_command', side_effect=self.command):
            return verify_preview(self.source, self.preview, subtitles or [self.subtitle])

    def test_all_streams_and_every_cue_verified(self):
        result = self.verify()
        self.assertTrue(result['passed'], result['errors'])
        self.assertEqual(result['subtitle_tracks'][0]['expected_cues'], 3)
        self.assertEqual(result['subtitle_tracks'][0]['extracted_cues'], 3)
        self.assertEqual(len(result['av_streams']), 2)
        self.assertEqual(len(result['subtitle_tracks'][0]['expected_sha256']), 64)

    def test_interior_text_mismatch_detected(self):
        self.extracted = self.srt.replace('中間句', '錯誤句')
        result = self.verify()
        self.assertFalse(result['passed'])
        self.assertEqual(result['subtitle_tracks'][0]['mismatched_cues'],
                         [dict(cue=2, reasons=['text'], timing_error_seconds=0.)])

    def test_interior_timing_mismatch_detected(self):
        self.extracted = self.srt.replace('00:00:00,500', '00:00:00,510')
        result = self.verify()
        self.assertFalse(result['passed'])
        self.assertEqual(result['subtitle_tracks'][0]['mismatched_cues'][0]['reasons'], ['timestamps'])

    def test_small_consistent_mux_shift_and_rounding_accepted(self):
        for row in self.output_streams[:2]:
            row['start_time'] = '.021'
        for before, after in (('200', '221'), ('400', '421'), ('500', '521'),
                              ('700', '721'), ('800', '821'), ('100', '121')):
            self.extracted = self.extracted.replace(',' + before, ',' + after)
        result = self.verify()
        self.assertTrue(result['passed'], result['errors'])
        self.assertAlmostEqual(result['mux_offset_seconds'], .021)

    def test_shift_must_agree_with_av_and_remain_bounded(self):
        self.output_streams[0]['start_time'] = '.021'
        self.assertFalse(self.verify()['passed'])
        for row in self.output_streams[:2]:
            row['start_time'] = '.3'
        self.assertIn('100 ms', ' '.join(self.verify()['errors']))

    def test_unknown_start_times_allow_unshifted_cues(self):
        for row in self.source_streams + self.output_streams:
            row.pop('start_time', None)
        self.assertTrue(self.verify()['passed'])
        self.extracted = self.srt.replace('00:00:00,500', '00:00:00,510')
        self.assertFalse(self.verify()['passed'])

    def test_wrong_audio_video_codec_and_missing_track_rejected(self):
        self.output_streams[0]['codec_name'] = 'hevc'
        self.output_streams[1]['codec_name'] = 'opus'
        result = self.verify()
        self.assertFalse(result['passed'])
        self.assertTrue(any('video codec_name differs' in e for e in result['errors']))
        self.assertTrue(any('audio codec_name differs' in e for e in result['errors']))
        self.output_streams.pop()
        self.assertTrue(any('subtitle track count' in e for e in self.verify()['errors']))

    def test_missing_middle_cue_and_extra_track_rejected(self):
        self.extracted = self.srt.replace('2\n00:00:00,500 --> 00:00:00,700\n中間句\n\n', '')
        self.assertFalse(self.verify()['passed'])
        self.extracted = self.srt
        self.output_streams.append(dict(index=3, codec_type='subtitle', codec_name='subrip'))
        self.assertFalse(self.verify()['passed'])

    def test_probe_failure_is_a_failed_serializable_report(self):
        with mock.patch.object(round4, 'resolve_ffmpeg', side_effect=FileNotFoundError('missing ffmpeg')):
            result = verify_preview(self.source, self.preview, [self.subtitle])
        self.assertFalse(result['passed'])
        self.assertIn('missing ffmpeg', result['errors'][0])
        json.dumps(result, allow_nan=False)

    def test_prepare_report_persists_verification_and_raises_on_failure(self):
        source = dict(segments=[dict(start=0., end=1., text_ja='部長', text='部長')], metrics={})
        results = {'sakura-v1': source, 'other': source}
        for passed in (True, False):
            with self.subTest(passed=passed), \
                    mock.patch.object(round4, 'read_json', return_value=source), \
                    mock.patch('oh_streamer.subtitles.blind_ab.write_blind_review'), \
                    mock.patch.object(round4, 'mux_subtitles'), \
                    mock.patch.object(round4, 'verify_preview', return_value=dict(passed=passed, errors=[])) as verify:
                if passed:
                    round4.prepare_report(self.root, self.root, results, video=self.source)
                else:
                    with self.assertRaisesRegex(ValueError, 'artifact verification'):
                        round4.prepare_report(self.root, self.root, results, video=self.source)
                saved = json.loads((self.root / 'artifact_verification.json').read_text('utf-8'))
                self.assertEqual(saved['passed'], passed)
                self.assertEqual(len(verify.call_args.args[2]), 2)

    def test_prepare_report_reuses_existing_preview_only_after_full_verification(self):
        source = dict(segments=[dict(start=0., end=1., text_ja='部長', text='部長')], metrics={})
        preview = self.root / 'quality_review.mkv'
        preview.write_bytes(b'existing preview')
        for passed in (True, False):
            with self.subTest(passed=passed), \
                    mock.patch.object(round4, 'read_json', return_value=source), \
                    mock.patch('oh_streamer.subtitles.blind_ab.write_blind_review'), \
                    mock.patch.object(round4, 'mux_subtitles') as mux, \
                    mock.patch.object(round4, 'verify_preview', return_value=dict(passed=passed, errors=[])) as verify:
                if passed:
                    round4.prepare_report(self.root, self.root, {'sakura-v1': source, 'other': source}, video=self.source)
                else:
                    with self.assertRaises(ValueError):
                        round4.prepare_report(self.root, self.root, {'sakura-v1': source, 'other': source}, video=self.source)
                mux.assert_not_called()
                verify.assert_called_once()
                self.assertEqual(preview.read_bytes(), b'existing preview')

    def test_report_retry_preserves_blind_mapping_and_votes_binding(self):
        source = dict(segments=[dict(start=0., end=1., text_ja='部長', text='部長')], metrics={})
        (self.root / 'sakura.zh-TW.json').write_text(json.dumps(source), 'utf-8')
        results = {'sakura-v1': source, 'other': source}
        round4.prepare_report(self.root, self.root, results)
        key = (self.root / 'answer_key.json').read_bytes()
        page = (self.root / 'blind_ab.html').read_bytes()
        seed = json.loads((self.root / 'experiment.json').read_text('utf-8'))['blind_seed']
        round4.prepare_report(self.root, self.root, results)
        self.assertEqual((self.root / 'answer_key.json').read_bytes(), key)
        self.assertEqual((self.root / 'blind_ab.html').read_bytes(), page)
        self.assertEqual(json.loads((self.root / 'experiment.json').read_text('utf-8'))['blind_seed'], seed)
        round4.prepare_report(self.root, self.root, results, seed=1234)
        self.assertEqual(json.loads((self.root / 'experiment.json').read_text('utf-8'))['blind_seed'], 1234)

    @unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'ffmpeg/ffprobe unavailable')
    def test_real_cpu_mux_extracts_all_tracks(self):
        second = self.root / 'second.srt'
        second.write_text(self.srt.replace('句', '行'), 'utf-8')
        # Tiny synthetic CPU-only fixture: no model, GPU, network or user media.
        command = [round4.resolve_ffmpeg(), '-nostdin', '-v', 'error', '-f', 'lavfi', '-i',
                   'color=c=black:size=64x48:rate=10', '-f', 'lavfi', '-i',
                   'anullsrc=r=48000:cl=stereo', '-t', '1.5', '-c:v', 'mpeg4', '-threads', '1',
                   '-c:a', 'pcm_s16le', '-y', str(self.source)]
        subprocess.run(command, check=True, capture_output=True, timeout=30)
        self.preview.unlink()
        round4.mux_subtitles(self.source, [('one', self.subtitle, 'chi'), ('two', second, 'chi')],
                             self.preview, seconds=2)
        result = verify_preview(self.source, self.preview, [self.subtitle, second])
        self.assertTrue(result['passed'], result['errors'])
        self.assertEqual([r['extracted_cues'] for r in result['subtitle_tracks']], [3, 3])


if __name__ == '__main__':
    unittest.main()
