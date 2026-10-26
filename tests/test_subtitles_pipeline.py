"""CPU model-process seams for production dual/single ASR and independent caches."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from oh_streamer.subtitles import pipeline, fusion, runtime
from oh_streamer.subtitles import cli, combos, environment
from oh_streamer.subtitles.quality import load_quality_config


class ProductionASRTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.models = self.root / 'models'
        for alias in (pipeline.PRIMARY_ASR, pipeline.SECONDARY_ASR):
            path = self.models / runtime.model_dir_name(alias)
            path.mkdir(parents=True)
            (path/'config.json').write_text('{}', 'utf-8')
            (path/'model.safetensors').write_bytes(b'weights')
        self.rows_a1 = [dict(start=0, end=1, text='こんにちは', avg_logprob=-.3, gap=False),
                        dict(start=1, end=2, text='大丈夫です', avg_logprob=-.9, no_speech_prob=.2,
                             compression_ratio=1.0, gap=True),
                        dict(start=2, end=3, text='', gap=False)]
        self.rows_a2 = [dict(start=0, end=1, text='こんにちは', avg_logprob=-.2, gap=False),
                        dict(start=1, end=2, text='大丈夫です', avg_logprob=-.4, no_speech_prob=.01,
                             compression_ratio=1.0, gap=True),
                        dict(start=2, end=3, text='ありがとう', avg_logprob=-.2, gap=False)]
        self.events, self.jobs, self.prepare_calls = [], [], []
        self.fail = set()

    def plan(self, source, folder, **kwargs):
        self.prepare_calls.append(kwargs)
        return dict(plan_key='spans', settings=dict(source_key='audio',
            vad=dict(threshold=kwargs.get('vad_threshold')), gap_fill=kwargs['quality_config']['gap_fill']),
            audio_path=str(self.root/'shared.wav'), audio_duration_seconds=3,
            source={'path': str(source)}, speech_intervals=[dict(start=0, end=1)],
            groups=[dict(start=i, end=i+1, gap=i==1) for i in range(3)])

    def child(self, job, callback):
        alias = job['asr']
        self.events.append(('start', alias))
        self.jobs.append(job)
        if alias in self.fail:
            self.events.append(('exit', alias))
            raise RuntimeError(f'{alias} fixture failure')
        rows = copy.deepcopy(self.rows_a1 if alias==pipeline.PRIMARY_ASR else self.rows_a2)
        target = Path(job['folder'])/'ja.json'
        runtime.atomic_write_json(target, dict(raw_segments=rows, segments=[r for r in rows if r['text']],
                                               metrics={'fixture': True}))
        self.events.append(('exit', alias))

    def run_pipeline(self, **kwargs):
        return pipeline.run_production_asr('source.mp4', self.root/'out', models_dir=self.models,
            child_runner=self.child, prepare_plan=self.plan, gpu_managed=True, **kwargs)

    def test_dual_runs_once_per_model_after_previous_exit_then_fuses_exact_raw_spans(self):
        real_fusion = fusion.fuse_segments

        def fuse(*args, **kwargs):
            self.events.append(('fusion', 'cpu'))
            return real_fusion(*args, **kwargs)

        with patch.object(fusion, 'fuse_segments', side_effect=fuse):
            result = self.run_pipeline()
        self.assertEqual(self.events, [('start', pipeline.PRIMARY_ASR), ('exit', pipeline.PRIMARY_ASR),
            ('start', pipeline.SECONDARY_ASR), ('exit', pipeline.SECONDARY_ASR), ('fusion', 'cpu')])
        self.assertEqual(len(self.prepare_calls), 1)
        self.assertEqual(self.jobs[0]['shared_plan'], self.jobs[1]['shared_plan'])
        self.assertEqual(self.jobs[0]['generate_overrides'], {'no_repeat_ngram_size': 6})
        self.assertEqual(result['segments'][2]['text'], 'ありがとう')
        self.assertEqual(result['segments'][2]['selected_model'], 'A2')
        self.assertFalse(result['segments'][1]['low_confidence'])  # same-text rescue
        self.assertEqual(result['segments'][1]['alt_text'], '大丈夫です')

    def test_single_runs_only_a1_and_disables_agreement_rescue(self):
        self.rows_a1[1].update(alt_text='大丈夫です', asr_agreement=1.0)
        result = self.run_pipeline(asr_mode='single')
        self.assertEqual([j['asr'] for j in self.jobs], [pipeline.PRIMARY_ASR])
        self.assertTrue(result['segments'][1]['low_confidence'])
        self.assertIsNone(result['segments'][1]['alt_text'])
        self.assertIsNone(result['segments'][1]['asr_agreement'])

    def test_model_failures_fall_back_to_available_model_and_record_reason(self):
        for failed, expected in ((pipeline.PRIMARY_ASR, pipeline.SECONDARY_ASR),
                                 (pipeline.SECONDARY_ASR, pipeline.PRIMARY_ASR)):
            with self.subTest(failed=failed):
                self.fail = {failed}
                result = self.run_pipeline(force=True)
                self.assertEqual(result['model'], expected)
                self.assertTrue(result['asr_warnings'])
                self.assertIn('fixture failure', result['asr_warnings'][0])
                self.assertTrue(result['metrics']['fallback'])
                self.assertTrue(all(not r.get('alt_text') for r in result['segments']))

    def test_both_failures_and_single_a1_failure_are_clear_errors(self):
        self.fail = {pipeline.PRIMARY_ASR, pipeline.SECONDARY_ASR}
        with self.assertRaisesRegex(RuntimeError, 'ASR 模型皆失敗'):
            self.run_pipeline()
        self.jobs.clear()
        with self.assertRaisesRegex(RuntimeError, 'ASR 模型皆失敗'):
            self.run_pipeline(asr_mode='single')
        self.assertEqual([j['asr'] for j in self.jobs], [pipeline.PRIMARY_ASR])

    def test_cached_raw_is_reused_across_fusion_f7_and_mode_changes(self):
        baseline = self.run_pipeline()
        self.jobs.clear()
        repeated = self.run_pipeline()
        self.assertEqual(self.jobs, [])
        self.assertEqual(repeated['cache_key'], baseline['cache_key'])
        with patch.object(fusion, 'FUSION_VERSION', 'changed-rule-version'):
            changed = self.run_pipeline()
        self.assertEqual(self.jobs, [])
        self.assertNotEqual(changed['cache_key'], baseline['cache_key'])
        config = load_quality_config({'f7': {'avg_logprob_min': -.95}})
        self.run_pipeline(quality_config=config)
        self.assertEqual(self.jobs, [])
        single = self.run_pipeline(asr_mode='single')
        self.assertEqual(self.jobs, [])
        self.assertEqual(single['model'], pipeline.PRIMARY_ASR)

    def test_model_change_invalidates_only_its_raw_cache(self):
        self.run_pipeline()
        self.jobs.clear()
        (self.models/pipeline.SECONDARY_ASR/'model.safetensors').write_bytes(b'changed weights')
        self.run_pipeline()
        self.assertEqual([j['asr'] for j in self.jobs], [pipeline.SECONDARY_ASR])

    def test_vad_settings_invalidate_raw_even_if_actual_spans_stay_equal(self):
        self.run_pipeline(vad_threshold=.3)
        self.jobs.clear()
        self.run_pipeline(vad_threshold=.4)
        self.assertEqual([j['asr'] for j in self.jobs], [pipeline.PRIMARY_ASR, pipeline.SECONDARY_ASR])

    def test_custom_vad_settings_are_forwarded_to_both_children(self):
        self.run_pipeline(vad_threshold=.4, vad_rescue_chunk_seconds=45)
        for job in self.jobs:
            self.assertEqual(job['vad_threshold'], .4)
            self.assertEqual(job['vad_rescue_chunk_seconds'], 45)

    def test_shared_cache_survives_new_combo_output_folder(self):
        self.run_pipeline(cache_dir=self.root/'common')
        self.jobs.clear()
        pipeline.run_production_asr('source.mp4', self.root/'new-output', models_dir=self.models,
            child_runner=self.child, prepare_plan=self.plan, gpu_managed=True, cache_dir=self.root/'common')
        self.assertEqual(self.jobs, [])

    def test_fusion_error_uses_a1_and_keeps_reason_on_cached_run(self):
        self.rows_a2.pop()
        first = self.run_pipeline()
        self.assertEqual(first['model'], pipeline.PRIMARY_ASR)
        self.assertIn('融合失敗', first['asr_warnings'][0])
        second = self.run_pipeline()
        self.assertEqual(second['asr_warnings'], first['asr_warnings'])

    def test_invalid_mode_fails_before_any_model_process(self):
        with self.assertRaisesRegex(ValueError, 'dual or single'):
            self.run_pipeline(asr_mode='other')
        self.assertEqual(self.jobs, [])


class ProductionDispatchTests(unittest.TestCase):
    def test_cli_default_and_explicit_modes_use_production_pipeline(self):
        for options, mode in (([], 'dual'), (['--asr-mode', 'dual'], 'dual'),
                              (['--asr-mode', 'single'], 'single')):
            with self.subTest(mode=mode, options=options), \
                    patch.object(pipeline, 'run_production_asr', return_value={'metrics': {}}) as run:
                self.assertEqual(cli.main(['poc', 'source.mp4', '--out', 'out', *options]), 0)
                self.assertEqual(run.call_args.kwargs['asr_mode'], mode)

    def test_cli_explicit_model_retains_diagnostic_path(self):
        from oh_streamer.subtitles import asr
        with patch.object(asr, 'run_asr', return_value={'metrics': {}}) as run, \
                patch.object(pipeline, 'run_production_asr') as production:
            self.assertEqual(cli.main(['poc', 'source.mp4', '--out', 'out', '--asr', 'anime-whisper']), 0)
        self.assertEqual(run.call_args.kwargs['model'], 'anime-whisper')
        production.assert_not_called()


    def test_environment_dual_accepts_either_model_single_requires_a1(self):
        for aliases in ((), (pipeline.PRIMARY_ASR,), (pipeline.SECONDARY_ASR,),
                        (pipeline.PRIMARY_ASR, pipeline.SECONDARY_ASR)):
            with self.subTest(aliases=aliases), tempfile.TemporaryDirectory() as tmp:
                base = Path(tmp) / 'testdata' / 'subtitle_poc'
                python = base / '.venv' / 'Scripts' / 'python.exe'
                python.parent.mkdir(parents=True)
                python.write_bytes(b'python')
                models = base / 'models'
                for alias in aliases:
                    model = models / alias
                    model.mkdir(parents=True)
                    (model / 'config.json').write_text('{}')
                    (model / 'model.safetensors').write_bytes(b'weights')
                gguf = models / 'sakura' / 'Sakura-Galtransl-7B-v3.7-IQ4_XS.gguf'
                gguf.parent.mkdir(parents=True)
                gguf.write_bytes(b'model')
                server = models / 'llama-cpp' / 'llama-server.exe'
                server.parent.mkdir(parents=True)
                server.write_bytes(b'server')
                with patch.object(environment, '_probe_python', return_value=True):
                    dual = environment.detect(root=tmp)
                    single = environment.detect({'subtitle_asr_mode': 'single'}, root=tmp)
                self.assertEqual(dual['available'], bool(aliases), dual['reason'])
                self.assertEqual(single['available'], pipeline.PRIMARY_ASR in aliases, single['reason'])
