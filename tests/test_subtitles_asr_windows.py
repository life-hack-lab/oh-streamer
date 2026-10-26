"""CPU-only resident model ownership and independent canonical ASR windows."""
from contextlib import nullcontext
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from oh_streamer.subtitles import asr, pipeline, runtime, worker


class ResidentASRTests(unittest.TestCase):
    def test_worker_releases_failed_traceback_model_before_loading_next_window(self):
        import gc
        import weakref

        class Model:
            def fail(self):
                raise RuntimeError('inference failed with model in traceback')

        was_enabled = gc.isenabled()
        gc.disable()
        if was_enabled:
            self.addCleanup(gc.enable)
        references, releases = [], []
        torch = SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda: releases.append(references[0]() is None)))

        def run(source, _folder, *, session, **kwargs):
            if source == 'first':
                model = Model()
                references.append(weakref.ref(model))
                session.acquire('model', lambda: (None, model, None, 'float32'), torch, 0)
                model.fail()  # Both this frame and Model.fail retain the model until except exits.
            else:
                self.assertIsNone(references[0](), 'failed inference traceback retained the previous model')
                session.acquire('model', lambda: ('new pipeline', None, None, 'float32'), torch, 0)

        jobs = [dict(source=name, folder=name, asr='anime-whisper', model_path='model')
                for name in ('first', 'second')]
        with patch.object(asr, 'run_asr', side_effect=run), patch.object(runtime, 'atomic_write_json'):
            errors = worker.run_asr_model_windows(dict(folder='unused', jobs=jobs))
        self.assertEqual(errors, {'first': 'inference failed with model in traceback'})
        self.assertEqual(releases, [True, True])

    def test_close_collects_confidence_hook_model_cycle_before_emptying_cuda_cache(self):
        import gc
        import weakref

        class Model:
            def generate(self):
                return None

        # Automatic GC must not happen to make this pass: close owns cleanup before reloading.
        was_enabled = gc.isenabled()
        gc.disable()
        if was_enabled:
            self.addCleanup(gc.enable)
        released = []
        model = Model()
        reference = weakref.ref(model)
        hook = asr._WhisperConfidence(None, model, object())
        hook.close()  # model.generate is now its own bound method: model -> method -> model
        torch = SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda: released.append(reference() is None)))
        session = asr.ASRSession()
        session.acquire('model', lambda: (None, model, None, 'float32'), torch, 0)
        del hook, model
        session.close()
        self.assertIsNone(reference())
        self.assertEqual(released, [True])

    def test_session_loads_once_and_preserves_each_windows_standalone_raw_output(self):
        import numpy as np
        torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False), inference_mode=nullcontext)
        audios = {'first.wav': np.full(3 * 16000, .02, dtype=np.float32),
                  'second.wav': np.full(5 * 16000, .04, dtype=np.float32)}
        loads = []
        def make_pipeline(*args):
            loads.append(args)
            def infer(payload, **kwargs):
                return {'text': 'こんにちは' if payload['array'][0] < .03 else 'ありがとう'}
            return infer, None, None, 'float32'
        def plan(name, start, end):
            return dict(plan_key=name, audio_path=name, audio_duration_seconds=len(audios[name])/16000,
                        speech_intervals=[dict(start=start, end=end)],
                        groups=[dict(start=start, end=end, gap=False)])
        plans = [plan('first.wav', .5, 2), plan('second.wav', 2, 4.5)]
        with tempfile.TemporaryDirectory() as folder, \
                patch.dict(sys.modules, {'torch': torch}), \
                patch.object(runtime, 'source_identity', side_effect=lambda source: (str(source), {})), \
                patch.object(runtime, 'local_model_fingerprint', return_value='model'), \
                patch.object(runtime, 'load_wav_float32', side_effect=lambda path: audios[Path(path).name]), \
                patch.object(asr, '_read_shared_plan', side_effect=lambda value, _settings: value), \
                patch.object(asr, '_make_pipeline', side_effect=make_pipeline):
            root = Path(folder)
            def run(index, target, session=None):
                return asr.run_asr(str(index), root / target, cache_dir=root / target / 'cache',
                                   segmentation='utterance', shared_plan=plans[index], session=session)
            standalone = [run(i, f'standalone-{i}') for i in range(2)]
            self.assertEqual(len(loads), 2)
            with asr.ASRSession() as session:
                resident = [run(i, f'batch-{i}', session) for i in range(2)]
                self.assertIsNotNone(session.pipeline)
                self.assertEqual(len(loads), 3)
            self.assertIsNone(session.pipeline)
            for first, second in zip(standalone, resident):
                self.assertEqual(first['raw_segments'], second['raw_segments'])
                self.assertEqual(first['cache_key'], second['cache_key'])
            self.assertEqual(resident[0]['raw_segments'][0]['start'], .5)
            self.assertEqual(resident[1]['raw_segments'][0]['start'], 2)
            self.assertEqual(resident[1]['raw_segments'][0]['text'], 'ありがとう')

    def test_session_rejects_mixed_model_identity_and_releases_on_error(self):
        torch = SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda: None))
        with asr.ASRSession() as session:
            session.acquire('first', lambda: ('pipe', None, None, 'float32'), torch, -1)
            with self.assertRaisesRegex(ValueError, 'cannot mix'):
                session.acquire('second', lambda: self.fail('must not load'), torch, -1)
        self.assertIsNone(session.pipeline)

    def test_owner_releases_cuda_cache_even_after_partial_model_creation_failure(self):
        releases = []
        torch = SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda: releases.append('released')))
        with asr.ASRSession() as session:
            def failing_factory():
                raise RuntimeError('model initialization failed')
            with self.assertRaisesRegex(RuntimeError, 'initialization failed'):
                session.acquire('model', failing_factory, torch, 0)
            self.assertEqual(releases, [])
        self.assertEqual(releases, ['released'])
        session.close()
        self.assertEqual(releases, ['released'])


class ProductionWindowsTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.models = self.root / 'models'
        for alias in (pipeline.PRIMARY_ASR, pipeline.SECONDARY_ASR):
            model = self.models / runtime.model_dir_name(alias)
            model.mkdir(parents=True)
            (model / 'config.json').write_text('{}')
            (model / 'model.safetensors').write_bytes(b'fixture')
        self.windows = [dict(source=f'window-{i}', folder=str(self.root / f'window-{i}')) for i in range(2)]
        self.prepared, self.jobs, self.events = [], [], []
        self.failed_windows = set()

    def plan(self, source, folder, **kwargs):
        self.prepared.append(source)
        index = int(source[-1])
        return dict(plan_key=source, settings={'source': source}, audio_path=source + '.wav',
                    audio_duration_seconds=3 + index, speech_intervals=[dict(start=index, end=index + 2)],
                    source={'path': source}, groups=[dict(start=index, end=index + 2, gap=False)])

    def child(self, batch, callback):
        alias = batch['jobs'][0]['asr']
        self.events.append(('start', alias))
        self.jobs.append(batch)
        errors = {}
        for job in batch['jobs']:
            if (job['source'], alias) in self.failed_windows:
                errors[job['folder']] = 'window fixture failure'
                continue
            index = int(job['source'][-1])
            rows = [dict(start=index, end=index + 2, text='こんにちは' if index == 0 else 'ありがとう',
                         gap=False, avg_logprob=-.1)]
            runtime.atomic_write_json(Path(job['folder']) / 'ja.json',
                                      dict(raw_segments=rows, segments=rows, metrics={}))
        runtime.atomic_write_json(Path(batch['folder']) / 'window-errors.json', errors)
        self.events.append(('exit', alias))

    def run_windows(self, **kwargs):
        return pipeline.run_production_asr_windows(self.windows, models_dir=self.models,
            prepare_plan=self.plan, child_runner=self.child, gpu_managed=True, **kwargs)

    def test_each_model_gets_one_process_all_windows_then_cpu_fusion_and_cache_reuse(self):
        real_finish = pipeline._finish_asr
        def finish(*args, **kwargs):
            self.events.append(('fusion', args[2].name))
            return real_finish(*args, **kwargs)
        with patch.object(pipeline, '_finish_asr', side_effect=finish):
            docs = self.run_windows()
        self.assertEqual(self.events, [('start', pipeline.PRIMARY_ASR), ('exit', pipeline.PRIMARY_ASR),
            ('start', pipeline.SECONDARY_ASR), ('exit', pipeline.SECONDARY_ASR),
            ('fusion', 'window-0'), ('fusion', 'window-1')])
        self.assertEqual(self.prepared, ['window-0', 'window-1'])
        self.assertEqual([len(job['jobs']) for job in self.jobs], [2, 2])
        self.assertNotEqual(self.jobs[0]['jobs'][0]['shared_plan'], self.jobs[0]['jobs'][1]['shared_plan'])
        self.assertEqual([doc['segments'][0]['start'] for doc in docs], [0, 1])
        self.jobs.clear()
        repeated = self.run_windows()
        self.assertEqual(self.jobs, [])
        self.assertEqual([doc['cache_key'] for doc in docs], [doc['cache_key'] for doc in repeated])
        # The original single-window coordinator uses exactly the same raw identity.
        for window, doc in zip(self.windows, docs):
            cached = pipeline.run_production_asr(window['source'], window['folder'],
                models_dir=self.models, prepare_plan=self.plan,
                child_runner=lambda *_: self.fail('raw must already be cached'))
            self.assertEqual(cached['cache_key'], doc['cache_key'])

    def test_single_window_failure_falls_back_only_that_window(self):
        self.failed_windows = {('window-0', pipeline.PRIMARY_ASR)}
        docs = self.run_windows()
        self.assertEqual(docs[0]['model'], pipeline.SECONDARY_ASR)
        self.assertTrue(docs[0]['asr_warnings'])
        self.assertEqual(docs[1]['model'], 'A1+A2')
        self.assertFalse(docs[1]['asr_warnings'])

    def test_both_models_fail_one_window_without_discarding_successful_neighbor(self):
        self.failed_windows = {('window-0', alias) for alias in
                               (pipeline.PRIMARY_ASR, pipeline.SECONDARY_ASR)}
        docs = self.run_windows()
        self.assertEqual(len(docs), 2)
        self.assertIn('ASR 模型皆失敗', docs[0]['error'])
        self.assertIn('window fixture failure', docs[0]['error'])
        self.assertEqual(docs[0]['segments'], [])
        self.assertEqual(docs[0]['asr_warnings'], [docs[0]['error']])
        self.assertTrue(all(item['status'] == 'failed' for item in docs[0]['asr_models'].values()))
        self.assertNotIn('error', docs[1])
        self.assertEqual(docs[1]['model'], 'A1+A2')
        self.assertEqual(docs[1]['segments'][0]['text'], 'ありがとう')
        self.assertTrue((Path(self.windows[1]['folder']) / 'ja.json').exists())
        # Retry only the failed window. The neighbor remains a valid reusable cache.
        self.jobs.clear()
        self.failed_windows.clear()
        retried = self.run_windows()
        self.assertEqual([len(job['jobs']) for job in self.jobs], [1, 1])
        self.assertTrue(all(job['jobs'][0]['source'] == 'window-0' for job in self.jobs))
        self.assertNotIn('error', retried[0])
        self.assertEqual(retried[1]['cache_key'], docs[1]['cache_key'])

    def test_prepare_failure_is_reported_per_window_without_starting_its_models(self):
        prepare = self.plan
        def broken_plan(source, *args, **kwargs):
            if source == 'window-0':
                raise RuntimeError('invalid audio window')
            return prepare(source, *args, **kwargs)
        with patch.object(self, 'plan', side_effect=broken_plan):
            docs = self.run_windows()
        self.assertEqual(docs[0]['error'], 'invalid audio window')
        self.assertEqual(docs[0]['segments'], [])
        self.assertEqual(docs[1]['model'], 'A1+A2')
        self.assertTrue(all(len(job['jobs']) == 1 and job['jobs'][0]['source'] == 'window-1'
                            for job in self.jobs))

    def test_force_runs_all_windows_again_single_mode_only_loads_primary(self):
        self.run_windows(asr_mode='single')
        self.run_windows(asr_mode='single', force=True)
        self.assertEqual(len(self.jobs), 2)
        self.assertTrue(all(job['jobs'][0]['asr'] == pipeline.PRIMARY_ASR for job in self.jobs))

    def test_worker_reuses_one_session_and_records_per_window_failure(self):
        jobs = [dict(source=f'{i}', folder=str(self.root / str(i)), asr='anime-whisper', model_path='model')
                for i in range(3)]
        sessions = []
        def run(source, folder, **kwargs):
            sessions.append(kwargs['session'])
            if source == '1':
                raise RuntimeError('bad audio window')
        with patch.object(asr, 'run_asr', side_effect=run):
            errors = worker.run_asr_model_windows(dict(folder=str(self.root), jobs=jobs))
        self.assertEqual(len(sessions), 3)
        self.assertTrue(all(item is sessions[0] for item in sessions))
        self.assertEqual(errors, {jobs[1]['folder']: 'bad audio window'})
        self.assertIsNone(sessions[0].pipeline)
        self.assertEqual(json.loads((self.root / 'window-errors.json').read_text('utf-8')), errors)



class PrepareOnlyTests(unittest.TestCase):
    def test_prepare_only_prepares_every_window_and_starts_no_model(self):
        calls = []

        def prepare(source, out_dir, **kwargs):
            calls.append(source)
            return dict(plan_key=source)

        with tempfile.TemporaryDirectory() as tmp, patch.object(runtime, 'atomic_write_json'),                 patch.object(pipeline, 'run_asr_child') as child:
            windows = [dict(source=name, folder=str(Path(tmp) / name)) for name in ('a', 'b')]
            result = pipeline.run_production_asr_windows(
                windows, models_dir=tmp, prepare_plan=prepare, prepare_only=True)
        self.assertEqual((result, calls), ([], ['a', 'b']))
        child.assert_not_called()

    def test_prepare_child_hides_cuda_and_uses_its_own_stage(self):
        seen = {}

        def fake_run(job, job_path, log_path, on_phase=None, *, python=None, on_process=None, env_extra=None):
            seen.update(job=job, env=env_extra)

        with tempfile.TemporaryDirectory() as tmp, patch.object(pipeline, 'run_worker_job', side_effect=fake_run):
            pipeline.run_production_asr_prepare_child(
                [dict(source=Path(tmp) / 'w.wav', folder=Path(tmp) / 'f')], workdir=Path(tmp) / 'work',
                models_dir=Path(tmp), vad_threshold=0.4)
        self.assertEqual(seen['job']['stage'], 'asr_prepare')
        self.assertEqual(seen['env'], {'CUDA_VISIBLE_DEVICES': ''})
        self.assertEqual(seen['job']['options']['vad_threshold'], 0.4)

if __name__ == '__main__':
    unittest.main()
