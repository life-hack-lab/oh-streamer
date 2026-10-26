"""Finalized translation batches can be published before the next model call."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from oh_streamer.subtitles import cli


class BatchPublishTests(unittest.TestCase):
    def setUp(self):
        conversion = patch('oh_streamer.subtitles.core._default_opencc',
                           return_value=lambda text: text.replace('软件', '軟體'))
        conversion.start()
        self.addCleanup(conversion.stop)
        sampler = patch.object(cli, "NvmlMemorySampler")
        sampler.start().return_value.__enter__.return_value.to_dict.return_value = {}
        self.addCleanup(sampler.stop)

    def run_translation(self, root, sources, on_batch=None, **options):
        source, model = root / 'ja.json', root / 'model.gguf'
        if not source.exists():
            source.write_text(json.dumps({'segments': sources}), encoding='utf-8')
        if not model.exists():
            model.write_bytes(b'fixture')
        return cli.translate_file(
            source, root, model='sakura', model_path=model, llama_server='unused',
            gpu_managed=True, batch_size=1, fixed_cues=True,
            on_batch=on_batch, **options,
        )

    def model_fixture(self, snapshots, *, fail_first=False):
        class Client:
            metrics = {}
            model = 'sakura'
            translate_model = 'sakura-v1'
            calls = 0

            def translate(self, prompt, **kwargs):
                self.calls += 1
                if fail_first and self.calls == 1:
                    raise ValueError('force single-cue fallback')
                if self.calls == (3 if fail_first else 2):
                    # First finalized batch is already visible when model starts
                    # the next one, including repeated occurrences' own timing.
                    assert len(snapshots) == 1
                    assert [r['start'] for r in snapshots[0]] == [0.0, 6.0]
                return '1. 软件'
        return Client()

    def test_batches_are_final_after_retry_and_conversion_with_duplicate_timing(self):
        snapshots = []
        client = self.model_fixture(snapshots, fail_first=True)
        sources = [dict(start=0, end=2, text='こんにちは'),
                   dict(start=3, end=5, text='ありがとう'),
                   dict(start=6, end=8, text='こんにちは')]
        with tempfile.TemporaryDirectory() as folder:
            result = self.run_translation(Path(folder), sources, snapshots.append,
                                          shared_client=client)
        self.assertEqual(len(snapshots), 2)
        self.assertEqual(snapshots[-1], result['segments'])
        self.assertEqual(snapshots[0][0]['text'], '軟體')
        self.assertEqual(snapshots[0][0], snapshots[-1][0])
        self.assertEqual(snapshots[0][1], snapshots[-1][2])
        self.assertIsNot(snapshots[0][0], snapshots[-1][0])
        snapshots[0][0]['text'] = 'consumer mutation'
        self.assertEqual(result['segments'][0]['text'], '軟體')

    def test_cache_replays_once_without_changing_cache_identity(self):
        snapshots = []
        sources = [dict(start=0, end=2, text='こんにちは')]
        client = self.model_fixture(snapshots)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            first = self.run_translation(root, sources, shared_client=client)
            second = self.run_translation(root, sources, snapshots.append,
                                          shared_client=client)
        self.assertEqual(client.calls, 1)
        self.assertEqual(first['cache_key'], second['cache_key'])
        self.assertEqual(snapshots, [second['segments']])
        snapshots[0][0]['text'] = 'consumer mutation'
        self.assertEqual(second['segments'][0]['text'], '軟體')

    def test_consumer_error_propagates_without_model_retry_and_preserves_memo(self):
        client = self.model_fixture([])
        sources = [dict(start=0, end=2, text='こんにちは'),
                   dict(start=3, end=5, text='ありがとう')]
        def abort(rows):
            raise RuntimeError('publisher stopped')
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with self.assertRaisesRegex(RuntimeError, 'publisher stopped'):
                self.run_translation(root, sources, abort, shared_client=client)
            self.assertTrue((root / 'sakura.translate.partial.json').exists())
            self.assertFalse((root / 'sakura.zh-TW.json').exists())
        self.assertEqual(client.calls, 1)

    def test_lookup_and_empty_input_publish_once(self):
        for sources in ([], [dict(start=0, end=2, text='こんにちは')]):
            snapshots = []
            with self.subTest(sources=sources), tempfile.TemporaryDirectory() as folder:
                result = self.run_translation(
                    Path(folder), sources, snapshots.append,
                    translation_lookup={'こんにちは': {'text': '你好'}},
                )
            self.assertEqual(snapshots, [result['segments']])


if __name__ == '__main__':
    unittest.main()
