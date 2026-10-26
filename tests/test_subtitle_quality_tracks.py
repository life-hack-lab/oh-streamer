"""CPU acceptance seams for deterministic translation and the two display tracks."""
import json
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

from oh_streamer.subtitles import cli, combos
from oh_streamer.subtitles.core import parse_srt
from oh_streamer.subtitles.tracks import CONCISE_TITLE, FULL_TITLE, deterministic_sampling
from oh_streamer.queue.subtitle_jobs import subtitle_paths


try:
    import opencc  # noqa: F401
    HAVE_OPENCC = True
except ImportError:
    HAVE_OPENCC = False
needs_opencc = unittest.skipUnless(HAVE_OPENCC, "OpenCC (opencc) is not installed in this venv")


class FakeTranslator:
    model = 'sakura'
    translate_model = 'sakura-galtransl-7b-v3.7'

    def __init__(self, *args, **kwargs):
        self.calls = []
        self.options = kwargs
        self.metrics = {}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        pass

    def translate(self, prompt, **kwargs):
        lines = re.findall(r'^(\d+)\. (.+)$', prompt[-1]['content'], re.M)
        self.calls.append(lines)
        return '\n'.join(f'{number}. 譯文{number}' for number, _ in lines)


class QualityTrackTests(unittest.TestCase):
    def test_missing_seed_remains_fixed_and_sampling_cannot_enable_randomness(self):
        self.assertEqual(deterministic_sampling({'seed': None, 'temperature': .9, 'cache_prompt': True}),
                         {'temperature': 0.0, 'seed': 42, 'cache_prompt': False})

    @needs_opencc

    def test_cache_identity_covers_seed_model_glossary_and_taiwan_terms_version(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, model, glossary = root/'ja.json', root/'model.gguf', root/'terms.json'
            source.write_text(json.dumps({'segments': [dict(start=0, end=1, text='大丈夫です')]}), 'utf-8')
            model.write_bytes(b'model')
            glossary.write_text('[{"src":"部長","dst":"经理"}]', 'utf-8')
            kwargs = dict(model='sakura', model_path=model, llama_server='fake', gpu_managed=True,
                          shared_client=FakeTranslator(), translate_model=FakeTranslator.translate_model,
                          glossary=glossary)
            baseline = cli.translate_file(source, root, **kwargs)['cache_key']
            self.assertEqual(cli.translate_file(source, root, **kwargs)['cache_key'], baseline)
            seeded = cli.translate_file(source, root, sampling={'seed': 7}, **kwargs)['cache_key']
            self.assertNotEqual(seeded, baseline)
            with patch('oh_streamer.subtitles.tracks.TAIWAN_TERMS_VERSION', 'changed'):
                self.assertNotEqual(cli.translate_file(source, root, **kwargs)['cache_key'], baseline)
            glossary.write_text('[{"src":"部長","dst":"部长"}]', 'utf-8')
            terms_key = cli.translate_file(source, root, **kwargs)['cache_key']
            self.assertNotEqual(terms_key, baseline)
            model.write_bytes(b'changed model')
            self.assertNotEqual(cli.translate_file(source, root, **kwargs)['cache_key'], terms_key)

    @needs_opencc

    def test_exact_text_dedup_repeatability_complete_candidates_and_subset(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            source, model = folder / 'ja.json', folder / 'model.gguf'
            model.write_bytes(b'model')
            rows = [
                dict(start=0, end=.4, text='大丈夫です'),
                dict(start=1, end=1.4, text='ちゅっ、ちゅぷっ……'),
                dict(start=2, end=2.5, text='大丈夫です'),
                dict(start=3, end=3.5, text='わかりました', gap=True, avg_logprob=-.9),
                dict(start=4, end=4.5, text='大丈夫です', gap=True, avg_logprob=-.9),
                dict(start=5, end=5.5, text='ご視聴ありがとうございました'),
            ]
            source.write_text(json.dumps({'segments': rows}), 'utf-8')
            clients = []

            def client(*args, **kwargs):
                value = FakeTranslator(*args, **kwargs)
                clients.append(value)
                return value

            results = []
            with patch('oh_streamer.subtitles.llm.LlamaServerClient', client):
                for index in range(2):
                    output = folder / str(index)
                    results.append(cli.translate_file(source, output, model='sakura', model_path=model,
                        llama_server='fake', gpu_managed=True, sentence_mode=True,
                        translate_model=FakeTranslator.translate_model, force=True))
            self.assertEqual(results[0]['segments'], results[1]['segments'])
            self.assertEqual(len(clients[0].calls), 1)
            self.assertEqual([text for _, text in clients[0].calls[0]],
                             ['大丈夫です', 'ちゅっ、ちゅぷっ……', 'わかりました'])
            self.assertEqual({k: clients[0].options[k] for k in ('temperature', 'seed', 'cache_prompt')},
                             {'temperature': 0.0, 'seed': 42, 'cache_prompt': False})
            repeated = [r['text'] for r in results[0]['segments'] if r['text_ja'] == '大丈夫です']
            self.assertEqual(len(set(repeated)), 1)
            concise = parse_srt((folder / '0/sakura.zh-TW.srt').read_text('utf-8'))
            full = parse_srt((folder / '0/sakura.zh-TW.full.srt').read_text('utf-8'))
            self.assertEqual(len(concise), 2)
            self.assertEqual(len(full), 5)
            self.assertTrue(set((s.start, s.end, s.text) for s in concise).issubset(
                set((s.start, s.end, s.text) for s in full)))
            self.assertEqual(sum(s.text.startswith('（低信心）') for s in full), 2)
            tracks = subtitle_paths(folder / '0')
            self.assertEqual([t[2] for t in tracks], [CONCISE_TITLE, FULL_TITLE])


