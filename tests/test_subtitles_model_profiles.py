"""CPU-only profile, request, cache, and resume regressions."""
import json
import re
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from oh_streamer.subtitles import cli
from oh_streamer.subtitles.glossary import (
    DEFAULT_TRANSLATE_MODEL, GALTRANSL_PREFIX, GALTRANSL_SYSTEM,
    PREFIX, SAKURA_PROFILES, SYSTEM, get_sakura_profile, sakura_messages,
)
from oh_streamer.subtitles.llm import LlamaServerClient, build_translation_prompt


class ProfileClient:
    instances = []

    def __init__(self, *args, **kwargs):
        self.options = kwargs
        self.calls = []
        self.metrics = {}
        self.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def translate(self, messages, **kwargs):
        self.calls.append(messages)
        # Parse only the final input section, ignoring any historical rows.
        content = messages[-1]['content']
        rows = [match.group(1) for line in content.splitlines()
                if (match := re.search(r'(\d+)\. ', line))]
        return '\n'.join(number + '. 你好' for number in rows)


class ModelProfileTests(unittest.TestCase):
    def test_official_galtransl_prompt_exactly_and_empty_glossary(self):
        expected = (
            '历史翻译：先前译文\n' + GALTRANSL_PREFIX + '部長->部长 #称谓\n'
            '根据以上术语表的对应关系和备注，结合历史剧情和上下文，'
            '将下面的文本从日文翻译成简体中文：\n1. 部長'
        )
        for name in SAKURA_PROFILES:
            if name == DEFAULT_TRANSLATE_MODEL:
                continue
            with self.subTest(name=name):
                messages = sakura_messages('1. 部長', [dict(src='部長', dst='部长', info='称谓')],
                                           '先前译文', translate_model=name)
                self.assertEqual(messages, [dict(role='system', content=GALTRANSL_SYSTEM),
                                            dict(role='user', content=expected)])
                self.assertEqual(build_translation_prompt('sakura', messages, translate_model=name), messages)
                empty = sakura_messages('原文', translate_model=name)
                self.assertTrue(empty[-1]['content'].startswith(GALTRANSL_PREFIX + '\n'))
                self.assertNotIn('历史翻译', empty[-1]['content'])
                with self.assertRaises(ValueError):
                    build_translation_prompt('sakura', messages)
                messages[0]['content'] += 'extra'
                with self.assertRaises(ValueError):
                    build_translation_prompt('sakura', messages, translate_model=name)

    def test_v1_default_and_profile_sampling(self):
        self.assertEqual(get_sakura_profile('v1.0').name, DEFAULT_TRANSLATE_MODEL)
        self.assertEqual(sakura_messages('原文'), [dict(role='system', content=SYSTEM),
            dict(role='user', content=PREFIX + '\n将下面的日文文本根据对应关系和备注翻译成中文：原文')])
        for name, profile in SAKURA_PROFILES.items():
            client = LlamaServerClient('sakura', 'unused.gguf', 'unused', translate_model=name)
            self.assertEqual((client.temperature, client.top_p), (profile.temperature, profile.top_p))
            self.assertIsNone(client.top_k)
            override = LlamaServerClient('sakura', 'unused', 'unused', translate_model=name,
                                         temperature=.4, top_p=.6)
            self.assertEqual((override.temperature, override.top_p), (.4, .6))

    def test_cli_default_and_explicit_profile(self):
        base = ['translate', 'ja.json', '--model', 'model.gguf', '--llama-server', 'unused', '--out', 'out']
        with patch.object(cli, 'translate_file', return_value={'metrics': {}}) as translate:
            self.assertEqual(cli.main(base), 0)
            self.assertEqual(translate.call_args.kwargs['translate_model'], 'sakura-galtransl-7b-v3.7')
            self.assertEqual(cli.main(base + ['--translate-model', 'sakura-galtransl-14b-v3.8',
                                               '--gpu-layers', '12', '--context-size', '4096']), 0)
            self.assertEqual(translate.call_args.kwargs['gpu_layers'], 12)
            self.assertEqual(translate.call_args.kwargs['context_size'], 4096)

    def test_profile_cache_isolation_and_history_resume(self):
        temp_root = Path(__file__).resolve().parents[1] / '.test-tmp'
        temp_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temp_root) as folder:
            root = Path(folder)
            source, model = root / 'ja.json', root / 'model.gguf'
            model.write_bytes(b'fixture')
            source.write_text(json.dumps({'segments': [dict(start=i*3, end=i*3+2, text=f'こんにちは{i}')
                                                         for i in range(21)]}), encoding='utf-8')
            kwargs = dict(model='sakura', model_path=model, llama_server='unused',
                          gpu_managed=True, batch_size=20, context_mode='user')
            ProfileClient.instances.clear()
            with patch('oh_streamer.subtitles.llm.LlamaServerClient', ProfileClient), \
                    patch('oh_streamer.subtitles.core._default_opencc', return_value=lambda text: text), \
                    patch.object(cli, 'NvmlMemorySampler') as sampler:
                sampler.return_value.__enter__.return_value.to_dict.return_value = {}
                baseline = cli.translate_file(source, root, **kwargs)
                name = 'sakura-galtransl-7b-v3.7'
                with patch.object(cli._CallMemo, 'clear'):
                    result = cli.translate_file(source, root, translate_model=name, **kwargs)
                client = ProfileClient.instances[-1]
                self.assertNotEqual(baseline['cache_key'], result['cache_key'])
                self.assertEqual(result['translate_model'], name)
                self.assertEqual(client.options['translate_model'], name)
                self.assertEqual(len(client.calls), 2)
                history = client.calls[1][-1]['content'].split(GALTRANSL_PREFIX)[0]
                self.assertTrue(history.startswith('历史翻译：你好'))
                self.assertNotIn('こんにちは', history)
                with patch.object(cli._CallMemo, 'clear'):
                    resumed = cli.translate_file(source, root, force=True, translate_model=name, **kwargs)
                self.assertEqual(resumed['segments'], result['segments'])
                self.assertEqual(ProfileClient.instances[-1].calls, [])
                changed = cli.translate_file(source, root, translate_model='sakura-galtransl-14b-v3.8',
                                             gpu_layers=12, context_size=4096, **kwargs)
                self.assertNotEqual(changed['cache_key'], result['cache_key'])
                self.assertEqual(len(ProfileClient.instances[-1].calls), 2)
                self.assertEqual(ProfileClient.instances[-1].options['gpu_layers'], 12)
                self.assertEqual(ProfileClient.instances[-1].options['context_size'], 4096)

    def test_fixed_cues_bypass_cleanup_and_keep_exact_rows(self):
        temp_root = Path(__file__).resolve().parents[1] / '.test-tmp'
        temp_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temp_root) as folder:
            root = Path(folder)
            source, model = root / 'sentence_cues.json', root / 'model.gguf'
            rows = [dict(start=0, end=.1, text='あっ'), dict(start=.1, end=10, text='あっ')]
            source.write_text(json.dumps({'segments': rows}), encoding='utf-8')
            model.write_bytes(b'fixture')
            kwargs = dict(model='sakura', model_path=model, llama_server='unused',
                          gpu_managed=True, translate_model='sakura-galtransl-7b-v3.7', fixed_cues=True)
            with patch('oh_streamer.subtitles.llm.LlamaServerClient', ProfileClient), \
                    patch('oh_streamer.subtitles.core._default_opencc', return_value=lambda text: text), \
                    patch.object(cli, 'NvmlMemorySampler') as sampler, \
                    patch.object(cli, 'clean_segments', side_effect=AssertionError('must preserve fixed rows')):
                sampler.return_value.__enter__.return_value.to_dict.return_value = {}
                result = cli.translate_file(source, root, **kwargs)
                self.assertEqual([(r['start'], r['end'], r['text_ja']) for r in result['segments']],
                                 [(r['start'], r['end'], r['text']) for r in rows])
                self.assertTrue(result['recipe_options']['fixed_cues'])
                japanese = (root / 'ja.srt').read_text('utf-8')
                self.assertIn('00:00:00,100 --> 00:00:10,000', japanese)
                (root / 'ja.srt').unlink()
                cli.translate_file(source, root, **kwargs)
                self.assertEqual((root / 'ja.srt').read_text('utf-8'), japanese)
                with self.assertRaisesRegex(ValueError, 'mutually exclusive'):
                    cli.translate_file(source, root, sentence_mode=True, **kwargs)


if __name__ == '__main__':
    unittest.main()
