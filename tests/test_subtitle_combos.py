"""Subtitle combos settings tests: presets, normalization (no models, GPU or network)."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from oh_streamer.subtitles import combos


class PresetTests(unittest.TestCase):
    def test_builtin_presets_load(self):
        with tempfile.TemporaryDirectory() as folder:
            presets, errors = combos.load_presets(Path(folder) / 'missing.json')
        self.assertEqual(errors, [])
        self.assertEqual(list(presets), [raw['id'] for raw in combos.BUILTIN_PRESETS])
        self.assertTrue(all(combo['source'] == 'builtin' for combo in presets.values()))
        self.assertEqual(presets['anime-gt14b']['translation']['gpu_layers'], 28)
        self.assertEqual(presets['anime-sakura1']['translation']['gpu_layers'], 'auto')
        self.assertEqual(presets['anime-llmfix-sakura1']['correction']['mode'], 'llm')
        self.assertEqual(presets['anime-sakura1']['correction'], {'mode': 'none'})

    def test_normalize_does_not_mutate_template(self):
        import copy
        before = copy.deepcopy(combos.COMBO_TEMPLATE)
        combos.normalize_combo({'id': 'a', 'label': 'a'})
        combos.normalize_combo({'id': 'b', 'label': 'b', 'asr': {'model': 'turbo'},
                                'translation': {'profile': 'sakura-galtransl-14b-v3.8'}})
        self.assertEqual(combos.COMBO_TEMPLATE, before)

    def write_user(self, folder, combos_list, **extra):
        path = Path(folder) / 'subtitle_combos.json'
        path.write_text(json.dumps({'schema_version': 1, 'combos': combos_list, **extra}), 'utf-8')
        return path

    def test_user_file_valid_combo_and_defaults(self):
        with tempfile.TemporaryDirectory() as folder:
            path = self.write_user(folder, [{'id': 'my-turbo', 'label': 'turbo', 'asr': {'model': 'TURBO'}}])
            presets, errors = combos.load_presets(path)
        self.assertEqual(errors, [])
        mine = presets['my-turbo']
        self.assertEqual(mine['source'], 'user')
        self.assertEqual(mine['asr']['model'], 'turbo')
        self.assertEqual(mine['asr']['decode'], {'no_repeat_ngram_size': 6})
        self.assertEqual(mine['translation']['backend'], 'sakura')

    def test_user_duplicate_of_builtin_is_rejected_but_others_load(self):
        with tempfile.TemporaryDirectory() as folder:
            path = self.write_user(folder, [
                {'id': 'anime-sakura1', 'label': 'override'},
                {'id': 'ok', 'label': 'ok'},
                {'id': 'ok', 'label': 'dup'},
            ])
            presets, errors = combos.load_presets(path)
        self.assertEqual(presets['anime-sakura1']['source'], 'builtin')
        self.assertNotEqual(presets['anime-sakura1']['label'], 'override')
        self.assertEqual(presets['ok']['label'], 'ok')
        self.assertEqual(len(errors), 2)
        self.assertIn('already defined', errors[0])
        self.assertIn('builtin', errors[0])

    def test_user_unknown_asr_alias_and_bad_fields_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = self.write_user(folder, [
                {'id': 'bad-asr', 'label': 'x', 'asr': {'model': 'whisper-xyz'}},
                {'id': 'Bad ID', 'label': 'x'},
                {'id': 'bad-key', 'label': 'x', 'nonsense': 1},
                {'id': 'bad-field', 'label': 'x', 'asr': {'nope': 1}},
                {'id': 'bad-profile', 'label': 'x', 'translation': {'profile': 'gpt'}},
                {'id': 'bad-cand', 'label': 'x', 'correction': {'mode': 'llm', 'candidates': ['zzz']}},
                {'id': 'qwen-grouped', 'label': 'x', 'asr': {'model': 'qwen3-asr-1.7b', 'segmentation': 'grouped'}},
                {'id': 'fine', 'label': 'x'},
            ])
            presets, errors = combos.load_presets(path)
        self.assertEqual(len(errors), 7)
        self.assertIn('unknown ASR model', errors[0])
        self.assertIn('fine', presets)
        for bad in ('bad-asr', 'Bad ID', 'bad-key', 'bad-field', 'bad-profile', 'bad-cand', 'qwen-grouped'):
            self.assertNotIn(bad, presets)

    def test_corrupt_or_wrong_version_user_file_keeps_builtins(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'subtitle_combos.json'
            path.write_text('{not json', 'utf-8')
            presets, errors = combos.load_presets(path)
            self.assertEqual(len(errors), 1)
            self.assertIn('anime-sakura1', presets)
            path = self.write_user(folder, [], schema_version=99)
            presets, errors = combos.load_presets(path)
            self.assertEqual(len(errors), 1)
            self.assertIn('schema_version', errors[0])


class ReproducibleTranslationValidationTests(unittest.TestCase):
    def test_validation(self):
        for bad in ({'seed': 5}, {'reproducible': 'yes'}, {'reproducible': True, 'seed': -1},
                    {'reproducible': True, 'seed': True}, {'temperature': 3}, {'temperature': 'hot'}):
            with self.assertRaises(combos.ComboError, msg=bad):
                combos.normalize_combo({'id': 'a', 'label': 'a', 'translation': bad})
