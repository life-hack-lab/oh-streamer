"""CPU-only integration seams: translation retry/cache, comparison and soft mux."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from oh_streamer.subtitles import cli
from oh_streamer.subtitles.core import parse_srt


try:
    import opencc  # noqa: F401
    HAVE_OPENCC = True
except ImportError:
    HAVE_OPENCC = False
needs_opencc = unittest.skipUnless(HAVE_OPENCC, "OpenCC (opencc) is not installed in this venv")


class FakeClient:
    instances = []
    def __init__(self, *args, **kwargs):
        self.calls = []
        self.closed = False
        self.metrics = {'completion_tokens': 10, 'tokens_per_second': 5}
        self.instances.append(self)
    def __enter__(self):
        return self
    def __exit__(self, *args):
        self.closed = True
    def translate(self, prompt, **kwargs):
        self.calls.append(prompt)
        # Malformed batch forces single-cue retry and strict numbering parsing.
        contents = prompt if isinstance(prompt, str) else '\n'.join(m['content'] for m in prompt)
        if '逐句' in contents:
            return '1. 错误\n3. 缺句'
        return '1. 软件鼠标视频'


class CLITests(unittest.TestCase):
    @needs_opencc
    def test_queue_translation_uses_admission_while_standalone_checks_idle_gpu(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, model = root / "ja.json", root / "model.gguf"
            source.write_text(json.dumps({"segments": [{"start": 0, "end": 2, "text": "こんにちは"}]}), encoding="utf-8")
            model.write_bytes(b"fixture")
            with patch("oh_streamer.subtitles.llm.LlamaServerClient", FakeClient), \
                    patch.object(cli, "assert_gpu_safe", side_effect=RuntimeError("mpv occupied")) as safety, \
                    patch.object(cli, "NvmlMemorySampler") as sampler:
                sampler.return_value.__enter__.return_value.to_dict.return_value = {}
                with self.assertRaisesRegex(RuntimeError, "mpv occupied"):
                    cli.translate_file(source, root, model="sakura", model_path=model, llama_server="unused")
                result = cli.translate_file(source, root, model="sakura", model_path=model,
                                            llama_server="unused", gpu_managed=True)
                self.assertEqual(safety.call_count, 1)
                self.assertEqual(result["gpu_safety"]["status"], "queue worker GPU admission")
                self.assertEqual(result["segments"][0]["text"], "軟體滑鼠影片")

    def test_translation_call_memo_resumes_without_server_calls_and_rejects_other_key(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, model = root / "ja.json", root / "model.gguf"
            source.write_text(json.dumps({"segments": [{"start": 0, "end": 2, "text": "こんにちは"}]}), encoding="utf-8")
            model.write_bytes(b"fixture")
            partial = root / "sakura.translate.partial.json"
            kwargs = dict(model="sakura", model_path=model, llama_server="unused", gpu_managed=True)
            with patch("oh_streamer.subtitles.llm.LlamaServerClient", FakeClient),                     patch("oh_streamer.subtitles.core._default_opencc", return_value=lambda text: text),                     patch.object(cli, "NvmlMemorySampler") as sampler:
                sampler.return_value.__enter__.return_value.to_dict.return_value = {}
                # Normal completion removes the checkpoint.
                cli.translate_file(source, root, **kwargs)
                self.assertFalse(partial.exists())
                # Keep it to simulate a stop before the final result was written.
                with patch.object(cli._CallMemo, "clear"):
                    cli.translate_file(source, root, force=True, **kwargs)
                self.assertTrue(partial.exists())
                self.assertTrue(json.loads(partial.read_text("utf-8"))["calls"])
                with patch.object(cli._CallMemo, "clear"):
                    result = cli.translate_file(source, root, force=True, **kwargs)
                self.assertEqual(FakeClient.instances[-1].calls, [])
                self.assertEqual(result["segments"][0]["text"], "软件鼠标视频")
                # A checkpoint from another key is ignored.
                data = json.loads(partial.read_text("utf-8"))
                data["cache_key"] = "other"
                partial.write_text(json.dumps(data), encoding="utf-8")
                cli.translate_file(source, root, force=True, **kwargs)
                self.assertTrue(FakeClient.instances[-1].calls)

    def test_on_fail_hide_and_japanese_cache_replay(self):
        class FailingClient(FakeClient):
            def translate(self, *args, **kwargs):
                raise ValueError('fixture failure')
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            source,model=root/'ja.json',root/'model.gguf'
            model.write_bytes(b'fixture')
            source.write_text(json.dumps({'segments':[dict(start=0,end=2,text='こんにちは')]}),encoding='utf-8')
            with patch('oh_streamer.subtitles.llm.LlamaServerClient',FailingClient), \
                 patch.object(cli,'assert_gpu_safe',return_value={}) as safety, \
                 patch.object(cli,'NvmlMemorySampler') as sampler:
                sampler.return_value.__enter__.return_value.to_dict.return_value={}
                result=cli.translate_file(source,root,model='sakura',model_path=model,llama_server='unused')
                self.assertTrue(result['segments'][0]['preserved_original'])
                self.assertIn('[原文]',result['segments'][0]['text'])
                self.assertEqual((root/'sakura.zh-TW.srt').read_text('utf-8'),'')
                cli.translate_file(source,root,model='sakura',model_path=model,llama_server='unused',on_fail='japanese')
                visible=(root/'sakura.zh-TW.srt').read_text('utf-8')
                self.assertIn('こんにちは',visible)
                self.assertNotIn('[原文]',visible)
                self.assertEqual(safety.call_count,1)

    @needs_opencc

    def test_translation_fallback_opencc_cache_and_rebuild(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, gguf = root / 'ja.json', root / 'fake.gguf'
            gguf.write_bytes(b'fixture only')
            source.write_text(json.dumps({'segments': [
                {'start': 0, 'end': 2, 'text': 'こんにちは'},
                {'start': 3, 'end': 5, 'text': 'ありがとう'}],
                'speech_intervals': [{'start': 0, 'end': 5}]}, ensure_ascii=False), encoding='utf-8')
            with patch('oh_streamer.subtitles.llm.LlamaServerClient', FakeClient), \
                 patch.object(cli, 'assert_gpu_safe', return_value={'mock': True}) as safety, \
                 patch.object(cli, 'NvmlMemorySampler') as sampler:
                sampler.return_value.__enter__.return_value.to_dict.return_value = {'available': False, 'mock': True}
                result = cli.translate_file(source, root, model='sakura', model_path=gguf,
                                            llama_server='unused')
                self.assertEqual(result['metrics']['mismatch_batches'], 1)
                self.assertEqual(result['metrics']['fallback_sentences'], 2)
                self.assertEqual(result['segments'][0]['text'], '軟體滑鼠影片')
                self.assertTrue(FakeClient.instances[-1].closed)
                batch_request = FakeClient.instances[-1].calls[0]
                self.assertEqual(batch_request[1]['role'], 'user')
                self.assertTrue(batch_request[1]['content'].endswith('1. こんにちは\n2. ありがとう'))
                # JSON cache regenerates deleted SRT without touching GPU/client.
                (root / 'sakura.zh-TW.srt').unlink()
                second = cli.translate_file(source, root, model='sakura', model_path=gguf,
                                            llama_server='unused')
                self.assertEqual(second['cache_key'], result['cache_key'])
                self.assertEqual(safety.call_count, 1)
                self.assertEqual(len(parse_srt((root / 'sakura.zh-TW.srt').read_text('utf-8'))), 2)
                # Changed source content invalidates translation cache.
                source.write_text(source.read_text('utf-8').replace('こんにちは', 'こんばんは'), encoding='utf-8')
                cli.translate_file(source, root, model='sakura', model_path=gguf, llama_server='unused')
                self.assertEqual(safety.call_count, 2)

    def test_compare_matches_filtered_rows_and_escapes(self):
        source = {'segments': [{'start': 0, 'end': 1, 'text': 'あっ'},
                               {'start': 1, 'end': 3, 'text': 'A|B'}]}
        translated = {'segments': [{'start': 1, 'end': 3, 'text_ja': 'A|B', 'text': '譯文'}]}
        text = cli.compare_markdown(source, {'sakura': translated})
        self.assertIn('A\\|B | 譯文', text)
        self.assertIn('（過濾／未翻譯）', text)
        # Cleanup may change text while retaining timestamps; do not misreport an omitted translation.
        translated['segments'][0]['text_ja'] = 'cleaned source'
        self.assertIn('A\\|B | 譯文', cli.compare_markdown(source, {'sakura': translated}))

    def test_matrix_releases_all_asr_processes_before_translation(self):
        with tempfile.TemporaryDirectory() as folder:
            events = []
            source = {'segments': [{'start': 0, 'end': 2, 'text': 'こんにちは'}],
                      'speech_intervals': [[0, 2]],
                      'metrics': {'wall_seconds': 1, 'raw_segment_count': 1}}
            def asr_process(command, **kwargs):
                events.append('asr')
                self.assertIn('PYTHONPATH', kwargs['env'])
                out = Path(command[command.index('--out') + 1])
                out.mkdir(parents=True)
                (out / 'ja.json').write_text(json.dumps(source), encoding='utf-8')
                (out / 'ja.srt').write_text('fixture', encoding='utf-8')
            def translator(*args, **kwargs):
                self.assertEqual(events.count('asr'), 6)
                events.append('translate')
                return {'segments': [{'start': 0, 'end': 2, 'text_ja': 'こんにちは', 'text': '你好'}],
                        'metrics': {'input_sentences': 1}}
            args = SimpleNamespace(out=folder, models='models', video='video.mkv',
                                   force=False, llama_server='llama-server.exe')
            with patch.object(cli.subprocess, 'run', side_effect=asr_process), \
                 patch.object(cli, 'translate_file', side_effect=translator), \
                 patch.object(cli, 'mux_subtitles') as mux:
                cli.run_matrix(args)
            self.assertEqual(events, ['asr'] * 6 + ['translate'] * 12)
            self.assertEqual(mux.call_count, 6)
            self.assertTrue((Path(folder) / 'measurements.md').exists())
            self.assertEqual(len(json.loads((Path(folder) / 'matrix.metrics.json').read_text())['asr']), 6)

    @unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'ffmpeg/ffprobe required')
    def test_mux_real_stream_copy_soft_tracks(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            video, preview = root / 'input.mkv', root / 'preview.mkv'
            subprocess.run([shutil.which('ffmpeg'), '-nostdin', '-loglevel', 'error',
                            '-f', 'lavfi', '-i', 'color=size=160x90:rate=10:duration=2',
                            '-f', 'lavfi', '-i', 'anullsrc=r=16000:cl=mono', '-t', '2',
                            '-c:v', 'mpeg4', '-c:a', 'aac', str(video)], check=True)
            ja, zh = root / 'ja.srt', root / 'zh.srt'
            ja.write_text('1\n00:00:00,000 --> 00:00:01,000\nこんにちは\n', encoding='utf-8')
            zh.write_text('1\n00:00:00,000 --> 00:00:01,000\n你好\n', encoding='utf-8')
            cli.mux_subtitles(video, [('日本語', ja, 'jpn'), ('繁體中文（AI）', zh, 'chi')], preview, 2)
            data = json.loads(subprocess.check_output([shutil.which('ffprobe'), '-v', 'error',
                              '-show_streams', '-of', 'json', str(preview)], encoding='utf-8'))
            tracks = [s for s in data['streams'] if s['codec_type'] == 'subtitle']
            self.assertEqual([s['codec_name'] for s in tracks], ['subrip', 'subrip'])
            self.assertEqual([s['tags']['language'] for s in tracks], ['jpn', 'chi'])
            self.assertEqual(next(s['codec_name'] for s in data['streams'] if s['codec_type'] == 'video'), 'mpeg4')
            with self.assertRaises(FileExistsError):
                cli.mux_subtitles(video, [], preview)

if __name__ == '__main__':
    unittest.main()
