import json
from io import BytesIO
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

_STREAMER_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_ROOT = _STREAMER_ROOT / "src"
_TEMP_ROOT = _STREAMER_ROOT / ".test-tmp"
_TEMP_ROOT.mkdir(parents=True, exist_ok=True)
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from oh_streamer.subtitles.llm import (
    LlamaServerClient,
    LlamaServerError,
    SAKURA_SYSTEM_PROMPT,
    SAKURA_USER_PREFIX,
    build_translation_prompt,
    strip_think_content,
)


class FakeResponse:
    def __init__(self, payload, status=200):
        self.payload = payload
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self):
        return self.payload


class FakeProcess:
    def __init__(self):
        self.pid = 4567
        self.return_code = None
        self.terminate_calls = 0
        self.kill_calls = 0
        self.wait_calls = []

    def poll(self):
        return self.return_code

    def terminate(self):
        self.terminate_calls += 1

    def kill(self):
        self.kill_calls += 1
        self.return_code = -9

    def wait(self, timeout=None):
        self.wait_calls.append(timeout)
        self.return_code = self.return_code if self.return_code is not None else 0
        return self.return_code


class SubtitleLlmTests(unittest.TestCase):
    def test_token_limit_truncation_is_failed_and_tokens_are_still_measured(self):
        client = LlamaServerClient('sakura', 'unused.gguf', 'llama-server')
        client.start = lambda: None
        response = {'choices': [{'message': {'content': '1. 截斷'}, 'finish_reason': 'length'}],
                    'usage': {'prompt_tokens': 8, 'completion_tokens': 128, 'total_tokens': 136}}
        with patch('oh_streamer.subtitles.llm.urlopen', return_value=FakeResponse(
                json.dumps(response, ensure_ascii=False).encode('utf-8'))):
            with self.assertRaisesRegex(LlamaServerError, 'truncated'):
                client.translate('1. 原文')
        self.assertEqual(client.metrics['requests'], 1)
        self.assertEqual(client.metrics['failures'], 1)
        self.assertEqual(client.metrics['completion_tokens'], 128)

    def test_sakura_prompt_uses_official_system_and_user_prefix(self):
        messages = build_translation_prompt("sakura", "1. こんにちは")

        self.assertEqual(messages[0], {"role": "system", "content": SAKURA_SYSTEM_PROMPT})
        self.assertEqual(messages[1]["content"], SAKURA_USER_PREFIX + "1. こんにちは")

    def test_sakura_prompt_keeps_caller_system_constraints(self):
        messages = build_translation_prompt(
            "sakura",
            [
                {"role": "system", "content": "保留編號"},
                {"role": "user", "content": "1. こんにちは"},
            ],
        )

        self.assertIn(SAKURA_SYSTEM_PROMPT, messages[0]["content"])
        self.assertIn("保留編號", messages[0]["content"])
        self.assertEqual(messages[1]["content"], SAKURA_USER_PREFIX + "1. こんにちは")

    def test_qwen3_prompt_disables_thinking_and_strips_all_think_blocks(self):
        messages = build_translation_prompt("qwen3", "1. こんにちは")

        self.assertEqual(messages[-1]["content"], "1. こんにちは /no_think")
        self.assertEqual(
            strip_think_content("<think>first</think>1. 你好<think>second</think>"),
            "1. 你好",
        )
        self.assertEqual(strip_think_content("visible<think>unfinished"), "visible")

    def test_translate_posts_openai_chat_request_and_accumulates_metrics(self):
        client = LlamaServerClient("sakura", "unused.gguf", "llama-server", port=8765)
        client.start = lambda: None
        response = {
            "choices": [{"message": {"content": "1. 你好"}}],
            "usage": {"prompt_tokens": 18, "completion_tokens": 4, "total_tokens": 22},
            "timings": {"prompt_n": 18, "predicted_n": 4, "predicted_ms": 500},
        }
        seen = {}

        def fake_urlopen(request, timeout):
            seen["url"] = request.full_url
            seen["body"] = json.loads(request.data.decode("utf-8"))
            seen["timeout"] = timeout
            return FakeResponse(json.dumps(response, ensure_ascii=False).encode("utf-8"))

        with patch("oh_streamer.subtitles.llm.urlopen", side_effect=fake_urlopen):
            result = client("1. こんにちは")

        self.assertEqual(result, "1. 你好")
        self.assertEqual(seen["url"], "http://127.0.0.1:8765/v1/chat/completions")
        self.assertEqual(seen["body"]["model"], "sakura")
        self.assertEqual(seen["body"]["messages"][1]["content"], SAKURA_USER_PREFIX + "1. こんにちは")
        self.assertFalse(seen["body"]["stream"])
        self.assertEqual(seen["timeout"], 300.0)
        self.assertEqual(client.metrics["requests"], 1)
        self.assertEqual(client.metrics["tokens"], 4)
        self.assertEqual(client.metrics["total_tokens"], 22)
        self.assertEqual(client.metrics["generation_tokens_per_second"], 8.0)

    def test_seed_and_cache_prompt_only_sent_when_set(self):
        response = {"choices": [{"message": {"content": "1. x"}}], "usage": {}, "timings": {}}
        bodies = []

        def fake_urlopen(request, timeout):
            bodies.append(json.loads(request.data.decode("utf-8")))
            return FakeResponse(json.dumps(response).encode("utf-8"))

        legacy = LlamaServerClient("sakura", "unused.gguf", "llama-server", port=8765)
        fixed = LlamaServerClient("sakura", "unused.gguf", "llama-server", port=8765,
                                  seed=42, cache_prompt=False, temperature=0.0)
        for client in (legacy, fixed):
            client.start = lambda: None
            with patch("oh_streamer.subtitles.llm.urlopen", side_effect=fake_urlopen):
                client("1. a")
        self.assertNotIn("seed", bodies[0])
        self.assertNotIn("cache_prompt", bodies[0])
        self.assertEqual(bodies[0]["temperature"], 0.1)
        self.assertEqual(bodies[1]["seed"], 42)
        self.assertIs(bodies[1]["cache_prompt"], False)
        self.assertEqual(bodies[1]["temperature"], 0.0)

    def test_qwen3_request_sets_server_thinking_override_and_strips_output(self):
        client = LlamaServerClient("qwen3", "unused.gguf", "llama-server", port=8766)
        client.start = lambda: None
        response = {"choices": [{"message": {"content": "<think>hidden</think>1. 你好"}}]}
        seen = {}

        def fake_urlopen(request, timeout):
            seen["body"] = json.loads(request.data.decode("utf-8"))
            return FakeResponse(json.dumps(response, ensure_ascii=False).encode("utf-8"))

        with patch("oh_streamer.subtitles.llm.urlopen", side_effect=fake_urlopen):
            result = client.translate("1. こんにちは")

        self.assertEqual(result, "1. 你好")
        self.assertEqual(seen["body"]["chat_template_kwargs"], {"enable_thinking": False})
        self.assertIn("/no_think", seen["body"]["messages"][-1]["content"])

    def test_missing_model_path_fails_before_starting_a_process(self):
        with tempfile.TemporaryDirectory(dir=_TEMP_ROOT) as temporary_directory:
            model_path = Path(temporary_directory) / "missing.gguf"
            client = LlamaServerClient("sakura", model_path, "llama-server")
            with patch("oh_streamer.subtitles.llm.subprocess.Popen") as popen:
                with self.assertRaisesRegex(FileNotFoundError, "local GGUF model does not exist"):
                    client.start()

        popen.assert_not_called()

    def test_shutdown_terminates_only_owned_process_and_closes_log(self):
        client = LlamaServerClient("sakura", "unused.gguf", "llama-server")
        owned = FakeProcess()
        other = FakeProcess()
        log_file = BytesIO()
        client._process = owned
        client._server_log_file = log_file

        client.close()

        self.assertEqual(owned.terminate_calls, 1)
        self.assertEqual(owned.wait_calls, [5.0])
        self.assertEqual(owned.kill_calls, 0)
        self.assertEqual(other.terminate_calls, 0)
        self.assertTrue(log_file.closed)
        self.assertIsNone(client.pid)

    def test_shutdown_kills_only_owned_process_after_terminate_timeout(self):
        client = LlamaServerClient("sakura", "unused.gguf", "llama-server")
        owned = FakeProcess()
        original_wait = owned.wait

        def wait_with_one_timeout(timeout=None):
            if not owned.wait_calls:
                owned.wait_calls.append(timeout)
                raise subprocess.TimeoutExpired("llama-server", timeout)
            return original_wait(timeout)

        owned.wait = wait_with_one_timeout
        client._process = owned

        client.close(terminate_timeout=0.01)

        self.assertEqual(owned.terminate_calls, 1)
        self.assertEqual(owned.kill_calls, 1)
        self.assertEqual(owned.wait_calls, [0.01, None])

    def test_start_routes_stderr_to_optional_log_and_closes_it(self):
        with tempfile.TemporaryDirectory(dir=_TEMP_ROOT) as temporary_directory:
            root = Path(temporary_directory)
            model_path = root / "model.gguf"
            model_path.write_bytes(b"test model placeholder")
            log_path = root / "logs" / "llama-server.log"
            client = LlamaServerClient(
                "qwen3",
                model_path,
                "llama-server",
                port=8767,
                startup_timeout=0.2,
                server_log_path=log_path,
            )
            process = FakeProcess()

            with patch("oh_streamer.subtitles.llm.shutil.which", return_value="llama-server"), \
                 patch("oh_streamer.subtitles.llm.subprocess.Popen", return_value=process) as popen, \
                 patch("oh_streamer.subtitles.llm.urlopen", return_value=FakeResponse(b"{}")):
                client.start()
                server_command = popen.call_args.args[0]
                self.assertEqual(popen.call_args.kwargs["stderr"], client._server_log_file)
                self.assertIn("--reasoning", server_command)
                self.assertEqual(server_command[server_command.index("--reasoning") + 1], "off")
                self.assertTrue(log_path.exists())
                client.close()

            self.assertIsNone(client.pid)
            self.assertIsNone(client._server_log_file)


if __name__ == "__main__":
    unittest.main()
