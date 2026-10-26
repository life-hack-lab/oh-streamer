"""Unit, integration, and regression tests for M4: LLM translation enhancements.

Covers:
- general-llm prompt construction (system, user, glossary, context, single-cue)
- Format tolerance in parse_numbered_translations (strict vs lenient, commentary, out-of-order, extra lines, JSON)
- Tiered ladder retry (retry -> bisection -> single-cue fallback)
- Request budget enforcement (preventing request storms)
- Safety refusal detection, content_blocked classification, and refusal_threshold abort
- CloudTranslationClient attributes cleanup (prompt_family, model alias, remote_model, seed, temperature)
- Bad cache solidification prevention
"""

import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch, MagicMock

from oh_streamer.subtitles import combos, core, cli, glossary, model_adapters, model_registry
from oh_streamer.subtitles.core import (
    Segment,
    TranslationBatch,
    parse_numbered_translations,
    translate_segments,
    is_refusal_text,
)
from oh_streamer.subtitles.glossary import (
    general_llm_messages,
    general_llm_single_messages,
)
from oh_streamer.subtitles.model_adapters import (
    CloudTranslationClient,
    ModelAdapterError,
)
from oh_streamer.subtitles.providers.http import ProviderError

_ORIGINAL_DEFAULT_OPENCC = core._default_opencc


def setUpModule():
    # Without OpenCC installed, convert as identity -- only while this module runs, so other modules still see it.
    try:
        import opencc  # noqa: F401
    except ImportError:
        core._default_opencc = lambda: (lambda text: text)


def tearDownModule():
    core._default_opencc = _ORIGINAL_DEFAULT_OPENCC



class GeneralLLMPromptTests(unittest.TestCase):
    def test_general_llm_messages_basic(self):
        japanese = "1. おはよう。\n2. さようなら。"
        messages = general_llm_messages(japanese)
        self.assertEqual(len(messages), 2)
        sys_msg = messages[0]["content"]
        user_msg = messages[1]["content"]

        # Check system prompt requirements
        self.assertIn("繁體中文（台灣習慣用詞）", sys_msg)
        self.assertIn("數字. 譯文", sys_msg)
        self.assertIn("不得保留任何日文假名", sys_msg)
        self.assertIn("絕對不要輸出任何解釋", sys_msg)

        # Check user prompt contains the numbered Japanese cues
        self.assertIn("1. おはよう。", user_msg)
        self.assertIn("2. さようなら。", user_msg)

    def test_general_llm_messages_with_glossary_and_context(self):
        japanese = "1. 会長、おはようございます。"
        glossary_items = [
            {"src": "会長", "dst": "會長", "info": "學生會長"},
            {"src": "おはよう", "dst": "早安", "info": ""},
        ]
        context = "昨日學校發生了不可思議的事件。"
        messages = general_llm_messages(japanese, glossary=glossary_items, context=context)
        sys_msg = messages[0]["content"]
        user_msg = messages[1]["content"]

        self.assertIn("会長 -> 會長 #學生會長", sys_msg)
        self.assertIn("おはよう -> 早安", sys_msg)
        self.assertIn("前文參考", user_msg)
        self.assertIn("昨日學校發生了不可思議的事件", user_msg)

    def test_general_llm_single_messages(self):
        text = "よろしくお願いします。"
        glossary_items = [{"src": "よろしく", "dst": "請多指教", "info": ""}]
        messages = general_llm_single_messages(text, glossary=glossary_items)
        self.assertEqual(len(messages), 2)
        sys_msg = messages[0]["content"]
        user_msg = messages[1]["content"]

        self.assertIn("1. 譯文", sys_msg)
        self.assertIn("繁體中文（台灣習慣用詞）", sys_msg)
        self.assertIn("よろしく -> 請多指教", sys_msg)
        self.assertEqual(user_msg, "1. よろしくお願いします。")

    def test_combos_normalize_accepts_general_llm(self):
        cfg = combos.COMBO_TEMPLATE.copy()
        cfg["id"] = "test_general_llm"
        cfg["label"] = "Test General LLM"
        cfg["translation"]["prompt_mode"] = "general-llm"
        normalized = combos.normalize_combo(cfg)
        self.assertEqual(normalized["translation"]["prompt_mode"], "general-llm")


class FormatToleranceAndParsingTests(unittest.TestCase):
    def test_parse_numbered_translations_strict(self):
        # Strict mode accepts clean format
        clean = "1. 早安。\n2. 晚安。"
        self.assertEqual(parse_numbered_translations(clean, 2, strict=True), ["早安。", "晚安。"])

        # Strict mode rejects unnumbered commentary
        commentary = "以下是翻譯：\n1. 早安。\n2. 晚安。"
        with self.assertRaises(ValueError):
            parse_numbered_translations(commentary, 2, strict=True)

        # Strict mode rejects out of order
        out_of_order = "2. 晚安。\n1. 早安。"
        with self.assertRaises(ValueError):
            parse_numbered_translations(out_of_order, 2, strict=True)

        # Strict mode rejects missing lines
        missing = "1. 早安。"
        with self.assertRaises(ValueError):
            parse_numbered_translations(missing, 2, strict=True)

        # Strict mode rejects empty string
        with self.assertRaises(ValueError):
            parse_numbered_translations("   \n ", 2, strict=True)

    def test_parse_numbered_translations_lenient_commentary(self):
        output = (
            "好的，為您翻譯如下：\n"
            "```text\n"
            "1. 你好！\n"
            "2. 很高興見到你。\n"
            "```\n"
            "希望有幫助到您！"
        )
        parsed = parse_numbered_translations(output, (1, 2), strict=False)
        self.assertEqual(parsed, ["你好！", "很高興見到你。"])

    def test_parse_numbered_translations_lenient_out_of_order_and_extra(self):
        output = (
            "2. 第二句\n"
            "3. 第三句（額外）\n"
            "1. 第一句\n"
        )
        parsed = parse_numbered_translations(output, (1, 2), strict=False)
        self.assertEqual(parsed, ["第一句", "第二句"])

    def test_parse_numbered_translations_lenient_json(self):
        # JSON list
        json_output = json.dumps([
            {"id": 1, "text": "早安"},
            {"id": 2, "zh": "午安"},
        ])
        self.assertEqual(parse_numbered_translations(json_output, (1, 2), strict=False), ["早安", "午安"])

        # JSON inside markdown
        fenced_json = f"```json\n{json_output}\n```"
        self.assertEqual(parse_numbered_translations(fenced_json, (1, 2), strict=False), ["早安", "午安"])

        # JSON dict with translations key
        dict_json = json.dumps({
            "translations": [
                {"id": 1, "translation": "早安"},
                {"id": 2, "content": "午安"},
            ]
        })
        self.assertEqual(parse_numbered_translations(dict_json, (1, 2), strict=False), ["早安", "午安"])

    def test_parse_numbered_translations_lenient_missing_raises(self):
        output = "1. 只有第一句\n"
        with self.assertRaises(ValueError):
            parse_numbered_translations(output, (1, 2), strict=False)


class TieredLadderAndBudgetTests(unittest.TestCase):
    def test_tiered_retry_level1_same_batch_retry_success(self):
        """Level 1: First attempt returns bad format, retry attempt succeeds."""
        segments = [
            Segment(0.0, 1.0, "文1"),
            Segment(1.0, 2.0, "文2"),
            Segment(2.0, 3.0, "文3"),
            Segment(3.0, 4.0, "文4"),
        ]
        batch_attempts = [0]

        def translate_batch(b):
            batch_attempts[0] += 1
            if batch_attempts[0] == 1:
                # Missing line 4 on first attempt
                return "1. 譯1\n2. 譯2\n3. 譯3"
            # Retry succeeds with all 4 lines
            return "1. 譯1\n2. 譯2\n3. 譯3\n4. 譯4"

        def translate_one(text):
            self.fail("Should not reach single-cue fallback when Level 1 retry succeeds")

        run = translate_segments(
            segments,
            translate_batch,
            translate_one,
            batch_size=20,
            tiered_retry=True,
            budget_multiplier=2.0,
        )
        self.assertEqual(batch_attempts[0], 2)
        self.assertEqual(run.metrics["mismatch_batches"], 1)
        self.assertEqual(run.metrics["fallback_sentences"], 0)
        self.assertEqual([r.text for r in run.results], ["譯1", "譯2", "譯3", "譯4"])

    def test_tiered_retry_level2_bisection_success(self):
        """Level 2: Batch retry still fails, but bisected sub-batches succeed."""
        segments = [
            Segment(0.0, 1.0, "文1"),
            Segment(1.0, 2.0, "文2"),
            Segment(2.0, 3.0, "文3"),
            Segment(3.0, 4.0, "文4"),
        ]
        batch_calls = []

        def translate_batch(b):
            batch_calls.append(len(b.items))
            if len(b.items) == 4:
                # Full batch of 4 always fails format
                return "1. 只有一句"
            # Bisected batches (2 items each) succeed
            return "\n".join(f"{idx}. 譯_{seg.text}" for idx, seg in enumerate([s for _, s in b.items], 1))

        def translate_one(text):
            self.fail("Should not reach single-cue fallback when bisection succeeds")

        run = translate_segments(
            segments,
            translate_batch,
            translate_one,
            batch_size=20,
            tiered_retry=True,
            budget_multiplier=3.0,
        )
        # Full batch called twice (initial + Level 1 retry), then 2 bisected calls of size 2
        self.assertIn(4, batch_calls)
        self.assertIn(2, batch_calls)
        self.assertEqual(run.metrics["fallback_sentences"], 0)
        self.assertEqual(len(run.results), 4)
        for r in run.results:
            self.assertFalse(r.preserved_original)
            self.assertTrue(r.text.startswith("譯_"))

    def test_request_budget_enforcement_prevents_request_storm(self):
        """When everything fails, total calls must strictly not exceed the allocated budget."""
        segments = [Segment(float(i), float(i + 1), f"cue-{i}") for i in range(10)]
        call_count = [0]

        def fail_batch(b):
            call_count[0] += 1
            return "not-numbered"

        def fail_one(text):
            call_count[0] += 1
            raise RuntimeError("single call failed")

        budget_mult = 1.5
        run = translate_segments(
            segments,
            fail_batch,
            fail_one,
            batch_size=20,
            tiered_retry=True,
            budget_multiplier=budget_mult,
        )
        # Expected max budget for 10 items with multiplier 1.5 is ceil(1.5 * 10) = 15
        max_allowed = int(1.5 * 10)
        self.assertLessEqual(call_count[0], max_allowed)
        # All segments safely fallback to [原文]
        for r in run.results:
            self.assertTrue(r.preserved_original)
            self.assertTrue(r.text.startswith("[原文]"))


class RefusalDetectionAndAbortTests(unittest.TestCase):
    def test_is_refusal_text_variants(self):
        self.assertTrue(is_refusal_text("很抱歉，我無法協助此類請求。"))
        self.assertTrue(is_refusal_text("此內容違反安全政策"))
        self.assertTrue(is_refusal_text("I cannot translate this adult content"))
        self.assertTrue(is_refusal_text("I'm unable to fulfill this request per safety policy"))
        self.assertFalse(is_refusal_text("今天天氣真好，一起去散步吧。"))

    def test_content_filter_raises_provider_error_in_client(self):
        spec = {"provider": "openai-compatible", "remote_model": "test-llm", "base_url": "https://api.test/v1"}
        client = CloudTranslationClient(spec, "fake-key")

        # Mock finish_reason == 'content_filter'
        response_data = {
            "choices": [
                {
                    "message": {"role": "assistant", "content": None},
                    "finish_reason": "content_filter",
                }
            ]
        }
        with patch("oh_streamer.subtitles.model_adapters._request", return_value=response_data):
            with self.assertRaises(ProviderError) as ctx:
                client.translate("不當內容")
            self.assertEqual(ctx.exception.kind, "content_blocked")
            self.assertFalse(ctx.exception.retryable)
            self.assertEqual(client.metrics["content_blocked"], 1)

    def test_refusal_threshold_abort_prevents_cache_solidification(self):
        """When refusal rate exceeds 20%, translation raises content_blocked and aborts."""
        segments = [Segment(float(i), float(i + 1), f"adult-cue-{i}") for i in range(10)]

        # 3 out of 10 cues are refused (30% > 20% threshold)
        def batch_refusal(b):
            return "抱歉，此內容違反安全政策，無法提供翻譯。"

        def single_refusal(text):
            return "無法協助"

        with self.assertRaises(ProviderError) as ctx:
            translate_segments(
                segments,
                batch_refusal,
                single_refusal,
                batch_size=20,
                tiered_retry=True,
                refusal_threshold=0.20,
            )
        self.assertEqual(ctx.exception.kind, "content_blocked")
        self.assertIn("refused by provider safety filter", str(ctx.exception))

    def test_refusals_below_threshold_preserve_original_and_continue(self):
        """When refusal rate is <= 20% (e.g. 1 out of 10 cues), translation succeeds with [原文]."""
        segments = [Segment(float(i), float(i + 1), f"cue-{i}") for i in range(10)]

        def batch_partial(b):
            # Batch fails format to trigger single cue resolution
            return "bad format"

        def single_partial(text):
            if text == "cue-0":
                # Only 1 cue is refused (10% <= 20%)
                return "無法協助"
            return f"翻譯-{text}"

        run = translate_segments(
            segments,
            batch_partial,
            single_partial,
            batch_size=20,
            tiered_retry=False,
            refusal_threshold=0.20,
        )
        self.assertEqual(run.metrics["suspected_refusals"], 1)
        self.assertEqual(run.results[0].text, "[原文] cue-0")
        self.assertTrue(run.results[0].preserved_original)
        self.assertFalse(run.results[1].preserved_original)
        self.assertEqual(run.results[1].text, "翻譯-cue-1")


class CloudClientAttributesTests(unittest.TestCase):
    def test_cloud_translation_client_attributes(self):
        spec = {
            "id": "user-test",
            "provider": "openai-compatible",
            "remote_model": "llama-3.3-70b-versatile",
            "base_url": "https://api.groq.com/openai/v1",
        }
        client = CloudTranslationClient(
            spec,
            api_key="gsk-fake",
            prompt_family="general-llm",
            temperature=0.1,
            seed=42,
        )
        # Test attributes
        self.assertEqual(client.prompt_family, "general-llm")
        self.assertEqual(client.model, "general-llm")  # alias for backward compat
        self.assertEqual(client.remote_model, "llama-3.3-70b-versatile")
        self.assertEqual(client.provider, "openai-compatible")
        self.assertEqual(client.temperature, 0.1)
        self.assertEqual(client.seed, 42)

        # Setter works for both
        client.model = "sakura"
        self.assertEqual(client.prompt_family, "sakura")
        client.prompt_family = "general-llm"
        self.assertEqual(client.model, "general-llm")

    def test_sampling_options_passed_to_chat_completions(self):
        spec = {
            "provider": "openai-compatible",
            "remote_model": "test-model",
            "base_url": "https://example.com/v1",
        }
        client = CloudTranslationClient(spec, "key", temperature=0.2, seed=1234)

        captured_body = {}
        def mock_request(model, route, payload, api_key, content_type, transport=None):
            nonlocal captured_body
            captured_body = json.loads(payload.decode("utf-8"))
            return {
                "choices": [{"message": {"content": "1. 測試成功"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            }

        with patch("oh_streamer.subtitles.model_adapters._request", side_effect=mock_request):
            client.translate("1. テスト")

        self.assertEqual(captured_body.get("temperature"), 0.2)
        self.assertEqual(captured_body.get("seed"), 1234)
        self.assertEqual(captured_body.get("model"), "test-model")


class FullPipelineAndCacheProtectionTests(unittest.TestCase):
    def test_full_pipeline_with_general_llm_and_responses_logging(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "ja.json"
            source.write_text(
                json.dumps(
                    {
                        "segments": [
                            {"start": 0.0, "end": 2.0, "text": "こんにちは。"},
                            {"start": 2.0, "end": 4.0, "text": "良い天気ですね。"},
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            out_dir = root / "out"

            spec = {
                "id": "user-groq",
                "label": "Groq LLaMA",
                "kind": "translation",
                "provider": "openai-compatible",
                "base_url": "https://api.groq.com/openai/v1",
                "remote_model": "llama-3.3-70b-versatile",
                "builtin": False,
                "removable": True,
                "tested": True,
            }
            client = CloudTranslationClient(spec, "key", prompt_family="general-llm")

            mock_response = (
                "好的，以下是繁體中文翻譯：\n"
                "1. 你好。\n"
                "2. 天氣真好呢。\n"
                "祝您有美好的一天！"
            )
            with patch.object(client, "translate", return_value=mock_response):
                result = cli.translate_file(
                    source,
                    out_dir,
                    model="sakura",
                    model_path=None,
                    llama_server=None,
                    batch_size=20,
                    prompt_mode="general-llm",
                    shared_client=client,
                    gpu_managed=True,
                    model_identity="groq_identity",
                )

            # Assert translations parsed correctly despite commentary
            self.assertEqual(len(result["segments"]), 2)
            self.assertIn("你好", result["segments"][0]["text"])
            self.assertIn("天氣真好", result["segments"][1]["text"])

            # Verify responses file was written
            resp_file = out_dir / "sakura.responses.json"
            self.assertTrue(resp_file.exists())
            with open(resp_file, "r", encoding="utf-8") as f:
                saved_resps = json.load(f)
            self.assertGreater(len(saved_resps), 0)

class DynamicTokenAndPacingTests(unittest.TestCase):
    def test_estimate_translation_max_tokens_single_and_batch(self):
        from oh_streamer.subtitles.core import estimate_translation_max_tokens, Segment

        # Empty / None
        self.assertEqual(estimate_translation_max_tokens(None), 32)
        self.assertEqual(estimate_translation_max_tokens([]), 32)

        # Single short cue: "おはよう" (4 chars)
        # Content = max(ceil(4 * 1.25), 16) = 16
        # Overhead = 1 * 4 = 4
        # Margin = max(32, ceil(16 * 0.20)) = 32
        # Total = 16 + 4 + 32 = 52
        tok_short = estimate_translation_max_tokens("おはよう")
        self.assertEqual(tok_short, 52)

        # List of Segments (12 cues, average 15 chars)
        # Content = 12 * max(ceil(15 * 1.25), 16) = 12 * 19 = 228
        # Overhead = 12 * 4 = 48
        # Margin = max(32, ceil(228 * 0.20)) = 46
        # Total = 228 + 48 + 46 = 322
        cues = [Segment(float(i), float(i + 1), "あ" * 15) for i in range(12)]
        tok_batch = estimate_translation_max_tokens(cues)
        self.assertEqual(tok_batch, 322)

        # Chat message format
        messages = [
            {"role": "system", "content": "Translate to Traditional Chinese."},
            {"role": "user", "content": "1. おはよう。\n2. さようなら。"},
        ]
        tok_msg = estimate_translation_max_tokens(messages)
        self.assertGreater(tok_msg, 50)
        self.assertLess(tok_msg, 200)

    def test_filter_relevant_terms(self):
        from oh_streamer.subtitles.glossary import filter_relevant_terms

        glossary = [
            {"src": "会長", "dst": "會長", "info": ""},
            {"src": "生徒会", "dst": "學生會", "info": ""},
            {"src": "放課後", "dst": "放學後", "info": ""},
        ]
        text = "1. 会長、放課後に部室へ行きましょう。"
        filtered = filter_relevant_terms(glossary, text)
        self.assertEqual(len(filtered), 2)
        sources = {t["src"] for t in filtered}
        self.assertEqual(sources, {"会長", "放課後"})

        # Empty / non-matching
        self.assertEqual(filter_relevant_terms(glossary, "何もない"), [])
        self.assertEqual(filter_relevant_terms([], text), [])

    def test_cloud_translation_client_proactive_pacer(self):
        spec = {
            "kind": "translation",
            "provider": "openai-compatible",
            "base_url": "https://api.groq.com/openai/v1",
            "remote_model": "qwen/qwen3.8-27b",
            "api_family": "groq",
        }
        client = CloudTranslationClient(spec, "dummy_key")
        self.assertTrue(client._is_groq)
        self.assertEqual(client._pacer_otpm, 1000.0)

        # Available starts at 1000
        client._pacer_available = 1000.0
        client._pacer_last_refill = time.monotonic()
        client._pacer_last_req = time.monotonic() - 5.0  # past min interval

        # Acquire 300 tokens
        client._pacer_acquire(300)
        self.assertAlmostEqual(client._pacer_available, 700.0, delta=5.0)

        # Record usage with actual=180 -> refund 120 tokens
        client._pacer_record_usage(300, 180)
        self.assertAlmostEqual(client._pacer_available, 820.0, delta=5.0)


    def test_dual_filter_prompt_contains_skip_and_protected_slang(self):
        from oh_streamer.subtitles.model_adapters import GENERAL_LLM_SYSTEM_PROMPT
        from oh_streamer.subtitles.glossary import general_llm_messages

        messages = general_llm_messages("1. テスト")
        sys_msg = messages[0]["content"]

        for prompt_text in (sys_msg, GENERAL_LLM_SYSTEM_PROMPT):
            self.assertIn("[SKIP]", prompt_text)
            self.assertIn("純狀聲詞", prompt_text)
            self.assertIn("ボン、キュッ、ボン！", prompt_text)
            self.assertIn("プルンプルン", prompt_text)
            self.assertIn("チョメチョメ", prompt_text)
            self.assertIn("混合行", prompt_text)

    def test_translate_segments_skip_filtering(self):
        from oh_streamer.subtitles.core import Segment, translate_segments

        segments = [
            Segment(start=1.0, end=2.0, text="ちゅっ、ちゅぱっ"),
            Segment(start=2.5, end=4.0, text="こんにちは、元気ですか？"),
            Segment(start=4.5, end=6.0, text="ご視聴ありがとうございました"),
        ]

        def fake_translate_batch(batch):
            return "1. [SKIP]\n2. 你好，你好嗎？\n3. [SKIP]"

        def fake_translate_one(text):
            return "[SKIP]" if ("ちゅ" in text or "視聴" in text) else "你好，你好嗎？"

        run = translate_segments(segments, fake_translate_batch, fake_translate_one)
        results = run.results
        self.assertEqual(len(results), 3)

        self.assertTrue(results[0].is_filtered)
        self.assertEqual(results[0].text, "")
        self.assertEqual(results[0].filter_reason, "onomatopoeia_or_hallucination")

        self.assertFalse(results[1].is_filtered)
        self.assertEqual(results[1].text, "你好，你好嗎？")

        self.assertTrue(results[2].is_filtered)
        self.assertEqual(results[2].text, "")

    def test_is_pure_onomatopoeia_and_slang_protection(self):
        from oh_streamer.subtitles.core import is_pure_onomatopoeia, is_skip_text

        # Pure onomatopoeia should be recognized (including new spec tokens)
        self.assertTrue(is_pure_onomatopoeia("ちゅっ、ちゅぱっ……"))
        self.assertTrue(is_pure_onomatopoeia("ハハハハハ！うぇーい！"))
        self.assertTrue(is_pure_onomatopoeia("ふふっ、ああ…"))
        self.assertTrue(is_pure_onomatopoeia("んっ……じゅるっ"))
        self.assertTrue(is_pure_onomatopoeia("いえーい！イエーイ！"))

        # Protected comedic/character terms must NEVER be treated as pure sound
        self.assertFalse(is_pure_onomatopoeia("もう、ボン、キュッ、ボン！プルンプルン！"))
        self.assertFalse(is_pure_onomatopoeia("ボン・キュッ・ボン"))
        self.assertFalse(is_pure_onomatopoeia("プルンプルンですよぉ。"))
        self.assertFalse(is_pure_onomatopoeia("チョメチョメしちゃってくださいよ。"))
        self.assertFalse(is_pure_onomatopoeia("ほんま人たらしやわ！"))

        # Hallucinations
        self.assertTrue(is_skip_text("ご視聴ありがとうございました"))
        self.assertTrue(is_skip_text("チャンネル登録よろしくお願いします"))
        self.assertTrue(is_skip_text("チャンネル登録よろしく"))
        self.assertTrue(is_skip_text("[SKIP]"))
        self.assertTrue(is_skip_text("[skip]"))
        self.assertTrue(is_skip_text("[Skip]"))


if __name__ == "__main__":
    unittest.main()
