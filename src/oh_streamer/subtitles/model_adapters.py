"""Explicit local recipes and OpenAI-compatible inference adapters."""
from __future__ import annotations
import hashlib
import json
import logging
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
from urllib.parse import urlsplit
from urllib.request import Request, urlopen
import uuid
from . import runtime
from .core import estimate_translation_max_tokens
from .providers.http import (
    CloudAsrRatePacer,
    ProviderError,
    RequestGate,
    get_cloud_asr_pacer,
    http_get,
    http_post,
    parse_retry_after,
)

GROQ_SIBLING_MODELS = {
    "whisper-large-v3": "whisper-large-v3-turbo",
    "whisper-large-v3-turbo": "whisper-large-v3",
}

GENERAL_LLM_SYSTEM_PROMPT = (
    "你是一位精通日中影視字幕翻譯與審校的專家。\n"
    "請將以下日文字幕批次翻譯為繁體中文（台灣習慣用詞與正體中文）。\n\n"
    "【重要過濾與情節台詞保護規則】：\n"
    "1. 【純狀聲詞與幻聽過濾】：純粹為孤立歡呼、無語意笑聲、親吻吸吮、孤立喘息或呻吟（如：うぇーい、ハハハハハ、ちゅっ、ふふっ、ああ、ハァハァ等）以及 ASR 宣傳幻聽（ご視聴ありがとうございました、チャンネル登録等），明確指令輸出「[SKIP]」，不要輸出中文字幕。\n"
    "2. 【身材特徵形容與親熱暗語保護】：角色身材特徵形容（如「ボン、キュッ、ボン！」前凸後翹、「プルンプルン」Ｑ彈Ｑ彈）與親熱暗語（「チョメチョメ」嘿咻嘿咻/啪啪啪）為關鍵情節台詞，明訂嚴禁跳過，絕對不可輸出 [SKIP]，必須生動翻譯！\n"
    "3. 【混合行清理保留對話】：若句子開頭或結尾帶有狀聲詞（如「ぢゅぱっ、やめれない…」），去除句首句尾狀聲詞，保留實際台詞（如「停不下來……」）。\n"
    "4. 【輸出格式】：嚴格按照編號格式輸出，每行一個編號（格式為「數字. 譯文」或「數字. [SKIP]」），不得添加多餘解釋或代碼標記。"
)

class ModelAdapterError(ProviderError):
    def __init__(self, message='Cloud inference request failed', *, kind='server', **kwargs):
        super().__init__(kind, message, **kwargs)

def validate_spec(spec, settings=None):
    kind, provider = spec.get('kind'), spec.get('provider')
    if kind not in {'asr', 'translation'} or provider not in {'local', 'openai-compatible'}:
        raise ValueError('Unsupported model adapter')
    row = dict(kind=kind, provider=provider, label=str(spec.get('label') or spec.get('remote_model') or spec.get('recipe') or 'Model')[:120])
    if provider == 'local':
        recipe = str(spec.get('recipe') or '')
        if kind == 'asr':
            recipe, _ = runtime.resolve_model(recipe)
        else:
            from .combos import TRANSLATION_PROFILES
            if recipe not in TRANSLATION_PROFILES:
                raise ValueError('Unsupported local translation recipe')
        row.update(recipe=recipe, model_path=str(Path(spec['model_path']).expanduser().resolve()))
        if kind == 'translation':
            from .environment import detect
            row['llama_server_path'] = str(spec.get('llama_server_path') or detect(settings).get('server') or 'llama-server')
    else:
        base = str(spec.get('base_url') or '').rstrip('/')
        parsed = urlsplit(base)
        if parsed.scheme not in {'https', 'http'} or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError('Invalid endpoint')
        if parsed.scheme == 'http' and parsed.hostname not in {'localhost', '127.0.0.1', '::1'}:
            raise ValueError('Remote endpoints require HTTPS')
        remote = str(spec.get('remote_model') or '').strip()
        if not remote or len(remote) > 200 or any(ord(c) < 32 for c in remote):
            raise ValueError('Invalid remote model')
        row.update(base_url=base, remote_model=remote)
        api_family = str(spec.get('api_family') or '').strip()[:60]
        if api_family:
            row['api_family'] = api_family
        azure_region = str(spec.get('azure_region') or '').strip()[:60]
        if azure_region:
            row['azure_region'] = azure_region
    return row

class CloudTranslationClient:
    def __init__(self, model, api_key='', gate=None, prompt_family=None, temperature=0.0, seed=None):
        self._spec = model
        self._prompt_family = prompt_family or model.get('prompt_family') or 'sakura'
        self.translate_model = 'sakura-v1'
        self.remote_model = model.get('remote_model', '')
        self.provider = model.get('provider', 'openai-compatible')
        self.base_url = model.get('base_url', '')
        self._key = api_key
        self.gate = gate
        self.temperature = 0.0 if temperature is None else float(temperature)
        self.seed = seed
        self.metrics = dict(
            requests=0,
            failures=0,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            tokens=0,
            elapsed_seconds=0.0,
            content_blocked=0,
        )
        self._is_groq = (
            model.get('api_family') == 'groq'
            or 'api.groq.com' in self.base_url
        )
        self._pacer_lock = threading.Lock()
        if self._is_groq:
            # Groq free tier parameters: 1,000 OTPM, 30 RPM (2.0s min interval)
            self._pacer_otpm = 1000.0
            self._pacer_refill_rate = 1000.0 / 60.0  # 16.667 tokens/sec
            self._pacer_available = 1000.0
            self._pacer_last_refill = time.monotonic()
            self._pacer_last_req = 0.0
            self._pacer_min_interval = 60.0 / 30.0  # 2.0s min interval for 30 RPM

    def _pacer_acquire(self, requested_tokens: int):
        if not self._is_groq:
            return
        with self._pacer_lock:
            now = time.monotonic()
            elapsed = now - self._pacer_last_refill
            self._pacer_available = min(
                self._pacer_otpm,
                self._pacer_available + elapsed * self._pacer_refill_rate
            )
            self._pacer_last_refill = now

            time_since_last_req = now - self._pacer_last_req
            rpm_wait = max(0.0, self._pacer_min_interval - time_since_last_req)

            token_deficit = requested_tokens - self._pacer_available
            token_wait = (token_deficit / self._pacer_refill_rate) if token_deficit > 0 else 0.0

            wait_needed = max(rpm_wait, token_wait)
            if wait_needed > 0.05:
                time.sleep(wait_needed)
                now = time.monotonic()
                elapsed = now - self._pacer_last_refill
                self._pacer_available = min(
                    self._pacer_otpm,
                    self._pacer_available + elapsed * self._pacer_refill_rate
                )
                self._pacer_last_refill = now

            self._pacer_available = max(0.0, self._pacer_available - requested_tokens)
            self._pacer_last_req = time.monotonic()

    def _pacer_record_usage(self, requested_tokens: int, actual_tokens: int):
        if not self._is_groq:
            return
        with self._pacer_lock:
            refund = max(0, requested_tokens - actual_tokens)
            if refund > 0:
                self._pacer_available = min(self._pacer_otpm, self._pacer_available + refund)

    @property
    def prompt_family(self) -> str:
        return self._prompt_family

    @prompt_family.setter
    def prompt_family(self, val: str):
        self._prompt_family = str(val)

    @property
    def model(self) -> str:
        return self._prompt_family

    @model.setter
    def model(self, val: str):
        self._prompt_family = str(val)

    def __enter__(self):
        return self
    def __exit__(self, *args):
        self.close()
    def close(self):
        pass
    def __call__(self, prompt_or_messages, *, max_tokens=None):
        return self.translate(prompt_or_messages, max_tokens=max_tokens)
    def translate(self, prompt_or_messages, *, max_tokens=None):
        if self._spec.get('api_family') == 'azure-translator':
            return self._translate_azure(prompt_or_messages)
        if max_tokens is None:
            max_tokens = estimate_translation_max_tokens(prompt_or_messages)
        if self.gate:
            self.gate.acquire(requests=1, tokens=max_tokens)
        elif self._is_groq and not getattr(self, '_managed', False):
            self._pacer_acquire(max_tokens)
        if isinstance(prompt_or_messages, str):
            sys_content = (
                GENERAL_LLM_SYSTEM_PROMPT
                if self._prompt_family in ("general-llm", "qwen3")
                else "Translate Japanese to Traditional Chinese (Taiwan). Preserve every numbered line and its number. Return only translations."
            )
            messages = [
                {"role": "system", "content": sys_content},
                {"role": "user", "content": prompt_or_messages},
            ]
        else:
            messages = list(prompt_or_messages)
        body = dict(
            model=self._spec['remote_model'],
            messages=messages,
            temperature=self.temperature,
            stream=False,
            max_tokens=max_tokens,
        )
        if self.seed is not None:
            body['seed'] = self.seed
        max_retries = 0 if getattr(self, '_managed', False) else 3
        data = None
        for attempt in range(max_retries + 1):
            started = time.perf_counter()
            self.metrics['requests'] += 1
            try:
                data = _request(self._spec, '/chat/completions', json.dumps(body, ensure_ascii=False).encode(), self._key, 'application/json')
                break
            except ProviderError as pe:
                if pe.kind == 'rate_limit' and pe.retryable and attempt < max_retries:
                    wait_s = float(pe.retry_after) if pe.retry_after is not None else min(2.0 ** (attempt + 1), 30.0)
                    time.sleep(wait_s)
                    if not self.gate and self._is_groq:
                        with self._pacer_lock:
                            self._pacer_last_refill = time.monotonic()
                    continue
                self.metrics['failures'] += 1
                raise
            except Exception:
                self.metrics['failures'] += 1
                raise
            finally:
                self.metrics['elapsed_seconds'] += time.perf_counter() - started
        usage = data.get('usage') or {}
        for name in ('prompt_tokens','completion_tokens','total_tokens'):
            try:
                self.metrics[name] += max(0,int(usage.get(name,0)))
            except (ValueError,TypeError):
                pass
        self.metrics['tokens'] = self.metrics['completion_tokens']
        elapsed = self.metrics['elapsed_seconds']
        self.metrics['tokens_per_second'] = self.metrics['tokens'] / elapsed if elapsed else 0

        actual_tokens = 0
        try:
            actual_tokens = int(usage.get('completion_tokens', 0))
        except (ValueError, TypeError):
            pass
        if not self.gate and self._is_groq and not getattr(self, '_managed', False):
            self._pacer_record_usage(max_tokens, actual_tokens)

        try:
            choice = data['choices'][0]
            if choice.get('finish_reason') == 'content_filter':
                self.metrics['content_blocked'] += 1
                raise ProviderError('content_blocked', 'Content blocked by provider safety policy',
                                    retryable=False, scope='none')
            msg = choice.get('message') or {}
            text = msg.get('content')
            if not isinstance(text, str) or not text.strip():
                if data.get('prompt_feedback', {}).get('block_reason'):
                    self.metrics['content_blocked'] += 1
                    raise ProviderError('content_blocked', 'Content blocked by provider safety policy',
                                        retryable=False, scope='none')
                raise ValueError()
            if choice.get('finish_reason') == 'length':
                raise ValueError()
            import re
            cleaned = re.sub(r'(?s)<think>.*?</think>\s*', '', text).strip()
            return cleaned or text.strip()
        except ProviderError:
            raise
        except (KeyError,IndexError,TypeError,ValueError):
            raise ModelAdapterError('Invalid or truncated translation response', kind='bad_response') from None

    def _translate_azure(self, prompt_or_messages):
        """Translate numbered Japanese text using Azure Translator REST API v3.0."""
        import re, time
        # Extract the user input text
        if isinstance(prompt_or_messages, str):
            raw_text = prompt_or_messages
        elif isinstance(prompt_or_messages, list):
            user_msg = next((m.get('content', '') for m in reversed(prompt_or_messages) if m.get('role') == 'user'), '')
            raw_text = user_msg
        else:
            raw_text = str(prompt_or_messages)

        # Parse numbered lines: '1. こんにちは\n2. ありがとう'
        # Or if from general-llm prompt: '【請將以下日文...】\n1. ...'
        lines = []
        for match in re.finditer(r'(?m)^\s*(\d+)\s*[.．、:：)\-]\s*(.*)$', raw_text):
            idx = int(match.group(1))
            text = match.group(2).strip()
            lines.append((idx, text))

        if not lines:
            # Fallback for plain unnumbered text
            clean_lines = [l.strip() for l in raw_text.splitlines() if l.strip()]
            lines = [(i + 1, l) for i, l in enumerate(clean_lines)]

        if not lines:
            return ''

        # Prepare Azure Translator payload: [{'Text': '...'}, ...]
        payload_items = [{'Text': text} for _, text in lines]
        route = '/translate?api-version=3.0&from=ja&to=zh-Hant'
        max_retries = 0 if getattr(self, '_managed', False) else 3
        data = None
        started = time.perf_counter()
        self.metrics['requests'] += 1
        for attempt in range(max_retries + 1):
            try:
                data = _request(self._spec, route, json.dumps(payload_items, ensure_ascii=False).encode('utf-8'),
                                self._key, 'application/json')
                break
            except ProviderError as pe:
                if pe.kind == 'rate_limit' and pe.retryable and attempt < max_retries:
                    wait_s = float(pe.retry_after) if pe.retry_after is not None else min(2.0 ** (attempt + 1), 30.0)
                    time.sleep(wait_s)
                    continue
                self.metrics['failures'] += 1
                raise
            except Exception:
                self.metrics['failures'] += 1
                raise
            finally:
                self.metrics['elapsed_seconds'] += time.perf_counter() - started

        # Parse Azure response: array of objects with translations: [{'translations': [{'text': '...', 'to': 'zh-Hant'}]}]
        if not isinstance(data, list) or len(data) != len(lines):
            raise ModelAdapterError('Invalid or mismatched translation response from Azure', kind='bad_response')

        from .core import is_skip_text
        output_numbered = []
        for (idx, original_text), item in zip(lines, data):
            trans_list = item.get('translations') or []
            zh = trans_list[0].get('text', '').strip() if trans_list else ''
            if is_skip_text(original_text) or (zh and is_skip_text(zh)):
                output_numbered.append(f'{idx}. [SKIP]')
            else:
                output_numbered.append(f'{idx}. {zh}')

        return '\n'.join(output_numbered)

def _request(model, route, payload, api_key, content_type, transport=None):
    headers = {'Content-Type':content_type, 'Accept':'application/json', 'User-Agent':'OhStreamer/1.0'}
    if model.get('api_family') == 'azure-translator':
        if api_key:
            headers['Ocp-Apim-Subscription-Key'] = api_key
        region = model.get('azure_region')
        if region:
            headers['Ocp-Apim-Subscription-Region'] = region
    elif api_key:
        headers['Authorization'] = 'Bearer ' + api_key
    url = model['base_url'] + route
    try:
        from . import usage_log
        from . import usage_monitor

        def _observe(status, resp_headers, body):
            usage_monitor.record_for_row(model, route, status, resp_headers, body, model.get('_usage_audio_s'))
        return usage_log.timed_request(
            model, route, lambda: http_post(url, payload, headers, transport=transport, on_response=_observe))
    except ProviderError:
        raise
    except Exception:
        raise ModelAdapterError('Cloud inference request failed', kind='server') from None

def make_translation_client(model_id, settings=None, **kwargs):
    from .configuration import runtime_plan
    plan = runtime_plan(settings)
    role = 'translation' if plan and model_id == plan['models']['translation']['id'] else None
    if role:
        row = plan['models'][role]
        from .combos import is_cloud_enabled
        from .model_registry import is_cloud_model
        if is_cloud_model(row) and not is_cloud_enabled(settings):
            raise ModelAdapterError('Cloud subtitles disabled', kind='auth')
        client = FallbackTranslationClient(plan, settings, role=role, **kwargs)
        client.base_url = row.get('base_url', '')
        return client
    for name in ('gpu_managed', 'check_cancel', 'on_event'):
        kwargs.pop(name, None)
    row, key = _runtime_model(model_id, 'translation', settings)
    if row['kind'] != 'translation':
        raise ValueError('Model is not a translator')
    return _translation_client(row, key, settings=settings, **kwargs)


def _binding_credentials(row, settings):
    from .combos import is_cloud_enabled
    from .model_registry import connection_api_key, model_api_key
    from .configuration import runtime_plan
    if row['provider'] == 'local':
        return ''
    plan = runtime_plan(settings)
    from .model_registry import is_cloud_model
    if is_cloud_model(row) and (not is_cloud_enabled(settings) or (plan and plan['mode'] == 'local')):
        # A consent boundary is not an inference error eligible for fallback.
        raise PermissionError('Cloud subtitles disabled')
    try:
        return (connection_api_key(row['connection_id'], settings) if row.get('connection_id')
                else model_api_key(row))
    except (ValueError, KeyError):
        raise ProviderError('auth', 'Subtitle connection unavailable') from None


def _local_ready(row, gpu_managed, settings=None):
    from .environment import model_file
    mpath = model_file(row, settings)
    if row.get('recipe') == '__dual__':
        if not mpath:
            raise ProviderError('unsupported', 'Local subtitle model missing')
        if not gpu_managed:
            raise ProviderError('unsupported', 'Local subtitle model requires owned GPU worker')
        path = Path(mpath)
        from .runtime import local_model_fingerprint, model_dir_name
        p1 = path / model_dir_name('whisper-ja-1.5B')
        p2 = path / model_dir_name('anime-whisper')
        try:
            local_model_fingerprint(p1)
            local_model_fingerprint(p2)
        except OSError:
            raise ProviderError('unsupported', 'Local speech model incomplete') from None
        return
    if not mpath or not Path(mpath).exists():
        raise ProviderError('unsupported', 'Local subtitle model missing')
    if row.get('kind') == 'translation' and not Path(mpath).is_file():
        raise ProviderError('unsupported', 'Local translation model missing')
    if not gpu_managed:
        raise ProviderError('unsupported', 'Local subtitle model requires owned GPU worker')
    if row.get('kind') == 'asr':
        try:
            runtime.local_model_fingerprint(mpath)
        except OSError:
            raise ProviderError('unsupported', 'Local speech model incomplete') from None


def _translation_operation(prompt, max_tokens, client_kwargs, settings, gpu_managed, metrics_out):
    """operation(row, key) translating one numbered prompt on one binding (shared by the Policy client and the
    single-candidate client). Refusals are content_blocked; a broken line mapping is bad_response (no cooldown)."""
    import re
    from .core import parse_numbered_translations, is_refusal_text, numbered_mismatch
    text = prompt if isinstance(prompt, str) else next(
        (m.get('content', '') for m in reversed(prompt) if m.get('role') == 'user'), '')
    ids = [int(m.group(1)) for m in re.finditer(r'(?m)^\s*(\d+)\s*[.．、:：)\-][ \t]*', text)]

    def operation(row, key):
        row = dict(row)
        if row['provider'] == 'local':
            _local_ready(row, gpu_managed, settings)
            from .environment import runtime_paths
            row.setdefault('llama_server_path', runtime_paths(settings)['server'])
        client = _translation_client(row, key, settings=settings, **client_kwargs)
        client._managed = True
        try:
            result = client.translate(prompt, max_tokens=max_tokens)
            if is_refusal_text(result):
                raise ProviderError('content_blocked', 'Translation refused')
            if ids:
                try:
                    # one line cannot be shifted: any answer numbered 1 is its translation
                    parse_numbered_translations(result, ids, strict=len(ids) > 1)
                except (ValueError, TypeError) as exc:
                    from . import usage_log
                    why = str(exc)
                    reason = next((code for word, code in (('unnumbered', 'unnumbered'), ('empty', 'empty'),
                                                           ('do not match', 'ids'), ('missing', 'ids'))
                                   if word in why), 'other')
                    usage_log.request_event('mt_mismatch', reason=reason, **numbered_mismatch(result, ids))
                    mismatch = ProviderError('bad_response', 'Translation line mapping invalid')
                    mismatch.no_cooldown = True
                    raise mismatch from None
            return result
        except (InterruptedError, PermissionError, ProviderError):
            raise
        except Exception:
            if row['provider'] != 'local':
                raise
            raise ProviderError('server', 'Local translation failed', retryable=False) from None
        finally:
            metrics = getattr(client, 'metrics', {})
            for name, value in (metrics.items() if isinstance(metrics, dict) else ()):
                if isinstance(value, (int, float)):
                    metrics_out[name] = metrics_out.get(name, 0) + value
            client.close()
    return operation


class FallbackTranslationClient:
    """One policy operation per numbered batch, releasing local models between calls."""
    def __init__(self, plan, settings, role='translation', **kwargs):
        from .fallback import Policy
        self.settings = settings
        self.primary = plan['models'][role]
        self.gpu_managed = kwargs.pop('gpu_managed', False)
        self.policy = Policy(plan, role, plan['models'][role],
                             check_cancel=kwargs.pop('check_cancel', None),
                             on_event=kwargs.pop('on_event', None))
        self.kwargs = kwargs
        self.prompt_family = kwargs.get('prompt_family', 'general-llm')
        self.translate_model = kwargs.get('translate_model', 'sakura-v1')
        self.metrics = {}

    @property
    def last_source(self):
        return self.policy.last_source

    @property
    def _key(self):
        return _binding_credentials(self.primary, self.settings)

    def translate(self, prompt, *, max_tokens=None):
        operation = _translation_operation(prompt, max_tokens, self.kwargs, self.settings, self.gpu_managed,
                                           self.metrics)
        return self.policy.run(operation, lambda row: _binding_credentials(row, self.settings),
                               tokens=max_tokens or estimate_translation_max_tokens(prompt))

    def __call__(self, prompt, **kwargs):
        return self.translate(prompt, **kwargs)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        pass  # No model or server survives an individual attempt.


# --------------------------------------------------------------------------------------------------
# M4b: one candidate, one run (the slice scheduler's entry points; no Policy, never sleeps)
# --------------------------------------------------------------------------------------------------
#
# The slice executors pick the candidate themselves (fallback.CandidateChain) and hand in `run`:
# run(operation, *, tokens=0, audio=0) -> operation(row, key) executed once on that candidate, normally
# fallback.run_candidate (Ledger claim + Groq admission + request; quota waits raise `not_sent`).
# The Policy-based FallbackTranslationClient / run_registered_asr stay for the legacy paths (until M7).

class CandidateRunner:
    """Policy-shaped adapter (`.run(operation, credentials, *, tokens, audio)`, `.last_source`) around the
    executor's `run`, so _cloud_asr's managed per-chunk path runs exactly one candidate. `credentials` is
    ignored: the executor's `run` brings its own."""
    def __init__(self, row, run):
        from .fallback import identity
        self.row = row
        self._run = run
        self.last_source = identity(row)

    def run(self, operation, credentials=None, *, tokens=0, audio=0):
        return self._run(operation, tokens=tokens, audio=audio)


class CandidateTranslationClient:
    """Numbered-batch translation client bound to ONE candidate row: every translate() is one run of that
    candidate through `run` (no fallback, no sleeping). last_source = the candidate's identity."""
    def __init__(self, row, settings, run, **kwargs):
        from .fallback import identity
        self.row = dict(row)
        self.settings = settings
        self._run = run
        self.gpu_managed = kwargs.pop('gpu_managed', False)
        for name in ('check_cancel', 'on_event'):
            kwargs.pop(name, None)
        self.kwargs = kwargs
        self.prompt_family = kwargs.get('prompt_family', 'general-llm')
        self.translate_model = kwargs.get('translate_model', 'sakura-v1')
        self.base_url = self.row.get('base_url', '')
        self.metrics = {}
        self.last_source = identity(self.row, settings)

    def translate(self, prompt, *, max_tokens=None):
        operation = _translation_operation(prompt, max_tokens, self.kwargs, self.settings, self.gpu_managed,
                                           self.metrics)
        return self._run(operation, tokens=max_tokens or estimate_translation_max_tokens(prompt))

    def __call__(self, prompt, **kwargs):
        return self.translate(prompt, **kwargs)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        pass


def make_candidate_translation_client(row, run, settings=None, **kwargs):
    """Translation client for one frozen-plan candidate row (slice scheduler). Cloud consent is enforced by
    the executor's credentials lookup (_binding_credentials raises PermissionError), as in the Policy path."""
    if row.get('kind', 'translation') != 'translation':
        raise ValueError('Model is not a translator')
    return CandidateTranslationClient(row, settings, run, **kwargs)


def run_asr_candidate(row, input_path, out_dir, run, settings=None, **kwargs):
    """Cloud ASR of one file on ONE candidate row (no Policy, no sibling switching, no sleeping): every
    upload chunk is one `run`. The chunk checkpoint in out_dir is keyed by the candidate's identity, so a
    retry of the same candidate with the same out_dir resumes after the chunks already transcribed."""
    from .fallback import identity
    if row.get('provider') == 'local':
        raise ProviderError('unsupported', 'Local ASR runs in the local executor')
    kwargs.pop('on_event', None)
    kwargs.update(fallback_policy=CandidateRunner(row, run), runtime_plan={'candidate': identity(row)},
                  allow_model_fallback=False,
                  credentials=lambda candidate: _binding_credentials(candidate, settings))
    return _cloud_asr(row, input_path, out_dir, settings=settings, **kwargs)


def _runtime_model(model_id, kind, settings):
    from .configuration import frozen_binding, runtime_plan
    from .model_registry import resolve_model, model_api_key, connection_api_key
    from .combos import is_cloud_enabled
    row = frozen_binding(settings, model_id, kind)
    if row is None:
        row = resolve_model(model_id, kind, settings)
    if row['provider'] != 'local':
        from .model_registry import is_cloud_model
        if settings is not None and is_cloud_model(row) and not is_cloud_enabled(settings):
            raise ModelAdapterError('Cloud subtitles disabled', kind='auth')
        key = (connection_api_key(row['connection_id'], settings)
               if runtime_plan(settings) and row.get('connection_id') else model_api_key(row))
    else:
        key = ''
        if kind == 'translation':
            from .environment import runtime_paths
            row.setdefault('llama_server_path', runtime_paths(settings)['server'])
    return row, key

def _translation_client(row, api_key='', settings=None, **kwargs):
    if row['provider'] == 'openai-compatible':
        prompt_family = kwargs.get('prompt_family') or kwargs.get('model') or 'sakura'
        temp = kwargs.get('temperature', 0.0)
        seed = kwargs.get('seed')
        client = CloudTranslationClient(
            row,
            api_key,
            prompt_family=prompt_family,
            temperature=temp,
            seed=seed,
        )
        client.translate_model = 'qwen3' if prompt_family == 'qwen3' else kwargs.get('translate_model', 'sakura-v1')
        return client
    from .llm import LlamaServerClient
    from .environment import model_file
    profile = row['recipe']
    kwargs.pop('prompt_family', None)
    kwargs.pop('translate_model', None)
    kwargs.pop('model', None)
    kwargs.pop('model_path', None)
    kwargs.pop('llama_server_path', None)
    return LlamaServerClient(model='qwen3' if profile == 'qwen3' else 'sakura',
        model_path=model_file(row, settings), llama_server_path=row['llama_server_path'],
        translate_model=profile if profile != 'qwen3' else 'sakura-v1', **kwargs)

# Groq reports segment times on whole seconds for much of a long file; the word times (10 ms) are precise.
# Bump when the way cloud ASR times are produced changes: it is part of every ASR cache / checkpoint identity.
ASR_TIMING_VERSION = 2
_log = logging.getLogger(__name__)


def refine_segments_with_words(segments, words, *, label=''):
    """Re-time every segment that matches the words from those words (word_timing.words_span: the first word's
    stretched start is capped, the end gets a short tail pad). Segments whose text the words do not spell keep
    their own times. Every segment that matched the words also keeps them (``words``: word / start / end, the reply's own times) for later re-alignment.
    One summary line per call (info), the reason of every rejected or accepted boundary at debug level."""
    segments = [dict(s) for s in (segments or [])]
    if not segments:
        return segments
    if not isinstance(words, list) or not words:
        _log.info('Cloud ASR word refinement %s: no word timestamps in the reply, %d segments keep their '
                  'boundaries', label, len(segments))
        return segments
    stats = dict(refined=0, precise=0, no_match=0, short=0, degenerate=0)
    try:
        import difflib
        word_text, word_of = [], []      # whitespace-free word text, owning word index of every char
        for i, w in enumerate(words):
            piece = ''.join(str(w['word']).split())
            word_text.append(piece)
            word_of.extend([i] * len(piece))
        seg_text = [''.join(str(s['text']).split()) for s in segments]
        joined_words, joined_segs = ''.join(word_text), ''.join(seg_text)
        # Whisper may drop repeated / hallucinated text from the word list, so align instead of comparing.
        matched = [None] * len(joined_segs)      # segment char -> word index
        matcher = difflib.SequenceMatcher(None, joined_segs, joined_words, autojunk=False)
        for x, y, size in matcher.get_matching_blocks():
            for k in range(size):
                matched[x + k] = word_of[y + k]
        previous = None
        pos = 0
        for n, (seg, text) in enumerate(zip(segments, seg_text)):
            hits = [m for m in matched[pos:pos + len(text)] if m is not None]
            pos += len(text)
            start, end = float(seg['start']), float(seg['end'])
            if end <= start:
                stats['degenerate'] += 1
            if text and len(hits) * 2 >= len(text):
                seg['words'] = [dict(word=str(w['word']), start=float(w['start']), end=float(w['end']))
                                for w in words[hits[0]:hits[-1] + 1]]
                from .word_timing import clean_words, words_span
                span = words_span(clean_words(seg['words']))
                if span is not None and span[1] - span[0] >= 0.05:
                    if span != (start, end):
                        stats['refined'] += 1
                        _log.debug('Cloud ASR word refinement %s seg %d: %.3f-%.3f -> %.3f-%.3f', label, n,
                                   start, end, span[0], span[1])
                    else:
                        stats['precise'] += 1
                    start, end = span
                else:
                    stats['short'] += 1
                    _log.debug('Cloud ASR word refinement %s seg %d: kept %.3f-%.3f (no usable word span)',
                               label, n, start, end)
            else:
                stats['no_match'] += 1
                _log.debug('Cloud ASR word refinement %s seg %d: kept %.3f-%.3f (words match %d of %d chars)',
                           label, n, start, end, len(hits), len(text))
            if previous is not None and start < previous['end']:
                if start - previous['start'] >= 0.05:
                    previous['end'] = start          # pull the earlier cue back rather than overlap
                else:
                    start = previous['end']
                    end = max(end, start + 0.05)
            seg['start'], seg['end'] = start, end
            previous = seg
    except (KeyError, TypeError, ValueError):
        _log.info('Cloud ASR word refinement %s: malformed word list, segments left untouched', label)
        return [dict(s) for s in (segments or [])]
    _log.info('Cloud ASR word refinement %s: %d segments, %d refined, %d already precise; rejected: %d no word '
              'match, %d too short; %d degenerate in the reply', label, len(segments),
              stats['refined'], stats['precise'], stats['no_match'], stats['short'], stats['degenerate'])
    return segments


def _asr_form_fields(model_name):
    return [('model', model_name), ('language', 'ja'), ('response_format', 'verbose_json'),
            ('timestamp_granularities[]', 'segment'), ('timestamp_granularities[]', 'word')]


def _cloud_asr(row, source, out_dir, api_key='', **kwargs):
    from .core import Segment, render_srt
    from .cloud_chunks import (
        detect_silence_intervals,
        plan_chunks,
        clamp_and_validate_segments,
        deduplicate_segments,
        reconstruct_timeline,
        extract_chunk_audio,
        ChunkCheckpoint,
        ChunkSpec,
    )
    import dataclasses
    import wave

    segment_fields = {f.name for f in dataclasses.fields(Segment)}      # rows also carry words / timing_repair
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    wav = out / 'cloud-input.wav'
    runtime.resample_to_wav(source, wav)
    duration = runtime.wav_duration(wav)
    on_phase = kwargs.get('on_phase')
    on_progress = kwargs.get('on_progress')

    def report(completed, detail=None):
        percent = min(100, max(0, int(completed * 100 / duration))) if duration else 100
        msg = detail or f'雲端語音辨識：已完成 {percent}%'
        if callable(on_phase):
            on_phase('asr', percent, msg)
        if callable(on_progress):
            on_progress(completed, duration)

    report(0)

    # Compute audio hash for checkpoint validation
    audio_hash = hashlib.sha256(wav.read_bytes()).hexdigest()
    checkpoint_identity = dict(audio=audio_hash, model={k: row.get(k) for k in
                               ('id', 'connection_id', 'base_url', 'remote_model')},
                               allow_model_fallback=kwargs.get('allow_model_fallback', True),
                               asr_timing=ASR_TIMING_VERSION)
    if kwargs.get('fallback_policy'):
        from .fallback import RUNTIME_VERSION
        checkpoint_identity['runtime_version'] = RUNTIME_VERSION
        checkpoint_identity['runtime_plan'] = kwargs['runtime_plan']
    checkpoint = ChunkCheckpoint(out / 'chunks', hashlib.sha256(
        json.dumps(checkpoint_identity, sort_keys=True).encode()).hexdigest())

    # Rate pacer & sibling fallback setup
    base_url = str(row.get('base_url', ''))
    pacer = kwargs.get('pacer') or get_cloud_asr_pacer(base_url)
    primary_model = str(row.get('remote_model', ''))
    allow_model_fallback = kwargs.get('allow_model_fallback', True)
    sibling_model = GROQ_SIBLING_MODELS.get(primary_model) if allow_model_fallback else None
    active_model = primary_model
    managed_policy = kwargs.get('fallback_policy')

    def _transcribe_piece(piece_path: Path, piece_duration: float, model_name: str, label: str = 'chunk') -> list[dict[str, Any]]:
        boundary = 'subtitle-' + uuid.uuid4().hex
        pieces = []
        for key, value in _asr_form_fields(model_name):
            pieces.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n{value}\r\n'.encode())

        is_flac = piece_path.suffix.lower() == '.flac'
        filename = "chunk.flac" if is_flac else "audio.wav"
        mime = "audio/flac" if is_flac else "audio/wav"

        pieces += [
            f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{filename}"\r\nContent-Type: {mime}\r\n\r\n'.encode(),
            piece_path.read_bytes(),
            f'\r\n--{boundary}--\r\n'.encode(),
        ]
        req_row = dict(row, remote_model=model_name, _usage_audio_s=piece_duration)
        data = _request(req_row, '/audio/transcriptions', b''.join(pieces), api_key, 'multipart/form-data; boundary=' + boundary)
        raw_segments = data.get('segments')
        if isinstance(raw_segments, list):
            raw_segments = refine_segments_with_words(raw_segments, data.get('words'), label=label)
        return clamp_and_validate_segments(raw_segments, piece_duration, label=label)

    silences = detect_silence_intervals(wav)
    skip_long_silence_sec = kwargs.get('skip_long_silence_sec', 10.0)
    chunks_plan = plan_chunks(
        duration,
        target_chunk_sec=300.0,
        silence_intervals=silences,
        skip_long_silence_sec=skip_long_silence_sec,
    )

    combined = []
    chunk_results = []
    processed_chunks = []
    sources = []

    for chunk in chunks_plan:
        if kwargs.get("check_cancel"):
            kwargs["check_cancel"]()
        idx = chunk[0]
        chunk_start = chunk[1]
        chunk_end = chunk[2]
        chunk_duration = chunk_end - chunk_start

        # Check if this chunk is already cached in checkpoint
        cached_segments = checkpoint.get_chunk(idx)
        if cached_segments is not None:
            part_segments = cached_segments
            cached_source = checkpoint.get_source(idx)
            if cached_source:
                sources.append(cached_source)
        elif managed_policy:
            piece = out / f'cloud-part-{idx}.flac'
            extract_chunk_audio(wav, piece, chunk_start, chunk_duration, fmt='flac')
            def transcribe(candidate, key):
                if candidate['provider'] == 'local':
                    _local_ready(candidate, kwargs.get('gpu_managed', False), kwargs.get('settings'))
                    from .environment import model_file
                    candidate_path = model_file(candidate, kwargs.get('settings'))
                    local_options = {k: v for k, v in kwargs.items() if k in
                                     ('gpu_managed', 'segmentation', 'vad_threshold', 'generate_overrides', 'cache_dir')}
                    local_folder = out / ('local-' + hashlib.sha256(json.dumps(candidate, sort_keys=True).encode()).hexdigest()[:16]) / str(idx)
                    try:
                        if candidate.get('recipe') == '__dual__':
                            from .pipeline import run_production_asr
                            models_dir = candidate_path
                            doc = run_production_asr(piece, local_folder, models_dir=models_dir, asr_mode='dual',
                                                     force=True, **local_options)
                            result = doc if (doc and 'segments' in doc) else json.loads((local_folder / 'ja.json').read_text('utf-8'))
                        else:
                            from .asr import run_asr
                            result = run_asr(piece, local_folder, model=candidate['recipe'],
                                             model_path=candidate_path, **local_options)
                    except (InterruptedError, PermissionError, ProviderError):
                        raise
                    except Exception:
                        raise ProviderError('server', 'Local speech recognition failed', retryable=False) from None
                    return clamp_and_validate_segments(result['segments'], chunk_duration, label=f'chunk {idx} (local)')
                nonlocal row, api_key
                previous_row, previous_key = row, api_key
                try:
                    row, api_key = candidate, key
                    return _transcribe_piece(piece, chunk_duration, candidate['remote_model'],
                                             f"chunk {idx} ({candidate['remote_model']})")
                finally:
                    row, api_key = previous_row, previous_key
            try:
                part_segments = managed_policy.run(transcribe, kwargs['credentials'], audio=chunk_duration)
                source_record = dict(index=idx, start=chunk_start, end=chunk_end, model=managed_policy.last_source)
                checkpoint.save_chunk(idx, chunk_start, chunk_end, part_segments, source=source_record)
                sources.append(source_record)
            finally:
                piece.unlink(missing_ok=True)
            if kwargs.get('check_cancel'):
                kwargs['check_cancel']()
        else:
            # Check primary model ASH budget; switch to sibling if primary exhausted
            if sibling_model and active_model == primary_model:
                afford, wait_s = pacer.can_afford(primary_model, chunk_duration, ash_limit=6900.0)
                if not afford:
                    sib_afford, _ = pacer.can_afford(sibling_model, chunk_duration, ash_limit=6900.0)
                    if sib_afford:
                        active_model = sibling_model
                        report(chunk_start, f"主模型額度滿載，自動切換至 {active_model}")

            piece = out / f'cloud-part-{idx}.flac'
            extract_chunk_audio(wav, piece, chunk_start, chunk_duration, fmt="flac")

            max_attempts = 2 if sibling_model else 1
            part_segments = None

            for attempt in range(max_attempts):
                try:
                    pacer.acquire(active_model, chunk_duration, max_wait=30.0)
                    part_segments = _transcribe_piece(piece, chunk_duration, active_model, f'chunk {idx} ({active_model})')
                    pacer.record_usage(active_model, chunk_duration)
                    break
                except ProviderError as pe:
                    is_rate_limit = (pe.kind == 'rate_limit' or getattr(pe, 'http_status', None) == 429)
                    retry_after = getattr(pe, 'retry_after', None)
                    if is_rate_limit and sibling_model and active_model != sibling_model:
                        if retry_after is None or retry_after > 30.0:
                            active_model = sibling_model
                            report(chunk_start, f"主模型額度滿載，自動切換至 {active_model}")
                            continue
                        elif retry_after <= 30.0 and attempt == 0:
                            time.sleep(retry_after + 0.5)
                            continue
                    raise
                finally:
                    pass

            piece.unlink(missing_ok=True)
            checkpoint.save_chunk(idx, chunk_start, chunk_end, part_segments)
            if kwargs.get("check_cancel"):
                kwargs["check_cancel"]()

        chunk_results.append(part_segments)
        processed_chunks.append(chunk)
        report(chunk_end)

    combined = reconstruct_timeline(processed_chunks, chunk_results)

    result = dict(
        model=((sources[0]['model'].get('remote_model') or sources[0]['model']['id']) if managed_policy and sources and
               all(s['model'] == sources[0]['model'] for s in sources) else
               'fallback-mixed' if managed_policy else row.get('id', row.get('remote_model'))),
        sources=sources,
        segments=combined,
        raw_segments=combined,
        speech_intervals=[[s['start'], s['end']] for s in combined],
        metrics={},
        duration=duration,
    )
    runtime.atomic_write_json(out / 'ja.json', result)
    runtime.atomic_write_text(out / 'ja.srt', render_srt([Segment(**{k: v for k, v in s.items() if k in segment_fields}) for s in combined]))
    return result

def run_registered_asr(model_id, input_path, out_dir, settings=None, **kwargs):
    from .configuration import runtime_plan
    plan = runtime_plan(settings)
    if plan and model_id == plan['models']['asr']['id'] and plan['models']['asr'].get('provider') != 'local':
        from .fallback import Policy, cache_identity
        row = plan['models']['asr']
        policy = Policy(plan, 'asr', row, check_cancel=kwargs.get('check_cancel'), on_event=kwargs.get('on_event'))
        kwargs.update(fallback_policy=policy, runtime_plan=cache_identity(plan, settings), allow_model_fallback=False,
                      credentials=lambda candidate: _binding_credentials(candidate, settings))
        return _cloud_asr(row, input_path, out_dir, settings=settings, **kwargs)
    row, key = _runtime_model(model_id, 'asr', settings)
    if row['kind'] != 'asr':
        raise ValueError('Model is not ASR')
    if row['provider'] == 'openai-compatible':
        plan = runtime_plan(settings)
        if plan:
            sibling = GROQ_SIBLING_MODELS.get(row.get('remote_model'))
            kwargs['allow_model_fallback'] = plan['whisper_policy'] == 'auto' and sibling in plan['whisper_models']
        return _cloud_asr(row, input_path, out_dir, key, settings=settings, **kwargs)
    if row.get('recipe') == '__dual__':
        from .pipeline import run_production_asr
        from .environment import model_file
        models_dir = model_file(row, settings)
        return run_production_asr(input_path, out_dir, models_dir=models_dir, asr_mode='dual', **kwargs)
    from .asr import run_asr
    kwargs.pop('check_cancel', None)
    kwargs.pop('model_path',None)
    kwargs.pop('model',None)
    from .environment import model_file
    return run_asr(input_path,out_dir,model=row['recipe'],model_path=model_file(row, settings),**kwargs)

def _write_builtin_test_wav(target: Path, duration_sec: float = 1.2, rate: int = 16000) -> None:
    import struct
    import wave
    n_frames = int(duration_sec * rate)
    buf = bytearray(n_frames * 2)
    for i in range(n_frames):
        t = i / rate
        # Modulated speech-band envelope (0.2s..0.9s) with leading/trailing silence
        if 0.15 <= t <= 0.95:
            sample = int(6000 * math.sin(2 * math.pi * 440.0 * t) * (0.5 + 0.5 * math.sin(2 * math.pi * 6.0 * t)))
        else:
            sample = 0
        struct.pack_into('<h', buf, i * 2, sample)
    with wave.open(str(target), 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(bytes(buf))


def test_model(row, *, api_key='', test_audio_path=None, settings=None):
    use_builtin_audio = (
        row['kind'] == 'asr'
        and row.get('provider') != 'local'
        and str(test_audio_path or '').strip() == '__builtin__'
    )
    if row['kind'] == 'asr' and not use_builtin_audio and (not test_audio_path or not Path(test_audio_path).is_file()):
        raise ValueError('A real speech sample is required')
    if row['provider'] == 'local':
        # Isolate heavy model imports and release GPU allocations when testing ends.
        payload = dict(row=row,test_audio_path=str(test_audio_path) if test_audio_path else None)
        script = 'import json,sys; from oh_streamer.subtitles.model_adapters import _test_local; _test_local(**json.load(sys.stdin))'
        from .environment import detect
        import os
        detected = detect(settings)
        executable = detected['python']
        if not Path(executable).is_file():
            raise ModelAdapterError('Subtitle Python runtime is unavailable')
        environment = dict(os.environ)
        source_root = str(Path(__file__).resolve().parents[2])
        environment['PYTHONPATH'] = source_root + os.pathsep + environment.get('PYTHONPATH','')
        proc = subprocess.run([executable,'-c',script], input=json.dumps(payload), text=True, env=environment,
                              stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=600,
                              creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        if proc.returncode:
            raise ModelAdapterError('Local model inference failed')
    elif row['kind'] == 'translation':
        text = CloudTranslationClient(row,api_key).translate('1. こんにちは。\n2. ありがとう。')
        import re
        numbers = [int(m.group(1)) for m in re.finditer(r'(?m)^\s*(\d+)\s*[.．、:：)\-][ \t]*\S[^\n]*', text)]
        if sorted(numbers) != [1, 2]:
            raise ModelAdapterError('Translation must preserve numbered lines', kind='bad_response')
    else:
        with tempfile.TemporaryDirectory(prefix='subtitle-model-test-') as tmp:
            if use_builtin_audio:
                audio_file = Path(tmp) / 'builtin-probe.wav'
                _write_builtin_test_wav(audio_file)
                boundary = 'subtitle-' + uuid.uuid4().hex
                pieces = []
                for key, value in _asr_form_fields(row['remote_model']):
                    pieces.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n{value}\r\n'.encode())
                pieces += [f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="audio.wav"\r\nContent-Type: audio/wav\r\n\r\n'.encode(), audio_file.read_bytes(), f'\r\n--{boundary}--\r\n'.encode()]
                data = _request(row, '/audio/transcriptions', b''.join(pieces), api_key, 'multipart/form-data; boundary=' + boundary)
                if not isinstance(data, dict) or ('segments' not in data and 'text' not in data):
                    raise ModelAdapterError('Inference test requires valid transcription response', kind='bad_response')
            else:
                # Capability validation must exercise the requested model itself.
                result = _cloud_asr(row,test_audio_path,tmp,api_key,allow_model_fallback=False)
                if not result['segments']:
                    raise ModelAdapterError('Inference test requires speech with timestamps', kind='bad_response')

def _test_local(row, test_audio_path=None):
    from .environment import model_file
    if row['kind'] == 'translation':
        from . import runtime
        runtime.assert_gpu_safe()
        with _translation_client(row) as client:
            if not client.translate('こんにちは。').strip():
                raise ModelAdapterError('Empty translation')
    else:
        from .asr import run_asr
        with tempfile.TemporaryDirectory(prefix='subtitle-model-test-') as tmp:
            result = run_asr(test_audio_path,tmp,model=row['recipe'],model_path=model_file(row),segmentation='utterance')
            if not any(s.get('text','').strip() for s in result['segments']):
                raise ModelAdapterError('Empty transcription')


def model_fingerprint(path):
    """Fingerprint GGUF files as well as ASR snapshots (including absent peers)."""
    from .pipeline import _digest      # lazy: pipeline imports this module's siblings
    path = Path(path).resolve()
    if path.is_file():
        stat = path.stat()
        return _digest(dict(path=str(path), size=stat.st_size, mtime_ns=stat.st_mtime_ns))
    try:
        return runtime.local_model_fingerprint(path)
    except FileNotFoundError:
        return _digest(dict(missing=str(path)))
