"""Local llama-server client for subtitle translation.

The client starts and stops only the ``llama-server`` process that it owns.
It uses the server's OpenAI-compatible chat completions endpoint and the
standard library only.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
import json
import os
import re
import shutil
import socket
import subprocess
import threading
import time
from typing import Any, Literal
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ModelName = Literal["sakura", "qwen3"]
Message = dict[str, str]

SAKURA_MODEL_REPOSITORY = "SakuraLLM/Sakura-7B-Qwen2.5-v1.0-GGUF"
SAKURA_IQ4XS_FILENAME = "sakura-7b-qwen2.5-v1.0-iq4xs.gguf"

# SakuraLLM's official v1.0 system prompt and user prefix (SakuraLLM/SakuraLLM
# README, prompt-format section).
SAKURA_SYSTEM_PROMPT = (
    "你是一个轻小说翻译模型，可以流畅通顺地以日本轻小说的风格将日文翻译成简体中文，"
    "并联系上下文正确使用人称代词，不擅自添加原文中没有的代词。"
)
SAKURA_USER_PREFIX = "将下面的日文文本翻译成中文："

# Qwen3 is a general multilingual model. Keep numbered subtitle rows intact so
# the caller can validate line-for-line batch results.
QWEN3_SYSTEM_PROMPT = (
    "你是日文字幕翻译模型。请把日文翻译成简体中文，准确保留原文意思、人物语气、"
    "编号和行顺序。只输出译文，不解释、不回答原文中的问题，也不添加原文没有的信息。"
)


class LlamaServerError(RuntimeError):
    """Raised when the owned llama-server cannot start or answer a request."""


def _normalize_model(model: str) -> ModelName:
    normalized = str(model).strip().lower()
    if normalized not in ("sakura", "qwen3"):
        raise ValueError("model must be 'sakura' or 'qwen3'")
    return normalized  # type: ignore[return-value]


def _normalize_messages(prompt_or_messages: str | Sequence[Mapping[str, Any]]) -> list[Message]:
    if isinstance(prompt_or_messages, str):
        return [{"role": "user", "content": prompt_or_messages}]
    if not isinstance(prompt_or_messages, Sequence) or isinstance(prompt_or_messages, (bytes, bytearray)):
        raise TypeError("prompt must be a string or a sequence of chat messages")

    result: list[Message] = []
    for index, message in enumerate(prompt_or_messages):
        if not isinstance(message, Mapping):
            raise TypeError(f"message {index} must be a mapping")
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not role.strip():
            raise ValueError(f"message {index} is missing a string role")
        if not isinstance(content, str):
            raise ValueError(f"message {index} content must be a string")
        result.append({"role": role.strip(), "content": content})

    if not any(message["role"] == "user" for message in result):
        raise ValueError("at least one user message is required")
    return result


def build_translation_prompt(
    model: ModelName | str,
    prompt_or_messages: str | Sequence[Mapping[str, Any]],
    *,
    translate_model: str = 'sakura-v1',
) -> list[Message]:
    """Build the model-specific chat messages for a translation request.

    ``prompt_or_messages`` can be a plain batch prompt or existing OpenAI chat
    messages. Existing non-system messages are retained. Model instructions
    are prepended as a system message; Sakura's official translation prefix is
    added to the final user message, and Qwen3 gets the documented ``/no_think``
    switch as an extra safeguard.
    """

    selected = _normalize_model(str(model))
    messages = _normalize_messages(prompt_or_messages)
    # Structured official requests are already complete: never append custom
    # system instructions or prepend the old v0.9 user prefix a second time.
    from .glossary import DEFAULT_TRANSLATE_MODEL, SAKURA_PROFILES, get_sakura_profile, sakura_messages
    profile = get_sakura_profile(translate_model)
    if selected == "sakura" and any(
        any(p.prefix in message["content"] for p in SAKURA_PROFILES.values())
        for message in messages
        if message["role"] == "user"
    ):
        if [
            message["content"]
            for message in messages
            if message["role"] == "system"
        ] != [profile.system] or not any(
            profile.prefix in m['content'] for m in messages if m['role'] == 'user'
        ):
            raise ValueError("official Sakura requests require the unmodified system prompt")
        return messages
    if selected == 'sakura' and profile.name != DEFAULT_TRANSLATE_MODEL:
        if any(m['role'] == 'system' for m in messages) or len(messages) != 1:
            raise ValueError('GalTransl requests require the official profile prompt')
        return sakura_messages(messages[0]['content'], translate_model=profile.name)
    user_indexes = [i for i, message in enumerate(messages) if message["role"] == "user"]
    last_user_index = user_indexes[-1]
    user_content = messages[last_user_index]["content"]

    if selected == "sakura":
        if not user_content.startswith(SAKURA_USER_PREFIX):
            messages[last_user_index]["content"] = SAKURA_USER_PREFIX + user_content
        system_prompt = SAKURA_SYSTEM_PROMPT
    else:
        if not user_content.rstrip().endswith("/no_think"):
            messages[last_user_index]["content"] = user_content.rstrip() + " /no_think"
        system_prompt = QWEN3_SYSTEM_PROMPT

    # Keep caller system messages after our base instructions, so callers can
    # add batch-specific constraints without losing the model's core guidance.
    extra_system = [m["content"] for m in messages if m["role"] == "system"]
    non_system = [m for m in messages if m["role"] != "system"]
    if extra_system:
        system_prompt = system_prompt + "\n" + "\n".join(extra_system)
    return [{"role": "system", "content": system_prompt}, *non_system]


def strip_think_content(text: str) -> str:
    """Remove Qwen ``<think>`` output, including an unterminated block."""

    parts: list[str] = []
    cursor = 0
    while opening := re.search(r"<think(?:\s[^>]*)?>", text[cursor:], flags=re.IGNORECASE):
        opening_start = cursor + opening.start()
        opening_end = cursor + opening.end()
        parts.append(text[cursor:opening_start])
        closing = re.search(r"</think\s*>", text[opening_end:], flags=re.IGNORECASE)
        if closing is None:
            # Fail closed: never leak a partial internal reasoning block as a
            # subtitle if generation was truncated before its closing tag.
            cursor = len(text)
            break
        cursor = opening_end + closing.end()
    parts.append(text[cursor:])
    return "".join(parts).strip()


class LlamaServerClient:
    """A synchronous translation client backed by an owned llama-server.

    ``model_path`` must point to an already-present local GGUF. This class
    never downloads models. The ``translate_model`` profile must match the
    caller's local Sakura GGUF; the default retains v1.0 behavior.

    The first call to :meth:`translate` starts the server automatically. Use
    this as a context manager (or call :meth:`close`) so its own subprocess is
    reliably stopped after a translation stage.
    """

    def __init__(
        self,
        model: ModelName | str,
        model_path: str | os.PathLike[str],
        llama_server_path: str | os.PathLike[str],
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        context_size: int = 8192,
        gpu_layers: int | str = "auto",
        startup_timeout: float = 120.0,
        request_timeout: float = 300.0,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        max_tokens: int = 512,
        metrics_callback: Any = None,
        server_log_path: str | os.PathLike[str] | None = None,
        on_phase: Any = None,
        translate_model: str = 'sakura-v1',
        seed: int | None = None,
        cache_prompt: bool | None = None,
        device: str | None = None,
        backend: str = "cuda",
        on_process: Any = None,
    ) -> None:
        self.model: ModelName = _normalize_model(str(model))
        from .glossary import get_sakura_profile
        profile = get_sakura_profile(translate_model)
        if self.model != 'sakura' and profile.name != 'sakura-v1':
            raise ValueError('Sakura translation profiles require model=sakura')
        self.translate_model = profile.name if self.model == 'sakura' else 'qwen3'
        self.model_path = Path(model_path).expanduser()
        self.llama_server_path = str(llama_server_path)
        if backend not in ("cuda", "vulkan"):
            raise ValueError("backend must be cuda or vulkan")
        if device is not None and (not isinstance(device, str) or not device or any(c.isspace() for c in device) or device.startswith("-")):
            raise ValueError("device must be a single explicit device identifier")
        if backend == "vulkan" and (device is None or not re.fullmatch(r"Vulkan\d+", device)):
            raise ValueError("Vulkan translation requires an explicit Vulkan device")
        self.device, self.backend = device, backend
        if host not in ("127.0.0.1", "localhost", "::1"):
            raise ValueError("llama-server must bind to a loopback host")
        if not 0 <= int(port) <= 65535:
            raise ValueError("port must be between 0 and 65535")
        if int(context_size) <= 0:
            raise ValueError("context_size must be positive")
        if int(max_tokens) <= 0:
            raise ValueError("max_tokens must be positive")
        if startup_timeout <= 0 or request_timeout <= 0:
            raise ValueError("timeouts must be positive")

        self.host = host
        self._requested_port = int(port)
        self._port = int(port)
        self.context_size = int(context_size)
        if isinstance(gpu_layers, str):
            if gpu_layers.casefold() not in ("auto", "all"):
                raise ValueError("gpu_layers must be 'auto', 'all', or a non-negative integer")
            self.gpu_layers: int | str = gpu_layers.casefold()
        else:
            if int(gpu_layers) < -1:
                raise ValueError("gpu_layers must be 'auto', 'all', or a non-negative integer")
            self.gpu_layers = "all" if int(gpu_layers) == -1 else int(gpu_layers)
        self.startup_timeout = float(startup_timeout)
        self.request_timeout = float(request_timeout)
        self.temperature = float(temperature if temperature is not None else (profile.temperature if self.model == "sakura" else 0.2))
        self.top_p = float(top_p if top_p is not None else (profile.top_p if self.model == "sakura" else 0.8))
        self.top_k = top_k if top_k is not None else (None if self.model == "sakura" else 20)
        # seed=None / cache_prompt=None keep the historical request body (no
        # "seed", server default prompt caching), so existing runs are unchanged.
        self.seed = int(seed) if seed is not None else None
        self.cache_prompt = bool(cache_prompt) if cache_prompt is not None else None
        self.max_tokens = int(max_tokens)
        self.metrics_callback = metrics_callback
        self.on_phase = on_phase
        # on_process(process): called right after llama-server starts, so an owner (the queue's local
        # subtitle executor) can register the process and kill it on cancel.
        self.on_process = on_process
        self.server_log_path = Path(server_log_path).expanduser() if server_log_path is not None else None

        self._process: subprocess.Popen[bytes] | None = None
        self._server_log_file: Any | None = None
        self._lifecycle_lock = threading.RLock()
        self._request_lock = threading.Lock()
        self._requests = 0
        self._failures = 0
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._total_tokens = 0
        self._elapsed_seconds = 0.0
        self._generation_seconds = 0.0

    @property
    def port(self) -> int:
        """The actual port used by the server (assigned at :meth:`start`)."""

        return self._port

    @property
    def pid(self) -> int | None:
        """PID of this client's live subprocess, for process-scoped observers."""

        process = self._process
        if process is None or process.poll() is not None:
            return None
        return process.pid

    @property
    def base_url(self) -> str:
        host = f"[{self.host}]" if ":" in self.host else self.host
        return f"http://{host}:{self._port}/v1"

    @property
    def metrics(self) -> dict[str, int | float]:
        """Return cumulative request, token, throughput, and elapsed metrics."""

        with self._lifecycle_lock:
            wall_tps = self._completion_tokens / self._elapsed_seconds if self._elapsed_seconds else 0.0
            measured_tps = (
                self._completion_tokens / self._generation_seconds
                if self._generation_seconds
                else wall_tps
            )
            return {
                "requests": self._requests,
                "failures": self._failures,
                # ``tokens`` and tokens_per_second describe generated tokens.
                "tokens": self._completion_tokens,
                "prompt_tokens": self._prompt_tokens,
                "completion_tokens": self._completion_tokens,
                "total_tokens": self._total_tokens,
                "elapsed_seconds": self._elapsed_seconds,
                "generation_seconds": self._generation_seconds,
                "tokens_per_second": wall_tps,
                "generation_tokens_per_second": measured_tps,
            }

    def __enter__(self) -> LlamaServerClient:
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def __call__(
        self,
        prompt_or_messages: str | Sequence[Mapping[str, Any]],
        *,
        max_tokens: int | None = None,
    ) -> str:
        return self.translate(prompt_or_messages, max_tokens=max_tokens)

    def start(self) -> None:
        """Start the configured local server and wait for its health endpoint."""

        with self._lifecycle_lock:
            if self._process is not None and self._process.poll() is None:
                return
            if callable(self.on_phase):
                self.on_phase(
                    "prepare_translate",
                    None,
                    "正在檢查本機翻譯模型與伺服器執行檔",
                )
            if self._process is not None:
                # A process can exit after a successful health check. Release
                # its diagnostic file before a later call starts a new one.
                self._process = None
                self._close_log_file_locked()
            if not self.model_path.is_file():
                raise FileNotFoundError(f"local GGUF model does not exist: {self.model_path}")

            server = self.llama_server_path
            if not Path(server).exists() and shutil.which(server) is None:
                raise FileNotFoundError(f"llama-server executable not found: {server}")
            if callable(self.on_phase):
                self.on_phase(
                    "prepare_translate",
                    25,
                    "本機翻譯模型與伺服器執行檔已驗證",
                )

            if self._requested_port == 0:
                with socket.socket(socket.AF_INET6 if ":" in self.host else socket.AF_INET, socket.SOCK_STREAM) as sock:
                    sock.bind((self.host, 0))
                    self._port = int(sock.getsockname()[1])
            else:
                self._port = self._requested_port

            command = [
                server,
                "--model", str(self.model_path.resolve()),
                "--host", self.host,
                "--port", str(self._port),
                "--alias", self.model,
                "--ctx-size", str(self.context_size),
                "--n-gpu-layers", str(self.gpu_layers),
                "--parallel", "1",
            ]
            if self.device is not None:
                command.extend(("--device", self.device))
            if self.backend == "vulkan":
                # Avoid automatic placement/spill onto a second adapter; bound
                # CPU work so the APU does not monopolize repair's host threads.
                command.extend(("--fit", "off", "--threads", "4", "--threads-batch", "4"))
            if self.model == "qwen3":
                # Qwen's official model card documents enable_thinking=False;
                # this server-level switch avoids allocating/generating a
                # reasoning block at all for this one-model subprocess.
                command.extend(("--reasoning", "off"))
            if callable(self.on_phase):
                self.on_phase(
                    "prepare_translate",
                    40,
                    "本機伺服器啟動參數已準備完成",
                )

            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
            log_file = None
            try:
                if self.server_log_path is not None:
                    self.server_log_path.parent.mkdir(parents=True, exist_ok=True)
                    log_file = self.server_log_path.open("ab")
                self._process = subprocess.Popen(
                    command,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=log_file if log_file is not None else subprocess.DEVNULL,
                    close_fds=True,
                    creationflags=creationflags,
                )
                self._server_log_file = log_file
                if callable(self.on_process):
                    self.on_process(self._process)
                if callable(self.on_phase):
                    self.on_phase(
                        "prepare_translate",
                        60,
                        "本機翻譯伺服器程序已啟動，正在載入模型",
                    )
            except OSError as exc:
                self._process = None
                self._server_log_file = None
                if log_file is not None:
                    log_file.close()
                raise LlamaServerError(f"could not start llama-server: {exc}") from exc

        deadline = time.monotonic() + self.startup_timeout
        health_url = f"{self.base_url.rsplit('/v1', 1)[0]}/health"
        while time.monotonic() < deadline:
            process = self._process
            if process is None:
                raise LlamaServerError("llama-server process was not created")
            return_code = process.poll()
            if return_code is not None:
                self.close()
                raise LlamaServerError(f"llama-server exited during startup (exit code {return_code})")
            try:
                with urlopen(health_url, timeout=min(1.0, self.startup_timeout)) as response:
                    if 200 <= response.status < 300:
                        if callable(self.on_phase):
                            self.on_phase(
                                "prepare_translate",
                                100,
                                "本機翻譯伺服器已就緒",
                            )
                        return
            except (HTTPError, URLError, OSError, TimeoutError):
                pass
            time.sleep(0.2)

        self.close()
        raise LlamaServerError(f"llama-server did not become ready within {self.startup_timeout:g} seconds")

    def close(self, *, terminate_timeout: float = 5.0) -> None:
        """Stop and reap only this client's own llama-server process."""

        with self._lifecycle_lock:
            process = self._process
            self._process = None
            log_file = self._server_log_file
            self._server_log_file = None
            try:
                if process is not None and process.poll() is None:
                    try:
                        process.terminate()
                    except OSError:
                        if process.poll() is None:
                            raise
                    try:
                        process.wait(timeout=terminate_timeout)
                    except subprocess.TimeoutExpired:
                        # ``Popen.kill`` targets only the process created above;
                        # no process name lookup or broad tree termination occurs.
                        try:
                            process.kill()
                        except OSError:
                            if process.poll() is None:
                                raise
                        process.wait()
            finally:
                if log_file is not None:
                    try:
                        log_file.flush()
                    except OSError:
                        pass
                    try:
                        log_file.close()
                    except OSError:
                        pass

    def _close_log_file_locked(self) -> None:
        log_file = self._server_log_file
        self._server_log_file = None
        if log_file is not None:
            try:
                log_file.flush()
            except OSError:
                pass
            try:
                log_file.close()
            except OSError:
                pass

    def translate(
        self,
        prompt_or_messages: str | Sequence[Mapping[str, Any]],
        *,
        max_tokens: int | None = None,
    ) -> str:
        """Translate a prompt or OpenAI message list and return plain text."""

        messages = build_translation_prompt(
            self.model, prompt_or_messages,
            translate_model=self.translate_model if self.model == 'sakura' else 'sakura-v1',
        )
        token_limit = self.max_tokens if max_tokens is None else int(max_tokens)
        if token_limit <= 0:
            raise ValueError("max_tokens must be positive")
        self.start()

        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": token_limit,
            "stream": False,
        }
        if self.top_k is not None:
            body["top_k"] = self.top_k
        if self.seed is not None:
            body["seed"] = self.seed
        if self.cache_prompt is not None:
            body["cache_prompt"] = self.cache_prompt
        if self.model == "qwen3":
            body["min_p"] = 0.0
            body["chat_template_kwargs"] = {"enable_thinking": False}
        else:
            body["repeat_penalty"] = 1.0

        payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
        request = Request(
            f"{self.base_url}/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )

        with self._request_lock:
            started = time.perf_counter()
            try:
                try:
                    with urlopen(request, timeout=self.request_timeout) as response:
                        raw_response = response.read()
                except HTTPError as exc:
                    detail = exc.read().decode("utf-8", errors="replace")[:1000]
                    raise LlamaServerError(f"llama-server returned HTTP {exc.code}: {detail}") from exc
                except (URLError, OSError, TimeoutError) as exc:
                    raise LlamaServerError(f"llama-server request failed: {exc}") from exc

                try:
                    data = json.loads(raw_response.decode("utf-8"))
                    content = data["choices"][0]["message"]["content"]
                except (UnicodeDecodeError, json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
                    raise LlamaServerError("llama-server returned an invalid chat completion response") from exc
                if not isinstance(content, str):
                    raise LlamaServerError("llama-server response content was not a string")

                if self.model == "qwen3":
                    content = strip_think_content(content)
                usage = data.get("usage") or {}
                timings = data.get("timings") or {}
                elapsed = time.perf_counter() - started
                prompt_tokens = _nonnegative_int(usage.get("prompt_tokens", timings.get("prompt_n", 0)))
                completion_tokens = _nonnegative_int(
                    usage.get("completion_tokens", timings.get("predicted_n", 0))
                )
                total_tokens = _nonnegative_int(
                    usage.get("total_tokens", prompt_tokens + completion_tokens)
                )
                generation_ms = _nonnegative_float(timings.get("predicted_ms", 0))
                generation_seconds = generation_ms / 1000.0 if generation_ms > 0 else elapsed
            except Exception:
                elapsed = time.perf_counter() - started
                self._record_metrics(elapsed=elapsed, failed=True)
                raise

            self._record_metrics(
                elapsed=elapsed,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                generation_seconds=generation_seconds,
                failed=data["choices"][0].get("finish_reason") == "length",
            )
            if data["choices"][0].get("finish_reason") == "length":
                raise LlamaServerError("Translation was truncated at the output token limit")
            return content.strip()

    def translate_with_metrics(
        self,
        prompt_or_messages: str | Sequence[Mapping[str, Any]],
        *,
        max_tokens: int | None = None,
    ) -> tuple[str, dict[str, int | float]]:
        """Return a translation and the cumulative metrics snapshot."""

        result = self.translate(prompt_or_messages, max_tokens=max_tokens)
        return result, self.metrics

    def _record_metrics(
        self,
        *,
        elapsed: float,
        failed: bool = False,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int = 0,
        generation_seconds: float = 0.0,
    ) -> None:
        with self._lifecycle_lock:
            self._requests += 1
            self._failures += int(failed)
            self._elapsed_seconds += elapsed
            self._prompt_tokens += prompt_tokens
            self._completion_tokens += completion_tokens
            self._total_tokens += total_tokens
            self._generation_seconds += generation_seconds
            snapshot = self.metrics
        if self.metrics_callback is not None:
            self.metrics_callback(snapshot)


def _nonnegative_int(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError, OverflowError):
        return 0


def _nonnegative_float(value: Any) -> float:
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError, OverflowError):
        return 0.0


__all__ = [
    "LlamaServerClient",
    "LlamaServerError",
    "SAKURA_IQ4XS_FILENAME",
    "SAKURA_MODEL_REPOSITORY",
    "build_translation_prompt",
    "strip_think_content",
]
