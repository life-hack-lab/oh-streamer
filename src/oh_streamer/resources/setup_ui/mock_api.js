/*
 * Browser-preview stand-in for the pywebview js_api of the setup window. It does NOTHING inside the real app window
 * (a WebView2 / WKWebView host, unless ?mock is given), so it is safe to ship next to setup.js. Loaded before setup.js.
 *
 * Pick a scenario with ?case=<name>[,<name>...] (default: nvidia). Names combine, e.g. ?case=nvidia,failed.
 *   GPU scenarios (first one found wins):
 *     nvidia          RTX 5060, driver ok, flavor "nvidia"                (default)
 *     old_driver      NVIDIA with an old driver: driver_too_old + warning code driver_below_570
 *     cpu             no graphics card, recommendation.flavor = null
 *     intel / amd     Intel / AMD only, flavor = null
 *     legacy          GTX 1060, legacy_nvidia_deferred, flavor = null
 *     unknown_nvidia  NVIDIA of unknown generation, flavor = null
 *   Flags:
 *     no_space        free disk smaller than the download
 *     no_webview2     webview2_installed = false
 *     current         a verified install already exists (current + base_ready)
 *     in_use          the app is still running: detect() lists 2 processes and the first start_install() is refused
 *                     with reason "in_use" (the 重試 after that succeeds, as if the user had closed it)
 *     low_space       6 GB free and another version installed: too little for old + new side by side, enough after
 *                     deleting the old env; start_install() is refused with reason "low_space" unless delete_old_first
 *     unknown_warning an unknown warning code next to driver_below_570
 *     failed          the install ends in phase "failed" during the self-check (CUDA driver message)
 *     fail_network    the install fails half-way through the gpu phase with a network error (English, like uv)
 *     fail_disk       the install fails half-way through the gpu phase with "No space left on device"
 *     installing      an install is already running when the page opens (tests resume after reload)
 *     warm            nothing to download (warm uv cache): poll() sends phase_progress = null, the bar only pulses
 *     detect_error    detect() rejects
 *     export_fail     export_diagnostics() returns ok:false
 *     launch_fail     launch_main() returns ok:false
 *   ?speed=<n> scales install progress (2 = twice as fast, 0.5 = half).
 */
(function () {
'use strict';

// pywebview injects window.pywebview only after the page has loaded, so its absence here proves nothing. The host's
// native bridge is there from the start: WebView2 (chrome.webview) on Windows, WKWebView (webkit.messageHandlers) on macOS.
const params = new URLSearchParams(location.search);
const inWebviewHost = !!((window.chrome && window.chrome.webview) || (window.webkit && window.webkit.messageHandlers));
if (window.pywebview || (inWebviewHost && !params.has('mock'))) return;

const flags = new Set((params.get('case') || 'nvidia').toLowerCase().split(',').map((s) => s.trim()).filter(Boolean));
const speed = Math.min(20, Math.max(0.1, parseFloat(params.get('speed')) || 1));
const SCENARIOS = ['nvidia', 'old_driver', 'cpu', 'intel', 'amd', 'legacy', 'unknown_nvidia'];
const scenario = SCENARIOS.find((s) => flags.has(s)) || 'nvidia';

const delay = (ms) => new Promise((r) => setTimeout(r, ms));
const latency = () => delay(120 + Math.random() * 200);
const pad = (n) => String(n).padStart(2, '0');

/* ------------------------------------------------------------------ detect() */

const IN_USE = [{ pid: 4242, name: 'pythonw.exe' }, { pid: 4317, name: 'python.exe' }];
let inUseLeft = flags.has('in_use') ? 1 : 0;   // refusals left before the "user" has closed the app
const installed = () => flags.has('current') || flags.has('low_space');
const diskFree = () => (flags.has('no_space') ? 2.1 : flags.has('low_space') ? 6.0 : 120.4);

function plan() {                              // same shape as bridge.disk_plan
  const free = diskFree(), required = 9.0, freeable = installed() ? 4.4 : 0;
  const low = free < required;
  return { required_gb: required, reuses_installed: false, low_space: low, freeable_gb: low ? freeable : 0,
           can_delete_old_first: low && freeable > 0 && free + freeable >= required };
}

const OPTIONS = [                              // mirrors bridge._OPTIONS / SUPPORTED_NOW
  { flavor: 'nvidia', label: 'NVIDIA（CUDA 12.8）', supported: true },
  { flavor: 'intel', label: 'Intel（Arc／Core Ultra）', supported: false },
  { flavor: 'amd', label: 'AMD（ROCm，實驗性）', supported: false },
];

function buildDetect() {
  const gpu = (name, vendor, driver, cc) => ({ name, vendor, driver_version: driver, compute_capability: cc });
  let gpus = [];
  let rec = { flavor: null, status: 'cpu_only', warnings: [] };
  switch (scenario) {
    case 'nvidia':
      gpus = [gpu('NVIDIA GeForce RTX 5060 Laptop GPU', 'nvidia', '576.80', '12.0'), gpu('Intel(R) UHD Graphics', 'intel', '32.0.101.6458', null)];
      rec = { flavor: 'nvidia', status: 'ok', warnings: [] };
      break;
    case 'old_driver':
      gpus = [gpu('NVIDIA GeForce RTX 4070', 'nvidia', '531.41', '8.9')];
      rec = { flavor: 'nvidia', status: 'driver_too_old', warnings: ['driver_below_570'] };
      break;
    case 'cpu':
      rec = { flavor: null, status: 'cpu_only', warnings: [] };
      break;
    case 'intel':
      gpus = [gpu('Intel(R) Arc(TM) A770 Graphics', 'intel', '31.0.101.5333', null)];
      rec = { flavor: null, status: 'intel_not_offered', warnings: [] };
      break;
    case 'amd':
      gpus = [gpu('AMD Radeon RX 7800 XT', 'amd', null, null)];
      rec = { flavor: null, status: 'amd_not_offered', warnings: [] };
      break;
    case 'legacy':
      gpus = [gpu('NVIDIA GeForce GTX 1060 6GB', 'nvidia', '472.12', '6.1')];
      rec = { flavor: null, status: 'legacy_nvidia_deferred', warnings: [] };
      break;
    case 'unknown_nvidia':
      gpus = [gpu('NVIDIA Some Future Board', 'nvidia', null, null)];
      rec = { flavor: null, status: 'unknown_nvidia', warnings: [] };
      break;
  }
  if (flags.has('unknown_warning')) rec.warnings = rec.warnings.concat(['some_new_warning_code']);
  return {                                     // same shape as bridge.SetupApi.detect(): options and sizes are always sent
    ok: true,
    gpus,
    recommendation: rec,
    options: OPTIONS.map((o) => Object.assign({}, o)),
    supported_cards: ['NVIDIA GTX 16 系列、RTX 20～50 系列（驅動 570 以上）'],
    disk_free_gb: diskFree(),
    estimated_download_gb: 4.0,
    required_disk_gb: plan().required_gb,
    disk_plans: { nvidia: plan() },
    in_use: inUseLeft > 0 ? IN_USE.map((p) => Object.assign({}, p)) : [],
    webview2_installed: !flags.has('no_webview2'),
    current: installed() ? { flavor: 'nvidia', verified: true } : null,
    base_ready: installed(),
  };
}

/* ------------------------------------------------------------------ install simulation */

// [phase, duration ms at speed 1, log lines emitted during the phase]
const PHASES = [['base', 3000, 36], ['gpu', 3000, 36], ['selfcheck', 2000, 14], ['cleanup', 1200, 3]];
const TOTAL_MS = PHASES.reduce((a, p) => a + p[1], 0);
const CANCEL_LAG_MS = 1500;      // the "backend" needs a moment before it admits to being cancelled
const DEVICE = 'NVIDIA GeForce RTX 5060 Laptop GPU';
const FAIL_MESSAGE =
  '自我檢查失敗：無法在 GPU 上執行測試。\n' +
  'RuntimeError: CUDA driver version is insufficient for CUDA runtime version\n' +
  '（這是預覽模式的假訊息。）';

let run = null;                  // { t0, flavor, cancelAt }
if (flags.has('installing')) run = { t0: Date.now() - 2500, flavor: 'nvidia', cancelAt: null };

function linesAt(elapsedMs) {
  const out = [];
  let start = 0;
  for (const [phase, dur, count] of PHASES) {
    const emitted = Math.min(count, Math.floor(Math.max(0, elapsedMs - start) / dur * count));
    for (let i = 1; i <= emitted; i++) out.push(`[${phase}] (mock) step ${i}/${count} ...`);
    start += dur;
  }
  return out;
}

function phaseAt(elapsedMs) {
  let start = 0;
  for (const [phase, dur] of PHASES) {
    if (elapsedMs < start + dur) return phase;
    start += dur;
  }
  return null;
}

const MESSAGES = {                // same texts as bridge._PHASE_MESSAGES
  base: '正在安裝基礎環境…',
  gpu: '正在下載並安裝 GPU 版（約 3 GB，請保持網路連線）…',
  selfcheck: '正在檢查顯示卡能否正常運算…',
  cleanup: '顯示卡測試通過，正在刪除舊環境、釋放磁碟空間…',
};
// failures that happen half-way through the gpu phase (like the bridge: "GPU 版安裝失敗：<uv message>")
const EARLY_FAILS = {
  fail_network: 'GPU 版安裝失敗：error: Failed to download `nvidia-cuda-nvrtc-cu12==12.8.93`\n' +
    '  Caused by: error sending request for url (https://download.pytorch.org/whl/cu128/...)\n' +
    '  Caused by: operation timed out\n（這是預覽模式的假訊息。）',
  fail_disk: 'GPU 版安裝失敗：error: Failed to install `torch==2.8.0+cu128`\n' +
    '  Caused by: failed to write to file: No space left on device (os error 112)\n（這是預覽模式的假訊息。）',
};
const EARLY_FAIL_AT = PHASES[0][1] + PHASES[1][1] / 2;

function pollNow() {                // like bridge.poll(): phase_progress is always present (null when unknown)
  return Object.assign({ phase_progress: null }, pollState());
}

function pollState() {
  if (!run) return { phase: 'idle', lines: [], message: '', device_name: null };
  const now = Date.now();
  const el = (now - run.t0) * speed;
  if (run.cancelAt != null) {
    const lines = linesAt((run.cancelAt - run.t0) * speed);
    if (now - run.cancelAt >= CANCEL_LAG_MS) return { phase: 'cancelled', lines, message: '安裝已取消。', device_name: null };
    return { phase: phaseAt((run.cancelAt - run.t0) * speed) || 'selfcheck', lines, message: '正在取消…', device_name: null };
  }
  const early = Object.keys(EARLY_FAILS).find((k) => flags.has(k));
  if (early && el >= EARLY_FAIL_AT) return { phase: 'failed', lines: linesAt(EARLY_FAIL_AT), message: EARLY_FAILS[early], device_name: null };
  if (el >= TOTAL_MS) {
    const lines = linesAt(TOTAL_MS);
    if (flags.has('failed')) return { phase: 'failed', lines, message: FAIL_MESSAGE, device_name: null };
    return { phase: 'done', lines, message: '', device_name: DEVICE };
  }
  const phase = phaseAt(el);
  return { phase, lines: linesAt(el), message: MESSAGES[phase] || '', device_name: null, phase_progress: progressAt(el) };
}

// same shape as bridge._progress_dict; null when nothing is measurable (warm cache, the self-check)
const DOWNLOAD_BYTES = { base: 259734499, gpu: 3473539452 };     // the real cold-cache totals of uv.lock
const MOCK_SPEED_BPS = 11.6 * 2 ** 20;                           // shown speed; the preview itself runs much faster
function progressAt(elapsedMs) {
  if (flags.has('warm')) return null;
  let start = 0;
  for (const [phase, dur] of PHASES) {
    if (elapsedMs < start + dur) {
      const total = DOWNLOAD_BYTES[phase];
      if (!total) return null;
      const t = elapsedMs - start;
      const frac = Math.min(1, t / (dur * 0.85));                  // the last bit of the phase is "installing"
      const done = Math.round(total * frac);
      const warmingUp = t < dur * 0.15;                            // like the bridge: no speed in the first seconds
      const speed = warmingUp ? null : Math.round(MOCK_SPEED_BPS * (0.95 + 0.1 * Math.random()));
      return { downloaded_bytes: done, total_bytes: total, speed_bps: frac >= 1 ? null : speed,
               eta_s: frac >= 1 ? 0 : speed ? Math.round((total - done) / speed) : null };
    }
    start += dur;
  }
  return null;
}

function isRunning() {
  const p = pollNow().phase;
  return p === 'base' || p === 'gpu' || p === 'selfcheck' || p === 'cleanup';
}

/* ------------------------------------------------------------------ the fake api */

const api = {
  async detect() {
    await latency();
    if (flags.has('detect_error')) throw new Error('偵測失敗（預覽模式的假錯誤）：無法讀取 GPU 資訊。');
    return buildDetect();
  },

  async start_install(flavor, options) {
    await latency();
    const opt = buildDetect().options.find((o) => o.flavor === flavor);   // like the bridge: only the flavor list decides
    if (!opt || !opt.supported) return { ok: false, message: `不支援的版本：${flavor}` };
    if (isRunning()) return { ok: false, message: '已有安裝正在進行。' };
    if (inUseLeft > 0) {
      inUseLeft--;
      return { ok: false, reason: 'in_use', processes: IN_USE.map((p) => Object.assign({}, p)),
               message: '程式還在執行（預覽模式的假訊息）。' };
    }
    const p = plan();
    if (p.low_space && p.can_delete_old_first && !(options && options.delete_old_first === true)) {
      return { ok: false, reason: 'low_space', plan: p, message: '磁碟空間不夠同時放新、舊兩份環境。可以選擇先刪除舊環境再安裝。' };
    }
    run = { t0: Date.now(), flavor, cancelAt: null };
    return { ok: true, message: '已開始安裝。' };
  },

  async poll() {
    await latency();
    return pollNow();
  },

  async cancel() {
    await latency();
    if (!isRunning()) return { ok: false };
    run.cancelAt = Date.now();
    return { ok: true };
  },

  async export_diagnostics() {
    await delay(500);
    if (flags.has('export_fail')) return { ok: false, path: '', message: '匯出診斷包失敗：找不到可寫入的資料夾（預覽模式的假錯誤）。' };
    const d = new Date();
    const stamp = `${d.getFullYear()}${pad(d.getMonth() + 1)}${pad(d.getDate())}-${pad(d.getHours())}${pad(d.getMinutes())}${pad(d.getSeconds())}`;
    return { ok: true, path: `C:\\Users\\preview\\AppData\\Local\\oh_streamer\\diagnostics\\setup-${stamp}.zip`, message: '診斷包已匯出。' };
  },

  async launch_main() {
    await delay(400);
    if (flags.has('launch_fail')) return { ok: false, message: '無法啟動主程式：找不到執行檔（預覽模式的假錯誤）。' };
    return { ok: true, message: '主程式已啟動，這個視窗即將關閉。（預覽模式不會真的啟動。）' };
  },
};

window.pywebview = {
  api,
  _mock: {
    case: Array.from(flags),
    help: `預覽模式：使用假資料（case=${Array.from(flags).join(',')}，speed=${speed}）。
` +
      '用網址參數切換：?case=nvidia（預設）、old_driver、cpu、intel、amd、legacy、unknown_nvidia；' +
      '可再加 no_space、no_webview2、current、in_use、low_space、unknown_warning、failed、fail_network、fail_disk、installing、warm、detect_error、export_fail、launch_fail（以逗號組合）。',
  },
};
})();
