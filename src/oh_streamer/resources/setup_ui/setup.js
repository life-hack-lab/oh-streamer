/*
 * 首次設定 — front end of the first-run setup window.
 *
 * Talks to the Python side only through window.pywebview.api (see the API contract at the end of this comment).
 * Without pywebview (plain browser) mock_api.js installs a fake one, so this file never needs to know the difference
 * except for showing a "preview" banner.
 *
 *   detect()                 -> { gpus, recommendation:{flavor,status,warnings}, options, supported_cards:[text], disk_free_gb,
 *                                 estimated_download_gb, required_disk_gb, webview2_installed, current, base_ready,
 *                                 disk_plans:{flavor:{required_gb,low_space,freeable_gb,can_delete_old_first}},
 *                                 in_use:[{pid,name}] }
 *   start_install(flavor[, {delete_old_first}])
 *                            -> { ok, message[, reason: in_use|low_space, processes, plan] }
 *   poll()                   -> { phase: idle|base|gpu|selfcheck|cleanup|done|failed|cancelled, lines, message, device_name }
 *   cancel()                 -> { ok }
 *   export_diagnostics()     -> { ok, path, message }
 *   launch_main()            -> { ok, message }
 *
 * The product name is never written here: it comes from window.BRAND (queue_ui/branding.js).
 * All DOM text is set through textContent / createElement; nothing from the backend goes through innerHTML.
 */
(function () {
'use strict';

/* ------------------------------------------------------------------ zh-TW lookup tables */

// recommendation.status -> plain explanation under the graphics card list (tone picks the notice colour)
const STATUS_TEXT = {
  ok:                     { tone: 'ok',   text: '這張 NVIDIA 顯示卡可以使用，直接按下方的「安裝」即可。' },
  driver_too_old:         { tone: 'warn', text: '這張顯示卡可以使用，但顯示卡驅動程式太舊。建議先到 NVIDIA 官網更新驅動程式再安裝（也可以先安裝，之後再更新）。' },
  legacy_nvidia_deferred: { tone: 'warn', text: '這張 NVIDIA 顯示卡（GTX 9xx／10xx 系列）目前還不支援。' },
  unknown_nvidia:         { tone: 'warn', text: '偵測到 NVIDIA 顯示卡，但無法判斷型號世代。可以手動選擇 NVIDIA 版安裝，安裝最後的「測試顯示卡」會確認能不能用。' },
  intel_not_offered:      { tone: 'info', text: '偵測到 Intel 顯示卡，這個版本還不支援。' },
  amd_not_offered:        { tone: 'info', text: '偵測到 AMD 顯示卡，這個版本還不支援。' },
  cpu_only:               { tone: 'info', text: '找不到可以用來加速的顯示卡。' },
};
const STATUS_FALLBACK = { tone: 'info', text: '無法辨識偵測結果，請匯出診斷包提供給開發者。' };

// recommendation.warnings holds codes, not sentences. Unknown codes never show the code itself.
const WARNING_TEXT = {
  driver_below_570: 'NVIDIA 驅動程式版本低於 570，請先更新驅動程式。',
};
const WARNING_FALLBACK = '偵測到其他注意事項，詳情請匯出診斷包。';

const VENDOR_TEXT = { nvidia: 'NVIDIA', intel: 'Intel', amd: 'AMD', other: '其他' };

const PHASE_ORDER = ['base', 'gpu', 'selfcheck'];
const STEP_STATE_TEXT = { done: '完成', active: '進行中', pending: '等待中' };
const ACTIVE_PHASES = new Set(['base', 'gpu', 'selfcheck', 'cleanup']);   // cleanup: after the 3 steps, not a step of its own
// plain names (same as the step list in index.html) and what the user should know while the step runs
const PHASE_NAME = { base: '準備程式元件', gpu: '下載顯示卡加速元件（約 3 GB）', selfcheck: '測試顯示卡' };
const PHASE_HINT = {
  idle: '準備開始…',
  base: '正在下載並準備程式需要的基本元件。',
  gpu: '正在下載顯示卡加速元件，檔案約 3 GB，需要一段時間。請保持網路連線，不要關閉這個視窗。',
  selfcheck: '快完成了：正在確認顯示卡能正常運算。',
  cleanup: '顯示卡測試通過。正在刪除舊的環境，釋放磁碟空間，請稍候。',
};
// share of the progress bar each phase takes (base ~0.8 GB, gpu ~3.2 GB download, selfcheck is short)
const PHASE_WEIGHT = { base: 0.2, gpu: 0.7, selfcheck: 0.1 };
const PROGRESS_BLOCKS = 40;

const LOG_TAIL = 80;               // lines of install log kept on screen
const POLL_INTERVAL_MS = 1000;
const POLL_MAX_FAILS = 15;         // consecutive failed poll() calls before we give up
const API_WAIT_MS = 10000;         // how long to wait for window.pywebview.api
const BRAND_FALLBACK_PATHS = ['/queue_ui/branding.js', 'queue_ui/branding.js', 'branding.js'];

const VIEW_IDS = {
  detecting: 'vDetecting', detectError: 'vDetectError', ready: 'vReady',
  installing: 'vInstalling', done: 'vDone', failed: 'vFailed',
};
const DIAG_VIEWS = new Set(['detectError', 'ready', 'done', 'failed']);   // views that have an 匯出診斷包 button

/* ------------------------------------------------------------------ state */

const S = {
  api: null,
  view: 'detecting',
  info: null,            // normalized detect() result
  flavor: null,          // flavor currently chosen in the dropdown
  runFlavor: null,       // flavor of the install that is running / failed last
  busy: 0,               // user-initiated API calls in flight (detect, start_install, diagnostics, launch)
  cancelRequested: false,
  launched: false,
  inUse: [],             // running app processes that block the install ({pid, name}); from detect() or start_install()
  lastLines: [],
  lastPhase: null,       // last install phase seen (base / gpu / selfcheck), used to say where a failure happened
  pollTimer: null,       // the only timer that can be alive besides the one-shot ones noted below
  pollGen: 0,            // bumped by stopPolling(): a poll answer from an older generation is ignored
  pollFails: 0,
  closeTimer: null,      // one-shot, see closeWindow()
};

/* ------------------------------------------------------------------ small helpers */

const $ = (sel) => document.querySelector(sel);

function el(tag, cls, text) {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text != null) n.textContent = text;
  return n;
}

function errText(e) {
  const m = e && e.message ? e.message : String(e);
  return m || '未知的錯誤';
}

// own-property lookup: a backend string such as "constructor" must not hit Object.prototype
function pick(table, key) {
  return typeof key === 'string' && Object.prototype.hasOwnProperty.call(table, key) ? table[key] : undefined;
}

function finiteNum(v) { return typeof v === 'number' && isFinite(v) ? v : null; }
function fmtGb(v) { return v.toFixed(1); }

function hasBrand() {
  return !!(window.BRAND && typeof window.BRAND.name === 'string' && window.BRAND.name.trim());
}
function brandName() { return hasBrand() ? window.BRAND.name.trim() : ''; }

/* ------------------------------------------------------------------ branding (window.BRAND from queue_ui/branding.js) */

function loadScript(src, timeoutMs) {
  return new Promise((resolve) => {
    const s = document.createElement('script');
    let done = false;
    let timer = null;
    const end = (ok) => {
      if (done) return;
      done = true;
      clearTimeout(timer);
      s.onload = s.onerror = null;
      resolve(ok);
    };
    timer = setTimeout(() => end(false), timeoutMs);
    s.onload = () => end(true);
    s.onerror = () => end(false);
    s.src = src;
    document.head.appendChild(s);
  });
}

// index.html already tried ../queue_ui/branding.js; if the window's web root is different, try a few other places.
async function ensureBranding() {
  for (const src of BRAND_FALLBACK_PATHS) {
    if (hasBrand()) return;
    await loadScript(src, 1500);
  }
}

// Small wordmark next to the pixel logo, like the queue window's "STREAMER": the product name without its first word
// (the logo already stands for that word), upper-cased. A one-word name is shown whole.
function wordmark(name) {
  const words = name.split(/\s+/).filter(Boolean);
  return (words.length > 1 ? words.slice(1).join(' ') : name).toUpperCase();
}

function applyBranding() {
  const name = brandName();
  const logo = window.BRAND && typeof window.BRAND.logo === 'string' ? window.BRAND.logo.trim() : '';
  const setup = '首次設定';
  document.title = name ? `${name} ${setup}` : setup;
  $('#appTitle').textContent = setup;
  $('#appSub').textContent = '選擇並安裝所需的執行環境';
  const brand = $('#pxBrand');
  if (!logo && !name) { brand.hidden = true; return; }
  brand.hidden = false;
  brand.title = name;
  brand.setAttribute('role', 'img');
  brand.setAttribute('aria-label', name || logo);
  $('#pxSub').textContent = name ? wordmark(name) : '';
  if (logo) startLogo(logo); else $('#pxLogo').hidden = true;
}

/* ------------------------------------------------------------------ pixel logo (port of the queue window's, queue_ui/app.js) */

// BRAND.logo in an RGB-offset triad, drawn through a shrinking pixelation so it resolves over ~1 s; click replays.
// Afterwards: slow breathing; while detecting / installing faster breathing with an occasional micro-glitch.
// Reduced motion: only the final static frame. Smaller than the queue window's (40 px instead of 50 px high).
const LOGO_H = 40;
const LOGO_FONT_PX = 37;
const LOGO_PAD = 5;
const LOGO_FPS = 30;
const LOGO_INTRO = [18, 13, 10, 7, 5, 4, 3, 2, 1], LOGO_INTRO_MS = 95;
const LOGO_FLOW_MS = 2400;          // the r/g/b bar moves one bar width per this while busy (queue: same)
const LOGO_COLORS = ['#ff4766', '#36db86', '#4a86ff'];
const LOGO_FONT = (px) => `900 ${px}px "Arial Black", "Segoe UI Black", "Helvetica Neue", Arial, sans-serif`;
const reducedMotion = window.matchMedia ? window.matchMedia('(prefers-reduced-motion: reduce)') : { matches: false };
const lrand = (a, b) => a + Math.random() * (b - a);
const mkCanvas = (w, h) => { const c = document.createElement('canvas'); c.width = w; c.height = h; return c; };

const LG = { text: '', L: null, raf: 0, last: 0, t0: 0, run: null };

function buildLogo(text) {
  const cv = $('#pxLogo');
  const g = cv && cv.getContext ? cv.getContext('2d') : null;
  if (!g) return null;
  const dpr = window.devicePixelRatio || 1;
  const L = { cv, g, dpr };
  const m = mkCanvas(1, 1).getContext('2d'); m.font = LOGO_FONT(LOGO_FONT_PX);
  L.wCss = Math.ceil(m.measureText(text).width) + LOGO_PAD * 2;
  L.W = Math.round(L.wCss * dpr); L.H = Math.round(LOGO_H * dpr);
  cv.width = L.W; cv.height = L.H;
  cv.style.width = L.wCss + 'px'; cv.style.height = LOGO_H + 'px';
  $('#rgbBar').style.width = L.wCss + 'px';
  $('#rgbBar').style.setProperty('--bw', L.wCss + 'px');
  L.y = L.H - Math.round(5 * dpr); L.x0 = Math.round(LOGO_PAD * dpr);
  L.base = Math.max(1, Math.round(1.4 * dpr));
  const mask = (color) => {
    const c = mkCanvas(L.W, L.H), cg = c.getContext('2d');
    cg.font = LOGO_FONT(Math.round(LOGO_FONT_PX * dpr)); cg.textBaseline = 'alphabetic'; cg.fillStyle = color; cg.fillText(text, L.x0, L.y);
    return c;
  };
  L.masks = LOGO_COLORS.map(mask);
  L.buf = mkCanvas(L.W, L.H); L.bg = L.buf.getContext('2d'); L.tmp = mkCanvas(1, 1);
  L.rgb = (cg, off = L.base) => {
    cg.clearRect(0, 0, L.W, L.H);
    cg.globalCompositeOperation = 'lighter';
    cg.drawImage(L.masks[0], -off, 0); cg.drawImage(L.masks[1], 0, 0); cg.drawImage(L.masks[2], off, 0);
    cg.globalCompositeOperation = 'source-over';
  };
  L.pixelate = (cg, src, b) => {
    cg.clearRect(0, 0, L.W, L.H);
    if (b <= 1) { cg.drawImage(src, 0, 0); return; }
    const tw = Math.ceil(L.W / b), th = Math.ceil(L.H / b);
    L.tmp.width = tw; L.tmp.height = th; L.tmp.getContext('2d').drawImage(src, 0, 0, tw, th);
    cg.imageSmoothingEnabled = false; cg.drawImage(L.tmp, 0, 0, tw, th, 0, 0, tw * b, th * b); cg.imageSmoothingEnabled = true;
  };
  L.slices = (cg, src, bands) => {
    cg.clearRect(0, 0, L.W, L.H);
    let y = 0;
    for (const [h, dx] of bands) { cg.drawImage(src, 0, y, L.W, h, dx, y, L.W, h); y += h; }
  };
  // The r/g/b bar scrolls one bar width per period while busy. The position accumulates from 0 (no jump when it
  // starts) and, once busy ends (period null), runs on to the next whole width before resting (no snap back).
  const bar = $('#rgbBar');
  L.flowPos = 0; L.flowT = null;
  L.flow = (t, period) => {
    const dt = L.flowT == null ? 0 : t - L.flowT;
    L.flowT = t;
    if (period == null && L.flowPos === 0) return;
    L.flowPos += dt / (period || LOGO_FLOW_MS) * L.wCss;
    if (period == null && L.flowPos >= L.wCss) L.flowPos = 0; else L.flowPos %= L.wCss;
    bar.style.backgroundPosition = Math.round(L.flowPos / 2) * 2 + 'px 0';
  };
  bar.style.backgroundPosition = '0 0';
  L.bands = (maxShift, prob) => {
    const out = []; let y = 0;
    while (y < L.H) { const h = Math.round(lrand(3, 10) * dpr); out.push([h, Math.random() < prob ? Math.round(lrand(-maxShift, maxShift) * dpr) : 0]); y += h; }
    return out;
  };
  return L;
}

function logoBusy() { return S.view === 'detecting' || S.view === 'installing'; }

// The phase starts at 0 when the intro ends (offset = base, the intro's last frame) and accumulates, so neither the
// end of the intro nor a busy/idle switch (different period) makes the logo jump.
function logoBreathe(L, t, period, amp) {
  L.ph = (L.ph || 0) + (L.pt == null ? 0 : (t - L.pt) / period * Math.PI * 2);
  L.pt = t;
  L.rgb(L.g, Math.max(0, L.base + Math.round(Math.sin(L.ph) * amp * L.dpr * 2) / 2));
}

function logoTick(t) {
  LG.raf = requestAnimationFrame(logoTick);
  if (t - LG.last < 1000 / LOGO_FPS - 2) return;
  LG.last = t;
  const L = LG.L, R = LG.run;
  if (!L) return;
  // the first rAF timestamp can be a little earlier than performance.now() at scheduling time: clamp to frame 0
  const i = Math.max(0, Math.floor((t - LG.t0) / LOGO_INTRO_MS));
  L.flow(t, logoBusy() ? LOGO_FLOW_MS : null);                  // the bar runs during the intro too
  if (i < LOGO_INTRO.length) {
    L.rgb(L.bg); L.pixelate(L.g, L.buf, Math.max(1, Math.round(LOGO_INTRO[i] * L.dpr)));
    return;
  }
  if (logoBusy()) {
    if (t >= R.next && t > R.end) { R.end = t + lrand(90, 150); R.next = R.end + lrand(2200, 4500); }
    if (t < R.end) {
      if (t - R.lastB > 45) { R.lastB = t; R.b = L.bands(3, .35); }
      L.rgb(L.bg, L.base + Math.round(2 * L.dpr)); L.slices(L.g, L.buf, R.b);
    } else logoBreathe(L, t, 2400, 1);
  } else logoBreathe(L, t, 7000, .7);
}

function drawLogo() {
  if (LG.raf) cancelAnimationFrame(LG.raf);
  LG.raf = 0;
  const L = buildLogo(LG.text);
  if (!L) { $('#pxLogo').hidden = true; return; }
  LG.L = L;
  if (reducedMotion.matches || typeof requestAnimationFrame !== 'function') { L.rgb(L.g); return; }
  const t0 = performance.now();
  LG.t0 = t0;
  LG.run = { next: t0 + LOGO_INTRO.length * LOGO_INTRO_MS + 1500, end: 0, lastB: 0, b: [] };
  LG.raf = requestAnimationFrame(logoTick);
}

function startLogo(text) {
  LG.text = text;
  $('#pxLogo').hidden = false;
  try { drawLogo(); } catch (e) { $('#pxLogo').hidden = true; }
}

/* ------------------------------------------------------------------ notices / diagnostics box */

function notify(kind, text) {
  const box = $('#notices');
  const n = el('div', 'notice ' + kind, text);
  box.appendChild(n);
  while (box.children.length > 3) box.removeChild(box.firstChild);
  $('#content').scrollTop = 0;
}

function clearNotices() { $('#notices').replaceChildren(); }

function showDiag(kind, parts) {
  const box = $('#diagBox');
  box.className = 'diag-box' + (kind ? ' ' + kind : '');
  box.replaceChildren(...parts);
  box.hidden = !DIAG_VIEWS.has(S.view);
  if (!box.hidden && box.scrollIntoView) box.scrollIntoView({ block: 'nearest' });
}

function clearDiag() {
  const box = $('#diagBox');
  box.replaceChildren();
  box.hidden = true;
}

/* ------------------------------------------------------------------ views & buttons */

function setView(v) {
  S.view = v;
  for (const [name, id] of Object.entries(VIEW_IDS)) $('#' + id).hidden = name !== v;
  for (const g of document.querySelectorAll('.foot .actions[data-for]')) g.hidden = g.dataset.for !== v;
  const diag = $('#diagBox');
  diag.hidden = !(DIAG_VIEWS.has(v) && diag.childNodes.length > 0);
  $('#content').scrollTop = 0;
  refreshButtons();
}

// the space plan for the chosen version (old and new env side by side until the new one passed); older backends sent none
function planFor(info, flavor) { return info && flavor && info.plans ? info.plans[flavor] || null : null; }

// space the install needs at its peak (env + cache before the torch static libs are pruned); older backends only sent the download size
function requiredGb(info) {
  const plan = planFor(info, S.flavor);
  if (plan && plan.requiredGb != null) return plan.requiredGb;
  return info.requiredGb != null ? info.requiredGb : info.downloadGb;
}

/* The install folder (envs, uv cache) is chosen only in the installer, so the way out of a full drive is reinstalling elsewhere. */
function otherDiskHint() {
  const name = brandName();
  return `
也可以重新執行${name ? ` ${name} ` : ''}安裝程式，在「變更位置」選空間比較多的磁碟。`;
}

function diskInsufficient() {
  const info = S.info;
  return !!(info && requiredGb(info) != null && info.diskFree != null && info.diskFree < requiredGb(info));
}

// too little space for old + new, but deleting the old env first would be enough
function canDeleteOldFirst() {
  const plan = planFor(S.info, S.flavor);
  return !!(plan && plan.canDeleteOld && diskInsufficient());
}

// recommendation statuses that still let the user pick a version by hand (an NVIDIA card nvidia-smi could not classify)
const MANUAL_STATUSES = new Set(['unknown_nvidia']);
function canOffer(info) { return !!info.rec.flavor || MANUAL_STATUSES.has(info.rec.status); }

function findOption(flavor) {
  if (!S.info || !flavor) return null;
  return S.info.options.find((o) => o.flavor === flavor) || null;
}

function canInstall() {
  const info = S.info;
  if (!info || !canOffer(info) || !S.flavor) return false;
  const opt = findOption(S.flavor);
  return !!(opt && opt.supported) && !diskInsufficient();
}

// Every button carries data-act. While installing only 取消 is live; otherwise everything waits for in-flight calls.
function refreshButtons() {
  const installing = S.view === 'installing';
  for (const b of document.querySelectorAll('button[data-act]')) {
    const act = b.dataset.act;
    let off;
    if (act === 'cancel') {
      off = !installing || S.cancelRequested;
    } else if (installing) {
      off = true;
    } else {
      off = S.busy > 0;
      if (act === 'install') off = off || !canInstall();
      if (act === 'installDeleteOld') off = off || !canDeleteOldFirst();
      if (act === 'launch') off = off || S.launched;
      if (act === 'retry') off = off || !(S.runFlavor || S.flavor);
    }
    b.disabled = off;
  }
  $('#optSelect').disabled = installing || S.busy > 0;
  $('#cancelNote').hidden = !(installing && S.cancelRequested);
}

/* ------------------------------------------------------------------ backend access */

async function call(name, ...args) {
  const api = S.api;
  if (!api || typeof api[name] !== 'function') throw new Error(`後端沒有提供「${name}」功能`);
  return await api[name](...args);
}

function apiReady() {
  const a = window.pywebview && window.pywebview.api;
  return a && typeof a.detect === 'function' ? a : null;
}

function waitForApi() {
  return new Promise((resolve) => {
    let done = false;
    let timer = null;
    const finish = () => {
      if (done) return;
      done = true;
      clearTimeout(timer);
      window.removeEventListener('pywebviewready', finish);
      resolve(apiReady());
    };
    if (apiReady()) { finish(); return; }
    window.addEventListener('pywebviewready', finish);
    timer = setTimeout(finish, API_WAIT_MS);
  });
}

function normalizeInfo(raw) {
  if (!raw || typeof raw !== 'object') throw new Error('偵測結果的格式不正確');
  const rec = raw.recommendation && typeof raw.recommendation === 'object' ? raw.recommendation : {};
  const cur = raw.current && typeof raw.current === 'object' ? raw.current : null;
  return {
    gpus: Array.isArray(raw.gpus) ? raw.gpus.filter((g) => g && typeof g === 'object') : [],
    rec: {
      flavor: typeof rec.flavor === 'string' && rec.flavor ? rec.flavor : null,
      status: typeof rec.status === 'string' ? rec.status : '',
      warnings: Array.isArray(rec.warnings) ? rec.warnings.filter((w) => typeof w === 'string') : [],
    },
    supportedCards: (Array.isArray(raw.supported_cards) ? raw.supported_cards : []).filter((c) => typeof c === 'string' && c),
    options: (Array.isArray(raw.options) ? raw.options : [])
      .filter((o) => o && typeof o.flavor === 'string' && o.flavor)
      .map((o) => ({ flavor: o.flavor, label: typeof o.label === 'string' && o.label ? o.label : o.flavor, supported: o.supported === true })),
    diskFree: finiteNum(raw.disk_free_gb),
    downloadGb: finiteNum(raw.estimated_download_gb),
    requiredGb: finiteNum(raw.required_disk_gb),
    webview2: raw.webview2_installed !== false,    // only an explicit false is a problem
    current: cur ? { flavor: typeof cur.flavor === 'string' && cur.flavor ? cur.flavor : null, verified: cur.verified === true } : null,
    baseReady: raw.base_ready === true,
    plans: normalizePlans(raw.disk_plans),
    inUse: normalizeProcs(raw.in_use),
  };
}

function normalizePlan(p) {
  if (!p || typeof p !== 'object') return null;
  return { requiredGb: finiteNum(p.required_gb), freeableGb: finiteNum(p.freeable_gb), canDeleteOld: p.can_delete_old_first === true };
}

function normalizePlans(raw) {
  const out = {};
  if (!raw || typeof raw !== 'object') return out;
  for (const [flavor, p] of Object.entries(raw)) { const n = normalizePlan(p); if (n) out[flavor] = n; }
  return out;
}

function normalizeProcs(raw) {
  return (Array.isArray(raw) ? raw : []).filter((p) => p && typeof p === 'object')
    .map((p) => ({ pid: finiteNum(p.pid), name: typeof p.name === 'string' ? p.name : '' }));
}

function normalizePoll(raw) {
  if (!raw || typeof raw !== 'object') throw new Error('安裝進度的格式不正確');
  const download = normalizeDownload(raw.phase_progress);
  return {
    phase: typeof raw.phase === 'string' ? raw.phase : 'idle',
    lines: Array.isArray(raw.lines) ? raw.lines.map((x) => String(x)) : [],
    message: typeof raw.message === 'string' ? raw.message : '',
    device: typeof raw.device_name === 'string' && raw.device_name ? raw.device_name : null,
    // bridge.poll() sends phase_progress = {downloaded_bytes, total_bytes, speed_bps, eta_s} while the current phase's
    // download can be measured, else null (warm cache, or nothing measurable): then the active phase just pulses.
    download,
    phaseProgress: download ? Math.max(0, Math.min(1, download.done / download.total)) : null,
  };
}

function normalizeDownload(raw) {
  if (!raw || typeof raw !== 'object') return null;
  const total = finiteNum(raw.total_bytes), done = finiteNum(raw.downloaded_bytes);
  if (total == null || total <= 0 || done == null) return null;
  const speed = finiteNum(raw.speed_bps), eta = finiteNum(raw.eta_s);
  return {
    done: Math.max(0, Math.min(total, done)),
    total,
    speed: speed != null && speed > 0 ? speed : null,
    eta: eta != null && eta >= 0 ? eta : null,
  };
}

/* ------------------------------------------------------------------ detection */

async function runDetect(first) {
  stopPolling();
  clearNotices();
  clearDiag();
  S.cancelRequested = false;
  S.busy++;
  setView('detecting');
  let raw;
  try {
    raw = await call('detect');
  } catch (e) {
    S.busy--;
    showDetectErrorText(errText(e));
    return;
  }
  S.busy--;
  if (raw && raw.ok === false) {                               // the backend reports a failed detection as {ok:false, message}
    showDetectErrorText(raw.message || '偵測失敗');
    return;
  }
  S.info = normalizeInfo(raw);
  S.inUse = S.info.inUse;
  pickDefaultFlavor();
  // A reloaded page may find an install already running (or finished) in the backend: pick it up again.
  if (first && await tryResume()) return;
  renderReady();
  setView('ready');
}

// lead = plain explanation (default: generic retry advice); text = raw message, kept in 技術細節
function showDetectErrorText(text, lead) {
  $('#detectErrorLead').textContent = lead ||
    '檢查這台電腦的顯示卡時發生問題。可以按「重新偵測」再試一次；若一直失敗，請匯出診斷包提供給開發者。';
  $('#detectErrorMsg').textContent = text;
  $('#detectErrorDetails').hidden = !text;
  setView('detectError');
}

function pickDefaultFlavor() {
  const supported = S.info.options.filter((o) => o.supported);
  if (S.flavor && supported.some((o) => o.flavor === S.flavor)) return;
  const rec = S.info.rec.flavor && supported.find((o) => o.flavor === S.info.rec.flavor);
  S.flavor = (rec || supported[0] || {}).flavor || null;
}

function renderReady() {
  const info = S.info;
  const offer = canOffer(info);

  // graphics cards: name + vendor only; driver / compute capability go to 詳細資訊
  const list = $('#gpuList');
  list.replaceChildren();
  if (!info.gpus.length) {
    list.appendChild(el('li', 'empty', '未偵測到任何顯示卡。'));
  } else {
    for (const g of info.gpus) {
      const li = el('li');
      const vendor = pick(VENDOR_TEXT, g.vendor) ? g.vendor : 'other';
      li.appendChild(el('span', 'tag ' + vendor, VENDOR_TEXT[vendor]));
      li.appendChild(el('span', 'gpu-name', typeof g.name === 'string' && g.name ? g.name : '未知'));
      list.appendChild(li);
    }
  }

  // explanation + warnings
  const st = pick(STATUS_TEXT, info.rec.status) || STATUS_FALLBACK;
  const stBox = $('#recStatus');
  stBox.className = 'notice ' + st.tone;
  stBox.textContent = st.text;

  const wl = $('#recWarnings');
  wl.replaceChildren();
  const seen = new Set();
  for (const code of info.rec.warnings) {
    const text = pick(WARNING_TEXT, code) || WARNING_FALLBACK;
    if (seen.has(text)) continue;
    seen.add(text);
    wl.appendChild(el('li', null, text));
  }

  // nothing can be installed: say what that means and what the user can still do (there is no CPU install)
  const no = $('#noOffer');
  no.hidden = offer;
  if (!offer) {
    const cards = $('#supportedCards');
    cards.replaceChildren();
    for (const c of info.supportedCards) cards.appendChild(el('li', null, c));
    $('#supportedBox').hidden = !info.supportedCards.length;
  }

  $('#webview2Warn').hidden = info.webview2;

  // plain one-liner about what is already on this computer
  const envLine = $('#envLine');
  let envText = '', envOk = false;
  if (info.current && info.current.flavor) {
    const opt = findOption(info.current.flavor);
    const label = opt ? opt.label : info.current.flavor;
    if (info.current.verified) {
      envText = `這台電腦已經安裝過 ${label}，並通過顯示卡測試。重新安裝會裝到新的資料夾，通過測試後才刪除舊的；` +
        '中途失敗或取消，現在的版本照常可以用。';
      envOk = true;
    }
    else envText = `之前安裝過 ${label}，但還沒通過顯示卡測試，建議重新安裝。`;
  } else if (info.baseReady) {
    envText = '部分程式元件之前已經準備好，這次安裝會快一些。';
  }
  envLine.textContent = envText;
  envLine.className = 'env-line' + (envOk ? ' ok' : '');
  envLine.hidden = !envText;

  renderDetails(info);

  // install options
  $('#cInstall').hidden = !offer;
  $('#btnInstall').hidden = !offer;
  // with nothing to install, exporting diagnostics is the main thing left to do
  $('#btnDiagReady').classList.toggle('primary', !offer);
  if (offer) renderOptions();
  updateInstallUi();
}

// 詳細資訊: technical facts for whoever helps the user (driver version, compute capability, environment state)
function renderDetails(info) {
  const dl = $('#detailList');
  dl.replaceChildren();
  const row = (label, value, cls) => { dl.appendChild(el('dt', null, label)); dl.appendChild(el('dd', cls, value)); };
  info.gpus.forEach((g, i) => {
    const name = typeof g.name === 'string' && g.name ? g.name : '未知';
    dl.appendChild(el('dt', 'group', info.gpus.length > 1 ? `顯示卡 ${i + 1}：${name}` : `顯示卡：${name}`));
    row('驅動程式版本', typeof g.driver_version === 'string' && g.driver_version ? g.driver_version : '未知');
    if (typeof g.compute_capability === 'string' && g.compute_capability) row('運算能力（compute capability）', g.compute_capability);
  });
  dl.appendChild(el('dt', 'group', '安裝狀態'));
  const rec = info.rec.flavor ? findOption(info.rec.flavor) : null;
  row('推薦版本', rec ? rec.label : info.rec.flavor || '無');
  row('偵測結果代碼', info.rec.status || '未知');
  row('基礎環境', info.baseReady ? '已就緒' : '尚未建立', info.baseReady ? 'ok' : 'no');
  if (info.current && info.current.flavor) {
    const opt = findOption(info.current.flavor);
    row('已安裝的 GPU 版', `${opt ? opt.label : info.current.flavor}（${info.current.verified ? '已通過自我檢查' : '尚未通過自我檢查'}）`,
      info.current.verified ? 'ok' : 'no');
  } else {
    row('已安裝的 GPU 版', '尚未安裝', 'no');
  }
  if (info.downloadGb != null) row('預估下載量', `${fmtGb(info.downloadGb)} GB`);
  if (info.requiredGb != null) row('安裝時需要的空間', `${fmtGb(info.requiredGb)} GB`);
  row('磁碟剩餘空間', info.diskFree != null ? `${fmtGb(info.diskFree)} GB` : '未知');
  row('WebView2 執行階段', info.webview2 ? '已安裝' : '未偵測到', info.webview2 ? 'ok' : 'no');
}

function renderOptions() {
  const sel = $('#optSelect');
  sel.replaceChildren();
  for (const o of S.info.options) {
    let text = o.supported ? o.label : `${o.label}（尚未開放）`;
    if (o.supported && o.flavor === S.info.rec.flavor) text += '（推薦）';
    const opt = el('option', null, text);
    opt.value = o.flavor;
    opt.disabled = !o.supported;
    sel.appendChild(opt);
  }
  if (S.flavor) sel.value = S.flavor;
}

function updateInstallUi() {
  const info = S.info;
  if (!canOffer(info)) { refreshButtons(); return; }

  // download size vs. free space
  const free = info.diskFree != null ? fmtGb(info.diskFree) + ' GB' : '未知';
  const need = requiredGb(info) != null ? `，安裝時需要約 ${fmtGb(requiredGb(info))} GB 空間` : '';
  $('#sizeInfo').textContent = info.downloadGb != null
    ? `需要下載約 ${fmtGb(info.downloadGb)} GB${need}（目前剩餘 ${free}）。`
    : `下載大小未知${need}（目前剩餘 ${free}）。`;
  const low = diskInsufficient();
  const deleteOld = canDeleteOldFirst();
  $('#diskWarn').hidden = !low || deleteOld;
  if (low && !deleteOld) {
    $('#diskWarnText').textContent =
      `磁碟空間不夠：需要約 ${fmtGb(requiredGb(info))} GB，目前只剩 ${fmtGb(info.diskFree)} GB。請先刪除不需要的檔案清出空間，再按「重新偵測」。` +
      otherDiskHint();
  }
  // old + new side by side do not fit, deleting the old env first would: let the user choose, with the risk spelled out
  $('#lowSpaceBox').hidden = !deleteOld;
  $('#envLine').hidden = deleteOld || !$('#envLine').textContent;     // its "the old one keeps working" would contradict the box
  if (deleteOld) {
    const plan = planFor(info, S.flavor);
    const freed = plan.freeableGb != null ? `（可以釋放約 ${fmtGb(plan.freeableGb)} GB）` : '';
    $('#lowSpaceText').textContent =
      `磁碟空間不夠同時放新、舊兩份：需要約 ${fmtGb(requiredGb(info))} GB，目前只剩 ${fmtGb(info.diskFree)} GB。\n` +
      `可以先刪除舊的環境${freed}再安裝。注意：這樣如果安裝失敗或取消，舊的版本也不能用了，` +
      '必須重新下載安裝（約 3 GB）才能使用。也可以先清出空間，再按「重新偵測」。' + otherDiskHint();
  }
  renderInUse();

  // button label + "nothing installable" hint
  const opt = findOption(S.flavor);
  $('#btnInstall').textContent = opt && opt.supported ? `安裝 ${opt.label}` : '安裝';
  const hint = $('#optHint');
  const anySupported = info.options.some((o) => o.supported);
  hint.hidden = anySupported;
  if (!anySupported) hint.textContent = '目前沒有可以安裝的選項。';
  refreshButtons();
}

// the app is still running from the env the install would replace: say so, with a 重試 button
function renderInUse() {
  const box = $('#inUseBox');
  const n = S.inUse.length;
  box.hidden = n === 0;
  if (!n) return;
  const name = brandName();
  const app = name || '程式';
  const sp = name ? ' ' : '';            // a space between a Latin name and the Chinese around it
  $('#inUseText').textContent =
    `${app}${sp}還開著（${n} 個程式），安裝前要先關閉，才能換掉它正在使用的檔案。\n` +
    `請關閉${sp}${app}${sp}的所有視窗，也從系統匣圖示選擇結束，並關閉正在播放的影片，再按「重試」。`;
}

/* ------------------------------------------------------------------ install + polling */

async function startInstall(flavor, opts) {
  if (S.busy > 0 || S.view === 'installing' || !flavor) return;
  clearNotices();
  clearDiag();
  S.cancelRequested = false;
  S.busy++;
  refreshButtons();
  let res;
  try {
    res = opts ? await call('start_install', flavor, opts) : await call('start_install', flavor);
  } catch (e) {
    S.busy--;
    refreshButtons();
    notify('danger', `無法開始安裝：${errText(e)}`);
    return;
  }
  S.busy--;
  if (res && res.reason === 'in_use') {
    S.inUse = normalizeProcs(res.processes);
    if (!S.inUse.length) S.inUse = [{ pid: null, name: '' }];
    if (S.view !== 'ready') { renderReady(); setView('ready'); }
    renderInUse();
    refreshButtons();
    const box = $('#inUseBox');
    if (box.scrollIntoView) box.scrollIntoView({ block: 'nearest' });
    return;
  }
  if (res && res.reason === 'low_space' && S.info) {
    const plan = normalizePlan(res.plan);
    if (plan) S.info.plans[flavor] = plan;
    if (S.view !== 'ready') { renderReady(); setView('ready'); }
    updateInstallUi();
    notify('warn', res.message || '磁碟空間不夠。');
    return;
  }
  S.inUse = [];
  renderInUse();
  if (!res || res.ok !== true) {
    refreshButtons();
    notify('danger', (res && res.message) || '無法開始安裝。');
    return;
  }
  S.runFlavor = flavor;
  const opt = findOption(flavor);
  enterInstalling(null, opt ? opt.label : '');
}

function resetProgressUi(label) {
  $('#instTitle').textContent = label ? `正在安裝 ${label}` : '正在安裝';
  S.lastPhase = null;
  renderSteps('idle');
  setInstallProgress('idle', null);
  setDownloadLine(null);
  $('#instHint').textContent = PHASE_HINT.idle;
  $('#instMessage').textContent = '';
  $('#pollWarn').hidden = true;
  $('#pollWarn').textContent = '';
  setLog($('#logPre'), $('#logCount'), []);
  S.lastLines = [];
}

function enterInstalling(firstPoll, label) {
  resetProgressUi(label);
  setView('installing');
  if (firstPoll) applyPoll(firstPoll);
  if (S.view === 'installing') startPolling();
}

async function tryResume() {
  let p;
  try { p = normalizePoll(await call('poll')); } catch (e) { return false; }
  if (ACTIVE_PHASES.has(p.phase)) {
    S.runFlavor = null;
    enterInstalling(p, '');
    return true;
  }
  if (p.phase === 'done') { S.lastLines = p.lines; showDone(p); return true; }
  return false;
}

function stopPolling() {
  S.pollGen++;
  if (S.pollTimer) { clearTimeout(S.pollTimer); S.pollTimer = null; }
}

function startPolling() {
  stopPolling();
  S.pollFails = 0;
  const gen = S.pollGen;
  const schedule = () => {
    if (gen !== S.pollGen) return;
    S.pollTimer = setTimeout(tick, POLL_INTERVAL_MS);
  };
  const tick = async () => {
    S.pollTimer = null;
    if (gen !== S.pollGen) return;
    let p;
    try {
      p = normalizePoll(await call('poll'));
    } catch (e) {
      if (gen !== S.pollGen) return;
      onPollError(e);
      schedule();
      return;
    }
    if (gen !== S.pollGen) return;      // page left the installing state while we were waiting
    S.pollFails = 0;
    $('#pollWarn').hidden = true;
    applyPoll(p);                        // may stop polling (done / failed / cancelled)
    schedule();
  };
  tick();
}

function onPollError(e) {
  S.pollFails++;
  if (S.pollFails >= POLL_MAX_FAILS) {
    stopPolling();
    showFailed(`無法取得安裝進度（已連續失敗 ${POLL_MAX_FAILS} 次）：${errText(e)}`, S.lastLines, 'poll');
    return;
  }
  const w = $('#pollWarn');
  w.textContent = `暫時無法取得安裝進度（第 ${S.pollFails} 次），正在重試…\n${errText(e)}`;
  w.hidden = false;
}

function renderSteps(phase) {
  const idx = phase === 'cleanup' ? PHASE_ORDER.length : PHASE_ORDER.indexOf(phase);   // cleanup: all three steps done
  for (const li of document.querySelectorAll('#steps li')) {
    const i = PHASE_ORDER.indexOf(li.dataset.phase);
    const state = idx < 0 ? 'pending' : i < idx ? 'done' : i === idx ? 'active' : 'pending';
    li.className = state;
    li.querySelector('.st').textContent = STEP_STATE_TEXT[state];
    if (state === 'active') li.setAttribute('aria-current', 'step'); else li.removeAttribute('aria-current');
  }
}

/* Block progress bar. Completed phases are filled. setInstallProgress(phase, frac) is the single entry point: with
 * frac (0..1 inside the phase: the phase's downloaded / total bytes) the bar fills partially and shows a percentage;
 * with frac = null (nothing measurable) the active phase's blocks pulse instead. */
let progNodes = null;
function setInstallProgress(phase, frac) {
  const box = $('#progBlocks');
  if (!progNodes) {
    progNodes = [];
    for (let i = 0; i < PROGRESS_BLOCKS; i++) { const b = document.createElement('i'); progNodes.push(b); box.appendChild(b); }
  }
  const idx = PHASE_ORDER.indexOf(phase);
  const cleanup = phase === 'cleanup';
  const done = phase === 'done' || cleanup;
  let start = 0, span = 0;
  for (let i = 0; i < PHASE_ORDER.length; i++) {
    const w = PHASE_WEIGHT[PHASE_ORDER[i]];
    if (done || i < idx) start += w;
    else if (i === idx) { span = w; break; }
  }
  const known = frac != null && idx >= 0;
  const filled = Math.min(1, start + (known ? span * frac : 0)) * PROGRESS_BLOCKS;
  const full = Math.floor(filled + 1e-9);
  const part = filled - full;
  const runEnd = known ? 0 : Math.round((start + span) * PROGRESS_BLOCKS);
  progNodes.forEach((b, i) => {
    const on = i < full, isPart = i === full && part > 0.02;
    const run = !on && !isPart && i < runEnd;
    b.className = on ? 'on' : isPart ? 'part' : run ? 'run' : '';
    if (isPart) b.style.setProperty('--p', (part * 100).toFixed(0) + '%'); else b.style.removeProperty('--p');
    if (run) b.style.setProperty('--k', String(i - full)); else b.style.removeProperty('--k');
  });
  box.classList.toggle('ok', done && !cleanup);
  const pct = Math.round(filled / PROGRESS_BLOCKS * 100);
  box.setAttribute('aria-valuenow', String(pct));
  const pctEl = $('#progPct');              // number in the pixel font, "%" in the text font (pixel "%" is unreadable), like the queue
  if (known || done) {
    const u = document.createElement('span');
    u.className = 'u'; u.textContent = '%';
    pctEl.replaceChildren(String(pct), u);
  } else pctEl.replaceChildren();
  $('#progLabel').textContent = idx >= 0
    ? `第 ${idx + 1} 步，共 ${PHASE_ORDER.length} 步：${PHASE_NAME[phase]}`
    : cleanup ? '安裝完成，正在刪除舊環境…' : done ? '全部完成' : '準備開始…';
}

/* Download line under the bar: 「已下載 1.2 / 3.2 GB（37%） · 12.3 MB/s · 剩約 2 分 30 秒」 (the bar's own percentage
 * covers all three steps; this one is the current download only). Hidden when the phase has no
 * measurable download. Units are binary (GB = 2^30), like uv's own "Downloading torch (3.2GiB)". */
function fmtBytes(n) {
  if (n >= 2 ** 30) return `${(n / 2 ** 30).toFixed(1)} GB`;
  if (n >= 2 ** 20) return `${Math.round(n / 2 ** 20)} MB`;
  return `${Math.max(1, Math.round(n / 2 ** 10))} KB`;
}
function fmtPair(done, total) {           // "1.2 / 3.2 GB": one unit for both, taken from the total
  if (total >= 2 ** 30) return `${(done / 2 ** 30).toFixed(1)} / ${(total / 2 ** 30).toFixed(1)} GB`;
  if (total >= 2 ** 20) return `${Math.round(done / 2 ** 20)} / ${Math.round(total / 2 ** 20)} MB`;
  return `${fmtBytes(done)} / ${fmtBytes(total)}`;
}
function fmtSpeed(bps) {
  const mb = bps / 2 ** 20;
  return mb >= 1 ? `${mb.toFixed(1)} MB/s` : `${Math.max(1, Math.round(bps / 2 ** 10))} KB/s`;
}
function fmtEta(s) {                      // coarse on purpose: a jittery "剩 2 分 37 秒" reads worse than "2 分 40 秒"
  if (s < 10) return '剩不到 10 秒';
  if (s < 60) return `剩約 ${Math.round(s / 5) * 5} 秒`;
  if (s < 3600) {
    const t = Math.round(s / 10) * 10, m = Math.floor(t / 60), r = t % 60;
    return r ? `剩約 ${m} 分 ${r} 秒` : `剩約 ${m} 分`;
  }
  const t = Math.round(s / 60), h = Math.floor(t / 60), m = t % 60;
  return m ? `剩約 ${h} 小時 ${m} 分` : `剩約 ${h} 小時`;
}
function downloadText(d) {
  if (d.done >= d.total) return `已下載 ${fmtBytes(d.total)}，正在安裝…`;
  const parts = [`已下載 ${fmtPair(d.done, d.total)}（${Math.floor(d.done / d.total * 100)}%）`];
  if (d.speed != null) parts.push(fmtSpeed(d.speed));
  parts.push(d.eta != null ? fmtEta(d.eta) : '正在估計剩餘時間…');
  return parts.join(' · ');
}
function setDownloadLine(d) {
  const line = $('#progDl');
  line.hidden = !d;
  line.textContent = d ? downloadText(d) : '';
}

function setLog(pre, countEl, lines) {
  const tail = lines.slice(-LOG_TAIL);
  const text = tail.join('\n');
  if (pre.textContent !== text) {
    const stick = pre.scrollTop + pre.clientHeight >= pre.scrollHeight - 8;   // follow the end unless the user scrolled up
    pre.textContent = text;
    if (stick) pre.scrollTop = pre.scrollHeight;
  }
  countEl.textContent = String(tail.length);
}

function applyPoll(p) {
  S.lastLines = p.lines;
  if (p.phase === 'done') {
    stopPolling();
    showDone(p);
    return;
  }
  if (p.phase === 'failed') {
    stopPolling();
    showFailed(p.message, p.lines);
    return;
  }
  if (p.phase === 'cancelled') {
    stopPolling();
    S.cancelRequested = false;
    renderReady();
    setView('ready');
    notify('warn', '已取消安裝。已經下載的檔案會保留，之後重新安裝時不用再下載。');
    return;
  }
  // idle / base / gpu / selfcheck (an unknown phase just shows no step as active)
  if (ACTIVE_PHASES.has(p.phase)) S.lastPhase = p.phase;
  renderSteps(p.phase);
  setInstallProgress(p.phase, p.phaseProgress);
  setDownloadLine(p.download);
  $('#instHint').textContent = pick(PHASE_HINT, p.phase) || '';
  // the backend's own status line stays available under 詳細資訊
  $('#instMessage').textContent = p.message ? `目前狀態：${p.message}` : '';
  setLog($('#logPre'), $('#logCount'), p.lines);
}

function showDone(p) {
  S.launched = false;
  S.cancelRequested = false;
  $('#doneDevice').hidden = !p.device;
  $('#doneDeviceName').textContent = p.device || '';
  const lm = $('#launchMsg');
  lm.hidden = true;
  lm.textContent = '';
  setView('done');
}

/* Failed install: plain explanation first, raw backend message (often an English traceback) in 技術細節.
 * Simple substring checks; disk and network are checked before driver because package names such as
 * "nvidia-cuda-runtime" show up inside download errors too. */
const FAIL_RULES = [
  { test: ['no space', 'os error 112', 'not enough space', 'disk full', '空間'],
    lead: '磁碟空間不夠，安裝沒有完成。',
    next: '請先刪除不需要的檔案清出空間，再按「重試」。', hint: 'otherDisk' },
  { test: ['connect', 'timed out', 'timeout', 'dns', 'network', 'resolve host', 'error sending request', 'failed to download', 'failed to fetch', '連線', '網路'],
    lead: '下載時網路連線出了問題，安裝沒有完成。',
    // uv cannot resume a single file: a half-downloaded torch wheel (3.2 GiB for cu128) starts over
    next: '請確認網路正常後按「重試」。顯示卡加速元件主要是一個約 3 GB 的大檔案，沒下載完就中斷的話要從頭下載；其他已經下載完成的檔案不會重新下載。' },
  { test: ['driver', 'cuda'],
    lead: '顯示卡驅動程式可能太舊，或沒有正確安裝。',
    next: '請到 NVIDIA 官網下載並安裝最新的顯示卡驅動程式，重新開機後再按「重試」。' },
];
const FAIL_GENERIC = { lead: '可以按「重試」再試一次；若一直失敗，請匯出診斷包。', next: '' };
const FAIL_POLL = { lead: '和安裝程序失去聯繫，安裝可能已經中斷。', next: '可以按「重試」；若一直失敗，請匯出診斷包提供給開發者。' };

// bridge.py prefixes a step failure with the step's (technical) name; the last polled phase is only a fallback,
// because a fast step can start and fail between two polls.
const FAIL_PREFIX = [['基礎環境安裝失敗', 'base'], ['GPU 版安裝失敗', 'gpu'], ['自我檢查失敗', 'selfcheck']];
function failurePhase(message) {
  const m = String(message || '').trimStart();
  const hit = FAIL_PREFIX.find(([p]) => m.startsWith(p));
  return hit ? hit[1] : null;
}

function classifyFailure(message) {
  const m = String(message || '').toLowerCase();
  for (const r of FAIL_RULES) if (r.test.some((s) => m.includes(s))) return r;
  return FAIL_GENERIC;
}

function showFailed(message, lines, kind) {
  S.cancelRequested = false;
  const raw = message && message.trim() ? message : '安裝失敗，後端沒有提供詳細訊息。';
  const rule = kind === 'poll' ? FAIL_POLL : classifyFailure(raw);
  const failedPhase = failurePhase(raw) || S.lastPhase;
  const where = failedPhase && PHASE_NAME[failedPhase] ?`安裝停在「${PHASE_NAME[failedPhase]}」這一步。` : '';
  $('#failLead').textContent = rule.lead;
  $('#failNext').textContent = [where, rule.next, rule.hint === 'otherDisk' ? otherDiskHint() : ''].filter(Boolean).join('');
  $('#failMsg').textContent = raw;
  $('#failDetails').open = false;
  $('#failLogBox').hidden = !lines.length;
  setLog($('#failLogPre'), $('#failLogCount'), lines);
  setView('failed');
}

async function cancelInstall() {
  if (S.cancelRequested || S.view !== 'installing') return;
  S.cancelRequested = true;
  refreshButtons();
  try {
    const r = await call('cancel');
    if (!r || r.ok !== true) {
      S.cancelRequested = false;
      notify('warn', '目前無法取消安裝。');
    }
  } catch (e) {
    S.cancelRequested = false;
    notify('danger', `取消安裝失敗：${errText(e)}`);
  }
  refreshButtons();
}

/* ------------------------------------------------------------------ diagnostics, launch, close */

async function exportDiag() {
  if (S.busy > 0) return;
  S.busy++;
  refreshButtons();
  showDiag('', ['正在匯出診斷包…']);
  try {
    const r = await call('export_diagnostics');
    if (r && r.ok === true) {
      const parts = [(typeof r.message === 'string' && r.message) || '診斷包已匯出。'];
      if (typeof r.path === 'string' && r.path) {
        parts.push('\n位置：');
        parts.push(el('code', null, r.path));
      }
      parts.push('\n請把這個檔案傳給開發者。');
      showDiag('ok', parts);
    } else {
      showDiag('error', [(r && typeof r.message === 'string' && r.message) || '匯出診斷包失敗。']);
    }
  } catch (e) {
    showDiag('error', [`匯出診斷包失敗：${errText(e)}`]);
  }
  S.busy--;
  refreshButtons();
}

async function launchMain() {
  if (S.busy > 0 || S.launched) return;
  const lm = $('#launchMsg');
  lm.hidden = true;
  S.busy++;
  refreshButtons();
  let ok = false;
  let text;
  try {
    const r = await call('launch_main');
    ok = !!(r && r.ok === true);
    text = (r && typeof r.message === 'string' && r.message) || (ok ? '已啟動。' : '無法啟動。');
  } catch (e) {
    text = `無法啟動：${errText(e)}`;
  }
  S.busy--;
  if (ok) S.launched = true;
  lm.className = 'launch-msg' + (ok ? '' : ' error');
  lm.textContent = text;
  lm.hidden = false;
  refreshButtons();
}

function closeWindow() {
  try { window.close(); } catch (e) { /* handled below */ }
  // In a normal browser tab window.close() is ignored; tell the user instead of doing nothing.
  clearTimeout(S.closeTimer);
  S.closeTimer = setTimeout(() => {
    S.closeTimer = null;
    if (!window.closed) notify('info', '無法自動關閉視窗，請直接關閉這個視窗。');
  }, 400);
}

/* ------------------------------------------------------------------ wiring */

const ACTIONS = {
  install: () => startInstall(S.flavor),
  installDeleteOld: () => startInstall(S.flavor, { delete_old_first: true }),
  retryInUse: () => startInstall(S.runFlavor || S.flavor),
  retry: () => startInstall(S.runFlavor || S.flavor),
  reselect: () => { clearNotices(); clearDiag(); renderReady(); setView('ready'); },
  redetect: () => runDetect(false),
  cancel: cancelInstall,
  diag: exportDiag,
  launch: launchMain,
  close: closeWindow,
};

document.addEventListener('click', (e) => {
  const b = e.target.closest ? e.target.closest('button[data-act]') : null;
  if (!b || b.disabled) return;
  const fn = ACTIONS[b.dataset.act];
  if (fn) fn();
});

$('#optSelect').addEventListener('change', (e) => {
  S.flavor = e.target.value || null;
  updateInstallUi();
});

$('#pxLogo').addEventListener('click', () => { if (LG.text) startLogo(LG.text); });

// opening a closed log should land on its newest lines
for (const [det, pre] of [['#logDetails', '#logPre'], ['#failDetails', '#failLogPre']]) {
  $(det).addEventListener('toggle', (e) => { if (e.target.open) $(pre).scrollTop = $(pre).scrollHeight; });
}

// never leave a timer behind when the page goes away
window.addEventListener('pagehide', () => {
  stopPolling();
  clearTimeout(S.closeTimer);
  if (LG.raf) cancelAnimationFrame(LG.raf);
  LG.raf = 0;
});

async function boot() {
  await ensureBranding();
  // wait for fonts so the logo glyphs are the final ones (bounded: never hold up the page)
  if (document.fonts && document.fonts.ready) await Promise.race([document.fonts.ready, new Promise((r) => setTimeout(r, 800))]).catch(() => {});
  applyBranding();
  refreshButtons();

  const api = await waitForApi();
  if (!api) {
    const msg = '無法連線到安裝程式的後端（window.pywebview.api 沒有出現）。';
    showDetectErrorText(msg, '安裝程式沒有正常啟動。請關閉這個視窗後重新開啟；若一直出現，請聯絡開發者。');
    return;
  }
  S.api = api;

  const mock = window.pywebview && window.pywebview._mock;
  if (mock) {
    document.title += '（預覽模式）';
    const banner = $('#previewBanner');
    const help = typeof mock.help === 'string' ? mock.help : '預覽模式：目前使用假資料。';
    banner.textContent = help.split('\n')[0];
    banner.title = help;
    banner.hidden = false;
  }
  await runDetect(true);
}

boot().catch((e) => {
  showDetectErrorText(`初始化失敗：${errText(e)}`);
});

})();
