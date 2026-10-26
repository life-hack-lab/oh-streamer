/* 處理佇列 front-end (vanilla JS, offline).
 *
 * Bridge: window.pywebview.api.<method>(...) -> Promise; Python pushes window.ohStreamer.onState(state).
 * Without window.pywebview (plain browser) an in-file mock backend with sample items is used
 * (also forced with ?mock). Extras for previewing: ?busy=1 (player_busy), ?notice=text, ?perf=none|prompt|running|failed|cancelled|blocked|stale,
 * ?stream=sub|nosub|behind|stuck|none (the 串流中 card; default sub).
 * All dynamic text is written via textContent; innerHTML is only used for the static ICONS below, the intake panel templates and the subtitle combo checklists (every dynamic value there goes through esc()).
 */
(() => {
'use strict';

/* ------------------------------------------------------------------ helpers */

const $ = (sel, root = document) => root.querySelector(sel);

const ICONS = {
  play: '<svg viewBox="0 0 24 24"><path class="fill" d="M7.500 4.800v14.400a.8.8 0 0 0 1.200.7l11.300-7.200a.8.8 0 0 0 0-1.400L8.700 4.100a.8.8 0 0 0-1.200.7z"/></svg>',
  pause: '<svg viewBox="0 0 24 24"><path d="M8 5v14M16 5v14" stroke-width="2.600"/></svg>',
  retry: '<svg viewBox="0 0 24 24"><path d="M20 11a8 8 0 1 0-2.300 5.700"/><path d="M20 4v7h-7"/></svg>',
  export: '<svg viewBox="0 0 24 24"><path d="M12 15V3M8 7l4-4 4 4"/><path d="M5 12v7a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2v-7"/></svg>',
  trash: '<svg viewBox="0 0 24 24"><path d="M4 7h16M9 7V4h6v3M6 7l1 13h10l1-13M10 11v6M14 11v6"/></svg>',
  folder: '<svg viewBox="0 0 24 24"><path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/></svg>',
  key: '<svg viewBox="0 0 24 24"><circle cx="8" cy="15" r="4"/><path d="M11 12l9-9M16 7l3 3M14 9l2 2"/></svg>',
  lock: '<svg viewBox="0 0 24 24"><rect x="5" y="11" width="14" height="9" rx="2"/><path d="M8 11V8a4 4 0 0 1 8 0v3"/></svg>',
  eye: '<svg viewBox="0 0 24 24"><path d="M2 12s3.500-7 10-7 10 7 10 7-3.500 7-10 7-10-7-10-7z"/><circle cx="12" cy="12" r="3"/></svg>',
  grip: '<svg viewBox="0 0 24 24"><g class="fill" stroke="none"><circle cx="9" cy="6" r="1.500"/><circle cx="15" cy="6" r="1.500"/><circle cx="9" cy="12" r="1.500"/><circle cx="15" cy="12" r="1.500"/><circle cx="9" cy="18" r="1.500"/><circle cx="15" cy="18" r="1.500"/></g></svg>',
  alert: '<svg viewBox="0 0 24 24"><path d="M12 3.500l10 17.500H2z"/><path d="M12 10v4.500M12 17.500v.01"/></svg>',
  info: '<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"/><path d="M12 11v6M12 7.500v.01"/></svg>',
  globe: '<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"/><path d="M3 12h18M12 3c3 3.500 3 14.500 0 18M12 3c-3 3.500-3 14.500 0 18"/></svg>',
  file: '<svg viewBox="0 0 24 24"><path d="M6 3h8l5 5v13H6z"/><path d="M14 3v5h5"/></svg>',
  layers: '<svg viewBox="0 0 24 24"><path d="M12 3l9 5-9 5-9-5z"/><path d="M3 13l9 5 9-5"/></svg>',
};

function svgEl(name) {
  const t = document.createElement('template');
  t.innerHTML = ICONS[name];
  return t.content.firstChild;
}

function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text != null) e.textContent = text;
  return e;
}

function setText(node, text) { if (node.textContent !== text) node.textContent = text; }
function setHidden(node, hidden) { if (node.hidden !== hidden) node.hidden = hidden; }
function setAttr(node, name, value) { if (node.getAttribute(name) !== String(value)) node.setAttribute(name, value); }

const num = (v) => (typeof v === 'number' && isFinite(v) ? v : (typeof v === 'string' && v.trim() !== '' && isFinite(+v) ? +v : 0));

function fmtBytes(n) {
  n = num(n);
  if (n <= 0) return '0 B';
  const u = ['B', 'KB', 'MB', 'GB', 'TB'];
  let i = Math.min(u.length - 1, Math.floor(Math.log(n) / Math.log(1024)));
  const v = n / Math.pow(1024, i);
  return (v >= 100 || i === 0 ? v.toFixed(0) : v >= 10 ? v.toFixed(1) : v.toFixed(2)) + ' ' + u[i];
}
function fmtGB(n) { return (num(n) / 1073741824).toFixed(1); }

function fmtDur(sec) {
  sec = Math.max(0, Math.round(num(sec)));
  const h = Math.floor(sec / 3600), m = Math.floor((sec % 3600) / 60), s = sec % 60;
  const p = (x) => String(x).padStart(2, '0');
  return h ? `${h}:${p(m)}:${p(s)}` : `${m}:${p(s)}`;
}

function fmtTimeProgress(curSec, totSec) {
  const pad = (x) => String(x).padStart(2, '0');
  const dur = (s, forceHours) => {
    s = Math.max(0, Math.round(num(s)));
    const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = s % 60;
    if (forceHours || h > 0) {
      return `${h}:${pad(m)}:${pad(sec)}`;
    }
    return `${m}:${pad(sec)}`;
  };
  const hasH = num(totSec) >= 3600;
  return `${dur(curSec, hasH)} / ${dur(totSec, hasH)}`;
}

function basename(p) {
  const parts = String(p).split(/[\\/]/).filter(Boolean);
  return parts.length ? parts[parts.length - 1] : String(p);
}

function sourceLabel(item) {
  const t = String(item.target || '');
  if (item.kind !== 'file' && /^https?:\/\//i.test(t)) {
    try { return { kind: 'url', text: new URL(t).hostname.replace(/^www\./, '') }; } catch (e) { /* fall through */ }
  }
  return { kind: 'file', text: basename(t) };
}

function qualityLabel(q) {
  if (q == null || q === '' || q === false) return '';
  if (typeof q === 'number') return q + 'p';
  if (typeof q === 'string') return /^\d+$/.test(q) ? q + 'p' : q;
  if (typeof q === 'object') {
    if (q.label) return String(q.label);
    if (q.height) return q.height + 'p';
    if (q.name) return String(q.name);
    if (q.format) return String(q.format);
  }
  return '';
}

const STATE_LABEL = {
  waiting: '等待中', downloading: '下載中', queued: '排隊中', processing: '處理中',
  paused: '已暫停', done: '完成', failed: '失敗',
};
const ACTIVE_STATES = new Set(['waiting', 'downloading', 'queued', 'processing']);
const MODEL_EDIT_STATES = new Set(['waiting', 'downloading', 'queued', 'paused', 'failed']);
const SUBTITLE_EDIT_STATES = new Set(['waiting', 'downloading', 'queued', 'paused']);
const MODEL_KEYS = ['detection_model', 'restoration_model'];

const SUBTITLE_ACTIVE_STATES = new Set(['queued', 'waiting_gpu', 'asr', 'translate', 'combos', 'partial', 'mux']);
// Provider error kinds as the user should read them.
const FALLBACK_REASON = {
  rate_limit: '被限流', quota_daily: '今日額度用完', network: '網路問題', timeout: '逾時', server: '服務暫時故障',
  auth: '金鑰或權限錯誤', bad_response: '回應格式錯誤', content_blocked: '內容被拒絕', unsupported: '不支援或環境不符',
  error: '執行錯誤', oom: 'GPU 記憶體不足', exhausted: '所有候選都無法使用',
};
function fallbackReason(kind) { return FALLBACK_REASON[kind] || String(kind); }

// M5b: renders the scheduler's slice-table view (subtitle.view, v=2). It reads nothing but `view` (plus the
// coarse job fields detail/error), so stale event fields can never leak into the label.
// Rounded time-until so the text only changes every 10 s / minute: <60 s rounds UP to 10 s, then whole minutes.
function subtitleRoundWhen(at, nowS, verb) {
  const s = Math.ceil(Number(at) - nowS);
  if (!(s > 0)) return `即將${verb}`;
  if (s <= 50) return `約 ${Math.ceil(s / 10) * 10} 秒後${verb}`;
  return `約 ${Math.max(1, Math.round(s / 60))} 分鐘後${verb}`;
}
const SUBTITLE_SPACE_NOTICE = '空間不足，佇列已暫停';
// Queue-wide holds that stop subtitle work while the subtitle view itself is not running (cloud work can go on).
// Pure: reads only the item and the top-level get_state fields. Returns null or {cat, glyph, text, detail, tone}.
function subtitleHalt(item, state) {
  const sub = item && item.subtitle;
  const view = sub && sub.view;
  if (view && String(view.state || 'queued') === 'running') return null;
  if ((item && item.state === 'paused') || (sub && sub.status === 'paused')) return { cat: 'paused', glyph: '⏸', text: '已暫停', detail: '', tone: '' };
  const st = state || {};
  if (st.paused_all) return { cat: 'halt', glyph: '⏸', text: '全部暫停中', detail: '繼續後自動恢復', tone: '' };
  if (st.perf && st.perf.status === 'running') return { cat: 'halt', glyph: '⇄', text: '效能檢測中', detail: '檢測完成後自動繼續', tone: '' };
  if (st.player_busy) return { cat: 'halt', glyph: '⇄', text: '讓給播放器', detail: '播放結束後自動繼續', tone: '' };
  if (st.notice === SUBTITLE_SPACE_NOTICE) return { cat: 'attn', glyph: '!', text: '磁碟空間不足', detail: '釋放空間後自動繼續', tone: 'warn' };
  return null;
}
const SUBTITLE_HALT_SKIP_STATUS = new Set(['mux', 'ready', 'mux_pending']);
function subtitleApplyHalt(display, subtitle, halt) {
  if (!display || !halt) return display;
  if (display.cat !== 'wait' && display.cat !== 'paused') return display;
  if (SUBTITLE_HALT_SKIP_STATUS.has(subtitle && subtitle.status) || display.status === 'failed') return display;
  const view = subtitle && subtitle.view;
  const covered = view ? Math.max(0, Math.floor(Number(view.covered_s) || 0)) : 0;
  const mmss = `${String(Math.floor(covered / 60)).padStart(2, '0')}:${String(covered % 60).padStart(2, '0')}`;
  return {
    ...display,
    cat: halt.cat, glyph: halt.glyph, tone: halt.tone,
    body: halt.text,
    label: covered > 0 ? `字幕已到 ${mmss} · ${halt.text}` : `字幕：${halt.text}`,
    detail: halt.detail,
  };
}
const SUBTITLE_VIEW_STAGE = { asr: '語音辨識', mt: '翻譯' };
const SUBTITLE_VIEW_EXECUTOR = { local: '本地', cloud: '雲端' };
function subtitleViewDisplay(subtitle, nowS) {
  const view = subtitle.view;
  const state = String(view.state || 'queued');
  const now = Number.isFinite(Number(nowS)) ? Number(nowS) : Date.now() / 1000;
  const retryIn = (at) => subtitleRoundWhen(at, now, '重試');
  const stageName = (stage) => SUBTITLE_VIEW_STAGE[stage] || '字幕';
  const sliceNo = (w) => (w && Number.isFinite(Number(w.slice)) ? `第 ${Number(w.slice) + 1} 片` : '');
  const covered = Math.max(0, Math.floor(Number(view.covered_s) || 0));
  const coverage = `${String(Math.floor(covered / 60)).padStart(2, '0')}:${String(covered % 60).padStart(2, '0')}`;
  const wait = view.wait || null;
  let body = '';
  let detail = '';
  let waitingKind = false;
  let glyph = '⧗';
  if (state === 'complete') {
    body = '完成';
  } else if (state === 'stuck') {
    const s = view.stuck || {};
    const cands = (Array.isArray(s.candidates) ? s.candidates : []).map((c) => {
      const name = c.name || c.id || '';
      return c.kind ? `${name}（${fallbackReason(c.kind)}）` : name;
    }).filter(Boolean);
    body = `${stageName(s.stage)}卡住${cands.length ? '：' + cands.join('、') : (s.kind ? `（${fallbackReason(s.kind)}）` : '')}`;
  } else if (state === 'running') {
    const r = view.running || {};
    const where = SUBTITLE_VIEW_EXECUTOR[r.executor] || '';
    const sn = sliceNo(r);
    const extra = [where, sn].filter(Boolean).join('，');
    body = `${r.stage === 'mt' ? '翻譯中' : '語音辨識中'}${extra ? `（${extra}）` : ''}`;
    if (r.candidate_name) detail = r.candidate_name;
    if (r.quota_wait) {
      detail = r.quota_retry_at != null ? `部分等待額度（${retryIn(r.quota_retry_at)}）` : '部分等待額度';
    }
    glyph = '';
  } else if (state === 'waiting_quota') {
    waitingKind = true;
    if (wait && wait.also_waiting_download) {
      body = '等待下載';
      glyph = '↓';
    } else {
      body = '等待額度中';
      detail = wait && wait.retry_at != null ? retryIn(wait.retry_at) : '冷卻中';
    }
  } else if (state === 'retrying') {
    waitingKind = true;
    const why = wait && wait.reason ? fallbackReason(wait.reason) : '';
    const when = wait && wait.retry_at != null ? retryIn(wait.retry_at) : '稍後重試';
    body = `重試中${why ? `（${why}）` : ''}`;
    detail = when;
  } else if (state === 'waiting_gpu') {
    waitingKind = true;
    body = '等待 GPU';
    detail = '修復中，稍後輪到字幕';
  } else if (state === 'waiting_executor') {
    waitingKind = true;
    const to = SUBTITLE_VIEW_EXECUTOR[wait && wait.executor] || '備援';
    body = `已交給${to === '本地' ? '本地 GPU 處理' : to + '處理'}，等待開始`;
    if (wait && wait.reclaim_at != null) detail = `雲端${subtitleRoundWhen(wait.reclaim_at, now, '恢復')}`;
  } else if (state === 'waiting_cut') {
    waitingKind = true;
    body = '準備切片';
  } else if (state === 'waiting_download') {
    waitingKind = true;
    body = '等待下載';
    detail = '字幕已追上下載進度';
    glyph = '↓';
  } else {
    body = '排隊中';
  }
  const partialStuck = view.partial_stuck === true && state !== 'stuck';
  if (partialStuck) {
    const n = Number(view.failures_elsewhere) || 0;
    detail = `另有 ${n > 0 ? n + ' ' : ''}段失敗，可重試`;
  }
  const label = state === 'complete' ? '字幕：完成'
    : covered > 0 ? `字幕已到 ${coverage} · ${body}` : `字幕：${body}`;
  const running = state === 'running';
  const rawProgress = view.progress == null ? NaN : Number(view.progress);
  const percent = running && Number.isFinite(rawProgress) ? Math.max(0, Math.min(100, rawProgress * 100)) : null;
  const active = state !== 'complete' && state !== 'stuck';
  const cat = state === 'complete' ? 'done' : state === 'stuck' ? 'attn' : running ? 'run' : 'wait';
  const coverageFill = Number.isFinite(rawProgress) ? Math.max(0, Math.min(100, rawProgress * 100)) : 0;
  if (cat === 'done') glyph = '✓';
  else if (cat === 'attn') glyph = '!';
  return {
    cat,
    glyph,
    tone: '',
    dim: false,
    fill: cat === 'done' ? 100 : coverageFill,
    status: state === 'stuck' ? 'failed' : state === 'complete' ? 'done' : state,
    label,
    body,
    active,
    showTrack: true,
    indeterminate: false,
    showPercent: running && percent != null,
    percent,
    detail: detail || (subtitle.detail == null ? '' : String(subtitle.detail)),
    error: state === 'stuck' ? String(subtitle.error || '') : '',
    blocked: false,
    retryable: state === 'stuck' || partialStuck,
    restartable: state === 'stuck' || state === 'complete',
  };
}

// Without a view (older rows, or before the scheduler published one) only the item-level status decides the
// label; every scheduler-side phase (queued/asr/translate/partial/...) reads as "排隊中" until a view arrives.
const SUBTITLE_ITEM_STATUS_LABEL = {
  done: '完成', mux: '封裝中', ready: '等待封裝', mux_pending: '等關閉播放器後封裝',
  skipped: '略過', cancelled: '已取消', paused: '已暫停', failed: '失敗',
};
const SUBTITLE_PLAY_WAIT_VIEW_STATES = new Set(['queued', 'running', 'waiting_quota', 'retrying', 'waiting_gpu', 'waiting_executor', 'waiting_download', 'waiting_cut']);
function subtitleHasView(subtitle) {
  return !!(subtitle && subtitle.view && subtitle.view.v === 2);
}

// ctx (optional): {item, state} lets queue-wide holds override the waiting look (see subtitleHalt).
function subtitleDisplay(subtitle, nowS, ctx) {
  const halt = ctx ? subtitleHalt(ctx.item, ctx.state) : null;
  if (subtitleHasView(subtitle)) {
    const display = subtitleViewDisplay(subtitle, nowS);
    // the item-level status still owns the mux / closed / cancelled phases that follow the slice table
    const st = subtitle.status;
    if (display.status === 'done' && ['mux', 'ready', 'mux_pending'].includes(st)) {
      const rawPct = subtitle.percent == null ? NaN : Number(subtitle.percent);
      const pct = st === 'mux' && Number.isFinite(rawPct) ? Math.max(0, Math.min(100, rawPct)) : null;
      return {
        ...display, label: `字幕：${SUBTITLE_ITEM_STATUS_LABEL[st]}`, status: st, active: true, restartable: false,
        cat: st === 'mux' ? 'run' : 'wait', glyph: st === 'mux' ? '' : '⧗',
        dim: st === 'mux' && pct == null, fill: st === 'mux' && pct != null ? pct : 100,
        showPercent: pct != null, percent: pct, detail: subtitle.detail == null ? '' : String(subtitle.detail),
      };
    }
    return subtitleApplyHalt(display, subtitle, halt);
  }
  if (!subtitle || !subtitle.status) return null;
  const status = subtitle.status;
  const text = SUBTITLE_ITEM_STATUS_LABEL[status] || '排隊中';
  const known = !!SUBTITLE_ITEM_STATUS_LABEL[status];
  const rawPercent = subtitle.percent == null ? NaN : Number(subtitle.percent);
  const muxing = status === 'mux';
  const percent = muxing && Number.isFinite(rawPercent) ? Math.max(0, Math.min(100, rawPercent)) : null;
  const active = !known || ['mux', 'ready', 'mux_pending'].includes(status);
  const cat = status === 'mux' ? 'run' : status === 'done' ? 'done' : status === 'paused' ? 'paused'
    : status === 'failed' ? 'attn' : (status === 'skipped' || status === 'cancelled') ? 'plain' : 'wait';
  const base = {
    cat,
    glyph: { wait: '⧗', paused: '⏸', attn: '!', done: '✓' }[cat] || '',
    tone: '',
    dim: muxing && percent == null,
    fill: muxing ? (percent == null ? 100 : percent) : cat === 'done' ? 100 : 0,
    status: known ? status : 'queued',
    label: `字幕：${text}`,
    active,
    showTrack: muxing,
    indeterminate: false,
    showPercent: muxing && percent != null,
    percent,
    detail: subtitle.detail == null ? '' : String(subtitle.detail),
    error: status === 'failed' ? String(subtitle.error || '') : '',
    blocked: false,
    retryable: status === 'failed',
    restartable: ['failed', 'paused', 'cancelled', 'done'].includes(status),
  };
  return subtitleApplyHalt(base, subtitle, halt);
}

function subtitlePlayPrompt(item) {
  if (!item || !item.subtitle) return null;
  const subtitle = item.subtitle;
  if (subtitleHasView(subtitle)) {
    if (Number(subtitle.view.covered_s) > 0) return null;
    if (!SUBTITLE_PLAY_WAIT_VIEW_STATES.has(String(subtitle.view.state || 'queued'))) return null;
  } else if (!(subtitle.status && !SUBTITLE_ITEM_STATUS_LABEL[subtitle.status])) {
    return null;
  }
  const display = subtitleDisplay(subtitle);
  return [display.label, display.showPercent && display.percent != null ? `${Math.round(display.percent)}%` : '', display.detail].filter(Boolean).join(' ');
}

function subtitleToggleState(item, environment) {
  const canEdit = !!item && SUBTITLE_EDIT_STATES.has(item.state);
  const available = subtitleRuntimeAvailable(environment);
  const reason = subtitleRuntimeReason(environment);
  return {
    visible: canEdit,
    checked: !!(item && item.settings && item.settings.subtitles === true),
    disabled: !canEdit || !available,
    reason: !available ? (reason || '字幕 runtime 不可用') : (!canEdit ? '此項目已開始處理' : ''),
  };
}
function subtitleRuntimeAvailable(environment) {
  return !!environment && (environment.runtime_available != null ? environment.runtime_available === true : environment.available === true);
}
function subtitleRuntimeReason(environment) {
  return String(environment && (environment.runtime_reason || environment.reason) || '字幕 runtime 不可用');
}

/* ---- subtitle track status helpers ---- */
const COMBO_STATUS = {
  pending: { label: '等待中', icon: '○' }, running: { label: '進行中', icon: '▶' },
  done: { label: '完成', icon: '✓' }, failed: { label: '失敗', icon: '✕' },
};
const COMBO_STAGE = { asr: '辨識', correction: '校正', translation: '翻譯' };
function subtitleComboSummary(subtitle) {
  const cb = subtitle && subtitle.combos;
  if (!cb || !Array.isArray(cb.items) || !cb.items.length) return null;
  const items = cb.items.map((e) => {
    const status = COMBO_STATUS[e.status] ? e.status : 'pending';
    const stage = e.failed_stage ? (COMBO_STAGE[e.failed_stage] || String(e.failed_stage)) : '';
    return {
      id: String(e.id), label: String(e.label || e.id), status, icon: COMBO_STATUS[status].icon, statusLabel: COMBO_STATUS[status].label,
      cues: Number.isFinite(Number(e.cue_count)) && e.cue_count != null && status === 'done' ? Math.max(0, Math.floor(Number(e.cue_count))) : null,
      failure: status === 'failed' ? `${stage ? `失敗於${stage}` : '失敗'}${e.error ? `：${e.error}` : ''}` : '',
    };
  });
  const total = Number.isFinite(Number(cb.total)) ? Number(cb.total) : items.length;
  const done = Number.isFinite(Number(cb.done)) ? Number(cb.done) : items.filter((e) => e.status === 'done').length;
  const failed = Number.isFinite(Number(cb.failed)) ? Number(cb.failed) : items.filter((e) => e.status === 'failed').length;
  const running = !!subtitle && subtitle.status === 'combos';
  return {
    items, total, done, failed, running,
    text: `字幕 ${done}/${total} 組`,
    failText: failed > 0 ? `${failed} 組失敗` : '',
    tone: failed > 0 ? 'warn' : (running ? 'run' : (done >= total && total > 0 ? 'ok' : '')),
  };
}

function modelLabel(key, value) {
  const options = current && current.model_options && current.model_options[key];
  const option = Array.isArray(options) && options.find((o) => o.value === value);
  return option ? String(option.label || option.value) : String(value || '');
}

/* ------------------------------------------------------------------ bridge */

let backend = null;   // object with the api methods
let isMock = false;

function api(name, ...args) {
  return Promise.resolve()
    .then(() => backend[name](...args))
    .then((res) => {
      if (name === 'stream' || name === 'stream_resolved') return res;
      if (res && typeof res === 'object' && !Array.isArray(res)) {
        if (res.error) toast(String(res.error), true);
        else if (res.message && !res.items) toast(String(res.message));
        else if (name === 'play' && res.pending) toast('已安排播放，來源準備好後自動開啟');
      }
      return res;
    })
    .catch((e) => {
      const error = `${name} 失敗：${e && e.message ? e.message : e}`;
      if (name === 'stream' || name === 'stream_resolved') return { ok: false, error };
      toast(error, true);
      return { ok: false, error };
    });
}

function refresh() {
  return api('get_state').then((s) => { if (s && Array.isArray(s.items)) render(s); });
}
function act(name, ...args) { return api(name, ...args).then(() => refresh()); }
const hasApi = (name) => !!backend && typeof backend[name] === 'function';

// The model registry is the source for selectable subtitle engines and status labels.
let subtitleModels = [];
let subtitleModelsLoaded = false;
let subtitleModelLoadBusy = false;
const mdl = { autoTried: false, detailsOpen: false, data: null, selected: new Set(), plan: null, planSeq: 0, progress: null, timer: null, loading: false };
let subtitleComboFallback = null;   // {message,...} while new items run on another combo because models are missing
async function loadSubtitleComboFallback() {
  if (!hasApi('subtitle_combo_fallback')) { subtitleComboFallback = null; return; }
  try {
    const res = await backend.subtitle_combo_fallback();
    subtitleComboFallback = res && res.fallback && res.fallback.message ? res.fallback : null;
  } catch (e) { subtitleComboFallback = null; }
}
async function loadSubtitleModels(force = false) {
  if (subtitleModelLoadBusy || (subtitleModelsLoaded && !force)) return subtitleModels;
  if (!hasApi('list_subtitle_models')) return subtitleModels;
  subtitleModelLoadBusy = true;
  try {
    const res = await backend.list_subtitle_models();
    const rows = Array.isArray(res) ? res : (res && Array.isArray(res.models) ? res.models : []);
    subtitleModels = rows.filter((m) => m && m.id != null && ['asr', 'translation'].includes(m.kind)).map((m) => ({
      id: String(m.id), label: String(m.label || m.id), kind: m.kind,
      provider: String(m.provider || 'local'), recipe: m.recipe == null ? '' : String(m.recipe),
      builtin: !!m.builtin, removable: m.removable === true,
      status: String(m.status || ''), status_label: String(m.status_label || ''),
      is_default: m.is_default === true || m.default === true,
      model_path: m.model_path == null ? '' : String(m.model_path),
      base_url: m.base_url == null ? '' : String(m.base_url), remote_model: m.remote_model == null ? '' : String(m.remote_model),
      api_family: m.api_family == null ? '' : String(m.api_family),
      connection_id: m.connection_id == null ? '' : String(m.connection_id),
      auth_source: m.auth_source == null ? '' : String(m.auth_source),
      model_file: m.model_file == null ? '' : String(m.model_file),
    }));
    subtitleModelsLoaded = true;
  } catch (e) {
    subtitleModels = [];
  } finally {
    subtitleModelLoadBusy = false;
  }
  await loadSubtitleComboFallback();
  if (current) renderSubtitleSettings((current && current.settings) || {});
  return subtitleModels;
}

window.ohStreamer = {
  onState(s) { if (s && Array.isArray(s.items)) render(s); },
  notify(msg, isErr) { toast(String(msg), !!isErr); },
};

/* --- subtitle usage bridge shim (stage B) --- */
// window.ohStreamer.subtitleUsage() -> Promise<snapshot>; never rejects (errors resolve {version:1, providers:[], error}).
window.ohStreamer.subtitleUsage = function subtitleUsage() {
  const fail = (e) => ({ version: 1, providers: [], error: String(e && e.message ? e.message : e) });
  return Promise.resolve().then(() => {
    if (!hasApi('subtitle_usage')) return fail('subtitle_usage unavailable');
    return backend.subtitle_usage();
  }).then((res) => (res && typeof res === 'object' && !Array.isArray(res) ? res : fail('invalid usage snapshot')), fail);
};
/* --- end subtitle usage bridge shim --- */

/* ------------------------------------------------------------------ toast + confirm */

function toast(msg, isErr) {
  const box = $('#toasts');
  const t = el('div', 'toast' + (isErr ? ' err-t' : ''));
  t.appendChild(svgEl(isErr ? 'alert' : 'info'));
  t.appendChild(el('span', null, msg));
  box.appendChild(t);
  while (box.children.length > 4) box.firstChild.remove();
  setTimeout(() => { t.classList.add('out'); setTimeout(() => t.remove(), 350); }, isErr ? 5000 : 2800);
}

let modalResolve = null;
function confirmDlg({ title, body, ok = '確定', danger = false, choices = [] }) {
  return new Promise((resolve) => {
    if (modalResolve) modalResolve(false);
    modalResolve = resolve;
    setText($('#modalTitle'), title);
    setText($('#modalBody'), body);
    const group = $('#modalChoices');
    group.replaceChildren();
    group.hidden = choices.length === 0;
    choices.forEach((choice, i) => {
      const label = el('label', 'modal-choice');
      const input = document.createElement('input');
      input.type = 'radio'; input.name = 'modalChoice'; input.value = choice.value; input.checked = i === 0;
      label.append(input, el('span', null, choice.label));
      group.appendChild(label);
    });
    const okBtn = $('#modalOk');
    setText(okBtn, ok);
    okBtn.classList.toggle('danger-fill', danger);
    $('#modal').hidden = false;
    (danger ? $('#modalCancel') : group.querySelector('input') || okBtn).focus();
  });
}
function closeModal(result) {
  if (result === true && !$('#modalChoices').hidden) {
    result = $('#modalChoices').querySelector('input:checked')?.value || false;
  }
  $('#modal').hidden = true;
  const r = modalResolve; modalResolve = null;
  if (r) r(result);
}

/* ------------------------------------------------------------------ state + rendering */

let current = null;             // last rendered state
const cards = new Map();        // id -> card object
let dragId = null;              // id of card being dragged (internal reorder)
const openingIds = new Set();
const playErrors = new Map();
const busyIds = new Set();      // ids with a slow action in flight (remove compare track)
const list = $('#list');

/* ------------------------------------------------------------------ film strip (queue card thumbnails) */

/* A strip of real video frames that doubles as the progress display.  One global mosaic grid over all tiles (C columns x R rows per tile,
   timeline column-major); a block counts as processed / downloaded when >= 50 % of its time span lies inside the range.
   not downloaded = empty panel + faint grid, downloaded = dimmed mosaic, processed = blurred frame.  A soft lens follows the pointer and
   reveals the available sharp image inside processed blocks (source until a restored thumbnail arrives). */
const SV_BLUR = 10, SV_COLS = 12, SV_RADIUS = 70, SV_PX = 170, SV_FX_DUR = 300, SV_FX_MAX = 8;
const SV_EMPTY = '#131725';
const SV_REDUCED = window.matchMedia ? window.matchMedia('(prefers-reduced-motion: reduce)') : { matches: false };
const SV_HAS_FILTER = typeof CanvasRenderingContext2D !== 'undefined' && 'filter' in CanvasRenderingContext2D.prototype;
const strips = new Set();

const svClamp = (x, a, b) => Math.min(b, Math.max(a, x));
const svHMS = (s) => { s = Math.max(0, Math.floor(s)); return [s / 3600 | 0, s % 3600 / 60 | 0, s % 60].map((x) => String(x).padStart(2, '0')).join(':'); };
const svRnd = (a) => () => { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; };
const svStepped = (k, n) => Math.ceil(svClamp(k, 0, 1) * n) / n;   // pixel-feel fades: n discrete levels
const svEase = (p) => p * p * (3 - 2 * p);

// one frame: spec {at, url, restored, w, h}; the image is reloaded only when url / restored change (url carries a ?v= cache-buster)
function svFrame(spec, onLoad) {
  const f = { at: 0, w: 0, h: 0, key: '', restored: false, img: null, ready: false, sharp: false, mos: new Map(), blr: new Map() };
  svSet(f, spec, onLoad);
  return f;
}
function svSet(f, spec, onLoad) {
  f.at = num(spec.at);
  if (num(spec.w) > 0 && num(spec.h) > 0 && !f.ready) { f.w = num(spec.w); f.h = num(spec.h); }
  f.restored = !!spec.restored;
  f.cover = !!spec.cover;
  const key = String(spec.url || '') + '|' + (f.restored ? 1 : 0) + '|' + (f.cover ? 1 : 0);
  if (f.key === key) return;
  f.ready = false; f.sharp = false; f.img = null; f.mos.clear(); f.blr.clear();
  f.key = key;
  if (!spec.url) return;
  const im = new Image();
  im.onload = () => {
    if (f.key !== key) return;                           // superseded by a newer url
    f.img = im; f.ready = true; f.sharp = true;
    f.w = im.naturalWidth; f.h = im.naturalHeight; f.mos.clear(); f.blr.clear();
    if (onLoad) onLoad();
  };
  // the real window is served over http, which may not load file:// images: ask the backend for the bytes instead
  if (/^file:/i.test(spec.url) && backend && backend.thumb) {
    Promise.resolve().then(() => backend.thumb(spec.url))
      .then((data) => { if (f.key === key && data) im.src = data; })
      .catch(() => {});
  } else im.src = spec.url;
}
function svMosaic(f, N) {   // mosaic: N blocks across, cached per frame and N
  N = Math.max(1, Math.min(f.w, Math.round(N)));
  let c = f.mos.get(N); if (c) return c;
  const bh = Math.max(1, Math.round(N * f.h / f.w));
  c = document.createElement('canvas'); c.width = N; c.height = bh;
  const x = c.getContext('2d'); x.imageSmoothingEnabled = true; x.imageSmoothingQuality = 'high';
  x.drawImage(f.img, 0, 0, N, bh);
  f.mos.set(N, c); return c;
}
function svBlur(f, b, twq) {   // blurred copy at source resolution, edges clamped (no transparent halo), cached per frame / tile width
  if (b <= 0) return f.img;
  const key = b + '|' + twq; let c = f.blr.get(key); if (c) return c;
  if (f.blr.size > 8) f.blr.clear();
  const w = f.w, h = f.h, r = b * w / twq, p = Math.ceil(r * 3) + 1;
  const pad = document.createElement('canvas'); pad.width = w + 2 * p; pad.height = h + 2 * p;
  const q = pad.getContext('2d'), im = f.img;
  q.drawImage(im, p, p);
  q.drawImage(im, 0, 0, w, 1, p, 0, w, p); q.drawImage(im, 0, h - 1, w, 1, p, p + h, w, p);
  q.drawImage(im, 0, 0, 1, h, 0, p, p, h); q.drawImage(im, w - 1, 0, 1, h, p + w, p, p, h);
  q.drawImage(im, 0, 0, 1, 1, 0, 0, p, p); q.drawImage(im, w - 1, 0, 1, 1, p + w, 0, p, p);
  q.drawImage(im, 0, h - 1, 1, 1, 0, p + h, p, p); q.drawImage(im, w - 1, h - 1, 1, 1, p + w, p + h, p, p);
  c = document.createElement('canvas'); c.width = w; c.height = h;
  const x = c.getContext('2d');
  if (SV_HAS_FILTER) { x.filter = 'blur(' + r.toFixed(2) + 'px)'; x.drawImage(pad, -p, -p); }
  else {   // fallback: down/up-scale
    const k = Math.max(1, Math.round(r)), sm = document.createElement('canvas');
    sm.width = Math.max(1, Math.ceil(pad.width / k)); sm.height = Math.max(1, Math.ceil(pad.height / k));
    sm.getContext('2d').drawImage(pad, 0, 0, sm.width, sm.height);
    x.imageSmoothingEnabled = true; x.drawImage(sm, 0, 0, sm.width, sm.height, -p, -p, pad.width, pad.height);
  }
  f.blr.set(key, c); return c;
}

/* host: element whose width the strip follows.  opts.onPick(seconds).  update(m): m = {frames: [spec], duration, ranges: [[a,b]] (clamped to [0, dl]),
   worker: fraction | null, dl: fraction}; cheap when nothing changed. */
function createStrip(host, opts) {
  const root = el('div', 'sv');
  root.setAttribute('role', 'img'); root.setAttribute('aria-label', '影片縮圖與區塊進度');
  const base = el('canvas', 'sv-base'), top = el('canvas', 'sv-top'), fx = el('canvas', 'sv-fx');
  top.hidden = fx.hidden = true;     // allocated on first use (lens: first hover; effects: first completion)
  const wsw = el('div', 'sv-sw w'), wsb = el('div', 'sv-sb'), wbb = el('div', 'sv-bb w'); wsw.appendChild(wsb);   // worker: white scan light
  const dsw = el('div', 'sv-sw d'), dsb = el('div', 'sv-sb'), dbb = el('div', 'sv-bb d'); dsw.appendChild(dsb);   // download front: blue scan light
  const tip = el('div', 'sv-tip');
  root.append(base, top, fx, wsw, wbb, dsw, dbb, tip); host.appendChild(root);
  const bc = base.getContext('2d'), tc = top.getContext('2d'), xc = fx.getContext('2d');

  let M = { frames: [], duration: 0, ranges: [], worker: null, dl: 0 }, msig = '';
  let frames = [], tiles = [], W = 0, n = 1, tw = 0, th = 0, dpr = 1, dead = false;
  let C = SV_COLS, R = 7, U = 1, proc = new Uint8Array(1), dl = new Uint8Array(1), paths = [], dpaths = [];
  let lastWu = -2, lastDu = -2, topOn = false, topDirty = true, fxOn = false;
  let fxs = [], fxRaf = 0, wInd = null, dInd = null, scanRaf = 0, raf = 0;
  const ptr = { x: 0, y: 0, in: false };
  const twq = () => Math.max(20, Math.round(tw / 20) * 20);
  const blurV = (f) => svBlur(f, SV_BLUR, twq());
  const self = { root, destroy, update, redraw: () => redraw(true), tick, info: () => ({ W, n, tw, th, U, C, R, fx: fxs.length, topOn, fxOn, proc: proc.reduce((a, b) => a + b, 0), dl: dl.reduce((a, b) => a + b, 0), flags: () => [proc, dl], rect: unitRect, tl: tiles.map((f) => [f.at, f.restored, f.sharp, f.ready]) }) };

  // ---- block grid ----
  const cover = (rs, a, b) => { let s = 0; for (const r of rs) s += Math.max(0, Math.min(r[1], b) - Math.max(r[0], a)); return s / (b - a); };
  const bx = (k, c) => Math.round(k * tw + c * tw / C);   // left edge of column c of tile k (c = C: right edge)
  const by = (r) => Math.round(r * th / R);
  const unitRect = (u) => {
    const gc = Math.floor(u / R), r = u % R, k = Math.floor(gc / C), c = gc % C, x = bx(k, c), y = by(r);
    return { gc, k, c, r, x, y, w: bx(k, c + 1) - x, h: by(r + 1) - y };
  };
  const fracUnit = (v) => (v == null || v >= 1 ? -1 : svClamp(Math.floor(v * U), 0, U - 1));
  const workerUnit = () => fracUnit(M.worker);
  const dlUnit = () => fracUnit(M.dl);

  function sameBytes(a, b) { if (a.length !== b.length) return false; for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false; return true; }
  // recompute processed / downloaded flags; returns the previous flags when they changed (or `force`), else null
  function computeGrid(force) {
    C = SV_COLS; R = Math.max(1, Math.round(C * th / tw)); U = n * C * R;
    const pp = proc, pd = dl, np = new Uint8Array(U), nd = new Uint8Array(U), drs = [[0, svClamp(M.dl, 0, 1)]];
    for (let u = 0; u < U; u++) {
      np[u] = cover(M.ranges, u / U, (u + 1) / U) >= 0.5 - 1e-9 ? 1 : 0;
      nd[u] = cover(drs, u / U, (u + 1) / U) >= 0.5 - 1e-9 ? 1 : 0;
    }
    if (!force && sameBytes(np, pp) && sameBytes(nd, pd)) return null;
    proc = np; dl = nd;
    // per tile: union of block rects (vertical runs merged) = clip path; processed -> blur + lens, downloaded -> mosaic
    const pathsOf = (arr) => tiles.map((f, k) => {
      const p = new Path2D();
      for (let c = 0; c < C; c++) {
        const col = (k * C + c) * R, x0 = bx(k, c), w = bx(k, c + 1) - x0;
        for (let r = 0; r < R;) {
          if (!arr[col + r]) { r++; continue; }
          let r2 = r; while (r2 < R && arr[col + r2]) r2++;
          p.rect(x0, by(r), w, by(r2) - by(r)); r = r2;
        }
      }
      return p;
    });
    paths = pathsOf(proc); dpaths = pathsOf(dl);
    return { pp, pd };
  }
  function drawGrid(ctx, x0, y0, x1, y1, bottom, right) {   // faint 1px grid on a not-downloaded block
    ctx.fillStyle = 'rgba(255,255,255,.05)';
    ctx.fillRect(x0, y0, x1 - x0, 1); ctx.fillRect(x0, y0, 1, y1 - y0);
    if (bottom) ctx.fillRect(x0, y1 - 1, x1 - x0, 1);
    if (right) ctx.fillRect(x1 - 1, y0, 1, y1 - y0);
  }
  function drawBase() {
    bc.setTransform(dpr, 0, 0, dpr, 0, 0); bc.clearRect(0, 0, W, th);
    tiles.forEach((f, k) => {
      const x = bx(k, 0), w = bx(k + 1, 0) - x;
      bc.fillStyle = SV_EMPTY; bc.fillRect(x, 0, w, th);   // not downloaded: panel colour (grid below)
      if (!f.ready) return;
      bc.save(); bc.clip(dpaths[k]);   // downloaded, unprocessed: the mosaic on the same grid, dimmed to ~65 %
      bc.imageSmoothingEnabled = false; bc.drawImage(svMosaic(f, C), x, 0, w, th);
      bc.fillStyle = 'rgba(0,0,0,.35)'; bc.fillRect(x, 0, w, th);
      bc.restore();
      bc.save(); bc.clip(paths[k]); bc.imageSmoothingEnabled = true; bc.drawImage(blurV(f), x, 0, w, th); bc.restore();
    });
    for (let k = 0; k < n; k++) for (let c = 0; c < C; c++) for (let r = 0; r < R; r++) {
      if (dl[(k * C + c) * R + r]) continue;
      drawGrid(bc, bx(k, c), by(r), bx(k, c + 1), by(r + 1), r === R - 1, k === n - 1 && c === C - 1);
    }
  }
  function sizeCanvas(cv, on) { if (!on) return; cv.width = Math.round(W * dpr); cv.height = Math.round(th * dpr); cv.style.width = W + 'px'; cv.style.height = th + 'px'; }
  // Reveal only processed blocks, using the available image: source, restored frame, or cover.
  function drawTop() {
    if (!topOn) { topOn = true; top.hidden = false; sizeCanvas(top, true); }
    topDirty = false;
    tc.setTransform(dpr, 0, 0, dpr, 0, 0); tc.clearRect(0, 0, W, th); tc.imageSmoothingEnabled = true;
    tiles.forEach((f, k) => {
      if (!f.ready || !f.sharp) return;
      const x = bx(k, 0);
      tc.save(); tc.clip(paths[k]); tc.drawImage(f.img, x, 0, bx(k + 1, 0) - x, th); tc.restore();
    });
  }

  // ---- scan lights: white on the worker block, blue on the download-front block (+0.4 s phase) ----
  function placeScan() {
    if (scanRaf) { cancelAnimationFrame(scanRaf); scanRaf = 0; }
    for (const e of [wsw, wbb, dsw, dbb]) e.style.display = 'none';
    wInd = dInd = null; if (!tiles.length) return;
    const wu = workerUnit(), du = dlUnit(), anim = !SV_REDUCED.matches;
    lastWu = wu; lastDu = du;
    const put = (bb, sw, sb, u) => {
      const q = unitRect(u); q.bh = Math.max(3, Math.round(q.h * 0.6));
      Object.assign(bb.style, { display: 'block', width: q.w + 'px', height: q.h + 'px', transform: 'translate(' + q.x + 'px,' + q.y + 'px)' });
      if (anim) { Object.assign(sw.style, { display: 'block', left: q.x + 'px', top: q.y + 'px', width: q.w + 'px', height: q.h + 'px' }); sb.style.height = q.bh + 'px'; }
      return q;
    };
    if (wu >= 0 && wu !== du) wInd = put(wbb, wsw, wsb, wu);   // caught up with the download: the worker waits, only blue shows
    if (du >= 0) dInd = put(dbb, dsw, dsb, du);
    paintScan(performance.now(), anim);
    if (anim && (wInd || dInd) && !document.hidden) scanRaf = requestAnimationFrame(loopScan);
  }
  function paintScan(t, anim) {
    const one = (q, bb, sb, ph) => {
      if (!q) return; if (!anim) { bb.style.opacity = '.12'; return; }
      const p = ((t + ph) % 1600) / 1600, e = svEase(p);
      sb.style.transform = 'translateY(' + (-q.bh + (q.h + q.bh) * e).toFixed(1) + 'px)';
      bb.style.opacity = (0.12 * (0.35 + 0.65 * Math.sin(Math.PI * p))).toFixed(3);
    };
    one(wInd, wbb, wsb, 0); one(dInd, dbb, dsb, 400);
  }
  function loopScan(t) { scanRaf = 0; if (dead || document.hidden || SV_REDUCED.matches || !(wInd || dInd)) return; paintScan(t, true); scanRaf = requestAnimationFrame(loopScan); }

  // ---- block-completion effect: 3x3 random dissolve for newly downloaded (blue) and newly processed (white) blocks.
  //      Only small, natural steps (<= 8 blocks at once) animate: a jump, a re-layout or the first render does not. ----
  function diffFx(pp, pd) {
    if (SV_REDUCED.matches || !tiles.length || document.hidden) { fxs = []; return; }
    const np = [], nd = [];
    for (let u = 0; u < U; u++) { if (proc[u] && !pp[u]) np.push(u); else if (dl[u] && !pd[u]) nd.push(u); }
    const t0 = performance.now();
    if (np.length <= SV_FX_MAX) for (const u of np) addFx(u, t0, pd[u] ? 'mosaic' : 'empty', 'blur', '255,255,255');
    if (nd.length <= SV_FX_MAX) for (const u of nd) addFx(u, t0, 'empty', 'mosaic', '74,134,255');
    fxs = fxs.filter((e) => (e.post === 'blur' ? proc : dl)[e.u]);   // undone (range moved back): drop
    if (fxs.length > 48) fxs = fxs.slice(-48);
    if (fxs.length && !fxRaf) fxRaf = requestAnimationFrame(loopFx);
  }
  function addFx(u, t0, pre, post, col) {
    const e = Object.assign({ u, t0, pre, post, col }, unitRect(u)); e.f = tiles[e.k];
    if (!e.f || !e.f.ready) return;
    if (!fxOn) { fxOn = true; fx.hidden = false; sizeCanvas(fx, true); }
    const rn = svRnd(u * 2654435761 + 7); e.ord = [0, 1, 2, 3, 4, 5, 6, 7, 8];
    for (let i = 8; i > 0; i--) { const j = Math.floor(rn() * (i + 1)); [e.ord[i], e.ord[j]] = [e.ord[j], e.ord[i]]; }
    fxs = fxs.filter((q) => q.u !== u); fxs.push(e);
  }
  function crop(ctx, src, e, dx, dy, dw, dh, smooth) {   // the block's part of a tile-sized image, into a destination rect
    const sw = src.width, sh = src.height, x0 = bx(e.k, 0), tws = bx(e.k + 1, 0) - x0;
    ctx.imageSmoothingEnabled = smooth;
    ctx.drawImage(src, (e.x - x0) / tws * sw, e.y / th * sh, e.w / tws * sw, e.h / th * sh, dx, dy, dw, dh);
  }
  function paintState(e, st) {
    if (st === 'empty') { xc.fillStyle = SV_EMPTY; xc.fillRect(e.x, e.y, e.w, e.h); drawGrid(xc, e.x, e.y, e.x + e.w, e.y + e.h, e.r === R - 1, e.k === n - 1 && e.c === C - 1); return; }
    if (st === 'mosaic') { crop(xc, svMosaic(e.f, C), e, e.x, e.y, e.w, e.h, false); xc.fillStyle = 'rgba(0,0,0,.35)'; xc.fillRect(e.x, e.y, e.w, e.h); }
    else crop(xc, blurV(e.f), e, e.x, e.y, e.w, e.h, true);
  }
  const rgba = (e, a) => 'rgba(' + e.col + ',' + a.toFixed(3) + ')';
  function paintDissolve(e, a) {
    for (let i = 0; i < 9; i++) {
      const cx = i % 3, cy = (i / 3) | 0, x0 = e.x + Math.round(cx * e.w / 3), x1 = e.x + Math.round((cx + 1) * e.w / 3);
      const y0 = e.y + Math.round(cy * e.h / 3), y1 = e.y + Math.round((cy + 1) * e.h / 3), since = a - e.ord.indexOf(i) * 22;
      if (since < 0) { xc.save(); xc.beginPath(); xc.rect(x0, y0, x1 - x0, y1 - y0); xc.clip(); paintState(e, e.pre); xc.restore(); continue; }
      const k = 1 - since / 110; if (k <= 0) continue;
      xc.fillStyle = rgba(e, 0.55 * svStepped(k, 3)); xc.fillRect(x0, y0, x1 - x0, y1 - y0);
    }
  }
  function paintFx(t) {
    xc.setTransform(dpr, 0, 0, dpr, 0, 0); xc.clearRect(0, 0, W, th);
    for (const e of fxs) { const a = t - e.t0; if (a >= SV_FX_DUR) { e.dead = 1; continue; } if (a >= 0) paintDissolve(e, a); }
    fxs = fxs.filter((e) => !e.dead);
  }
  function loopFx() { fxRaf = 0; if (dead) return; paintFx(performance.now()); if (fxs.length && !document.hidden) fxRaf = requestAnimationFrame(loopFx); }
  // called by the shared visibility listener
  function tick(hidden) {
    if (hidden) { fxs = []; if (fxOn) { xc.setTransform(1, 0, 0, 1, 0, 0); xc.clearRect(0, 0, fx.width, fx.height); } }
    else if ((wInd || dInd) && !scanRaf && !SV_REDUCED.matches) scanRaf = requestAnimationFrame(loopScan);
  }

  // ---- layout / redraw ----
  function layout(w) {
    W = Math.max(40, Math.floor(w));
    const a = frames[0]; if (!a) return;
    n = svClamp(Math.round(W / SV_PX), 1, frames.length);
    tw = W / n; th = 96;
    tiles = Array.from({ length: n }, (_, j) => frames[j === 0 && M.frames[0].cover ? 0 : Math.min(frames.length - 1, Math.floor((j + 0.5) * frames.length / n))]);
    dpr = Math.min(2, window.devicePixelRatio || 1);
    sizeCanvas(base, true); sizeCanvas(top, topOn); sizeCanvas(fx, fxOn);
    root.style.height = th + 'px';
    root.style.setProperty('--r', SV_RADIUS + 'px'); root.style.setProperty('--core', Math.round(SV_RADIUS * 0.55) + 'px');
    redraw(true);
  }
  function redraw(force) {
    if (dead || !W || !tiles.length) return;
    const prev = computeGrid(force);
    if (prev) {
      if (!force && prev.pp.length === U) diffFx(prev.pp, prev.pd); else fxs = [];   // grid changed (resize / frame count): nothing to animate
      drawBase(); topDirty = true; if (topOn && ptr.in) drawTop();
      placeScan();
    } else if (workerUnit() !== lastWu || dlUnit() !== lastDu) placeScan();
    else return;
    if (ptr.in) schedule();
  }

  // ---- model / frames ----
  function update(m) {
    if (dead) return;
    const sig = m.duration + '|' + m.dl.toFixed(4) + '|' + (m.worker == null ? '-' : m.worker.toFixed(4)) + '|' + m.ranges.map((r) => r[0].toFixed(4) + ',' + r[1].toFixed(4)).join(';');
    let framesChanged = m.frames.length !== frames.length;
    const onLoad = () => { topDirty = true; if (W) redraw(true); };
    m.frames.forEach((spec, i) => {
      if (!frames[i]) { frames[i] = svFrame(spec, onLoad); return; }
      const before = frames[i].key; svSet(frames[i], spec, onLoad);
      if (frames[i].key !== before && !frames[i].ready) framesChanged = true;
    });
    if (frames.length > m.frames.length) frames.length = m.frames.length;
    if (!framesChanged && sig === msig) return;
    msig = sig; M = m;
    if (framesChanged && W) layout(W); else redraw(false);
  }

  // ---- pointer: lens + time tip + click to play ----
  const tileAt = (x) => svClamp(Math.floor(x / tw), 0, n - 1);
  function unitAt() {
    const k = tileAt(ptr.x), x0 = bx(k, 0), c = svClamp(Math.floor((ptr.x - x0) / ((bx(k + 1, 0) - x0) / C)), 0, C - 1);
    return (k * C + c) * R + svClamp(Math.floor(ptr.y / (th / R)), 0, R - 1);
  }
  function updateTip() {
    if (ptr.x < 0 || ptr.x > W || ptr.y < 0 || ptr.y > th || !U) { tip.style.display = 'none'; return; }
    const u = unitAt(), d = M.duration || 0;
    const txt = svHMS(u / U * d) + '–' + svHMS((u + 1) / U * d) + ' · ' + (proc[u] ? '已處理' : dl[u] ? '已下載' : '未下載');
    if (txt !== tip.textContent) tip.textContent = txt;
    tip.style.display = 'block';
    tip.style.transform = 'translate(' + svClamp(ptr.x + 12, 0, Math.max(0, W - tip.offsetWidth)) + 'px,' + (ptr.y > 30 ? ptr.y - 28 : ptr.y + 14) + 'px)';
  }
  function schedule() { if (!raf) raf = requestAnimationFrame(pframe); }
  function pframe() { raf = 0; if (!ptr.in || dead) return; root.style.setProperty('--x', ptr.x + 'px'); root.style.setProperty('--y', ptr.y + 'px'); updateTip(); }
  function setPtr(e) { const r = root.getBoundingClientRect(); ptr.x = e.clientX - r.left; ptr.y = e.clientY - r.top; }
  root.addEventListener('pointerenter', (e) => { setPtr(e); ptr.in = true; if (tiles.length && (!topOn || topDirty)) drawTop(); root.classList.add('on'); schedule(); });
  root.addEventListener('pointermove', (e) => { setPtr(e); ptr.in = true; schedule(); });
  const leave = () => { ptr.in = false; root.classList.remove('on'); tip.style.display = 'none'; };
  root.addEventListener('pointerleave', leave); root.addEventListener('pointercancel', leave);
  root.addEventListener('click', (e) => { if (!U || !tiles.length) return; setPtr(e); if (opts && opts.onPick) opts.onPick(unitAt() / U * (M.duration || 0)); });

  const ro = typeof ResizeObserver === 'function'
    ? new ResizeObserver((es) => { const w = es[0].contentRect.width; if (w >= 1 && Math.abs(w - W) >= 0.5 && frames.length) layout(w); })
    : null;
  if (ro) ro.observe(host);

  function destroy() {
    dead = true; strips.delete(self);
    if (ro) ro.disconnect();
    for (const id of [raf, scanRaf, fxRaf]) if (id) cancelAnimationFrame(id);
    raf = scanRaf = fxRaf = 0; fxs = []; frames = []; tiles = []; paths = []; dpaths = [];
    for (const cv of [base, top, fx]) { cv.width = 0; cv.height = 0; }   // release the canvas backing stores
    root.remove();
  }
  strips.add(self);
  root.__strip = self;   // debugging / tests
  return self;
}
document.addEventListener('visibilitychange', () => { for (const s of strips) s.tick(document.hidden); });
if (SV_REDUCED.addEventListener) SV_REDUCED.addEventListener('change', () => { for (const s of strips) s.redraw(); });

// card item -> strip model (null: no strip, the plain progress bar is shown)
function thumbsModel(item) {
  const th = item.thumbs;
  const coverEnabled = !(current && current.settings && current.settings.show_thumbnail_cover === false);
  const coverImage = coverEnabled && /^data:image\//i.test(item.cover || '') ? item.cover : null;
  if ((!th || !Array.isArray(th.frames) || !th.frames.length) && !coverImage) return null;
  const duration = th && num(th.duration) > 0 ? num(th.duration) : num(item.duration);
  // Keep empty timeline slots while the thumbnail worker fills them.
  const frameSpecs = Array.from({ length: 12 }, (_, i) => ({ at: (i + 0.5) / 12 * duration, url: '', restored: false }));
  for (const f of (th && th.frames || [])) {
    const slot = duration > 0 ? svClamp(Math.floor(num(f.at) / duration * 12), 0, 11) : 0;
    frameSpecs[slot] = f;
  }
  if (coverImage) frameSpecs[0] = { at: 0, url: coverImage, restored: false, cover: true };
  const st = item.state;
  const dl = svClamp(item.download_fraction == null ? (st === 'processing' || st === 'done' || st === 'queued' ? 1 : 0) : num(item.download_fraction), 0, 1);
  const src = Array.isArray(item.processed_ranges) ? item.processed_ranges : (st === 'done' ? [[0, 1]] : []);
  const ranges = [];
  for (const r of src) {
    if (!Array.isArray(r)) continue;
    const a = Math.min(svClamp(num(r[0]), 0, 1), dl), b = Math.min(svClamp(num(r[1]), 0, 1), dl);
    if (b - a > 1e-9) ranges.push([a, b]);
  }
  const worker = item.worker_fraction == null ? null : Math.min(svClamp(num(item.worker_fraction), 0, 1), dl);
  return { frames: frameSpecs, duration, ranges, worker, dl };
}
const thumbsEnabled = () => !(current && current.settings && current.settings.show_thumbnails === false);

let subtitlePlayChoice = null;
function updateSubtitlePlayDialog() {
  if (!subtitlePlayChoice) return;
  const item = current && current.items.find((x) => x.id === subtitlePlayChoice.id);
  setText($('#subtitlePlayWait'), item && item.subtitle && item.subtitle.view && Number(item.subtitle.view.slices_total) > 1
    ? '等前段字幕完成再播' : '等字幕完成再播');
  setText($('#subtitlePlayProgress'), subtitlePlayPrompt(item) || (item ? '字幕狀態已更新，可直接播放。' : '此項目已移除。'));
}
function chooseSubtitlePlayback(id) {
  if (subtitlePlayChoice) return Promise.resolve(null);
  return new Promise((resolve) => {
    subtitlePlayChoice = { id, resolve, focusReturn: document.activeElement };
    updateSubtitlePlayDialog();
    $('#subtitlePlayDialog').showModal();
    $('#subtitlePlayWait').focus();
  });
}
function closeSubtitlePlayback(choice) {
  const pending = subtitlePlayChoice;
  subtitlePlayChoice = null;
  $('#subtitlePlayDialog').close();
  if (pending) {
    if (pending.focusReturn && pending.focusReturn.isConnected) pending.focusReturn.focus();
    pending.resolve(choice);
  }
}
$('#subtitlePlayWait').addEventListener('click', () => closeSubtitlePlayback('wait'));
$('#subtitlePlaySkip').addEventListener('click', () => closeSubtitlePlayback('skip'));
$('#subtitlePlayDialog').addEventListener('cancel', (event) => { event.preventDefault(); closeSubtitlePlayback(null); });
$('#subtitlePlayDialog').addEventListener('click', (event) => { if (event.target === $('#subtitlePlayDialog')) closeSubtitlePlayback(null); });

async function openPlayer(id, start) {
  if (openingIds.has(id)) return;
  let waitForSubtitles = false;
  const item = current && current.items.find((x) => x.id === id);
  if (subtitlePlayPrompt(item)) {
    const choice = await chooseSubtitlePlayback(id);
    if (!choice) return;
    waitForSubtitles = choice === 'wait';
  }
  openingIds.add(id);
  playErrors.delete(id);
  if (current) render(current);
  try {
    const result = waitForSubtitles ? await api('play', id, start == null ? null : start, true) : await api('play', id, start);
    if (!result || result.error || result.ok === false) throw new Error('播放器開啟失敗');
    if (result.pending || waitForSubtitles) return; // Source/subtitle preparation is handled by the queue.
    if (!isMock) {
      const deadline = Date.now() + 60000;   // a cold player start (torch + engine imports) took ~32 s
      let ready = false;
      while (Date.now() < deadline) {
        const status = await api('player_window_ready');
        if (status && status.ready) { ready = true; break; }
        await new Promise((resolve) => setTimeout(resolve, 250));
      }
      if (!ready) throw new Error('播放器尚未回應，請重試');
    }
  } catch (error) {
    playErrors.set(id, error.message || '播放器開啟失敗');
  } finally {
    openingIds.delete(id);
    if (current) render(current);
    await refresh();
  }
}

function playFrom(id, sec) {
  // the service clamps to the duration; an unprocessed target makes the worker jump there (player shows 處理中)
  return Number.isFinite(sec) && sec >= 0 ? openPlayer(id, Math.round(sec * 10) / 10) : openPlayer(id);
}

function updateStrip(c, item) {
  const m = thumbsEnabled() ? thumbsModel(item) : null;
  if (!m) {
    if (c.strip) { c.strip.destroy(); c.strip = null; }
    setHidden(c.stripHost, true); c.root.classList.remove('has-strip');
    return;
  }
  setHidden(c.stripHost, false); c.root.classList.add('has-strip');
  if (!c.stripArea) { c.stripArea = el('div', 'strip-area'); c.stripHost.appendChild(c.stripArea); }
  if (!c.strip) c.strip = createStrip(c.stripArea, { onPick: (sec) => playFrom(c.id, sec) });
  c.strip.update(m);
}

function buildBtn(label, icon, cls, onClick, title) {
  const b = el('button', 'btn sm ' + (cls || ''));
  b.type = 'button';
  b.appendChild(svgEl(icon));
  if (label) b.appendChild(el('span', 'lbl', label)); else b.classList.add('icon-only');
  const tip = title || label;
  if (tip) { b.title = tip; b.setAttribute('aria-label', tip); }
  b.addEventListener('click', (e) => { e.stopPropagation(); onClick(); });
  return b;
}

function createCard(id) {
  const root = el('article', 'card');
  root.dataset.id = id;

  const handle = el('div', 'handle');
  handle.tabIndex = 0;
  handle.title = '拖曳排序（鍵盤：上／下方向鍵）';
  handle.setAttribute('role', 'button');
  handle.setAttribute('aria-label', '拖曳排序');
  handle.appendChild(svgEl('grip'));

  const main = el('div', 'main');
  const row1 = el('div', 'row1');
  const title = el('h3', 'title');
  const badge = el('span', 'badge');
  row1.append(title, badge);

  const meta = el('div', 'meta');

  const stripHost = el('div', 'strip-host'); stripHost.hidden = true;   // film strip (see createStrip); replaces the bar while thumbnails exist

  const prog = el('div', 'prog');
  const bar = el('div', 'bar'); const fill = el('i'); bar.appendChild(fill);
  const ptext = el('div', 'prog-text');
  const pPct = el('span', 'pct'), pDetail = el('span'), pRate = el('span'), pEta = el('span'), pFrag = el('span');
  ptext.append(pPct, pDetail, pRate, pEta, pFrag);
  prog.append(bar, ptext);

  const err = el('div', 'err'); err.appendChild(svgEl('alert')); const errText = el('span'); err.appendChild(errText);

  const subtitleProgress = el('div', 'subtitle-info'); subtitleProgress.hidden = true;
  const subtitleHead = el('div', 'subtitle-head');
  const subtitleGlyph = el('span', 'subtitle-glyph'); subtitleGlyph.setAttribute('aria-hidden', 'true');
  const subtitleLabel = el('span', 'subtitle-label'), subtitleDetail = el('span', 'subtitle-detail'), subtitlePercent = el('span', 'subtitle-percent');
  subtitleHead.append(subtitleGlyph, subtitleLabel, subtitleDetail, subtitlePercent);
  const subtitleTrack = el('div', 'subtitle-track');
  subtitleTrack.setAttribute('role', 'progressbar');
  subtitleTrack.setAttribute('aria-label', '字幕進度');
  subtitleTrack.setAttribute('aria-valuemin', '0');
  subtitleTrack.setAttribute('aria-valuemax', '100');
  const subtitleFill = el('i'); const subtitleCursor = el('b', 'subtitle-cursor'); subtitleTrack.append(subtitleFill, subtitleCursor);
  const subtitleActions = el('div', 'subtitle-actions'); subtitleActions.hidden = true;
  subtitleProgress.append(subtitleHead, subtitleTrack, subtitleActions);
  const subtitleError = el('div', 'subtitle-error'); subtitleError.hidden = true;
  subtitleError.setAttribute('role', 'alert');
  subtitleError.appendChild(svgEl('alert'));
  const subtitleErrorText = el('span'); subtitleError.appendChild(subtitleErrorText);

  const settingsRow = el('div', 'subtitle-settings-row'); settingsRow.hidden = true;
  const combosBox = el('div', 'subtitle-combos'); combosBox.hidden = true;

  const row3 = el('div', 'row3');
  const info = el('div', 'info');
  const size = el('span', 'size');
  const keep = el('button', 'chip-btn'); keep.type = 'button';
  keep.appendChild(svgEl('lock')); keep.appendChild(el('span', null, '保留'));
  keep.title = '保留：自動清除空間時不會刪除';
  const seen = el('span', 'seen'); seen.appendChild(svgEl('eye')); seen.appendChild(el('span', null, '已看過'));
  info.append(size, keep, seen);
  const actions = el('div', 'actions');
  row3.append(info, actions);

  main.append(row1, meta, stripHost, prog, err, subtitleProgress, subtitleError, settingsRow, combosBox, row3);
  root.append(handle, main);

  const c = { id, root, handle, title, badge, meta, stripHost, strip: null, prog, bar, fill, pPct, pDetail, pRate, pEta, pFrag, err, errText, size, keep, seen, actions,
              subtitleProgress, subtitleGlyph, subtitleLabel, subtitleDetail, subtitlePercent, subtitleTrack, subtitleFill, subtitleActions, subtitleError, subtitleErrorText, settingsRow, combosBox,
              sig: { combos: '', meta: '', actions: '', subtitleActions: '', settingsRow: '', item: null } };

  keep.addEventListener('click', () => {
    const item = c.sig.item; if (!item) return;
    act('set_keep', id, !item.keep);
  });

  // drag & drop reorder (only armed from the handle)
  handle.addEventListener('pointerdown', () => { root.draggable = true; });
  const disarm = () => { if (dragId == null) root.draggable = false; };
  handle.addEventListener('pointerup', disarm);
  handle.addEventListener('pointercancel', disarm);
  root.addEventListener('dragstart', (e) => {
    if (!root.draggable) { e.preventDefault(); return; }
    dragId = id;
    e.dataTransfer.effectAllowed = 'move';
    try { e.dataTransfer.setData('application/x-oh-item', id); } catch (err2) { /* ignore */ }
    requestAnimationFrame(() => root.classList.add('dragging'));
  });
  root.addEventListener('dragend', () => {
    dragId = null;
    root.draggable = false;
    root.classList.remove('dragging');
    clearDropMarks();
    if (current) render(current);
  });
  handle.addEventListener('keydown', (e) => {
    if (e.key !== 'ArrowUp' && e.key !== 'ArrowDown') return;
    e.preventDefault();
    const ids = [...list.children].map((n) => n.dataset.id);
    const i = ids.indexOf(id);
    const j = e.key === 'ArrowUp' ? i - 1 : i + 1;
    if (j < 0 || j >= ids.length) return;
    list.insertBefore(root, e.key === 'ArrowUp' ? list.children[j] : list.children[j].nextSibling);
    handle.focus();
    act('move', id, j);
  });

  return c;
}

function progressInfo(item) {
  if (item.subtitle && ['asr', 'translate', 'combos', 'mux'].includes(item.subtitle.status)) return null;
  const p = item.state === 'downloading' && Object.keys(item.download_progress || {}).length
    ? item.download_progress : (item.progress || {});
  const hasDl = num(p.total_bytes) > 0 || num(p.downloaded_bytes) > 0;
  const hasProc = num(p.frames) > 0 || num(p.frame) > 0;
  let kind = null;
  if (item.state === 'downloading') kind = 'dl';
  else if (item.state === 'processing') kind = 'proc';
  else if (item.state === 'paused' || item.state === 'queued' || item.state === 'waiting') kind = hasProc ? 'proc' : hasDl ? 'dl' : null;
  if (!kind) return null;
  if (kind === 'dl') {
    const t = num(p.total_bytes), d = num(p.downloaded_bytes);
    if (item.state !== 'downloading' && !(d > 0)) return null;
    if (item.state === 'queued') return null;
    const fc = Math.floor(num(p.fragment_count)), fi = Math.floor(num(p.fragment_index));
    const dl = item.state === 'downloading';
    return {
      pct: fc > 0 && fi > 0 ? Math.min(100, fi / fc * 100) : (t > 0 ? Math.min(100, d / t * 100) : null),
      detail: t > 0 ? `已下載 ${fmtBytes(d)} / ${p.total_approx ? '約 ' : ''}${fmtBytes(t)}` : (d > 0 ? `已下載 ${fmtBytes(d)}` : '連線中…'),
      rate: dl && num(p.speed) > 0 ? `${fmtBytes(p.speed)}/s` : '',
      eta: dl && num(p.eta) > 0 ? `剩餘 ${fmtDur(p.eta)}` : '',
      frag: dl && fc > 0 && fi > 0 ? `片段 ${fi} / ${fc}` : '',
    };
  }
  const f = Math.floor(num(p.frame)), t = Math.floor(num(p.frames));
  if (item.state === 'queued' || item.state === 'waiting') { if (!(f > 0)) return null; }
  const dl = item.source_status === 'downloading' ? (item.download_progress || {}) : null;
  const dlSpeed = dl && num(dl.speed) > 0 ? ` · ${fmtBytes(dl.speed)}/s` : '';
  const downloadDetail = dl ? `下載 ${fmtBytes(num(dl.downloaded_bytes))}${num(dl.total_bytes) > 0 ? ' / ' + fmtBytes(num(dl.total_bytes)) : ''}${dlSpeed}` : '';

  const duration = num(p.duration) > 0 ? num(p.duration) : (num(item.duration) > 0 ? num(item.duration) : (item.thumbs && num(item.thumbs.duration) > 0 ? num(item.thumbs.duration) : null));
  const srcFps = num(p.source_fps) > 0 ? num(p.source_fps) : (num(item.source_fps) > 0 ? num(item.source_fps) : (duration && t > 0 ? t / duration : null));
  let timeDetail = '';
  if (duration && duration > 0) {
    const curSec = t > 0 ? (f / t) * duration : (srcFps ? f / srcFps : 0);
    timeDetail = fmtTimeProgress(curSec, duration);
  } else if (srcFps && srcFps > 0) {
    const curSec = f / srcFps;
    const totSec = t > 0 ? t / srcFps : null;
    timeDetail = totSec ? fmtTimeProgress(curSec, totSec) : fmtDur(curSec);
  }

  const frameStr = t > 0 ? `${f.toLocaleString()} / ${t.toLocaleString()} 格` : '';
  return {
    pct: t > 0 ? Math.min(100, f / t * 100) : null,
    detail: timeDetail || (t > 0 ? frameStr : (item.state === 'processing' ? '準備中…' : '')),
    detailTitle: frameStr,
    rate: item.state === 'processing' && num(p.fps) > 0 ? `${num(p.fps).toFixed(1)} fps` : '',
    eta: item.state === 'processing' ? (num(p.eta) > 0 ? `剩餘 ${fmtDur(p.eta)}` : (num(item.eta_estimate_s) > 0 ? `預估約 ${Math.max(1, Math.round(num(item.eta_estimate_s) / 60))} 分` : '')) : '',
    frag: downloadDetail,
  };
}

const comboOpen = new Set();     // item ids whose extra subtitle track results are expanded
function redrawCombos(id) {
  const c = cards.get(id);
  if (c) { c.sig.combos = ''; c.sig.actions = ''; }
  if (current) render(current);
}

function updateCombos(c, item) {
  const id = c.id;
  const sub = subtitleComboSummary(item.subtitle);
  const expanded = !!sub && (sub.running || comboOpen.has(id));
  const detail = item.subtitle && item.subtitle.detail ? String(item.subtitle.detail) : '';
  const sig = JSON.stringify([sub, expanded, sub && sub.running ? detail : '']);
  if (sig === c.sig.combos) return;
  c.sig.combos = sig;
  const box = c.combosBox;
  box.replaceChildren();
  setHidden(box, !sub);
  if (!sub) return;
  const chip = el('button', `combo-chip ${sub.tone}`);
  chip.type = 'button';
  chip.setAttribute('aria-expanded', expanded ? 'true' : 'false');
  chip.title = '點一下展開／收合各字幕軌狀態';
  chip.append(el('span', 'combo-chip-main', `字幕軌 ${sub.text.replace(/^字幕\s*/, '')}`));
  if (sub.failText) chip.append(el('span', 'combo-chip-fail', sub.failText));
  chip.append(el('span', 'combo-chip-caret', expanded ? '▾' : '▸'));
  chip.addEventListener('click', () => { if (comboOpen.has(id)) comboOpen.delete(id); else comboOpen.add(id); redrawCombos(id); });
  box.appendChild(chip);
  if (expanded) {
    const ul = el('ul', 'combo-list');
    for (const e of sub.items) {
      const li = el('li', `combo-row ${e.status}`);
      li.append(el('span', 'combo-ico', e.icon), el('span', 'combo-lbl', e.label), el('span', 'combo-st', e.statusLabel));
      if (e.cues != null) li.append(el('span', 'combo-cues', `${e.cues} 句`));
      if (e.status === 'running' && detail) li.append(el('span', 'combo-detail', detail));
      if (e.failure) li.appendChild(el('span', 'combo-fail', e.failure));
      ul.appendChild(li);
    }
    box.appendChild(ul);
  }
}

async function retrySubtitles(id) {
  if (busyIds.has(id)) return;
  busyIds.add(id);
  if (current) render(current);
  try { await act('retry_subtitles', id); }
  finally { busyIds.delete(id); if (current) render(current); }
}

// Circular arrow: drops every subtitle result (speech recognition included) and starts over.
async function restartSubtitles(id) {
  if (busyIds.has(id)) return;
  if (typeof window.confirm === 'function'
      && !window.confirm('重新處理字幕會刪除此影片已完成的辨識與翻譯結果，並再次消耗雲端額度。要繼續嗎？')) return;
  busyIds.add(id);
  if (current) render(current);
  try { await act('restart_subtitles', id); }
  finally { busyIds.delete(id); if (current) render(current); }
}

function updateSubtitleActions(c, item, subtitle) {
  const show = !!subtitle && (subtitle.retryable || subtitle.restartable) && !!(item.settings && item.settings.subtitles);
  const sig = show ? [subtitle.retryable, subtitle.restartable, busyIds.has(c.id)].join('|') : '';
  if (sig === c.sig.subtitleActions) return;
  c.sig.subtitleActions = sig;
  c.subtitleActions.replaceChildren();
  setHidden(c.subtitleActions, !show);
  if (!show) return;
  const id = c.id;
  const busy = busyIds.has(id);
  if (subtitle.retryable) {
    const retry = buildBtn('立即重試', 'play', '', () => retrySubtitles(id), '用目前的字幕設定（含備援）重新嘗試，保留已完成的部分');
    retry.disabled = busy;
    c.subtitleActions.appendChild(retry);
  }
  if (subtitle.restartable) {
    const restart = buildBtn('', 'retry', '', () => restartSubtitles(id), '重新處理字幕：刪除已完成的辨識與翻譯並從頭開始');
    restart.disabled = busy;
    c.subtitleActions.appendChild(restart);
  }
}

// Queue subtitle settings redesign: pending-settings prompt (per item and queue-wide), 本片自訂 / 沿用舊設定 tags.
const SUBTITLE_OVERRIDE_LABELS = { subtitle_comparison_enabled: '多字幕比較' };
function subtitleOverrideLines(settings) {
  const ov = settings && settings.subtitle_overrides;
  if (!ov || typeof ov !== 'object') return [];
  return Object.keys(ov).map((k) => {
    const v = ov[k];
    return `${SUBTITLE_OVERRIDE_LABELS[k] || k}：${typeof v === 'boolean' ? (v ? '開' : '關') : String(v)}`;
  });
}

async function applySubtitleSettings(id, choice) {
  if (busyIds.has(id)) return;
  busyIds.add(id);
  if (current) render(current);
  try { await act('apply_subtitle_settings', id, choice); }
  finally { busyIds.delete(id); if (current) render(current); }
}

async function applySubtitleSettingsAll(choice) {
  if (busyIds.has('*')) return;
  busyIds.add('*');
  if (current) render(current);
  try { await act('apply_subtitle_settings_all', choice); }
  finally { busyIds.delete('*'); if (current) render(current); }
}

function updateSettingsRow(c, item) {
  const pending = item.subtitle && item.subtitle.settings_pending;
  const asrChanged = !!(pending && pending.asr_changed);
  const busy = busyIds.has(c.id) || busyIds.has('*');
  const sig = pending ? [asrChanged, busy].join('|') : '';
  if (sig === c.sig.settingsRow) return;
  c.sig.settingsRow = sig;
  c.settingsRow.replaceChildren();
  setHidden(c.settingsRow, !pending);
  if (!pending) return;
  const id = c.id;
  c.settingsRow.append(el('span', 'ssr-text', '字幕設定已變更'));
  const box = el('div', 'subtitle-actions');
  const mk = (label, choice, tip) => {
    const b = buildBtn(label, choice === 'keep' ? 'lock' : 'retry', '', () => applySubtitleSettings(id, choice), tip);
    b.dataset.choice = choice; b.disabled = busy; box.appendChild(b);
  };
  mk('剩下的用新設定', 'rest', '已完成的部分保留，尚未處理的部分改用新設定');
  mk('維持原設定', 'keep', '這部影片繼續使用原本的字幕設定，之後不再詢問這次的變更');
  if (!asrChanged) mk('已完成的也用新翻譯重做', 'redo_mt', '保留語音辨識結果，已完成的翻譯也用新設定重做');
  c.settingsRow.appendChild(box);
}

function renderPendingBar(items) {
  const bar = $('#subtitlePending');
  if (!bar) return;
  const n = items.filter((it) => it.subtitle && it.subtitle.settings_pending).length;
  const busy = busyIds.has('*');
  const sig = n ? `${n}|${busy}` : '';
  if (bar.dataset.sig === sig) return;
  bar.dataset.sig = sig;
  bar.replaceChildren();
  setHidden(bar, !n);
  if (!n) return;
  bar.append(el('span', 'ssr-text', `${n} 個處理中的項目設定已變更`));
  const box = el('div', 'subtitle-actions');
  const apply = buildBtn('全部套用', 'retry', '', () => applySubtitleSettingsAll('rest'), '全部改用新設定處理剩下的部分');
  const keep = buildBtn('全部維持', 'lock', '', () => applySubtitleSettingsAll('keep'), '全部維持原本的字幕設定');
  apply.dataset.choice = 'rest'; keep.dataset.choice = 'keep'; apply.disabled = keep.disabled = busy;
  box.append(apply, keep);
  bar.appendChild(box);
}

async function rerunSubtitles(id) {
  if (busyIds.has(id)) return;
  busyIds.add(id);
  if (current) render(current);
  try { await act('rerun_subtitles', id); }
  finally { busyIds.delete(id); if (current) render(current); }
}

function updateMeta(c, item) {
  const src = sourceLabel(item);
  const q = qualityLabel(item.quality);
  const s = item.settings || {};
  const enc = s.encoder ? (String(s.encoder).toLowerCase() === 'h264' ? 'H.264' : String(s.encoder).toUpperCase()) : '';
  const interp = s.interpolation === 'full';
  // finished items: item.has_compare is authoritative (a removed compare track must not keep the chip)
  const cmp = item.state === 'done' && typeof item.has_compare === 'boolean'
    ? item.has_compare
    : (s.save_compare === true || s.compare === true || s.compare === 'on');
  const cmpOrig = item.compare_original === 'encode' ? '原始畫面：重新壓縮' : (item.compare_original === 'copy' ? '原始畫面：沿用原檔' : '');
  const models = MODEL_KEYS.map((key) => modelLabel(key, s[key]));
  const reprocessed = !!item.parent_id;
  const overrideLines = subtitleOverrideLines(s);
  const hold = !!s.subtitle_hold;
  const sig = JSON.stringify([src, q, enc, interp, cmp, cmpOrig, models, reprocessed, overrideLines, hold]);
  if (sig === c.sig.meta) return;
  c.sig.meta = sig;
  c.meta.replaceChildren();
  if (src.text) {
    const e = el('span', 'src');
    e.appendChild(svgEl(src.kind === 'url' ? 'globe' : 'file'));
    e.appendChild(el('span', null, src.text));
    e.title = String(item.target || '');
    c.meta.appendChild(e);
  }
  if (q) c.meta.appendChild(el('span', 'chip', q));
  if (reprocessed) c.meta.appendChild(el('span', 'chip', '重新處理版本'));
  if (enc) c.meta.appendChild(el('span', 'chip enc', enc));
  models.forEach((label, index) => {
    if (!label) return;
    const chip = el('span', 'chip model', `${index === 0 ? '偵測' : '修復'}：${label}`);
    chip.title = `${index === 0 ? '偵測模型' : '修復模型'}：${s[MODEL_KEYS[index]]}`;
    c.meta.appendChild(chip);
  });
  if (interp) c.meta.appendChild(el('span', 'chip fps', '插幀'));
  if (overrideLines.length) {
    const chip = el('span', 'chip custom', '本片自訂');
    chip.title = `這部影片的字幕設定與全域不同：\n${overrideLines.join('\n')}`;
    c.meta.appendChild(chip);
  }
  if (hold) {
    const chip = el('span', 'chip hold', '沿用舊設定');
    chip.title = '這部影片維持原本的字幕設定，不跟隨新的全域變更';
    c.meta.appendChild(chip);
  }
  if (cmp) {
    const chip = el('span', 'chip', '對照');
    if (cmpOrig) chip.title = cmpOrig;
    c.meta.appendChild(chip);
  }
}

async function removeCompare(id) {
  if (busyIds.has(id)) return;
  const ok = await confirmDlg({
    title: '移除比對軌', ok: '移除', danger: true,
    body: '移除這個項目的比對軌？處理後的影片會保留，此動作無法復原。',
  });
  if (!ok) return;
  busyIds.add(id); if (current) render(current);
  try { await act('remove_compare', id); }
  finally { busyIds.delete(id); if (current) render(current); }
}

/* ------------------------------------------------------------------ reprocess dialog */

let reprocessDialogState = null;
let reprocessSequence = 0;
let reprocessFocusRestore = null;

function checkedValue(name, root = $('#reprocessModal')) {
  const input = root.querySelector(`input[name="${name}"]:checked`);
  return input ? input.value : '';
}

function setCheckedValue(name, value, root = $('#reprocessModal')) {
  for (const input of root.querySelectorAll(`input[name="${name}"]`)) input.checked = input.value === String(value);
}

function populateReprocessModel(key, value) {
  const select = $(`#reprocess_${key}`);
  const available = current && current.model_options && current.model_options[key];
  const options = Array.isArray(available) ? available.slice() : [];
  if (value != null && !options.some((o) => String(o.value) === String(value))) {
    options.push({ value, label: String(value) + '（目前未列出）', disabled: true });
  }
  select.replaceChildren();
  for (const option of options) {
    const node = el('option', null, String(option.label || option.value));
    node.value = option.value;
    node.disabled = !!option.disabled;
    select.appendChild(node);
  }
  if (!options.length) {
    const node = el('option', null, '沒有可用模型');
    node.value = '';
    select.appendChild(node);
  }
  select.disabled = !Array.isArray(available) || !available.length;
  if (value != null) select.value = String(value);
}

function normalizeQuality(value) {
  if (typeof value === 'number' && Number.isInteger(value)) return value;
  if (typeof value === 'string' && /^\d+$/.test(value)) return Number(value);
  return value;
}

function renderReprocessQuality(value) {
  const select = $('#reprocess_quality');
  select.replaceChildren();
  const keep = el('option', null, value == null ? '保留來源解析度' : `保留目前：${qualityLabel(value) || String(value)}`);
  keep.value = 'keep';
  select.appendChild(keep);
  const best = el('option', null, '最佳畫質'); best.value = 'best'; select.appendChild(best);
  const heights = [2160, 1440, 1080, 720, 480, 360];
  if (typeof value === 'number' && Number.isInteger(value) && value > 0 && !heights.includes(value)) heights.push(value);
  if (typeof value === 'string' && /^\d+$/.test(value) && !heights.includes(Number(value))) heights.push(Number(value));
  for (const height of [...new Set(heights)].sort((a, b) => b - a)) {
    const option = el('option', null, `${height}p`);
    option.value = `height:${height}`;
    select.appendChild(option);
  }
  select.value = 'keep';
  syncReprocessQualityNotice();
}

function syncReprocessQualityNotice() {
  const state = reprocessDialogState;
  const notice = $('#reprocessQualityNotice');
  if (!state || !state.data) { setHidden(notice, true); return; }
  const isUrl = state.isUrl;
  $('#reprocess_quality').disabled = !isUrl;
  setHidden($('#reprocessQualityLocalHint'), isUrl);
  const choice = $('#reprocess_quality').value;
  const original = normalizeQuality(state.data.quality !== undefined ? state.data.quality : state.item.quality);
  const selected = choice === 'keep' ? original : choice === 'best' ? 'best' : Number(choice.slice('height:'.length));
  const changed = choice !== 'keep' && selected !== original;
  setHidden(notice, !isUrl || !changed);
  if (state.redownloadRequired) {
    $('#reprocess_redownload').checked = true;
    $('#reprocess_redownload').disabled = true;
  } else if (isUrl && changed) {
    if (!state.redownloadManuallyChosen) {
      $('#reprocess_redownload').checked = true;
      state.qualityForcedRedownload = true;
    }
    $('#reprocess_redownload').disabled = true;
  } else {
    $('#reprocess_redownload').disabled = false;
    if (state.qualityForcedRedownload && !state.redownloadManuallyChosen) $('#reprocess_redownload').checked = false;
    state.qualityForcedRedownload = false;
  }
  syncReprocessSourcePresentation();
}

function syncReprocessSourcePresentation() {
  const state = reprocessDialogState;
  if (!state || !state.data) return;
  const source = state.data.source || {};
  const redownload = state.isUrl && $('#reprocess_redownload').checked;
  setText($('#reprocessSource'), redownload ? '將重新下載網址的原始來源。' : String(source.message || '來源資訊尚未提供。'));
  setHidden($('#reprocessLossy'), !source.lossy || redownload);
}

function syncReprocessCompare() {
  const enabled = $('#reprocess_save_compare').checked;
  const original = checkedValue('reprocess_compare_original');
  setHidden($('#reprocess_compare_settings'), !enabled);
  setHidden($('#reprocess_compare_quality_row'), !enabled || original !== 'encode');
}

function fillReprocessDialog(data, item) {
  const kind = data.kind || item.kind;
  const target = data.target || item.target;
  const isUrl = kind !== 'file' && /^https?:\/\//i.test(String(target || ''));
  reprocessDialogState.kind = kind;
  reprocessDialogState.target = target;
  reprocessDialogState.isUrl = isUrl;
  reprocessDialogState.redownloadRequired = isUrl && String((data.source || {}).mode || '').toLowerCase() === 'download';
  reprocessDialogState.redownloadManuallyChosen = false;
  reprocessDialogState.qualityForcedRedownload = false;
  const settings = data.settings || item.settings || {};
  const fallback = current && current.settings ? current.settings : {};
  for (const key of MODEL_KEYS) populateReprocessModel(key, settings[key] != null ? settings[key] : (item.settings || {})[key]);
  setCheckedValue('reprocess_encoder', settings.encoder || fallback.encoder || 'hevc');
  setCheckedValue('reprocess_interpolation', settings.interpolation || fallback.interpolation || 'full');
  $('#reprocess_save_compare').checked = settings.save_compare === true || settings.save_compare === 'on';
  setCheckedValue('reprocess_compare_original', settings.compare_original || fallback.compare_original || 'copy');
  setCheckedValue('reprocess_compare_quality', settings.compare_quality || fallback.compare_quality || 'small');
  renderReprocessQuality(data.quality !== undefined ? data.quality : item.quality);

  const source = data.source || {};
  setHidden($('#reprocessRedownloadRow'), !isUrl);
  $('#reprocess_redownload').checked = reprocessDialogState.redownloadRequired;
  $('#reprocess_redownload').disabled = reprocessDialogState.redownloadRequired;
  const mode = String(source.mode || '').toLowerCase();
  const missingLocal = kind === 'file' && (
    source.needs_path === true || /missing|lost|not_found|relocat|找不到|不存在|遺失|缺少/.test(mode) || /找不到|不存在|遺失|缺少/.test(String(source.message || ''))
  );
  setHidden($('#reprocessSourcePathRow'), !missingLocal);
  $('#reprocess_source_path').value = '';
  $('#reprocess_source_path').required = missingLocal;
  $('#reprocess_source_path').placeholder = missingLocal ? '輸入目前可用的本機檔案完整路徑' : '';
  $('#reprocess_stop_current').checked = false;
  const state = data.state || item.state;
  const editInPlace = data.edit_in_place === true;
  setHidden($('#reprocessStopRow'), editInPlace || (state !== 'downloading' && state !== 'queued' && state !== 'processing'));
  const stateMessage = editInPlace ? '尚未開始的項目會直接更新設定並保留排序。' : '新設定會建立重新處理版本；既有成品不會被覆寫。';
  setText($('#reprocessStateNote'), `目前狀態：${STATE_LABEL[state] || state || '未知'}。${stateMessage}`);
  setText($('#reprocessSubmit'), editInPlace ? '更新設定' : '開始重新處理');
  setText($('#reprocessError'), '');
  setHidden($('#reprocessError'), true);
  syncReprocessCompare();
  syncReprocessQualityNotice();
}

function restoreReprocessFocus(id, saved) {
  if (saved && saved.isConnected && !saved.disabled) { saved.focus(); return; }
  const card = cards.get(id);
  if (!card) return;
  const button = [...card.actions.querySelectorAll('button')].find((b) => b.textContent.includes('重新處理'));
  if (button && !button.disabled) button.focus();
}

function finishReprocessBusy(id) {
  busyIds.delete(id);
  if (current) render(current);
  if (reprocessFocusRestore && reprocessFocusRestore.id === id) {
    const saved = reprocessFocusRestore.focusReturn;
    reprocessFocusRestore = null;
    restoreReprocessFocus(id, saved);
  }
}

function closeReprocessDialog() {
  const state = reprocessDialogState;
  if (!state || state.submitting) return;
  reprocessSequence++;
  $('#reprocessModal').hidden = true;
  reprocessDialogState = null;
  if (busyIds.has(state.id)) reprocessFocusRestore = { id: state.id, focusReturn: state.focusReturn };
  else restoreReprocessFocus(state.id, state.focusReturn);
}

async function openReprocessDialog(id) {
  if (busyIds.has(id) || reprocessDialogState) return;
  const item = current && current.items.find((x) => x.id === id);
  if (!item) return;
  const seq = ++reprocessSequence;
  const modal = $('#reprocessModal');
  const fields = $('#reprocessFields');
  reprocessDialogState = { id, item, data: null, focusReturn: document.activeElement, loading: true, submitting: false, seq };
  $('#reprocessTitle').textContent = '重新處理';
  setText($('#reprocessSource'), '正在讀取來源與目前設定…');
  setText($('#reprocessStateNote'), '');
  setText($('#reprocessError'), '');
  setHidden($('#reprocessError'), true);
  fields.disabled = true;
  $('#reprocessCancel').disabled = false;
  $('#reprocessSubmit').disabled = true;
  setText($('#reprocessSubmit'), '讀取中…');
  modal.hidden = false;
  $('#reprocessCancel').focus();
  busyIds.add(id);
  if (current) render(current);
  try {
    const data = await api('reprocess_options', id);
    if (!reprocessDialogState || reprocessDialogState.seq !== seq) return;
    if (!data || data.ok === false || data.error) {
      setText($('#reprocessError'), String((data && (data.error || data.message)) || '無法讀取重新處理設定。'));
      setHidden($('#reprocessError'), false);
      return;
    }
    reprocessDialogState.data = data;
    reprocessDialogState.loading = false;
    fillReprocessDialog(data, item);
    fields.disabled = false;
    $('#reprocessSubmit').disabled = false;
    $('#reprocess_detection_model').focus();
  } finally {
    finishReprocessBusy(id);
  }
}

async function submitReprocess(event) {
  event.preventDefault();
  const state = reprocessDialogState;
  if (!state || state.loading || state.submitting || !state.data) return;
  const sourcePath = $('#reprocess_source_path').value.trim();
  if ($('#reprocess_source_path').required && !sourcePath) {
    setText($('#reprocessError'), '請輸入目前可用的本機檔案路徑。');
    setHidden($('#reprocessError'), false);
    $('#reprocess_source_path').focus();
    return;
  }
  const modelChanges = {};
  for (const key of MODEL_KEYS) {
    const select = $(`#reprocess_${key}`);
    if (select.disabled) continue; // no installed choices: preserve the backend's saved model
    const selected = select.selectedOptions[0];
    if (!selected || selected.disabled) {
      setText($('#reprocessError'), `請為「${key === 'detection_model' ? '偵測模型' : '修復模型'}」選擇已安裝的模型。`);
      setHidden($('#reprocessError'), false);
      select.focus();
      return;
    }
    modelChanges[key] = select.value;
  }
  const qChoice = $('#reprocess_quality').value;
  const quality = qChoice === 'keep' ? normalizeQuality(state.data.quality !== undefined ? state.data.quality : state.item.quality)
    : qChoice === 'best' ? 'best' : Number(qChoice.slice('height:'.length));
  const changes = {
    ...modelChanges,
    encoder: checkedValue('reprocess_encoder'),
    interpolation: checkedValue('reprocess_interpolation'),
    save_compare: $('#reprocess_save_compare').checked,
    compare_original: checkedValue('reprocess_compare_original'),
    compare_quality: checkedValue('reprocess_compare_quality'),
    quality,
  };
  const options = {
    redownload: $('#reprocess_redownload').checked,
    stop_current: $('#reprocess_stop_current').checked,
  };
  if (sourcePath) options.source_path = sourcePath;

  const id = state.id;
  const seq = state.seq;
  state.submitting = true;
  $('#reprocessFields').disabled = true;
  $('#reprocessCancel').disabled = true;
  $('#reprocessSubmit').disabled = true;
  setText($('#reprocessSubmit'), state.data.edit_in_place === true ? '正在更新設定…' : '正在建立版本…');
  setText($('#reprocessError'), ''); setHidden($('#reprocessError'), true);
  busyIds.add(id);
  if (current) render(current);
  try {
    const result = await api('reprocess', id, changes, options);
    if (!reprocessDialogState || reprocessDialogState.seq !== seq) return;
    if (!result || result.ok === false || result.error) {
      state.submitting = false;
      $('#reprocessFields').disabled = false;
      $('#reprocessCancel').disabled = false;
      $('#reprocessSubmit').disabled = false;
      setText($('#reprocessSubmit'), state.data.edit_in_place === true ? '更新設定' : '開始重新處理');
      setText($('#reprocessError'), String((result && (result.error || result.message)) || '重新處理失敗，請檢查來源後再試。'));
      setHidden($('#reprocessError'), false);
      return;
    }
    $('#reprocessModal').hidden = true;
    reprocessDialogState = null;
    reprocessFocusRestore = { id, focusReturn: state.focusReturn };
    await refresh();
  } finally {
    finishReprocessBusy(id);
  }
}

$('#reprocessForm').addEventListener('submit', submitReprocess);
$('#reprocessCancel').addEventListener('click', closeReprocessDialog);
$('#reprocessModal').addEventListener('mousedown', (event) => {
  if (event.target === $('#reprocessModal')) closeReprocessDialog();
});
$('#reprocessForm').addEventListener('change', (event) => {
  if (event.target.id === 'reprocess_save_compare' || event.target.name === 'reprocess_compare_original') syncReprocessCompare();
  if (event.target.id === 'reprocess_quality') syncReprocessQualityNotice();
  if (event.target.id === 'reprocess_redownload') {
    if (reprocessDialogState && !event.target.disabled) reprocessDialogState.redownloadManuallyChosen = event.target.checked;
    syncReprocessSourcePresentation();
  }
});
$('#reprocessModal').addEventListener('keydown', (event) => {
  if (event.key !== 'Tab') return;
  const focusable = [...$('#reprocessForm').querySelectorAll('button, input, select, textarea, [tabindex]')]
    .filter((node) => !node.disabled && node.tabIndex >= 0 && node.getClientRects().length);
  if (!focusable.length) return;
  const first = focusable[0], last = focusable[focusable.length - 1];
  if (event.shiftKey && (document.activeElement === first || !$('#reprocessForm').contains(document.activeElement))) {
    event.preventDefault(); last.focus();
  } else if (!event.shiftKey && (document.activeElement === last || !$('#reprocessForm').contains(document.activeElement))) {
    event.preventDefault(); first.focus();
  }
});

async function exportItem(id) {
  const item = current && current.items.find((x) => x.id === id);
  if (!item) return;
  let includeCompare = false;
  if (item.has_compare === true) {
    const version = await confirmDlg({
      title: '匯出影片', body: '選擇要匯出的版本，接著選擇儲存位置。', ok: '下一步',
      choices: [
        { value: 'processed', label: '處理後影片（不含對比軌）' },
        { value: 'compare', label: '含對比軌（可切換原始畫面）' },
      ],
    });
    if (!version) return;
    includeCompare = version === 'compare';
  }
  await act('export', id, includeCompare);
}

function updateActions(c, item) {
  const hasHeaders = !!(item.headers && Object.keys(item.headers).length);
  const canPlay = item.state !== 'failed' || !!item.source_path;     // failed: only when the source was downloaded
  const hasCmp = item.has_compare === true;
  const isBusy = busyIds.has(c.id);
  const opening = openingIds.has(c.id);
  const playError = playErrors.get(c.id);
  const env = current && current.subtitle_environment ? current.subtitle_environment : {};
  const subtitleToggle = subtitleToggleState(item, env);
  const subtitleStatus = item.subtitle && item.subtitle.status || '';
  const subtitleOnly = !!(item.subtitle && item.subtitle.only);
  const subtitleRerunPaused = subtitleOnly && subtitleStatus === 'paused';
  const subtitleRerunActive = subtitleOnly && (SUBTITLE_ACTIVE_STATES.has(subtitleStatus) || ['ready', 'mux_pending'].includes(subtitleStatus));
  const sig = [item.state, hasHeaders, canPlay, hasCmp, isBusy, opening, playError, subtitleToggle.checked,
    subtitleToggle.visible, subtitleToggle.disabled, subtitleToggle.reason, subtitleRuntimeAvailable(env), subtitleStatus, subtitleOnly, subtitleRerunPaused, item.play_after_subtitles].join('|');
  if (sig === c.sig.actions) return;
  c.sig.actions = sig;
  const id = c.id;
  const a = c.actions;
  a.replaceChildren();
  const st = item.state;
  if (subtitleToggle.visible) {
    const label = el('label', 'subtitle-toggle');
    const input = el('input'); input.type = 'checkbox'; input.checked = subtitleToggle.checked;
    input.disabled = subtitleToggle.disabled || isBusy;
    input.setAttribute('aria-label', '產生字幕');
    label.title = subtitleToggle.reason;
    label.append(input, el('span', null, '產生字幕'));
    input.addEventListener('change', (event) => {
      event.stopPropagation();
      act('set_subtitles', id, input.checked);
    });
    a.appendChild(label);
    if (subtitleToggle.disabled && !subtitleRuntimeAvailable(env)) a.appendChild(el('span', 'subtitle-disabled-note', subtitleToggle.reason));
  }
  if (st === 'done') {
    a.appendChild(buildBtn('播放', 'play', 'primary', () => openPlayer(id)));
    a.appendChild(buildBtn('匯出', 'export', '', () => exportItem(id)));
    const rerun = buildBtn(isBusy || subtitleRerunActive ? '字幕重跑中…' : '只重跑字幕', 'retry', '', () => rerunSubtitles(id),
      subtitleRuntimeAvailable(env) ? '只重新產生字幕，保留已完成的影片' : subtitleRuntimeReason(env));
    rerun.disabled = isBusy || subtitleRerunActive || subtitleRerunPaused || !subtitleRuntimeAvailable(env);
    if (subtitleRerunPaused) {
      a.appendChild(buildBtn('繼續字幕', 'play', 'primary', () => act('resume', id), '從暫停處繼續只重跑字幕'));
    }
    a.appendChild(rerun);
    if (subtitleRerunActive) {
      a.appendChild(buildBtn('暫停字幕', 'pause', '', () => act('pause', id), '暫停只重跑字幕，保留進度'));
    }
    if (subtitleRerunActive || subtitleRerunPaused) {
      a.appendChild(buildBtn('取消字幕', 'pause', '', () => act('cancel_subtitles', id), '取消只重跑字幕'));
    }
    if (hasCmp) {
      a.appendChild(buildBtn(isBusy ? '移除中…' : '移除比對軌', 'layers', '', () => removeCompare(id)));
    }
  } else if (st === 'failed') {
    if (canPlay) a.appendChild(buildBtn('播放', 'play', '', () => openPlayer(id)));
    a.appendChild(buildBtn('重試', 'retry', 'primary', () => act('retry', id)));
  } else if (st === 'paused') {
    a.appendChild(buildBtn('播放', 'play', '', () => openPlayer(id)));
    a.appendChild(buildBtn('繼續', 'play', 'primary', () => act('resume', id)));
  } else {
    // still being processed / downloaded: playing it shows the finished segments (or a live restore)
    a.appendChild(buildBtn('播放', 'play', '', () => openPlayer(id)));
    a.appendChild(buildBtn('暫停', 'pause', '', () => act('pause', id)));
  }
  if (MODEL_EDIT_STATES.has(st)) {
    a.appendChild(buildBtn('套用目前模型', 'layers', '', () => act('apply_model_settings', id), '套用設定中的偵測與修復模型，重新判定處理快取'));
  }
  a.appendChild(buildBtn('重新處理', 'retry', '', () => openReprocessDialog(id), '使用新設定重新處理此項目'));
  if (hasHeaders) {
    a.appendChild(buildBtn('', 'key', '', async () => {
      const cur = (current && current.items.find((x) => x.id === id)) || item;
      const ok = await confirmDlg({
        title: '清除登入資訊', ok: '清除', danger: true,
        body: `這會刪除「${cur.title || cur.target}」儲存的 headers / cookies。\n清除後若網址需要登入或已過期，之後可能無法繼續下載。`,
      });
      if (ok) act('clear_headers', id);
    }, '清除登入資訊'));
  }
  a.appendChild(buildBtn('', 'folder', '', () => api('open_folder', id), '開啟資料夾'));
  a.appendChild(buildBtn('', 'trash', 'danger', async () => {
    const cur = (current && current.items.find((x) => x.id === id)) || item;
    const ok = await confirmDlg({
      title: '刪除項目', ok: '刪除', danger: true,
      body: `要刪除「${cur.title || cur.target}」嗎？\n已下載與處理好的檔案也會一併刪除，無法復原。`,
    });
    if (ok) act('remove', id);
  }, '刪除'));
  const playButton = [...a.querySelectorAll('button')].find((button) => button.textContent === '播放');
  if (canPlay && playButton) {
    if (opening) {
      playButton.replaceChildren(el('span', 'opening-spinner'), el('span', null, '正在開啟播放器…'));
      playButton.disabled = true;
    } else if (playError) {
      playButton.replaceChildren(svgEl('retry'), el('span', null, '重試播放'));
    }
  }
  if (item.play_after_subtitles) {
    a.appendChild(el('span', 'subtitle-autoplay', '字幕完成後自動播放'));
    a.appendChild(buildBtn('取消自動播放', 'pause', '', () => act('cancel_play_after_subtitles', id)));
  }
  if (playError) {
    const message = el('span', 'player-open-error', playError);
    message.setAttribute('role', 'alert');
    a.appendChild(message);
  }
  if (isBusy) for (const b of a.children) b.disabled = true;   // slow action in flight: lock this card's buttons
}

function updateCard(c, item) {
  c.sig.item = item;
  c.root.dataset.state = item.state;
  c.root.classList.toggle('player-opening', openingIds.has(c.id));
  c.root.setAttribute('aria-busy', String(openingIds.has(c.id)));
  setText(c.title, item.title || item.target || '(未命名)');
  c.title.title = item.title || item.target || '';
  const overlapping = item.state === 'processing' && item.source_status === 'downloading';
  setText(c.badge, overlapping ? '下載並處理中' : (STATE_LABEL[item.state] || String(item.state)));

  updateMeta(c, item);
  updateStrip(c, item);

  // progress
  const pi = progressInfo(item);
  setHidden(c.prog, !pi);
  if (pi) {
    const indet = pi.pct == null;
    c.bar.classList.toggle('indet', indet);
    c.fill.style.width = indet ? '' : pi.pct.toFixed(2) + '%';
    if (indet) { c.pPct.dataset.sig = ''; c.pPct.replaceChildren(); } else setStat(c.pPct, Math.floor(pi.pct), '%');
    setText(c.pDetail, pi.detail);
    if (pi.detailTitle) c.pDetail.title = pi.detailTitle;
    else c.pDetail.removeAttribute('title');
    setText(c.pRate, pi.rate);
    setText(c.pEta, pi.eta);
    setText(c.pFrag, pi.frag || '');
  }

  // error
  const showErr = item.state === 'failed' && !!item.error;
  setHidden(c.err, !showErr);
  if (showErr) setText(c.errText, String(item.error));

  const subtitle = subtitleDisplay(item.subtitle, undefined, { item, state: current });
  setHidden(c.subtitleProgress, !subtitle);
  setHidden(c.subtitleError, !(subtitle && subtitle.error));
  if (subtitle) {
    setText(c.subtitleLabel, subtitle.label);
    setText(c.subtitleGlyph, subtitle.glyph || '');
    setHidden(c.subtitleGlyph, !subtitle.glyph);
    setAttr(c.subtitleProgress, 'data-cat', subtitle.cat || 'wait');
    if (subtitle.tone) setAttr(c.subtitleProgress, 'data-tone', subtitle.tone); else c.subtitleProgress.removeAttribute('data-tone');
    c.subtitleProgress.classList.toggle('still', !!subtitle.dim);
    setText(c.subtitleDetail, subtitle.detail);
    setHidden(c.subtitleDetail, !subtitle.detail);
    if (subtitle.detail) setAttr(c.subtitleDetail, 'title', subtitle.detail);
    else c.subtitleDetail.removeAttribute('title');
    setText(c.subtitlePercent, subtitle.showPercent && subtitle.percent != null ? `${Math.round(subtitle.percent)}%` : '');
    setHidden(c.subtitleTrack, !subtitle.showTrack);
    setAttr(c.subtitleTrack, 'aria-label', subtitle.label);
    if (subtitle.percent != null) setAttr(c.subtitleTrack, 'aria-valuenow', Math.round(subtitle.percent));
    else c.subtitleTrack.removeAttribute('aria-valuenow');
    const fillPct = Number.isFinite(subtitle.fill) ? subtitle.fill : (subtitle.percent == null ? 0 : subtitle.percent);
    c.subtitleFill.style.width = `${fillPct}%`;
    c.subtitleTrack.style.setProperty('--w', `${fillPct}%`);
    c.subtitleProgress.classList.toggle('failed', subtitle.status === 'failed');
    c.subtitleProgress.classList.toggle('blocked', subtitle.blocked);
    if (subtitle.error) setText(c.subtitleErrorText, `字幕錯誤：${subtitle.error}`);
  }
  updateSubtitleActions(c, item, subtitle);
  updateSettingsRow(c, item);
  updateCombos(c, item);

  // size / keep / watched
  let sizeText = '';
  const p = item.state === 'downloading' && Object.keys(item.download_progress || {}).length
    ? item.download_progress : (item.progress || {});
  if (num(item.size_bytes) > 0) sizeText = fmtBytes(item.size_bytes);
  else if (item.state === 'downloading' && num(p.total_bytes) > 0) sizeText = fmtBytes(p.total_bytes);
  setText(c.size, sizeText);
  setHidden(c.size, !sizeText);
  setAttr(c.keep, 'aria-pressed', !!item.keep);
  setHidden(c.seen, !(item.watched && item.state === 'done'));

  updateActions(c, item);
}

/* ---- streaming card (T3.5) ----
 * get_state().streaming is the live-mode stream that plays now (queue_ipc.py, "Live stream report": title, pos, duration, ahead_s, speed, ...)
 * plus, for a stream with subtitles, `subtitle` {status, view, detail, error, lead_s} (lead_s = how far the subtitles are ahead of the viewer).
 * streamingCardModel is pure; renderStreaming only writes the model into the static card of index.html. */
function fmtSpan(sec) {
  const s = Math.max(0, Math.round(num(sec)));
  if (s < 60) return `${s} 秒`;
  const m = Math.floor(s / 60), r = s % 60;
  if (m >= 120) return `${Math.floor(m / 60)} 小時${m % 60 ? ` ${m % 60} 分` : ''}`;
  return m >= 10 || r === 0 ? `${m} 分鐘` : `${m} 分 ${r} 秒`;
}

function streamingCardModel(st, nowS) {
  if (!st || typeof st !== 'object') return null;
  const opt = (v) => (v == null ? null : num(v));
  const pos = opt(st.pos), ahead = opt(st.ahead_s), speed = opt(st.speed);
  const dur = num(st.duration) > 0 ? num(st.duration) : 0;
  const pct = (v) => (dur && v != null ? Math.max(0, Math.min(100, (v / dur) * 100)) : null);
  const model = {
    title: String(st.title || '') || '串流播放',
    state: st.buffering ? '緩衝中' : st.playing ? '播放中' : '已暫停',
    timeText: pos == null ? '載入中…' : (dur ? fmtTimeProgress(pos, dur) : fmtDur(pos)),
    playedPct: pct(pos),
    aheadPct: pos != null && ahead != null ? pct(pos + ahead) : null,
    restore: '', restoreLevel: '', subtitle: false, subLabel: '', subNote: '', subLevel: '',
  };
  if (st.restore_done) {
    model.restore = '修復完成';
    model.restoreLevel = 'good';
  } else {
    model.restore = [ahead != null ? `修復領先 ${fmtSpan(ahead)}` : '修復準備中', speed != null ? `${speed.toFixed(1)}x` : ''].filter(Boolean).join('・');
    model.restoreLevel = speed == null ? '' : speed >= 1 ? 'good' : ahead != null && ahead < 5 ? 'bad' : 'warn';
  }
  const sub = st.subtitle;
  if (sub && typeof sub === 'object') {
    model.subtitle = true;
    const d = subtitleDisplay({ status: sub.status, view: sub.view, detail: sub.detail, error: sub.error }, nowS);
    const lead = opt(sub.lead_s);
    const covered = sub.view ? num(sub.view.covered_s) : 0;
    if (!d) {
      model.subLabel = '字幕：準備中';
    } else if (d.status === 'done') {
      model.subLabel = '字幕：完成';
      model.subLevel = 'good';
    } else if (d.status === 'failed') {
      model.subLabel = d.label;
      model.subLevel = 'bad';
      model.subNote = d.error;
    } else if (lead != null && covered > 0) {
      model.subLabel = lead > 0 ? `字幕：已領先 ${fmtSpan(lead)}` : '字幕：尚未跟上播放';
      model.subLevel = lead >= 30 ? 'good' : 'warn';
      model.subNote = [d.body, d.detail].filter(Boolean).join('・');
    } else {
      model.subLabel = d.label;
      model.subNote = d.detail;
    }
  }
  return model;
}

function renderStreaming(st) {
  const m = streamingCardModel(st);
  setHidden($('#streamCard'), !m);
  if (!m) return;
  setText($('#streamTitle'), m.title);
  setAttr($('#streamTitle'), 'title', m.title);
  setText($('#streamState'), m.state);
  setText($('#streamTime'), m.timeText);
  $('#streamPlayed').style.width = `${m.playedPct == null ? 0 : m.playedPct}%`;
  $('#streamAhead').style.width = `${m.aheadPct == null ? 0 : m.aheadPct}%`;
  if (m.playedPct == null) $('#streamBar').removeAttribute('aria-valuenow');
  else setAttr($('#streamBar'), 'aria-valuenow', Math.round(m.playedPct));
  setText($('#streamRestore'), m.restore);
  setAttr($('#streamRestore'), 'data-level', m.restoreLevel);
  setHidden($('#streamSubRow'), !m.subtitle);
  setText($('#streamSub'), m.subLabel);
  setAttr($('#streamSub'), 'data-level', m.subLevel);
  setText($('#streamSubNote'), m.subNote);
  setHidden($('#streamSubNote'), !m.subNote);
}
/* ---- end streaming card ---- */

function renderSummary(items) {
  const n = { active: 0, wait: 0, done: 0, failed: 0, paused: 0 };
  for (const it of items) {
    if (it.state === 'downloading' || it.state === 'processing') n.active++;
    else if (it.state === 'waiting' || it.state === 'queued') n.wait++;
    else if (it.state === 'done') n.done++;
    else if (it.state === 'failed') n.failed++;
    else if (it.state === 'paused') n.paused++;
  }
  const parts = [`共 ${items.length} 項`];
  if (n.active) parts.push(`進行中 ${n.active}`);
  if (n.wait) parts.push(`等待 ${n.wait}`);
  if (n.paused) parts.push(`暫停 ${n.paused}`);
  if (n.done) parts.push(`完成 ${n.done}`);
  if (n.failed) parts.push(`失敗 ${n.failed}`);
  setText($('#summary'), items.length ? parts.join(' · ') : '');
}

let noticeSig = '';
function renderNotices(state) {
  const notes = [];
  let n = state.notice;
  if (n && typeof n === 'object') n = n.message || n.text || '';
  if (n) notes.push({ cls: 'warn', icon: 'alert', text: String(n) });
  if (state.player_busy) notes.push({ cls: 'info', icon: 'info', text: '播放器正在即時修復，背景處理已暫停' });
  if (state.paused_all) notes.push({ cls: 'info', icon: 'pause', text: '佇列已全部暫停，按「全部繼續」恢復' });
  const sig = JSON.stringify(notes);
  if (sig === noticeSig) return;
  noticeSig = sig;
  const box = $('#notices');
  box.replaceChildren();
  for (const x of notes) {
    const d = el('div', 'notice ' + x.cls);
    d.setAttribute('role', 'status');
    d.appendChild(svgEl(x.icon));
    d.appendChild(el('span', null, x.text));
    box.appendChild(d);
  }
}

function renderStorage(state) {
  const settings = state.settings || {};
  const used = num(state.usage_bytes);
  const cap = num(state.cap_bytes) || num(settings.cap_gb) * 1073741824;
  setText($('#usedText'), fmtGB(used));
  setText($('#capText'), cap > 0 ? `${fmtGB(cap)} GB` : '—');
  const pct = cap > 0 ? Math.min(100, used / cap * 100) : 0;
  setText($('#usedPct'), cap > 0 ? `${pct.toFixed(pct < 10 ? 1 : 0)}%` : '');
  $('#meterFill').style.width = pct.toFixed(2) + '%';
  const m = $('#meter');
  m.classList.toggle('warn', pct >= 70 && pct < 90);
  m.classList.toggle('danger', pct >= 90);
  setAttr(m, 'aria-valuenow', Math.round(pct));
  m.title = `已用 ${fmtBytes(used)} / 上限 ${fmtBytes(cap)}`;
  renderBlocks(pct, m.title);
}

// pixel skin: 50-block storage meter drawn from the same percentage as #meter (partial last block, warn/danger colours)
const METER_BLOCKS = 50;
let blockNodes = null, blockSig = '';
function renderBlocks(pct, title) {
  const box = $('#meterBlocks');
  if (!box) return;
  if (!blockNodes) {
    blockNodes = [];
    for (let i = 0; i < METER_BLOCKS; i++) { const b = document.createElement('i'); blockNodes.push(b); box.appendChild(b); }
  }
  const filled = Math.max(0, Math.min(METER_BLOCKS, pct / 100 * METER_BLOCKS));
  const full = Math.floor(filled + 1e-9);
  const frac = filled - full;
  const sig = [full, frac > 0.02 ? frac.toFixed(2) : 0, pct >= 90 ? 2 : pct >= 70 ? 1 : 0].join('|');
  setAttr(box, 'aria-valuenow', Math.round(pct));
  box.title = title;
  if (sig === blockSig) return;
  blockSig = sig;
  box.classList.toggle('warn', pct >= 70 && pct < 90);
  box.classList.toggle('danger', pct >= 90);
  blockNodes.forEach((b, i) => {
    const on = i < full, part = i === full && frac > 0.02;
    b.classList.toggle('on', on);
    b.classList.toggle('part', part);
    if (part) b.style.setProperty('--p', (frac * 100).toFixed(0) + '%'); else b.style.removeProperty('--p');
  });
}

function renderHeader(state) {
  const btn = $('#btnPauseAll');
  const paused = !!state.paused_all;
  const lbl = $('.lbl', btn);
  setText(lbl, paused ? '全部繼續' : '全部暫停');
  const ico = $('svg', btn);
  if (ico.dataset.mode !== (paused ? 'play' : 'pause')) {
    const n = svgEl(paused ? 'play' : 'pause');
    n.dataset.mode = paused ? 'play' : 'pause';
    ico.replaceWith(n);
  }
  btn.disabled = !state.items.length;
}

function render(state) {
  current = state;
  updateSubtitlePlayDialog();
  renderStorage(state);
  renderNotices(state);
  renderHeader(state);
  renderSettings(state.settings || {});
  renderPerf(state.perf);
  renderStreaming(state.streaming);
  renderSummary(state.items);
  renderPendingBar(state.items);
  updateLogoState(state.items);

  const seen = new Set();
  for (const item of state.items) {
    seen.add(item.id);
    let c = cards.get(item.id);
    if (!c) { c = createCard(item.id); cards.set(item.id, c); list.appendChild(c.root); }
    updateCard(c, item);
  }
  for (const [id, c] of cards) {
    if (!seen.has(id) && id !== dragId) { if (c.strip) c.strip.destroy(); c.root.remove(); cards.delete(id); }
  }
  if (dragId == null) {
    // keep DOM order in sync without touching nodes that are already in place
    let i = 0;
    for (const item of state.items) {
      const node = cards.get(item.id).root;
      const expected = list.children[i];
      if (expected !== node) list.insertBefore(node, expected || null);
      i++;
    }
  }
  setHidden($('#empty'), state.items.length > 0);
  setHidden($('#list'), state.items.length === 0);
}

/* ------------------------------------------------------------------ drag reorder (list level) */

function clearDropMarks() {
  for (const c of list.children) c.classList.remove('drop-before', 'drop-after');
}

function dropTarget(clientY) {
  const others = [...list.children].filter((n) => n.dataset.id !== dragId);
  let index = others.length;
  for (let i = 0; i < others.length; i++) {
    const r = others[i].getBoundingClientRect();
    if (clientY < r.top + r.height / 2) { index = i; break; }
  }
  return { index, others };
}

list.addEventListener('dragover', (e) => {
  if (dragId == null) return;
  e.preventDefault();
  e.dataTransfer.dropEffect = 'move';
  const { index, others } = dropTarget(e.clientY);
  clearDropMarks();
  if (!others.length) return;
  if (index < others.length) others[index].classList.add('drop-before');
  else others[others.length - 1].classList.add('drop-after');
  // auto-scroll near viewport edges
  if (e.clientY < 120) window.scrollBy(0, -14);
  else if (e.clientY > window.innerHeight - 60) window.scrollBy(0, 14);
});
list.addEventListener('dragleave', (e) => {
  if (dragId != null && !list.contains(e.relatedTarget)) clearDropMarks();
});
list.addEventListener('drop', (e) => {
  if (dragId == null) return;
  e.preventDefault();
  e.stopPropagation();
  const id = dragId;
  const { index, others } = dropTarget(e.clientY);
  clearDropMarks();
  const node = cards.get(id) && cards.get(id).root;
  if (!node) return;
  dragId = null;                       // allow render to reorder again
  node.draggable = false;
  node.classList.remove('dragging');
  list.insertBefore(node, index < others.length ? others[index] : null);   // optimistic
  act('move', id, index);
});

/* ------------------------------------------------------------------ add box + intake panel (解析面板) + drop files/text */

const addInput = $('#addInput');
const entryStatus = $('#entryStatus');
const sourceStatus = $('#sourceStatus');
let sourceTimer = null;
let sourceRevision = 0;

function setSourceStatus(message, kind = 'info') {
  sourceStatus.className = `entry-status ${kind}`;
  setText(sourceStatus, message);
  setHidden(sourceStatus, !message);
}

function scheduleSourceCheck() {
  clearTimeout(sourceTimer);
  const revision = ++sourceRevision;
  const text = addInput.value.trim();
  setSourceStatus('');
  if (!text) return;
  sourceTimer = setTimeout(async () => {
    setSourceStatus('正在檢查網址規則…', 'pending');
    try {
      const result = await backend.check_source(text);
      if (revision !== sourceRevision || addInput.value.trim() !== text) return;
      const kind = result.status === 'recognized' ? 'success' : result.status === 'invalid' ? 'error' : 'pending';
      setSourceStatus(String(result.message || ''), kind);
    } catch (e) {
      if (revision === sourceRevision) setSourceStatus('無法檢查網址規則；仍可嘗試解析', 'pending');
    }
  }, 350);
}

function setEntryStatus(message, kind = 'info') {
  entryStatus.className = `entry-status ${kind}`;
  setText(entryStatus, message);
  setHidden(entryStatus, !message);
}

function autoGrow() {
  addInput.style.height = '34px';
  const h = Math.max(34, Math.min(addInput.scrollHeight, 110));
  addInput.style.height = h + 'px';
  addInput.classList.toggle('scroll', addInput.scrollHeight > 110);
}
addInput.addEventListener('input', () => { autoGrow(); scheduleSourceCheck(); syncStale(); });
addInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) { e.preventDefault(); submitResolve(); }
});
$('#addForm').addEventListener('submit', (e) => { e.preventDefault(); submitResolve(); });

/* ---- intake panel ----
 * Enter / 解析 -> parse_inputs(text) -> resolve(entry.text) per entry (max 3 at once) + defaults() -> estimate(tokens, overrides) on every settings
 * change -> stream_resolved / add_resolved / save_defaults.  `ip` is the open panel; ipRev invalidates results of a panel that was closed or replaced.
 * The panel is built from string templates: every dynamic value (titles, hosts, errors ...) goes through esc(). */

const intake = $('#intake');
const IP_CONCURRENCY = 3;
const IP_STEPS = ['連線', '讀取格式', '影片資訊', '縮圖'];
const IP_STAGE = ['連線中…', '讀取格式中…', '讀取影片資訊中…', '擷取縮圖中…'];
const IP_STEP_AT = [0, 1.2, 4, 7];                       // seconds into the resolve at which the (cosmetic) step advances
const DET_LABEL = { 'v4-fast': 'V4 快速', 'v4-accurate': 'V4 精準' };
const DET_NOTE = { 'v4-fast': '較快，偶爾漏抓小塊馬賽克', 'v4-accurate': '漏抓較少，慢約 8–15%' };
const VERDICT_CLS = { live: 'live', prebuffer: 'pre', queue: 'slow' };
const VERDICT_TXT = { live: '可即時', prebuffer: '需預載', queue: '太慢' };
const ENC_LABEL = { hevc: 'HEVC', h264: 'H.264' };

let ip = null;                 // {rev, inputText, entries, phase: 'resolving'|'ready'|'error', defaults, est, cur, advOpen, sentStream, streamBusy, addBusy, skipped}
let ipRev = 0;
let estTimer = null, estRev = 0;
let scanTimer = null;

const ESC_MAP = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' };
const esc = (s) => String(s == null ? '' : s).replace(/[&<>"']/g, (c) => ESC_MAP[c]);

function fmtEtaZh(sec) {
  const m = Math.round(num(sec) / 60);
  if (m < 1) return '不到 1 分';
  const h = Math.floor(m / 60);
  return h ? `${h} 小時 ${String(m % 60).padStart(2, '0')} 分` : `${m} 分`;
}
const fpsText = (f) => String(+num(f).toFixed(2));

/* ---- state helpers ---- */
const okEntries = () => (ip ? ip.entries.filter((e) => e.state === 'ok' && e.info) : []);
const isSingle = () => !!ip && ip.entries.length === 1;
const isHigh = (info) => num(info.fps) > 31;                           // 60 fps class: no 插幀 row, 修復幀率 instead
const heightsOf = (info) => (Array.isArray(info.heights) ? info.heights : []);
const heightKey = (info) => heightsOf(info).map((h) => h.height).join(',');
const defHeight = (info) => (info.default_height != null ? info.default_height : (heightsOf(info)[0] ? heightsOf(info)[0].height : null));
function showHeight() {
  const items = okEntries();
  return items.length > 0 && items.every((e) => e.info.kind === 'url' && heightsOf(e.info).length && heightKey(e.info) === heightKey(items[0].info));
}
// Streaming with subtitles fetches its own audio: audio-only rendition first, else the lowest
// quality, else (single quality) a second copy of the same URL -> about twice the traffic.  `subtitle_audio` ("audio_only" | "lowest" |
// "same" | "file") comes from the resolve entry when set; otherwise fall back to counting the selectable qualities.
const SINGLE_QUALITY_NOTE = '此來源只有一種畫質，串流時產生字幕需要另外下載一份（流量約加倍）';
function subtitleStreamDoubles(info) {
  if (!info || info.kind !== 'url') return false;                       // local files are read directly
  if (info.subtitle_audio != null && info.subtitle_audio !== '') return info.subtitle_audio === 'same';
  return heightsOf(info).length <= 1;
}
// Streaming runs the main subtitle on the cloud only for now (local models need the GPU the player is using: T3.1 / P4).
const STREAM_LOCAL_SUBTITLE_NOTE = '主字幕模型為本機時，串流播放中字幕會和修復輪流使用 GPU，影片或字幕可能跟不上';
function subtitlePrimaryIsLocal(settings) {                             // does the saved profile's main subtitle run on a local model?
  const profile = settings && settings.subtitle_profile;
  const active = profile && profile.profiles && profile.profiles[profile.active_mode];
  if (!active) return false;
  if (profile.active_mode === 'local') return true;
  if (profile.active_mode === 'cloud') return false;
  return [active.asr, active.translation].some((id) => {                 // custom: any local ASR / translation model
    if (!id) return false;
    if (id === SUBTITLE_DUAL_VALUE) return true;
    const model = subtitleModels.find((m) => m.id === id);
    return !!model && !isCloudModel(model);
  });
}
// T4.3: stream.subtitle_gpu (queue_service._stream_subtitle_gpu) = the stream verdict with local subtitles counted in.
const SUBTITLE_GPU_VRAM_NOTE = '顯示卡記憶體不足以同時修復與產生本機字幕';
const SUBTITLE_GPU_SUGGEST_LABEL = { queue: '加入佇列', cloud: '改用雲端字幕' };
function subtitleGpuNotice(g) {                                         // {lead, actions: [{kind, label}]} or null
  if (!g || typeof g !== 'object') return null;
  const suggest = Array.isArray(g.suggest) ? g.suggest : [];
  const actions = suggest.map((x) => {
    if (!x) return null;
    if (x.kind === 'prebuffer') return Number(x.seconds) > 0 ? { kind: 'prebuffer', label: `先緩衝約 ${Math.round(Number(x.seconds))} 秒` } : null;
    return SUBTITLE_GPU_SUGGEST_LABEL[x.kind] ? { kind: x.kind, label: SUBTITLE_GPU_SUGGEST_LABEL[x.kind] } : null;
  }).filter(Boolean);
  let lead;
  if (g.vram_ok === false) lead = SUBTITLE_GPU_VRAM_NOTE;
  else {
    const r = Number(g.ratio);
    if (!(r > 0)) return null;
    lead = `開本機字幕後，這台電腦邊看邊修復約 ${r.toFixed(1)}x 即時`;
    if (!actions.length) lead += g.verdict === 'live' ? '，仍可即時播放' : '';
  }
  return { lead, actions };
}
function subtitleGpuOf(info) {                                        // stream.subtitle_gpu of the entry's estimate, or null
  const it = info ? estOf(info.token) : null;
  return it && it.stream && it.stream.subtitle_gpu ? it.stream.subtitle_gpu : null;
}
function subtitleGpuNoticeText(g) {
  const n = subtitleGpuNotice(g);
  if (!n) return '';
  return n.actions.length ? `${n.lead}，建議：${n.actions.map((a) => a.label).join('／')}` : n.lead;
}
const hasLow = () => okEntries().some((e) => !isHigh(e.info));
const hasHigh = () => okEntries().some((e) => isHigh(e.info));
const pick = (k, side) => (ip.cur[k] !== undefined ? ip.cur[k] : ip.defaults[side][k]);
function curHeight() {                                                  // selected resolution (null: items have different defaults, none picked)
  if (!showHeight()) return null;
  if (ip.cur.height !== undefined) return ip.cur.height;
  const ds = okEntries().map((e) => defHeight(e.info));
  return ds.every((d) => d === ds[0]) ? ds[0] : null;
}
const subtitleComparisonSupported = () => hasApi('set_subtitle_configuration') && hasApi('add_resolved');
const subtitleComparisonDefault = () => !!(current && current.settings && current.settings.subtitle_comparison_enabled);
function subtitleComparisonNames(settings) {
  const rows = settings && Array.isArray(settings.subtitle_comparison_rows) ? settings.subtitle_comparison_rows : [];
  return rows.map((row, index) => String((row && (row.label || row.name)) || `字幕軌 ${index + 1}`));
}
function subtitleComparisonSummary(settings) {
  const names = subtitleComparisonNames(settings);
  if (!names.length) return '尚未新增額外字幕軌';
  return `額外字幕軌 ${names.length} 條：${names.slice(0, 3).join('、')}${names.length > 3 ? '…' : ''}`;
}
// overrides actually in effect: keys whose row is not shown are dropped
function activeCur() {
  const o = {};
  const subtitleAvailable = !!(ip && subtitleRuntimeAvailable(ip.defaults && ip.defaults.subtitle_environment));
  const subtitleEnabled = !!(ip && (ip.cur.subtitles !== undefined ? ip.cur.subtitles : !!(ip.defaults && ip.defaults.queue && ip.defaults.queue.subtitles)));
  const comparisonCanOverride = subtitleAvailable && subtitleEnabled && subtitleComparisonSupported();
  for (const [k, v] of Object.entries(ip.cur)) {
    if (k === 'height' && !showHeight()) continue;
    if (k === 'interpolation' && !hasLow()) continue;
    if (k === 'restoration_fps' && !(isSingle() && hasHigh())) continue;
    if (k === 'subtitles' && !subtitleAvailable) continue;
    if (k === 'subtitle_comparison_enabled' && !comparisonCanOverride) continue;
    o[k] = v;
  }
  return o;
}
// what the buttons use: the edits plus the resolution the row shows (the queue would otherwise download the best
// rendition while the panel shows the player's default, <= 1080p)
function sendOverrides() {
  const o = activeCur(), h = curHeight();
  if (h != null) o.height = h;
  const env = ip && ip.defaults && ip.defaults.subtitle_environment;
  o.subtitles = !!(subtitleRuntimeAvailable(env) && pick('subtitles', 'queue'));
  return o;
}
// an override equal to every saved default it would replace is no change
function normalizeCur() {
  const cur = ip.cur, d = ip.defaults;
  if (cur.height !== undefined) {
    const ds = okEntries().map((e) => defHeight(e.info));
    if (ds.length && ds.every((x) => x === cur.height)) delete cur.height;
  }
  for (const k of Object.keys(cur)) {
    if (k === 'height') continue;
    const sv = [d.stream[k], d.queue[k]].filter((v) => v !== undefined);
    if (sv.length && sv.every((v) => v === cur[k])) delete cur[k];
  }
  if (cur.subtitle_comparison_enabled !== undefined && cur.subtitle_comparison_enabled === subtitleComparisonDefault()) {
    delete cur.subtitle_comparison_enabled;
  }
  for (const k of Object.keys(cur)) if (!(k in activeCur())) delete cur[k];
}
const detLabel = (v) => DET_LABEL[v] || modelLabel('detection_model', v) || String(v || '');
const choiceLabel = (key, v) => {
  const list = ip && ip.defaults && ip.defaults.choices && ip.defaults.choices[key];
  const o = Array.isArray(list) && list.find((x) => x.value === v);
  return o ? String(o.label || o.value) : String(v || '');
};
const estOf = (token) => (ip && ip.est && Array.isArray(ip.est.items) ? ip.est.items.find((x) => x.token === token) : null);
const profileKnown = () => (ip && ip.est && !ip.est.failed ? !!ip.est.has_profile : !!(ip && ip.defaults && ip.defaults.has_profile));

/* ---- templates ---- */
function seg(name, options, value, extra = '') {
  return `<div class="seg inline ${extra}" role="radiogroup">${options.map((o) =>
    `<label class="${o.cls || ''}"><input type="radio" name="${name}" value="${esc(o.v)}" ${String(o.v) === String(value) ? 'checked' : ''}><span class="opt">${o.html}</span></label>`).join('')}</div>`;
}

function thumbHtml(info, small) {
  const ok = info.thumb && /^data:image\//i.test(info.thumb);
  const dur = num(info.duration) > 0 ? `<span class="dur">${fmtDur(info.duration)}</span>` : '';
  return `<div class="thumb"${ok ? ' role="button" tabindex="0" aria-label="放大縮圖"' : ''}>${ok ? `<img alt="" src="${esc(info.thumb)}">` : '<span class="nothumb">無縮圖</span>'}${small || !ok ? '' : '<span class="peek">點選放大</span>'}${dur}</div>`;
}

let imagePreview = null, imagePreviewFocus = null;
function closeImagePreview() {
  if (!imagePreview) return;
  imagePreview.remove(); imagePreview = null;
  if (imagePreviewFocus && imagePreviewFocus.isConnected) imagePreviewFocus.focus();
}
function showImagePreview(src) {
  closeImagePreview(); imagePreviewFocus = document.activeElement;
  const dialog = document.createElement('dialog'); dialog.className = 'image-preview';
  const image = el('img'); image.src = src; image.alt = '縮圖放大預覽';
  const close = el('button', 'btn', '關閉'); close.type = 'button';
  const menu = el('div', 'preview-menu'); menu.hidden = true;
  const copy = el('button', 'btn', '複製圖片'), save = el('button', 'btn', '儲存圖片…');
  copy.type = save.type = 'button';
  const status = el('span', 'preview-status'); status.setAttribute('role', 'status');
  copy.addEventListener('click', async () => {
    menu.hidden = true;
    try {
      const canvas = document.createElement('canvas'); canvas.width = image.naturalWidth; canvas.height = image.naturalHeight;
      canvas.getContext('2d').drawImage(image, 0, 0);
      const blob = await new Promise((resolve) => canvas.toBlob(resolve, 'image/png'));
      if (!blob) throw new Error('無法讀取圖片');
      await navigator.clipboard.write([new ClipboardItem({ 'image/png': blob })]);
      status.textContent = '圖片已複製';
    } catch (error) { status.textContent = '複製失敗：' + (error.message || '剪貼簿無法使用'); }
  });
  save.addEventListener('click', async () => {
    menu.hidden = true;
    try {
      if (!isMock) {
        const result = await api('save_preview_image', src);
        if (result.cancelled) return;
        if (!result.ok) throw new Error(result.error || '儲存失敗');
      } else {
        const link = el('a'); link.href = src; link.download = '封面.' + (src.startsWith('data:image/jpeg') ? 'jpg' : src.startsWith('data:image/webp') ? 'webp' : 'png');
        document.body.appendChild(link); link.click(); link.remove();
      }
      status.textContent = '圖片已儲存';
    } catch (error) { status.textContent = '儲存失敗：' + error.message; }
  });
  menu.append(copy, save);
  image.addEventListener('contextmenu', (e) => {
    e.preventDefault(); menu.hidden = false; copy.focus();
  });
  close.addEventListener('click', closeImagePreview);
  dialog.addEventListener('click', (e) => { if (e.target === dialog) closeImagePreview(); });
  dialog.addEventListener('cancel', (e) => { e.preventDefault(); closeImagePreview(); });
  dialog.append(image, menu, close, status); document.body.appendChild(dialog); imagePreview = dialog;
  dialog.showModal(); close.focus();
}
document.addEventListener('click', (e) => {
  const thumb = e.target.closest('.intake .thumb[role="button"]');
  if (thumb) showImagePreview(thumb.querySelector('img').src);
});
document.addEventListener('keydown', (e) => {
  if (imagePreview && e.key === 'Escape') { e.stopImmediatePropagation(); e.preventDefault(); closeImagePreview(); return; }
  if ((e.key === 'Enter' || e.key === ' ') && e.target.matches('.intake .thumb[role="button"]')) {
    e.preventDefault(); showImagePreview(e.target.querySelector('img').src);
  }
});

function factsHtml(info) {
  const file = info.kind === 'file';
  const top = heightsOf(info)[0];
  const h = file ? defHeight(info) : (top ? top.height : defHeight(info));
  const fps = file || !top || top.fps == null ? info.fps : top.fps;
  const parts = [`<span class="fact site">${file ? '📁 ' : ''}${esc(info.site || '')}</span>`];
  if (h) parts.push(`<span class="fact ${num(fps) > 31 ? 'fps60' : ''}"><b>${file ? '' : '最高 '}${h}p${fps != null ? fpsText(fps) : ''}</b></span>`);
  const codecs = [info.vcodec, info.acodec].filter(Boolean);
  if (codecs.length) parts.push(`<span class="fact">${codecs.map((c) => `<b>${esc(c)}</b>`).join(' / ')}</span>`);
  if (info.hls) parts.push('<span class="fact">HLS</span>');
  if (num(info.headers_count) > 0) parts.push(`<span class="fact hdr" title="從 cURL 帶入的標頭會用在下載和播放">🔒 標頭 ×${num(info.headers_count)}</span>`);
  return parts.join('');
}

function headHtml() {
  const items = okEntries();
  if (isSingle()) {
    const info = items[0].info;
    const d = info.dupe;
    const dupe = d ? `<div class="dupe"><span>這部影片<b>已在佇列</b>（${esc(STATE_LABEL[d.state] || d.state || '')}${d.progress != null ? ' ' + Math.round(num(d.progress) * (num(d.progress) <= 1 ? 100 : 1)) + '%' : ''}）</span><a class="link" href="#" data-a="dupe">前往項目</a></div>` : '';
    return `<div class="ihead">${thumbHtml(info)}
      <div class="imeta">
        <h3 class="ititle" title="${esc(info.title || '')}">${esc(info.title || basename(info.target))}</h3>
        <div class="facts">${num(info.duration) > 0 ? `<span class="fact">長度 <b>${fmtDur(info.duration)}</b></span>` : ''}${factsHtml(info)}</div>
        ${dupe}
      </div>
      <button class="btn ghost sm icon-only iclose" type="button" title="關閉" aria-label="關閉" data-a="close">✕</button></div>${skippedHtml()}`;
  }
  const total = items.reduce((a, e) => a + num(e.info.duration), 0);
  const failed = ip.entries.filter((e) => e.state === 'err').length;
  return `<div class="bhead"><h3>${ip.entries.length} 部影片</h3><span class="note">${total > 0 ? `總長 ${fmtDur(total)} · ` : ''}${failed ? `${failed} 部解析失敗 · ` : ''}多部時只能加入佇列</span>
      <button class="btn ghost sm icon-only iclose" type="button" title="關閉" aria-label="關閉" data-a="close">✕</button></div>
    <div class="blist">${ip.entries.map((e, i) => {
      if (e.state !== 'ok') {
        return `<div class="brow err"><div class="thumb"><span class="nothumb">×</span></div>
          <div style="min-width:0"><div class="bt" title="${esc(e.target)}">${esc(e.target || e.text)}</div><div class="bm">${e.state === 'busy' ? '<span class="fact">解析中…</span>' : `<span class="fact err" title="${esc(e.detail)}">解析失敗：${esc(e.error)}</span>`}</div></div>
          <span class="eta">${e.state === 'busy' ? '' : `<a class="link" href="#" data-a="retry${i}">重試</a>`}</span>
          <button class="btn ghost sm icon-only" type="button" title="從這批移除" aria-label="從這批移除" data-a="rm${i}">✕</button></div>`;
      }
      const info = e.info;
      return `<div class="brow">${thumbHtml(info, true)}<div style="min-width:0"><div class="bt" title="${esc(info.title || '')}">${esc(info.title || basename(info.target))}</div>
        <div class="bm">${factsHtml(info)}${info.dupe ? '<span class="fact dupe-tag">已在佇列</span>' : ''}</div></div>
        <span class="eta" data-tok="${esc(info.token)}">${etaText(info.token)}</span>
        <button class="btn ghost sm icon-only" type="button" title="從這批移除" aria-label="從這批移除" data-a="rm${i}">✕</button></div>`;
    }).join('')}</div>${skippedHtml()}`;
}

function skippedHtml() {
  return ip.skipped && ip.skipped.length ? `<div class="ip-note">略過：${ip.skipped.map(esc).join('；')}</div>` : '';
}

function etaText(token) {
  if (!profileKnown()) return '—';
  const it = estOf(token);
  return it && it.queue && it.queue.eta_s != null ? `約 ${fmtEtaZh(it.queue.eta_s)}` : '—';
}

function settingsHtml() {
  const items = okEntries(), single = isSingle();
  const it0 = items[0].info, D = ip.defaults, cur = ip.cur;
  const changed = Object.keys(activeCur());
  const advChanged = changed.filter((k) => ['detection_model', 'restoration_model', 'encoder', 'save_compare'].includes(k)).length;
  const mark = (k) => (changed.includes(k) ? 'changed' : '');
  const est1 = single ? estOf(it0.token) : null;
  const byH = est1 && est1.stream_by_height ? est1.stream_by_height : null;
  let rows = '';

  if (showHeight()) {
    const h = curHeight();
    const sel = heightsOf(it0).find((x) => x.height === h);
    rows += `<div class="irow ${mark('height')}"><span class="nm">解析度</span><div class="ctl">${seg('ip_h', heightsOf(it0).map((x) => {
      const v = byH && profileKnown() ? byH[String(x.height)] : null;
      const dot = v ? `<i class="vd ${VERDICT_CLS[v] || ''}" title="串流：${VERDICT_TXT[v] || ''}"></i>` : '';
      return { v: x.height, html: `${dot}${x.height}p${num(x.fps) > 31 ? `<small>${Math.round(num(x.fps))}</small>` : ''}` };
    }), h)}${single && sel && num(sel.size_bytes) > 0 ? `<span class="note">約 ${fmtBytes(sel.size_bytes)}</span>` : ''}</div></div>`;
  }
  if (hasLow()) {
    const lows = items.filter((e) => !isHigh(e.info)).length;
    const v = pick('interpolation', 'queue');
    const applies = items.length > 1 && hasHigh() ? `<span class="applies">30fps 的 ${lows} 部</span>` : '';
    rows += `<div class="irow ${mark('interpolation')}"><span class="nm"><span>插幀${applies}</span></span><div class="ctl">
      ${seg('ip_interp', [{ v: 'off', html: '關' }, { v: 'auto', html: '自動<small>→60</small>' }], v)}
      <span class="note">${v === 'auto' ? (single ? '<em>串流</em>：效能夠時才插 · <em>佇列</em>：每一格都插' : '每一格都插成 60 fps') : '維持原幀率，處理較快'}</span></div></div>`;
  }
  if (single && isHigh(it0)) {
    const v = pick('restoration_fps', 'stream');
    rows += `<div class="irow ${mark('restoration_fps')}"><span class="nm">修復幀率</span><div class="ctl">
      ${seg('ip_rfps', [{ v: 'original', html: `原始 ${fpsText(Math.round(num(it0.fps) * 100) / 100)}` }, { v: '30', html: '30' }], v)}
      <span class="tagonly">僅串流</span><span class="note">${v === '30' ? '以 30 fps 修復，速度約快一倍；效能夠時自動補回 60' : '每一格都修復'}</span></div></div>`;
  }
  const dets = (D.choices && D.choices.detection_model) || [];
  let detRow = '';
  if (dets.length) {                       // stream and queue share the setting-dialog value, so there is one default
    const detVal = pick('detection_model', 'queue');
    const opts = dets.map((o) => ({ v: o.value, html: esc(DET_LABEL[o.value] || String(o.label || o.value).replace(/（建議）$/, '')) }));
    detRow = `<div class="irow ${mark('detection_model')}"><span class="nm">偵測模型</span><div class="ctl">
      ${seg('ip_det', opts, detVal)}
      <span class="note">${esc(DET_NOTE[detVal] || '')}</span></div></div>`;
  }

  const subtitleEnv = D.subtitle_environment || {};
  const subtitleAvailable = subtitleRuntimeAvailable(subtitleEnv);
  const subtitleEnabled = !!pick('subtitles', 'queue');
  const subtitleReason = subtitleRuntimeReason(subtitleEnv);
  rows += `<div class="irow ${mark('subtitles')} subtitle-intake-row"><span class="nm">產生字幕</span><div class="ctl">
    <label class="subtitle-option"><input type="checkbox" data-k="subtitles" aria-label="產生字幕" ${subtitleEnabled ? 'checked' : ''} ${subtitleAvailable ? '' : 'disabled'}><span>${single ? '串流與加入佇列都產生字幕' : '加入佇列時產生字幕'}</span></label>
    <span class="note">${subtitleAvailable ? '使用設定中的 ASR 與翻譯模型' : esc(subtitleReason)}</span>
    ${single && subtitleAvailable && subtitleEnabled && subtitleStreamDoubles(it0) ? `<span class="note traffic-note" role="note">${esc(SINGLE_QUALITY_NOTE)}</span>` : ''}
    ${single && subtitleAvailable && subtitleEnabled && subtitlePrimaryIsLocal(current && current.settings) && !subtitleGpuOf(it0) ? `<span class="note traffic-note" role="note" id="ipStreamLocalNote">${esc(STREAM_LOCAL_SUBTITLE_NOTE)}</span>` : ''}</div></div>`;

  if (subtitleAvailable && subtitleEnabled && subtitleComparisonSupported()) {
    const comparisonOn = cur.subtitle_comparison_enabled !== undefined ? !!cur.subtitle_comparison_enabled : subtitleComparisonDefault();
    rows += `<div class="irow ${mark('subtitle_comparison_enabled')} subtitle-intake-row subtitle-comparison-intake"><span class="nm">多字幕比較</span><div class="ctl">
      <label class="subtitle-option"><input type="checkbox" data-k="subtitle_comparison_enabled" aria-label="這部影片產生額外字幕軌" ${comparisonOn ? 'checked' : ''}><span>這部影片產生額外字幕軌</span></label>
      ${single ? '<span class="tagonly">僅佇列</span>' : ''}<span class="note">${esc(subtitleComparisonSummary(current && current.settings))}；主字幕與預設播放不變。</span></div></div>`;
  }

  const rm = pick('restoration_model', 'queue');
  const rmOpts = ((D.choices && D.choices.restoration_model) || []).map((o) => ({ value: o.value, label: o.label || o.value }));
  if (rm && !rmOpts.some((o) => o.value === rm)) rmOpts.push({ value: rm, label: rm });
  const enc = pick('encoder', 'queue'), cmp = !!pick('save_compare', 'queue');
  const dirty = changed.length
    ? `<span class="dirty">已調整 <b>${changed.length}</b> 項，只套用${single ? '這部' : '這批'} · <a class="link" href="#" data-a="savedef">設為預設</a><a class="link" href="#" data-a="reset">還原</a></span>`
    : '<span class="dirty">使用已儲存的設定</span>';
  return `<div class="iset">${rows}
    <div class="ibar">
      <button class="adv-toggle" type="button" aria-expanded="${ip.advOpen}" data-a="adv"><i class="car"></i>進階${advChanged ? `<span class="cnt">${advChanged}</span>` : ''}
        <span class="sum">${esc(detLabel(pick('detection_model', 'queue')))} · ${esc(choiceLabel('restoration_model', rm))} · ${esc(ENC_LABEL[enc] || String(enc).toUpperCase())} · ${cmp ? '含比對軌' : '無比對軌'}</span></button>
      ${dirty}
    </div>
    <div class="adv" ${ip.advOpen ? '' : 'hidden'}>
      ${detRow}
      <div class="irow ${mark('restoration_model')}"><span class="nm">修復模型</span><div class="ctl">
        <select class="model-select" data-k="restoration_model" aria-label="修復模型">${rmOpts.map((o) => `<option value="${esc(o.value)}" ${o.value === rm ? 'selected' : ''}>${esc(o.label)}</option>`).join('')}</select></div></div>
      <div class="irow ${mark('encoder')}"><span class="nm">輸出編碼</span><div class="ctl">
        ${seg('ip_enc', [{ v: 'hevc', html: 'HEVC' }, { v: 'h264', html: 'H.264' }], enc)}
        ${single ? '<span class="tagonly">僅佇列</span>' : ''}<span class="note">HEVC 檔案較小；H.264 相容性較好</span></div></div>
      <div class="irow ${mark('save_compare')}"><span class="nm">比對軌</span><div class="ctl">
        <label class="switch"><input type="checkbox" data-k="save_compare" aria-label="比對軌" ${cmp ? 'checked' : ''}><span></span></label>
        ${single ? '<span class="tagonly">僅佇列</span>' : ''}<span class="note">另存原始畫面，播放時可切換對照（檔案約大 50%）</span></div></div>
    </div></div>`;
}

const QUEUE_ICON = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 5v14M5 12h14" stroke="currentColor" stroke-width="2.500" fill="none"/></svg>';
const STREAM_ICON = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M7 5l12 7-12 7z" fill="currentColor" stroke="none"/></svg>';

function queueBtn(n) {
  return `<button class="btn primary" type="button" data-a="queue" ${ip.addBusy ? 'disabled' : ''}>${QUEUE_ICON}${n > 1 ? `全部加入佇列（${n}）` : '加入佇列'}</button>`;
}

function queuePosText() {
  const pos = ip.est && ip.est.queue_position;
  const ahead = ip.est && ip.est.queue_ahead_s;
  if (!(pos > 1)) return '排在最前面';
  return `排第 <span class="num">${pos}</span> 位${num(ahead) > 0 ? `，前面約 ${fmtEtaZh(ahead)}` : ''}`;
}

function estNone(l1, l2) { return `<div class="est none"><div class="l1"><i class="vd"></i>${l1}</div><div class="l2">${l2}</div></div>`; }

function actsHtml() {
  const items = okEntries();
  const loading = !ip.est;
  const failed = !!(ip.est && ip.est.failed);
  const profile = profileKnown();
  const perfLink = perfLast ? ' · <a class="link" href="#" data-a="perf">立即檢測（約 3 分）</a>' : '';
  const hasUrl = items.some((e) => e.info.kind === 'url');
  if (!isSingle()) {
    const qInterp = pick('interpolation', 'queue') === 'auto' && hasLow();
    let est;
    if (loading) est = estNone('估算中…', '正在預估處理時間');
    else if (failed) est = estNone('無法預估', '預估失敗；仍可加入佇列');
    else if (!profile) est = estNone('尚未做效能檢測', `總長 ${fmtDur(items.reduce((a, e) => a + num(e.info.duration), 0))}，完成時間未知${perfLink}`);
    else {
      const etas = items.map((e) => { const it = estOf(e.info.token); return it && it.queue ? num(it.queue.eta_s) : null; });
      const tot = etas.reduce((a, b) => a + (b || 0), 0);
      est = `<div class="est live"><div class="l1"><i class="vd live"></i>全部處理約 ${fmtEtaZh(tot)}</div><div class="l2">${esc(detLabel(pick('detection_model', 'queue')))}${qInterp ? ' · 30fps 的每格插幀' : ''} · ${queuePosText()}${hasUrl ? ' · 不含下載' : ''}</div></div>`;
    }
    return `<div class="act">${queueBtn(items.length)}${est}</div>`;
  }
  const info = items[0].info, it = estOf(info.token);
  const sDet = detLabel((it && it.stream && it.stream.detector) || pick('detection_model', 'stream'));
  const qDet = detLabel(pick('detection_model', 'queue'));
  let sEst, qEst;
  if (loading) { sEst = estNone('估算中…', '正在預估串流速度'); qEst = estNone('估算中…', '正在預估處理時間'); }
  else if (failed || !it) { sEst = estNone('無法預估', '預估失敗；仍可串流播放'); qEst = estNone('無法預估', '預估失敗；仍可加入佇列'); }
  else if (!profile) {
    sEst = estNone('尚未做效能檢測', `無法預估速度${perfLink}`);
    qEst = estNone('尚未做效能檢測', `長度 ${num(info.duration) > 0 ? fmtDur(info.duration) : '未知'}，完成時間未知`);
  } else {
    const s = it.stream;
    if (!s) sEst = estNone('無法預估', '這個來源沒有串流預估');
    else {
      const v = VERDICT_CLS[s.verdict] || 'slow';
      const label = { live: '可即時播放', prebuffer: `需先預載約 ${Math.round(num(s.prebuffer_s))} 秒`, queue: '這台電腦太慢，建議加入佇列' }[s.verdict] || '';
      let tip = '';
      if (s.verdict !== 'live' && s.suggest) {
        if (s.suggest.restoration_fps) tip = `<div class="tip">修復幀率改 ${esc(s.suggest.restoration_fps)} 可即時播放<a class="link" href="#" data-a="sug" data-k="restoration_fps" data-v="${esc(s.suggest.restoration_fps)}">套用</a></div>`;
        else if (s.suggest.height) tip = `<div class="tip">改 ${num(s.suggest.height)}p 可即時播放<a class="link" href="#" data-a="sug" data-k="height" data-v="${num(s.suggest.height)}">套用</a></div>`;
      }
      const sg = subtitleGpuNotice(s.subtitle_gpu);
      if (sg) {
        const links = sg.actions.map((a) => (a.kind === 'queue' ? '<a class="link" href="#" data-a="queue">加入佇列</a>' : esc(a.label))).join('／');
        tip += `<div class="tip" id="ipSubtitleGpuNote" role="note">${esc(sg.lead)}${sg.actions.length ? `，建議：${links}` : ''}</div>`;
      }
      const interp = s.interp_share != null
        ? (num(s.interp_share) >= 0.05 ? ` · 約 <span class="num">${Math.round(num(s.interp_share) * 100)}%</span> 時間插成 60` : ' · 效能不足，幾乎不會插幀') : '';
      sEst = `<div class="est ${v}"><div class="l1"><i class="vd ${v}"></i>${label}</div>
        <div class="l2">修復 <span class="num">${fpsText(s.restore_fps)}</span> fps · 需要 <span class="num">${fpsText(s.need_fps)}</span> fps（<span class="num">${num(s.ratio).toFixed(2)}×</span>）${interp} · ${esc(sDet)}</div>${tip}</div>`;
    }
    const q = it.queue;
    if (!q) qEst = estNone('無法預估', '這個來源沒有佇列預估');
    else {
      const ok = num(q.fps) >= num(info.fps) ? 'live' : 'pre';
      const qInterp = !isHigh(info) && pick('interpolation', 'queue') === 'auto';
      qEst = `<div class="est ${ok}"><div class="l1"><i class="vd ${ok}"></i>處理約 ${fmtEtaZh(q.eta_s)}</div>
        <div class="l2">預估 <span class="num">${fpsText(q.fps)}</span> fps${qInterp ? '（每格插成 60）' : ''} · ${esc(qDet)} · ${queuePosText()}${info.kind === 'url' ? ' · 不含下載' : ''}</div></div>`;
    }
  }
  return `<div class="act ${ip.sentStream ? 'sent' : ''}">
      <button class="btn go-stream" type="button" data-a="stream" ${ip.streamBusy ? 'disabled' : ''}>${STREAM_ICON}串流播放</button>
      ${sEst}<div class="after">已送至播放器 · 仍可加入佇列</div>
    </div>
    <div class="act">${queueBtn(1)}${qEst}</div>`;
}

function resolvingHtml() {
  return `<div class="resolving">
      <div class="rline"><span class="stage"></span><span class="src"></span><button class="btn sm" type="button" data-a="cancel">取消</button></div>
      <div class="rscan">${'<i></i>'.repeat(40)}</div>
      <div class="rsteps">${IP_STEPS.map((t) => `<span>${t}</span>`).join('')}</div>
      <div class="skel"><div class="sk-b" style="aspect-ratio:16/9"></div><div class="sk-t"><div class="sk-b" style="height:18px;width:80%"></div><div class="sk-b" style="height:18px;width:55%"></div><div class="sk-b" style="height:22px;width:70%"></div></div></div>
    </div>`;
}

function errorHtml() {
  const bad = ip.entries.filter((e) => e.state === 'err');
  const one = bad.length === 1 && ip.entries.length === 1;
  const body = one
    ? `<div class="ierr-msg">${esc(bad[0].error)}</div>${bad[0].detail ? `<pre>${esc(bad[0].detail)}</pre>` : ''}`
    : `<pre>${bad.map((e) => `${esc(e.target || e.text)}\n  ${esc(e.error)}`).join('\n')}</pre>`;
  const url = bad.some((e) => e.kind !== 'file');
  return `<div class="ierr">
      <h3>${one ? '解析失敗' : `解析失敗（${bad.length} 部）`}</h3>
      ${body}
      ${url ? '<div class="ierr-tip">網站可能需要登入或標頭。可以從瀏覽器開發者工具「複製為 cURL」貼上，會一併帶入 Cookie。</div>' : ''}
      <div class="row"><button class="btn primary sm" type="button" data-a="retry">重試</button><button class="btn ghost sm" type="button" data-a="close">關閉</button></div>
    </div>`;
}

/* ---- rendering ---- */
function focusKey() {
  const a = document.activeElement;
  if (!a || !intake.contains(a)) return null;
  if (a.name) return `[name="${a.name}"][value="${a.value}"]`;
  if (a.dataset.k) return `[data-k="${a.dataset.k}"]`;
  if (a.dataset.a) return `[data-a="${a.dataset.a}"]${a.dataset.k ? `[data-k="${a.dataset.k}"]` : ''}`;
  return null;
}
function restoreFocus(key) {
  if (!key) return;
  const n = intake.querySelector(key);
  if (n && typeof n.focus === 'function') n.focus({ preventScroll: true });
}
function setRegion(name, html) {
  const r = intake.querySelector(`[data-r="${name}"]`);
  if (r) r.innerHTML = html;
}

function renderIntake() {
  stopScan();
  if (!ip) { intake.hidden = true; intake.replaceChildren(); intake.classList.remove('stale'); return; }
  const key = focusKey();
  intake.hidden = false;
  if (ip.phase === 'resolving') { intake.innerHTML = resolvingHtml(); updateResolving(); startScan(); }
  else if (ip.phase === 'error') intake.innerHTML = errorHtml();
  else {
    intake.innerHTML = `<div data-r="head">${headHtml()}</div><div data-r="set">${settingsHtml()}</div>
      <div class="iact ${isSingle() ? '' : 'single'}" data-r="act">${actsHtml()}</div>
      <div class="notice warn stale-bar"><span>網址已變更 — 按「解析」重新取得資訊</span><button class="btn sm ghost" type="button" data-a="close">關閉</button></div>`;
  }
  syncStale();
  restoreFocus(key);
}

// settings changed or an estimate arrived: refresh the parts that depend on it without rebuilding the batch list
function renderReady(full) {
  if (!ip || ip.phase !== 'ready') return;
  const key = focusKey();
  const a = document.activeElement;
  const selectOpen = a && a.tagName === 'SELECT' && intake.contains(a);
  if (full) setRegion('head', headHtml());
  else for (const n of intake.querySelectorAll('.brow .eta[data-tok]')) setText(n, etaText(n.dataset.tok));
  if (!selectOpen) setRegion('set', settingsHtml());
  const act = intake.querySelector('[data-r="act"]');
  if (act) { act.className = 'iact ' + (isSingle() ? '' : 'single'); act.innerHTML = actsHtml(); }
  syncStale();
  restoreFocus(key);
}

function stopScan() { clearInterval(scanTimer); scanTimer = null; }
function startScan() {
  let p = 0;
  stopScan();
  scanTimer = setInterval(() => {
    const cells = intake.querySelectorAll('.rscan i');
    if (!cells.length) { stopScan(); return; }
    cells.forEach((c, i) => { c.className = i === p ? 'on' : (i < p && i >= p - 4) ? 'trail' : ''; });
    p = (p + 1) % 44;
    updateResolving();
  }, 45);
}
function updateResolving() {
  if (!ip || ip.phase !== 'resolving') return;
  const secs = (Date.now() - ip.t0) / 1000;
  let step = 0;
  IP_STEP_AT.forEach((t, i) => { if (secs >= t) step = i; });
  const n = ip.entries.length, done = ip.entries.filter((e) => e.state === 'ok' || e.state === 'err').length;
  const stage = intake.querySelector('.stage');
  if (stage) setText(stage, n > 1 ? `解析中 ${Math.min(n, done + 1)} / ${n}…` : IP_STAGE[step]);
  const busy = ip.entries.find((e) => e.state === 'busy') || ip.entries[0];
  const src = intake.querySelector('.rline .src');
  if (src && busy) setText(src, String(busy.target || busy.text).split('\n')[0]);
  intake.querySelectorAll('.rsteps span').forEach((s, i) => {
    const cls = i < step ? 'done' : i === step ? 'now' : '';
    if (s.className !== cls) s.className = cls;
  });
}

function syncStale() {
  const stale = !!ip && ip.phase !== 'resolving' && addInput.value.trim() !== ip.inputText;
  intake.classList.toggle('stale', stale);
  for (const c of intake.children) {
    if (c.classList.contains('stale-bar')) continue;
    if (stale) c.setAttribute('inert', ''); else c.removeAttribute('inert');
  }
}

/* ---- life cycle ---- */
function closeIntake(rescan) {
  ipRev++; estRev++;
  clearTimeout(estTimer);
  ip = null;
  renderIntake();
  if (rescan) scheduleSourceCheck();
}

async function submitResolve() {
  const text = addInput.value.trim();
  if (!text) return;
  clearTimeout(sourceTimer); sourceRevision++;
  setEntryStatus('');
  const rev = ++ipRev;
  estRev++; clearTimeout(estTimer);
  const res = await api('parse_inputs', text);
  if (rev !== ipRev) return;
  if (!res || res.ok === false) { ip = null; renderIntake(); return; }       // api() already toasted the error
  if (!Array.isArray(res.entries) || !res.entries.length) {
    ip = null; renderIntake();
    setSourceStatus(Array.isArray(res.errors) && res.errors.length ? res.errors.join('；') : '沒有可解析的內容', 'error');
    return;
  }
  ip = {
    rev, inputText: text, phase: 'resolving', t0: Date.now(), defaults: null, est: null, cur: {}, advOpen: false,
    sentStream: false, streamBusy: false, addBusy: false, skipped: Array.isArray(res.errors) ? res.errors.map(String) : [],
    entries: res.entries.map((e) => ({ text: String(e.text), kind: e.kind === 'file' ? 'file' : 'url', target: String(e.target || ''),
      headers_count: num(e.headers_count), state: 'wait', info: null, error: '', detail: '' })),
  };
  startResolve(ip, ip.entries);
}

function startResolve(p, todo) {
  p.phase = 'resolving'; p.t0 = Date.now(); p.est = null;
  for (const e of todo) { e.state = 'wait'; e.info = null; e.error = ''; e.detail = ''; }
  setSourceStatus('解析中…', 'pending');
  const defsP = Promise.resolve().then(() => backend.defaults()).catch(() => null);
  renderIntake();
  let next = 0;
  const work = async () => {
    while (next < todo.length) {
      const e = todo[next++];
      await resolveEntry(p, e);
      if (p.rev !== ipRev || ip !== p) return;
      updateResolving();
    }
  };
  Promise.all(Array.from({ length: Math.min(IP_CONCURRENCY, todo.length) }, work)).then(() => finishResolve(p, defsP));
}

async function resolveEntry(p, e) {
  e.state = 'busy';
  try {
    const r = await Promise.resolve().then(() => backend.resolve(e.text));
    if (p.rev !== ipRev || ip !== p) return;
    if (r && r.ok && r.info) { e.state = 'ok'; e.info = r.info; }
    else { e.state = 'err'; e.error = String((r && r.error) || '解析失敗'); e.detail = String((r && r.detail) || ''); }
  } catch (err) {
    e.state = 'err'; e.error = '解析失敗'; e.detail = String(err && err.message ? err.message : err);
  }
}

async function finishResolve(p, defsP) {
  if (p.rev !== ipRev || ip !== p) return;
  if (!okEntries().length) { p.phase = 'error'; renderIntake(); setSourceStatus('解析失敗', 'error'); return; }
  const defs = await defsP;
  if (p.rev !== ipRev || ip !== p) return;
  if (!defs || !defs.stream || !defs.queue) {
    for (const e of p.entries) if (e.state === 'ok') { e.state = 'err'; e.error = '無法取得已儲存的設定'; e.detail = ''; }
    p.phase = 'error'; renderIntake(); setSourceStatus('解析失敗', 'error'); return;
  }
  p.defaults = defs;
  p.phase = 'ready';
  normalizeCur();
  const n = p.entries.length;
  setSourceStatus(n > 1 ? `已解析 ${okEntries().length} / ${n} 部` : (p.entries[0].kind === 'file' ? '本機檔案' : '已解析'), 'success');
  renderIntake();
  scheduleEstimate(0);
}

function scheduleEstimate(delay = 150) {
  clearTimeout(estTimer);
  if (!ip || ip.phase !== 'ready') return;
  const p = ip, rev = ++estRev;
  estTimer = setTimeout(async () => {
    const tokens = okEntries().map((e) => e.info.token);
    if (!tokens.length) return;
    let res = null;
    try { res = await Promise.resolve().then(() => backend.estimate(tokens, sendOverrides())); } catch (err) { res = null; }
    if (rev !== estRev || p !== ip || p.phase !== 'ready') return;
    p.est = res && Array.isArray(res.items) ? res : { failed: true, items: [] };
    renderReady(false);
  }, delay);
}

function settingsChanged() {
  if (!ip) return;
  ip.sentStream = false;
  normalizeCur();
  renderReady(false);
  scheduleEstimate();
}

/* ---- panel events ---- */
intake.addEventListener('change', (e) => {
  if (!ip || ip.phase !== 'ready') return;
  const t = e.target, cur = ip.cur;
  if (t.name === 'ip_h') cur.height = +t.value;
  else if (t.name === 'ip_interp') cur.interpolation = t.value;
  else if (t.name === 'ip_det') cur.detection_model = t.value;
  else if (t.name === 'ip_rfps') cur.restoration_fps = t.value;
  else if (t.name === 'ip_enc') cur.encoder = t.value;
  else if (t.dataset.k === 'restoration_model') cur.restoration_model = t.value;
  else if (t.dataset.k === 'save_compare') cur.save_compare = t.checked;
  else if (t.dataset.k === 'subtitles') cur.subtitles = t.checked;
  else if (t.dataset.k === 'subtitle_comparison_enabled') cur.subtitle_comparison_enabled = t.checked;
  else return;
  settingsChanged();
});

intake.addEventListener('click', (e) => {
  const a = e.target.closest('[data-a]');
  if (!a || !ip) return;
  e.preventDefault();
  if (intake.classList.contains('stale') && a.dataset.a !== 'close' && a.dataset.a !== 'cancel') return;
  const act = a.dataset.a;
  if (act === 'close' || act === 'cancel') { closeIntake(true); setEntryStatus(''); }
  else if (act === 'adv') { ip.advOpen = !ip.advOpen; renderReady(false); }
  else if (act === 'reset') { ip.cur = {}; settingsChanged(); }
  else if (act === 'savedef') ipSaveDefaults();
  else if (act === 'sug') {
    ip.cur[a.dataset.k] = a.dataset.k === 'height' ? +a.dataset.v : a.dataset.v;
    settingsChanged();
  } else if (act === 'perf') {
    setPerfOpen(true);
    if (perfLast && perfLast.can_start !== false && perfLast.status !== 'running') perfStart();
    $('#perfPanel').scrollIntoView({ block: 'nearest' });
  } else if (act === 'dupe') ipGotoDupe();
  else if (act === 'stream') ipStream();
  else if (act === 'queue') ipQueue();
  else if (act === 'retry') {
    const bad = ip.entries.filter((x) => x.state === 'err');
    if (bad.length) startResolve(ip, bad);
  } else if (/^retry\d+$/.test(act)) {
    const en = ip.entries[+act.slice(5)];
    if (en) retryOne(en);
  } else if (/^rm\d+$/.test(act)) ipRemove(+act.slice(2));
});

async function retryOne(en) {
  const p = ip;
  en.state = 'busy'; en.error = ''; en.detail = '';
  renderReady(true);
  await resolveEntry(p, en);
  if (ip !== p) return;
  renderReady(true);
  scheduleEstimate(0);
}

function ipRemove(i) {
  const en = ip.entries[i];
  if (!en) return;
  ip.entries.splice(i, 1);
  // drop the same line from the add box so the panel does not turn stale
  const lines = addInput.value.split(/\r?\n/);
  const at = lines.findIndex((l) => l.trim() === en.text.trim());
  if (at >= 0) { lines.splice(at, 1); addInput.value = lines.join('\n'); autoGrow(); scheduleSourceCheck(); }
  ip.inputText = addInput.value.trim();
  if (!ip.entries.length) { closeIntake(true); return; }
  if (!okEntries().length) { ip.phase = 'error'; renderIntake(); return; }
  ip.est = null;
  normalizeCur();
  renderIntake();
  scheduleEstimate(0);
}

function ipGotoDupe() {
  const info = okEntries()[0] && okEntries()[0].info;
  const c = info && info.dupe && cards.get(info.dupe.id);
  if (!c) { toast('在佇列中找不到這個項目'); return; }
  c.root.scrollIntoView({ block: 'center', behavior: 'smooth' });
  c.root.classList.add('ip-flash');
  setTimeout(() => c.root.classList.remove('ip-flash'), 1800);
}

async function ipStream() {
  const p = ip;
  if (!p || p.phase !== 'ready' || !isSingle() || p.streamBusy || !okEntries().length) return;
  p.streamBusy = true;
  renderReady(false);
  setEntryStatus('正在開啟播放器…', 'pending');
  const overrides = sendOverrides();
  // streaming now carries the subtitle switch (T3.4); the extra comparison tracks stay queue-only
  delete overrides.subtitle_comparison_enabled;
  const res = await api('stream_resolved', okEntries()[0].info.token, overrides);
  p.streamBusy = false;
  if (!res || res.ok !== true) setEntryStatus(String((res && (res.error || res.message)) || '無法開啟播放器。'), 'error');
  else { p.sentStream = true; setEntryStatus(String(res.message || '播放器已開啟；正在緩衝並啟動 AI 修復。'), 'success'); }
  if (ip === p) renderReady(false);
}

async function ipQueue() {
  const p = ip;
  if (!p || p.phase !== 'ready' || p.addBusy) return;
  const tokens = okEntries().map((e) => e.info.token);
  if (!tokens.length) return;
  p.addBusy = true;
  renderReady(false);
  const res = await api('add_resolved', tokens, sendOverrides());
  p.addBusy = false;
  const errs = res && Array.isArray(res.errors) ? res.errors : [];
  for (const m of errs) toast(String(m), true);
  const added = res && res.ok !== false && Array.isArray(res.ids) && res.ids.length > 0;
  if (added && ip === p) {
    closeIntake(false);
    addInput.value = ''; autoGrow(); scheduleSourceCheck();
    setEntryStatus('');
  } else if (ip === p) renderReady(false);
  refresh();
}

async function ipSaveDefaults() {
  const p = ip;
  if (!p || p.phase !== 'ready') return;
  const o = activeCur();
  const comparisonEnabled = o.subtitle_comparison_enabled;
  delete o.subtitle_comparison_enabled;      // saved with the complete subtitle configuration
  const res = await api('save_defaults', o);
  if (!res || res.ok === false || res.error) return;
  if (comparisonEnabled !== undefined && subtitleComparisonSupported()) {
    const settings = current && current.settings || {};
    const r2 = await api('set_subtitle_configuration', settings.subtitle_primary, comparisonEnabled, settings.subtitle_comparison_rows || []);
    if (r2 && r2.ok === false) return;
    delete p.cur.subtitle_comparison_enabled;
  }
  if (!res.message) toast('已設為預設');
  const defs = await Promise.resolve().then(() => backend.defaults()).catch(() => null);
  if (ip !== p) return;
  if (defs && defs.stream && defs.queue) p.defaults = defs;
  for (const k of Object.keys(o)) if (k !== 'height') delete p.cur[k];
  normalizeCur();
  renderReady(false);
  scheduleEstimate(0);
  refresh();
}

/* ---- header buttons ---- */
let lastOpened = { key: '', at: 0 };
function openText(text) {
  text = String(text || '').trim();
  if (!text) return;
  addInput.value = text; autoGrow();
  submitResolve();
}
// paths (加入檔案, dropped files; also callable by the host as window.ohStreamer.openPaths): one panel for 1 or many files
function openPaths(paths) {
  paths = (Array.isArray(paths) ? paths : [paths]).map((p) => String(p || '').trim()).filter(Boolean);
  if (!paths.length) return;
  const key = paths.join('\n'), now = Date.now();
  if (key === lastOpened.key && now - lastOpened.at < 1500) return;      // the same drop reported by both the page and the host
  lastOpened = { key, at: now };
  openText(paths.join('\n'));
}
// Ctrl/Cmd+V anywhere outside an editable field or open dialog -> the pasted text goes through the add box (same as typing it and pressing Enter)
function pasteTarget(el) {
  return !!(el && el.closest && el.closest('input, textarea, select, [contenteditable]:not([contenteditable="false"])'));
}
function pasteToAddBox(e) {
  if (e.defaultPrevented || pasteTarget(e.target) || pasteTarget(document.activeElement)) return;
  if (document.querySelector('.modal:not([hidden]), dialog[open], #drawer:not([hidden])')) return;
  const text = String((e.clipboardData && e.clipboardData.getData('text/plain')) || '').trim();
  if (!text) return;                    // files only / empty clipboard: leave it alone
  e.preventDefault();
  openText(text);
}
document.addEventListener('paste', pasteToAddBox);
window.ohStreamer.openPaths = openPaths;
window.ohStreamer.onDrop = openPaths;

$('#btnAddFiles').addEventListener('click', async () => {
  const res = await api('add_files');
  if (res && Array.isArray(res.paths) && res.paths.length) openPaths(res.paths);
  else refresh();
});
$('#btnPauseAll').addEventListener('click', () => act(current && current.paused_all ? 'resume_all' : 'pause_all'));
$('#btnFolder').addEventListener('click', () => api('open_folder', null));

let dragDepth = 0;
const dropzone = $('#dropzone');
function isExternalDrag(e) {
  if (dragId != null || !e.dataTransfer) return false;
  const t = Array.from(e.dataTransfer.types || []);
  return t.includes('Files') || t.includes('text/plain') || t.includes('text/uri-list');
}
window.addEventListener('dragenter', (e) => {
  if (!isExternalDrag(e)) return;
  e.preventDefault();
  dragDepth++; dropzone.hidden = false;
});
window.addEventListener('dragover', (e) => {
  if (dragId != null) { e.preventDefault(); return; }   // keep drop cursor valid anywhere during reorder
  if (!isExternalDrag(e)) return;
  e.preventDefault(); e.dataTransfer.dropEffect = 'copy';
});
window.addEventListener('dragleave', (e) => {
  if (!isExternalDrag(e)) return;
  dragDepth = Math.max(0, dragDepth - 1);
  if (!dragDepth) dropzone.hidden = true;
});
window.addEventListener('drop', (e) => {
  if (dragId != null) { e.preventDefault(); return; }
  e.preventDefault();
  dragDepth = 0; dropzone.hidden = true;
  const dt = e.dataTransfer; if (!dt) return;
  const files = Array.from(dt.files || []);
  const uri = (dt.getData('text/uri-list') || '').split(/\r?\n/).map((s) => s.trim()).filter((s) => s && !s.startsWith('#')).join('\n');
  const text = (uri || dt.getData('text/plain') || '').trim();
  // pywebview fills in file.pywebviewFullPath from its own drop handler; give it a tick.
  setTimeout(() => {
    const paths = files.map((f) => f.pywebviewFullPath || f.path || '').filter(Boolean);
    if (paths.length) openPaths(paths);
    else if (files.length) { if (isMock) toast('無法取得檔案的完整路徑，請改用「加入檔案」按鈕', true); }   // real window: the host reports the paths via window.ohStreamer.openPaths
    else if (text) openText(text);
  }, 60);
});

/* ------------------------------------------------------------------ settings dialog (category rail + pane) */

const drawer = $('#drawer');            // the dialog element (id kept from the old drawer)
const scrim = $('#settingsScrim');
const SETTINGS_EXPANDED_KEY = 'ohStreamer.queue.settingsExpanded';
const lastEdit = {};          // key -> ms timestamp of last local edit
const EDIT_GRACE = 1500;
const DEFAULT_ON = new Set(['show_thumbnails', 'show_thumbnail_cover']);   // switches that are on while the backend has not sent the key

// 10x10 pixel glyphs of the category rail ('#' = filled); drawn as one SVG path, coloured by currentColor
const CAT_GLYPH = {
  proc: ['..########', '..#......#', '###.####.#', '#.#.#..#.#', '#.#.####.#', '#.#......#', '#.########', '#........#', '##########', '..........'],
  out: ['##########', '#.#....#.#', '##########', '#.#....#.#', '#.#....#.#', '#.#....#.#', '##########', '#.#....#.#', '##########', '..........'],
  store: ['.########.', '#........#', '.########.', '#........#', '#.......##', '.########.', '#........#', '#.......##', '.########.', '..........'],
  gen: ['.#....#...', '.#....#...', '###...#...', '.#...###..', '.#....#...', '.#....#...', '.#..#.#...', '.#.###....', '.#..#.....', '..........'],
  subtitles: ['..######..', '.#......#.', '#..####..#', '#.#....#.#', '#.#.##.#.#', '#.#.##.#.#', '#.#....#.#', '#..####..#', '.#......#.', '..######..'],
};
for (const holder of drawer.querySelectorAll('.pxg[data-glyph]')) {
  const rows = CAT_GLYPH[holder.dataset.glyph] || [];
  let d = '';
  rows.forEach((row, y) => { for (const m of row.matchAll(/#+/g)) d += `M${m.index} ${y}h${m[0].length}v1h-${m[0].length}z`; });
  holder.innerHTML = `<svg viewBox="0 0 10 10" width="20" height="20" shape-rendering="crispEdges" aria-hidden="true" focusable="false"><path d="${d}" fill="currentColor"/></svg>`;
}

const settingsTabs = [...drawer.querySelectorAll('.sdlg-rail [role="tab"]')];
let settingsCat = 'proc';               // last selected category (kept for the session)
function selectSettingsCat(cat, focus) {
  if (!settingsTabs.some((t) => t.dataset.cat === cat)) cat = 'proc';
  settingsCat = cat;
  for (const t of settingsTabs) {
    const on = t.dataset.cat === cat;
    t.setAttribute('aria-selected', String(on));
    t.tabIndex = on ? 0 : -1;
    if (on && focus) t.focus();
  }
  for (const pane of drawer.querySelectorAll('[role="tabpanel"][data-cat]')) setHidden(pane, pane.dataset.cat !== cat);
  $('.sdlg-pane', drawer).scrollTop = 0;
}
function setSettingsExpanded(expanded, persist = false) {
  drawer.classList.toggle('expanded', expanded);
  const button = $('#btnExpandSettings');
  const label = expanded ? '還原設定視窗' : '放大設定視窗';
  button.setAttribute('aria-pressed', String(expanded));
  button.title = label;
  if (persist) lsSet(SETTINGS_EXPANDED_KEY, expanded ? '1' : '0');
}
setSettingsExpanded(lsGet(SETTINGS_EXPANDED_KEY) === '1');
$('#btnExpandSettings').addEventListener('click', () => setSettingsExpanded(!drawer.classList.contains('expanded'), true));
for (const t of settingsTabs) t.addEventListener('click', () => selectSettingsCat(t.dataset.cat, false));
$('.sdlg-rail', drawer).addEventListener('keydown', (e) => {
  const i = settingsTabs.indexOf(document.activeElement);
  if (i < 0) return;
  let n = -1;
  if (e.key === 'ArrowDown' || e.key === 'ArrowRight') n = (i + 1) % settingsTabs.length;
  else if (e.key === 'ArrowUp' || e.key === 'ArrowLeft') n = (i - 1 + settingsTabs.length) % settingsTabs.length;
  else if (e.key === 'Home') n = 0;
  else if (e.key === 'End') n = settingsTabs.length - 1;
  if (n < 0) return;
  e.preventDefault();
  selectSettingsCat(settingsTabs[n].dataset.cat, true);
});

const settingsOpen = () => !drawer.hidden;
function openSettings() {
  mdl.autoTried = false;
  loadSubtitleModels();
  loadResetupRow();
  loadUpdateRow();
  selectSettingsCat(settingsCat, false);
  scrim.hidden = false; drawer.hidden = false; setAttr(drawer, 'aria-hidden', 'false');
  const t = settingsTabs.find((x) => x.dataset.cat === settingsCat);
  if (t) t.focus();
}
function closeSettings() {
  subtitleLocalPeek = false;
  scrim.hidden = true; drawer.hidden = true; setAttr(drawer, 'aria-hidden', 'true');
  $('#btnSettings').focus();
}
$('#btnSettings').addEventListener('click', () => (settingsOpen() ? closeSettings() : openSettings()));
$('#btnCloseSettings').addEventListener('click', closeSettings);
scrim.addEventListener('click', closeSettings);
$('#btnFolder2').addEventListener('click', () => api('open_folder', null));

// ---- 顯示卡版本 / 重新偵測：installed copies only (backend runtime_info().available) ----
async function loadResetupRow() {
  const row = $('#resetupRow'), infoEl = $('#resetupInfo'), btn = $('#btnResetup');
  if (!hasApi('runtime_info')) { row.hidden = true; return; }
  let info = null;
  try { info = await backend.runtime_info(); } catch (e) { info = null; }
  if (!info || typeof info !== 'object') { row.hidden = true; return; }
  row.hidden = false;
  btn.disabled = !info.available;
  setText(infoEl, info.available
    ? `目前：${info.label || '尚未選擇'}${info.verified ? '' : '（尚未通過檢查）'}。換顯示卡、或想改用其他版本時使用。`
    : (info.note || '開發環境不提供'));
}
$('#btnResetup').addEventListener('click', async () => {
  const busy = current && Array.isArray(current.items)
    && current.items.some((i) => i.state === 'processing' || i.state === 'downloading');
  const ok = await confirmDlg({
    title: '重新偵測／切換版本', ok: '關閉並開啟設定畫面',
    body: `${BRAND_NAME} 會先關閉，接著開啟設定畫面重新偵測顯示卡，並可切換版本；過程中可能需要下載檔案。`
      + (busy ? ' 注意：目前有工作正在處理或下載，關閉後會中斷，之後重新開啟時可從佇列繼續。' : ''),
  });
  if (!ok) return;
  const r = await api('resetup');
  if (r && r.ok) toast('正在開啟設定畫面…');
});

// ---- 程式版本 / 檢查更新：manual only (never contacts the network until the button is pressed) ----
let updateUrl = '';
async function loadUpdateRow() {
  const row = $('#updateRow');
  if (!hasApi('app_version') || !hasApi('check_update')) { row.hidden = true; return; }
  let v = '';
  try { v = await backend.app_version(); } catch (e) { v = ''; }
  row.hidden = false;
  setText($('#updateVersion'), v ? `目前版本：${v}` : '');
}
$('#btnCheckUpdate').addEventListener('click', async () => {
  const btn = $('#btnCheckUpdate'), res = $('#updateResult'), notes = $('#updateNotes'), go = $('#btnOpenRelease');
  btn.disabled = true; setText(btn, '查詢中…');
  go.hidden = true; notes.hidden = true; updateUrl = '';
  setText(res, '正在查詢…'); res.hidden = false;
  let r = null;
  try { r = await backend.check_update(); } catch (e) { r = null; }
  btn.disabled = false; setText(btn, '檢查更新');
  if (!r || typeof r !== 'object') r = { status: 'error', message: '查詢失敗，請稍後再試。' };
  res.dataset.status = r.status || 'error';
  setText(res, r.message || '');
  if (r.status === 'update') {
    if (r.notes) { setText(notes, r.notes); notes.hidden = false; }
    if (r.url) { updateUrl = r.url; go.hidden = false; }
  }
});
$('#btnOpenRelease').addEventListener('click', () => { if (updateUrl) api('open_release_page', updateUrl); });

// ==== BEGIN cloud usage panel (Settings -> 字幕 -> 雲端用量) ====
// Statistics + warning only. Reads window.ohStreamer.subtitleUsage() (guarded: it may be missing or reject).
const USAGE_POLL_MS = 30000;
const usageUi = (() => {
  const num = (v) => (typeof v === 'number' && isFinite(v) ? v : null);
  // Remaining share (0-100) of a {limit, remaining} pair; null when unknown.
  function remainingPct(limit, remaining) {
    const l = num(limit), r = num(remaining);
    if (l == null || r == null || l <= 0) return null;
    return Math.max(0, Math.min(100, Math.round((r / l) * 100)));
  }
  // Used share (0-100) of used/limit; null when unknown.
  function usedPct(used, limit) {
    const u = num(used), l = num(limit);
    if (u == null || l == null || l <= 0) return null;
    return Math.max(0, Math.min(100, Math.round((u / l) * 100)));
  }
  const warnClass = (w) => (w === 'ok' || w === 'low' || w === 'critical' ? w : 'unknown');
  function worstWarning(models) {
    let rank = -1, out = 'unknown';
    const order = { unknown: 0, ok: 1, low: 2, critical: 3 };
    for (const m of models || []) {
      const w = warnClass(m && m.warning);
      if (w === 'low' || w === 'critical') { if (order[w] > rank) { rank = order[w]; out = w; } }
    }
    return rank < 0 ? 'ok' : out;
  }
  function fmtResetIn(s) {
    const v = num(s);
    if (v == null) return '重置時間未知';
    if (v < 1) return '即將重置';
    if (v < 90) return `約 ${Math.round(v)} 秒後重置`;
    if (v < 5400) return `約 ${Math.round(v / 60)} 分鐘後重置`;
    return `約 ${Math.round(v / 360) / 10} 小時後重置`;
  }
  function fmtCount(n) {
    const v = num(n);
    if (v == null) return '—';
    const a = Math.abs(v);
    if (a >= 1e6) return `${Math.round(v / 1e5) / 10}M`;
    if (a >= 1e4) return `${Math.round(v / 100) / 10}K`;
    return String(Math.round(v));
  }
  function fmtSeconds(s) {
    const v = num(s);
    if (v == null) return '—';
    if (v < 90) return `${Math.round(v)} 秒`;
    if (v < 5400) return `${Math.round(v / 60)} 分`;
    return `${Math.round(v / 360) / 10} 小時`;
  }
  function fmtHour(t) {
    const d = new Date(t * 1000);
    return `${d.getMonth() + 1}/${d.getDate()} ${String(d.getHours()).padStart(2, '0')}:00`;
  }
  // Sum the hourly series of all providers into `hours` consecutive buckets ending at the newest hour.
  function seriesView(snapshot, hours) {
    const sum = new Map();
    for (const p of (snapshot && snapshot.providers) || []) {
      for (const pt of (p && p.series && p.series.points) || []) {
        const t = num(pt && pt.t);
        if (t == null) continue;
        const k = Math.floor(t / 3600) * 3600;
        const row = sum.get(k) || { t: k, tokens: 0, audio_s: 0, requests: 0 };
        row.tokens += num(pt.tokens) || 0; row.audio_s += num(pt.audio_s) || 0; row.requests += num(pt.requests) || 0;
        sum.set(k, row);
      }
    }
    if (!sum.size) return [];
    const end = Math.max(...sum.keys());
    const out = [];
    for (let i = hours - 1; i >= 0; i--) {
      const t = end - i * 3600;
      out.push(sum.get(t) || { t, tokens: 0, audio_s: 0, requests: 0 });
    }
    return out;
  }
  const flatModels = (snapshot) => {
    const rows = [];
    for (const p of (snapshot && snapshot.providers) || []) for (const m of (p && p.models) || []) if (m) rows.push(m);
    return rows;
  };
  return { remainingPct, usedPct, warnClass, worstWarning, fmtResetIn, fmtCount, fmtSeconds, fmtHour, seriesView, flatModels };
})();
window.__usageUi = usageUi;   // pure helpers, exposed for tests

(() => {
  const root = $('#usageSection');
  if (!root) return;
  const pane = $('#spane-subtitles');
  const body = $('#usageBody'), chip = $('#usageChip'), empty = $('#usageEmpty'), rows = $('#usageModels');
  const chartBox = $('#usageChart'), note = $('#usageAsrNote'), table = $('#usageTable');
  const btn24 = $('#usageRange24'), btn7d = $('#usageRange7d'), audioToggle = $('#usageShowAudio');
  const U = usageUi;
  let snapshot = null, hours = 24, timer = null, seq = 0;
  const SVGNS = 'http://www.w3.org/2000/svg';
  const svgNode = (tag, attrs) => { const n = document.createElementNS(SVGNS, tag); for (const k in attrs) n.setAttribute(k, attrs[k]); return n; };

  function bar(pct, cls, label) {
    const track = el('span', 'usage-bar');
    const fill = el('span', `usage-fill ${cls}`);
    fill.style.width = `${pct == null ? 0 : pct}%`;
    track.appendChild(fill);
    track.setAttribute('role', 'img');
    track.setAttribute('aria-label', label);
    return track;
  }
  function metricRow(name, pct, cls, text) {
    const r = el('div', 'usage-metric');
    r.append(el('span', 'usage-mname', name), bar(pct, cls, `${name} ${text}`), el('span', 'usage-mtext', text));
    return r;
  }
  function modelCard(m) {
    const cls = U.warnClass(m.warning);
    const card = el('div', `usage-model w-${cls}`);
    const head = el('div', 'usage-mhead');
    head.append(el('strong', 'usage-modelname', String(m.model || '—')), el('span', 'usage-kind', m.kind === 'asr' ? '語音辨識' : '翻譯'));
    const state = { ok: '正常', low: '偏低', critical: '即將耗盡', unknown: '未知' }[cls];
    head.appendChild(el('span', `usage-state ${cls}`, state));
    card.appendChild(head);
    for (const [key, name] of [['requests', '今日請求(RPD)'], ['tokens', '每分鐘 token(TPM)']]) {
      const q = m[key] || {};
      const pct = U.remainingPct(q.limit, q.remaining);
      const text = pct == null ? '尚無資料'
        : `剩餘 ${pct}%（${U.fmtCount(q.remaining)} / ${U.fmtCount(q.limit)}）· ${U.fmtResetIn(q.reset_in_s)}`;
      card.appendChild(metricRow(name, pct, cls, text));
    }
    if (m.tokens_day && (m.tokens_day.limit != null || m.tokens_day.used > 0)) {
      const td = m.tokens_day;
      const pct = U.usedPct(td.used, td.limit);
      const text = pct == null ? (td.used ? `已用 ${U.fmtCount(td.used)}` : '尚無資料')
        : `已用 ${pct}%（${U.fmtCount(td.used)} / ${U.fmtCount(td.limit)}）`;
      card.appendChild(metricRow('今日 token(TPD)', pct, cls, text));
    }
    if (m.requests_minute && (m.requests_minute.limit != null || m.requests_minute.used > 0)) {
      const rm = m.requests_minute;
      const pct = U.usedPct(rm.used, rm.limit);
      const text = pct == null ? (rm.used ? `已用 ${U.fmtCount(rm.used)}` : '尚無資料')
        : `已用 ${pct}%（${U.fmtCount(rm.used)} / ${U.fmtCount(rm.limit)}）`;
      card.appendChild(metricRow('每分鐘請求(RPM)', pct, cls, text));
    }
    if (m.kind === 'asr' && m.audio) {
      for (const [key, name] of [['hour', '音訊（小時）'], ['day', '音訊（日）']]) {
        const a = m.audio[key] || {};
        const pct = U.usedPct(a.used_s, a.limit_s);
        const text = pct == null ? '尚無資料' : `已用 ${pct}%（${U.fmtSeconds(a.used_s)} / ${U.fmtSeconds(a.limit_s)}）`;
        card.appendChild(metricRow(name, pct, cls, text));
      }
    }
    return card;
  }

  function drawChart() {
    chartBox.textContent = ''; table.textContent = '';
    const pts = U.seriesView(snapshot, hours);
    const total = pts.reduce((a, p) => a + p.tokens, 0), audio = pts.reduce((a, p) => a + p.audio_s, 0);
    const summary = pts.length
      ? `近 ${hours === 24 ? '24 小時' : '7 天'}：共 ${U.fmtCount(total)} Token、音訊 ${U.fmtSeconds(audio)}`
      : '尚無歷史資料';
    chartBox.setAttribute('aria-label', summary);
    if (!pts.length) { chartBox.appendChild(el('p', 'usage-muted', summary)); return; }
    const W = 600, H = 150, L = 8, R = 8, T = 8, B = 20, pw = W - L - R, ph = H - T - B;
    const maxTok = Math.max(1, ...pts.map((p) => p.tokens)), maxAud = Math.max(1, ...pts.map((p) => p.audio_s));
    const svg = svgNode('svg', { viewBox: `0 0 ${W} ${H}`, class: 'usage-svg', role: 'img', 'aria-label': summary, focusable: 'false' });
    const step = pw / pts.length;
    svg.appendChild(svgNode('line', { x1: L, x2: W - R, y1: T + ph, y2: T + ph, class: 'usage-axis' }));
    pts.forEach((p, i) => {
      const h = (p.tokens / maxTok) * ph;
      const r = svgNode('rect', { x: L + i * step + step * 0.12, width: Math.max(1, step * 0.76), y: T + ph - h, height: h, class: 'usage-tokbar' });
      const tip = svgNode('title', {}); tip.textContent = `${U.fmtHour(p.t)}  Token ${p.tokens}  音訊 ${Math.round(p.audio_s)} 秒  請求 ${p.requests}`;
      r.appendChild(tip); svg.appendChild(r);
    });
    if (audioToggle.checked) {
      const d = pts.map((p, i) => `${i ? 'L' : 'M'}${(L + i * step + step / 2).toFixed(1)} ${(T + ph - (p.audio_s / maxAud) * ph).toFixed(1)}`).join('');
      svg.appendChild(svgNode('path', { d, class: 'usage-audioline', fill: 'none' }));
    }
    for (const i of [0, Math.floor((pts.length - 1) / 2), pts.length - 1]) {
      const tx = svgNode('text', { x: L + i * step + step / 2, y: H - 5, class: 'usage-tick', 'text-anchor': i === 0 ? 'start' : (i === pts.length - 1 ? 'end' : 'middle') });
      tx.textContent = U.fmtHour(pts[i].t); svg.appendChild(tx);
    }
    chartBox.appendChild(svg);
    const cap = el('p', 'usage-legend');
    cap.append(el('span', 'usage-key tok', `Token（最高 ${U.fmtCount(maxTok)}/小時）`));
    if (audioToggle.checked) cap.append(el('span', 'usage-key aud', `音訊秒數（最高 ${U.fmtSeconds(maxAud)}/小時）`));
    chartBox.appendChild(cap);
    // screen-reader fallback
    const head = el('tr'); for (const h of ['時間', 'Token', '音訊秒數', '請求數']) head.appendChild(el('th', null, h));
    table.appendChild(head);
    for (const p of pts) {
      const tr = el('tr'); for (const v of [U.fmtHour(p.t), p.tokens, Math.round(p.audio_s), p.requests]) tr.appendChild(el('td', null, String(v)));
      table.appendChild(tr);
    }
  }

  function render() {
    const models = U.flatModels(snapshot);
    const has = models.length > 0 || U.seriesView(snapshot, 24).length > 0;
    setHidden(empty, has); setHidden(rows, !has); setHidden($('#usageChartWrap'), !has);
    rows.textContent = '';
    for (const m of models) rows.appendChild(modelCard(m));
    setHidden(note, !models.some((m) => m.kind === 'asr' && m.audio && m.audio.limits_verified === false));
    const worst = U.worstWarning(models);
    const warn = worst === 'low' || worst === 'critical';
    setHidden(chip, !warn);
    chip.className = `usage-chip ${warn ? worst : ''}`;
    setText(chip, worst === 'critical' ? '用量即將耗盡' : '用量偏低');
    root.classList.toggle('has-warning', warn);
    if (has) drawChart();
  }

  async function refresh() {
    const my = ++seq;
    let snap = null;
    try {
      const fn = window.ohStreamer && window.ohStreamer.subtitleUsage;
      if (typeof fn === 'function') snap = await fn.call(window.ohStreamer);
    } catch (_) { snap = null; }
    if (my !== seq) return;
    snapshot = snap && typeof snap === 'object' && !snap.error && Array.isArray(snap.providers) ? snap : null;
    render();
  }

  const visible = () => !document.hidden && !drawer.hidden && !pane.hidden && root.getClientRects().length > 0;
  function sync() {
    if (visible()) { if (!timer) { refresh(); timer = setInterval(() => { if (visible()) refresh(); else sync(); }, USAGE_POLL_MS); } }
    else if (timer) { clearInterval(timer); timer = null; seq++; }
  }
  new MutationObserver(sync).observe(drawer, { attributes: true, attributeFilter: ['hidden'] });
  new MutationObserver(sync).observe(pane, { attributes: true, attributeFilter: ['hidden'] });
  const cloudHome = $('#subtitleCloudHome');
  if (cloudHome) new MutationObserver(sync).observe(cloudHome, { attributes: true, attributeFilter: ['hidden'] });
  document.addEventListener('visibilitychange', sync);
  for (const [b, h] of [[btn24, 24], [btn7d, 168]]) b.addEventListener('click', () => {
    hours = h; btn24.setAttribute('aria-pressed', String(h === 24)); btn7d.setAttribute('aria-pressed', String(h === 168)); if (snapshot) drawChart();
  });
  audioToggle.addEventListener('change', () => { if (snapshot) drawChart(); });
  root.addEventListener('toggle', () => { if (root.open) sync(); });
  window.__usageUiRefresh = refresh;   // test hook
  render();
})();
// ==== END cloud usage panel ====
// keep Tab inside the dialog while it is open
const FOCUSABLE = 'button, input, select, textarea, a[href], [tabindex]';
drawer.addEventListener('keydown', (e) => {
  if (e.key !== 'Tab') return;
  const items = [...drawer.querySelectorAll(FOCUSABLE)].filter((el) => !el.disabled && el.tabIndex >= 0 && el.getClientRects().length);
  if (!items.length) return;
  const first = items[0], last = items[items.length - 1];
  if (e.shiftKey && (document.activeElement === first || !drawer.contains(document.activeElement))) { e.preventDefault(); last.focus(); }
  else if (!e.shiftKey && (document.activeElement === last || !drawer.contains(document.activeElement))) { e.preventDefault(); first.focus(); }
});
document.addEventListener('focusin', (e) => {
  if (settingsOpen() && $('#modal').hidden && !drawer.contains(e.target)) {
    const t = settingsTabs.find((x) => x.dataset.cat === settingsCat);
    if (t) t.focus();
  }
});

function pushSetting(key, value) {
  lastEdit[key] = Date.now();
  if (current && current.settings) current.settings[key] = value;
  if ((key === 'show_thumbnails' || key === 'show_thumbnail_cover') && current) render(current);
  act('set_settings', { [key]: value });
}
const fresh = (key) => Date.now() - (lastEdit[key] || 0) > EDIT_GRACE;
for (const select of document.querySelectorAll('.model-select[data-key]')) {
  select.addEventListener('change', () => pushSetting(select.dataset.key, select.value));
}

// The engine selector exists only when more than one user-visible engine is installed (state.engine_options).
function renderEngineSelect(value) {
  const options = (current && Array.isArray(current.engine_options)) ? current.engine_options : [];
  const row = $('#engineRow'), select = $('#engineSelect');
  row.hidden = options.length < 2;
  if (row.hidden) return;
  const signature = JSON.stringify(options);
  if (select.dataset.options !== signature) {
    select.dataset.options = signature;
    select.replaceChildren();
    for (const option of options) {
      const node = el('option', null, String(option.label || option.value));
      node.value = option.value;
      select.appendChild(node);
    }
  }
  if (fresh('engine') && value != null) select.value = value;
}

function renderModelSelect(key, value) {
  const select = document.querySelector(`.model-select[data-key="${key}"]`);
  const available = current && current.model_options && current.model_options[key];
  const options = Array.isArray(available) ? available.slice() : [];
  if (value && !options.some((o) => o.value === value)) options.push({ value, label: value + '（未安裝）', disabled: true });
  const signature = JSON.stringify(options);
  if (select.dataset.options !== signature) {
    select.dataset.options = signature;
    select.replaceChildren();
    for (const option of options) {
      const node = el('option', null, String(option.label || option.value));
      node.value = option.value;
      node.disabled = !!option.disabled;
      select.appendChild(node);
    }
    if (!options.length) {
      const node = el('option', null, '沒有可用模型');
      node.value = '';
      select.appendChild(node);
    }
  }
  select.disabled = !Array.isArray(available) || !available.length;
  if (fresh(key) && value != null) select.value = value;
}

const CLIP_OPTIONS = [30, 60, 90, 120, 180, 240, 300, 360];
const clipSelect = $('#clipLength');
clipSelect.addEventListener('change', () => pushSetting('max_clip_length', +clipSelect.value));
function renderClipSelect(value) {
  const v = Number.isFinite(+value) && value != null ? +value : null;
  const list = v != null && !CLIP_OPTIONS.includes(v) ? [...CLIP_OPTIONS, v].sort((a, b) => a - b) : CLIP_OPTIONS;
  const sig = list.join(',');
  if (clipSelect.dataset.options !== sig) {
    clipSelect.dataset.options = sig;
    clipSelect.replaceChildren(...list.map((n) => { const o = el('option', null, `${n} 幀`); o.value = String(n); return o; }));
  }
  if (fresh('max_clip_length') && v != null) clipSelect.value = String(v);
}

const capNum = $('#capNum'), capRange = $('#capRange');
function paintRange() {
  const min = +capRange.min, max = +capRange.max;
  const v = Math.min(max, Math.max(min, +capRange.value));
  capRange.style.setProperty('--fill', ((v - min) / (max - min) * 100) + '%');
}
capRange.addEventListener('input', () => { capNum.value = capRange.value; paintRange(); });
capRange.addEventListener('change', () => pushSetting('cap_gb', +capRange.value));
capNum.addEventListener('change', () => {
  let v = Math.round(+capNum.value);
  if (!isFinite(v) || v < 1) v = 1;
  if (v > 5000) v = 5000;
  capNum.value = v;
  capRange.value = v; paintRange();
  pushSetting('cap_gb', v);
});
for (const r of document.querySelectorAll('input[name="encoder"], input[name="interpolation"], input[name="compare_original"], input[name="compare_quality"]')) {
  r.addEventListener('change', () => { if (r.checked) { pushSetting(r.name, r.value); syncCompareOpts(); } });
}
for (const cb of document.querySelectorAll('input[type="checkbox"][data-key]')) {
  cb.addEventListener('change', () => pushSetting(cb.dataset.key, cb.checked));
}
// compare sub-options: shown only while 保存對照畫面 is on; 比對軌畫質 only for 重新壓縮
const compareOpts = $('#compareOpts'), compareQualityField = $('#compareQualityField');
function syncCompareOpts() {
  setHidden(compareOpts, !$('#swCompare').checked);
  setHidden(compareQualityField, !$('input[name="compare_original"][value="encode"]').checked);
}
$('#swCompare').addEventListener('change', syncCompareOpts);

function renderSettings(s) {
  for (const key of MODEL_KEYS) renderModelSelect(key, s[key]);
  renderEngineSelect(s.engine);
  renderClipSelect(s.max_clip_length);
  const shutdownAvailable = !current || !current.capabilities || current.capabilities.shutdown !== false;
  $('#swShutdown').disabled = !shutdownAvailable;
  $('#swShutdown').title = shutdownAvailable ? '' : '此平台尚未支援自動關機';
  const rifeAvailable = !current || !current.capabilities || current.capabilities.rife !== false;
  const rifeFull = $('input[name="interpolation"][value="full"]');
  rifeFull.disabled = !rifeAvailable;
  rifeFull.parentElement.title = rifeAvailable ? '' : 'macOS 首版暫不支援 RIFE 插幀';
  if (fresh('cap_gb') && document.activeElement !== capNum && s.cap_gb != null) {
    const v = num(s.cap_gb);
    if (String(capNum.value) !== String(v)) capNum.value = v;
    const clamped = Math.min(+capRange.max, Math.max(+capRange.min, v));
    if (String(capRange.value) !== String(clamped)) capRange.value = clamped;
    paintRange();
  }
  for (const key of ['encoder', 'interpolation', 'compare_original', 'compare_quality']) {
    if (!fresh(key) || s[key] == null) continue;
    for (const r of document.querySelectorAll(`input[name="${key}"]`)) r.checked = (r.value === s[key]);
  }
  for (const cb of document.querySelectorAll('input[type="checkbox"][data-key]')) {
    const k = cb.dataset.key;
    if (!fresh(k)) continue;
    if (s[k] != null) cb.checked = !!s[k];
    else if (DEFAULT_ON.has(k)) cb.checked = true;
  }
  syncCompareOpts();
  renderSubtitleSettings(s);
}

const SUBTITLE_DUAL_VALUE = '__dual__';
const DEFAULT_TRANSLATION_RECIPE = 'sakura-galtransl-7b-v3.7';
let subtitleSubview = 'main';
let subtitleModelKind = 'asr';
let subtitleRowIdSequence = 0;
let subtitleSaveChain = Promise.resolve();
let subtitleConnections = [];
let subtitleConnectionsLoaded = false;
let subtitleConnectionsBusy = false;
let subtitleProfileBusy = false;
let subtitleProfileDraft = null;
let subtitleProfileError = '';
let subtitleServiceBusy = '';
const subtitleConnectionChoice = {};
const subtitleServiceMessages = {};
let subtitlePoolEditing = null; // null: closed; '': new connection; otherwise stable connection ID

function subtitleProfileValue() {
  return cloneConfig(subtitleProfileDraft || (current && current.settings && current.settings.subtitle_profile));
}
async function loadSubtitleConnections(force = false) {
  if (subtitleConnectionsBusy || (subtitleConnectionsLoaded && !force) || !hasApi('list_subtitle_connections')) return;
  subtitleConnectionsBusy = true;
  try {
    const result = await api('list_subtitle_connections');
    if (Array.isArray(result)) { subtitleConnections = result; subtitleConnectionsLoaded = true; }
  } finally { subtitleConnectionsBusy = false; }
  if (current) renderSubtitleHome(current.settings || {});
}
async function saveSubtitleProfile(profile) {
  if (!profile || !hasApi('set_subtitle_profile') || subtitleProfileBusy) return false;
  subtitleProfileBusy = true; subtitleProfileDraft = cloneConfig(profile); subtitleProfileError = '';
  renderSubtitleHome(current.settings || {});
  try {
    await subtitleSaveChain.catch(() => {});
    const result = await api('set_subtitle_profile', profile);
    if (!result || result.ok !== true) {
      subtitleProfileError = String(result && result.error || '字幕方案儲存失敗，請重試。');
      return false;
    }
    current.settings.subtitle_profile = result.profile;
    current.settings.subtitle_runtime_contract = result.runtime_contract;
    return true;
  } finally {
    subtitleProfileBusy = false; subtitleProfileDraft = null;
    renderSubtitleHome(current.settings || {});
  }
}
function subtitleConnectionFor(family, profile) {
  const key = 'groq';
  if (subtitleConnectionChoice[key] !== undefined) return subtitleConnectionChoice[key];
  const model = subtitleModels.find((m) => m.id === profile.profiles.cloud.asr);
  if (model && model.connection_id) return model.connection_id;
  const available = subtitleConnections.filter((c) => c.api_family === family);
  if (available.length > 0) return available[0].id;
  return '';
}
function subtitleCapabilityText(connection, role) {
  const validation = connection && connection.validation && connection.validation[role];
  const status = validation && validation.status;
  const labels = { verified: '已驗證', reachable: '已連線驗證',
    failed: '驗證失敗', unverified: '待驗證', legacy_tested: '舊版測試（已驗證）' };
  return `${role === 'asr' ? '語音辨識' : '翻譯'}：${labels[status] || '待設定'}${validation && validation.error ? ' — ' + validation.error : ''}`;
}
function subtitlePoolValue(profile) {
  const cloud = profile.profiles.cloud;
  const saved = cloud.connection_pool;
  const connections = subtitleConnections.filter((c) => c.api_family === 'groq');
  if (saved) return {
    auto_switch: saved.auto_switch,
    connections: [...saved.connections.map((c) => ({ ...c })),
      ...connections.filter((c) => !saved.connections.some((r) => r.id === c.id)).map((c) => ({ id: c.id, enabled: false }))],
  };
  const ids = [cloud.asr, cloud.translation, ...cloud.fallback.asr, ...cloud.fallback.translation];
  const priority = [...new Set(ids.map((id) => subtitleModels.find((m) => m.id === id)?.connection_id).filter(Boolean))];
  const ordered = [...priority.map((id) => connections.find((c) => c.id === id)).filter(Boolean),
    ...connections.filter((c) => !priority.includes(c.id))];
  return { auto_switch: true, connections: ordered.map((c, i) => ({ id: c.id, enabled: priority.length ? priority.includes(c.id) : i === 0 })) };
}
function openSubtitlePoolEditor(id) {
  if (subtitleServiceBusy || subtitleProfileBusy) return;
  subtitlePoolEditing = id;
  subtitleConnectionChoice.groq = id;
  const connection = subtitleConnections.find((c) => c.id === id);
  $('#subtitleGroqName').value = connection?.label || (id ? `Groq 連線 ${subtitleConnections.filter((c) => c.api_family === 'groq').indexOf(connection) + 1}` : '');
  $('#subtitleGroqKey').value = '';
  $('#subtitleGroqKey').type = 'password';
  setText($('#subtitleGroqShowKey'), '顯示');
  const pool = subtitlePoolValue(subtitleProfileValue());
  const priority = String(id ? pool.connections.findIndex((c) => c.id === id) : pool.connections.length);
  subtitleServiceMessages.groq = '';
  renderSubtitleHome(current.settings || {});
  $('#subtitleGroqPriority').value = priority;
  $('#subtitleGroqName').focus();
}
async function saveSubtitlePool(pool) {
  const profile = subtitleProfileValue();
  if (!profile || subtitleServiceBusy || subtitleProfileBusy) return false;
  profile.profiles.cloud.connection_pool = pool;
  const ok = await saveSubtitleProfile(profile);
  if (!ok) subtitleServiceMessages.groq = subtitleProfileError;
  renderSubtitleHome(current.settings || {});
  return ok;
}
function renderSubtitleGroqPool(profile) {
  const box = $('#subtitleGroqPool');
  // Older embedded frontends may not have the pool yet.
  if (!box || typeof box.replaceChildren !== 'function') return;
  const form = $('#subtitleGroqForm');
  const pool = subtitlePoolValue(profile);
  const busy = !!subtitleServiceBusy || subtitleProfileBusy || !subtitleConnectionsLoaded || !subtitleModelsLoaded;
  // Queue progress refreshes frequently. Keep the live editor connected so typing
  // and keyboard focus survive unrelated state updates.
  const signature = JSON.stringify([pool, busy, subtitlePoolEditing,
    subtitleConnections.map((c) => [c.id, c.label, c.auth_source, c.validation])]);
  if (box.dataset.renderSignature === signature) return;
  box.dataset.renderSignature = signature;
  box.replaceChildren();
  let editorPlaced = false;
  pool.connections.forEach((entry, index) => {
    const connection = subtitleConnections.find((c) => c.id === entry.id);
    if (!connection) return;
    const item = el('div', 'subtitle-pool-item');
    const row = el('div', 'subtitle-pool-row');
    const enabled = el('input'); enabled.type = 'checkbox'; enabled.checked = entry.enabled; enabled.disabled = busy;
    const label = connection.label || `Groq 連線 ${subtitleConnections.filter((c) => c.api_family === 'groq').indexOf(connection) + 1}`;
    enabled.setAttribute('aria-label', `啟用${label}`);
    enabled.addEventListener('change', async () => {
      const next = subtitlePoolValue(subtitleProfileValue());
      next.connections.find((c) => c.id === entry.id).enabled = enabled.checked;
      try { await saveSubtitlePool(next); } catch (_) { subtitleServiceMessages.groq = '連線池儲存失敗，請重試。'; renderSubtitleHome(current.settings || {}); }
    });
    const name = el('div', 'subtitle-pool-name', label);
    name.appendChild(el('small', null, connection.auth_source === 'none' ? '尚未設定金鑰' : '金鑰已儲存 · ••••'));
    const states = ['asr', 'translation'].map((role) => {
      const val = connection.validation?.[role];
      const title = role === 'asr' ? '辨識' : '翻譯';
      return `${title}：${['verified', 'reachable', 'legacy_tested'].includes(val?.status) ? '已驗證' : val?.status === 'failed' ? '驗證失敗' : '待驗證'}`;
    });
    const status = el('span', 'subtitle-pool-state', entry.enabled ? states.join('\n') : '已停用');
    const edit = el('button', 'btn sm ghost', subtitlePoolEditing === entry.id ? '收合' : '編輯'); edit.type = 'button'; edit.disabled = busy;
    edit.setAttribute('aria-expanded', String(subtitlePoolEditing === entry.id));
    edit.setAttribute('aria-controls', 'subtitleGroqForm');
    edit.addEventListener('click', () => {
      if (subtitlePoolEditing === entry.id) { subtitlePoolEditing = null; $('#subtitleGroqKey').value = ''; renderSubtitleHome(current.settings || {}); }
      else openSubtitlePoolEditor(entry.id);
    });
    row.append(enabled, name, status, edit); item.appendChild(row);
    if (subtitlePoolEditing === entry.id) { item.appendChild(form); editorPlaced = true; }
    box.appendChild(item);
  });
  if (!pool.connections.length) box.appendChild(el('p', 'snote', '尚無連線，按「＋ 新增」加入金鑰。'));
  if (!editorPlaced) box.appendChild(form);
  form.hidden = subtitlePoolEditing === null;
  $('#subtitleGroqAdd').disabled = busy || pool.connections.length >= 10;
  $('#subtitleGroqAdd').title = pool.connections.length >= 10 ? '連線池最多保存 10 組連線' : '';
  $('#subtitleGroqAuto').checked = pool.auto_switch;
  $('#subtitleGroqAuto').disabled = busy;
  $('#subtitleGroqDelete').hidden = !subtitlePoolEditing;
  $('#subtitleGroqDelete').disabled = busy;
  $('#subtitleGroqCancel').disabled = busy;
  $('#subtitleGroqName').disabled = busy;
  $('#subtitleGroqPriority').disabled = busy;
  setText($('#subtitleGroqEditorTitle'), subtitlePoolEditing ? '編輯連線' : '新增連線');
  const position = $('#subtitleGroqPriority').value;
  fillSelect($('#subtitleGroqPriority'), Array.from({ length: pool.connections.length + (subtitlePoolEditing === '' ? 1 : 0) }, (_, i) => ({ value: String(i), label: `第 ${i + 1} 順位` })), position);
}
async function deleteSubtitlePoolConnection() {
  if (!subtitlePoolEditing || subtitleServiceBusy || subtitleProfileBusy) return;
  const cid = subtitlePoolEditing;
  if (!(await confirmDlg({ title: '刪除連線', body: '刪除此連線與所屬模型？佇列工作或其他字幕設定仍在使用時，會保留連線。', ok: '刪除', danger: true }))) return;
  const profile = subtitleProfileValue();
  // Explicitly migrate legacy bindings before the backend performs guarded deletion.
  if (!profile.profiles.cloud.connection_pool && !(await saveSubtitlePool(subtitlePoolValue(profile)))) return;
  subtitleServiceBusy = 'groq'; renderSubtitleHome(current.settings || {});
  try {
    const result = await api('remove_subtitle_pool_connection', cid);
    if (!result?.ok) { subtitleServiceMessages.groq = String(result?.error || '刪除失敗，請重試。'); return; }
    current.settings.subtitle_profile = result.profile;
    subtitleConnections = subtitleConnections.filter((c) => c.id !== cid);
    subtitlePoolEditing = null; delete subtitleConnectionChoice.groq;
    $('#subtitleGroqKey').value = '';
    subtitleServiceMessages.groq = '連線已刪除。';
    await loadSubtitleModels(true);
  } catch (_) { subtitleServiceMessages.groq = '刪除失敗，請重試。'; }
  finally { subtitleServiceBusy = ''; renderSubtitleHome(current.settings || {}); }
}
let subtitleLocalPeek = false;   // true after 前往下載: shows the local card even when another mode is active (no setting change)
function renderSubtitleHome(settings) {
  const profile = subtitleProfileValue();
  if (!profile) {
    setText($('#subtitleProfileStatus'), '字幕方案尚未載入；請重新開啟設定或更新後端。');
    $('#subtitleMode').disabled = true;
    return;
  }
  const mode = profile.active_mode, active = profile.profiles[mode];
  $('#subtitleMode').value = mode;
  $('#subtitleMode').disabled = subtitleProfileBusy || !!subtitleServiceBusy || !hasApi('set_subtitle_profile');
  setHidden($('#subtitleCloudHome'), mode !== 'cloud');
  setHidden($('#subtitleLocalHome'), mode !== 'local' && !subtitleLocalPeek);
  if ((mode === 'local' || subtitleLocalPeek) && typeof ensureModelDlLoaded === 'function') ensureModelDlLoaded();
  setHidden($('#subtitleCustomEditor'), mode !== 'custom');

  if (mode === 'cloud' && !settings.subtitle_cloud_enabled) {
    settings.subtitle_cloud_enabled = true;
    pushSetting('subtitle_cloud_enabled', true);
  }

  const cloudNotice = $('#subtitleCloudNotice');
  const stratRow = $('#subtitleCloudStrategyRow');
  if (cloudNotice) {
    setHidden(cloudNotice, mode !== 'cloud');
    setText(cloudNotice, '☁️ 雲端模式會透過 HTTPS 加密傳送音訊與字幕文字至雲端服務（服務可能依使用量計費）。');
  }
  if (stratRow) {
    const showStrategy = mode === 'cloud' || (mode === 'custom' && !!settings.subtitle_cloud_enabled);
    setHidden(stratRow, !showStrategy);
  }

  const describe = (id) => {
    if (!id) return '待設定';
    const model = subtitleModels.find((m) => m.id === id);
    return model ? `${model.label}（${subtitleModelStatus(model)}）` : `${id}（清單載入後確認）`;
  };
  const flow = `${describe(active.asr)} → ${describe(active.translation)}`;
  const cloudNote = mode !== 'local' && !settings.subtitle_cloud_enabled ? '（雲端外連尚未允許，不會上傳）' : '';
  setText($('#subtitleProfileStatus'), subtitleProfileBusy ? '正在儲存字幕方案…' : `${subtitleProfileError ? subtitleProfileError + '\n' : ''}目前流程：${flow}${cloudNote}。僅影響新排程工作。`);
  const fbBox = $('#subtitleComboFallback');
  const fb = typeof subtitleComboFallback === 'undefined' ? null : subtitleComboFallback;
  if (fbBox) { setText(fbBox, fb ? fb.message : ''); setHidden(fbBox, !fb); setHidden($('#subtitleComboFallbackGo'), !fb); }
  setText($('#subtitleCloudPlan'), `辨識方案：${profile.profiles.cloud.whisper_policy === 'auto' ? 'Groq Whisper 自動策略（' + profile.profiles.cloud.whisper_models.join('／') + '）' : '固定模型'}；翻譯依已保存方案。進階選項與模型備援由下方「進階字幕設定」配置。`);
  setText($('#subtitleLocalStatus'), `辨識：${describe(profile.profiles.local.asr)}\n翻譯：${describe(profile.profiles.local.translation)}`);
  $('#subtitleLocalRecommend').disabled = subtitleProfileBusy || !hasApi('set_subtitle_profile');
  for (const [key, family] of [['groq', 'groq']]) {
    const prefix = 'subtitleGroq';
    const cid = subtitleConnectionFor(family, profile);
    const connections = subtitleConnections.filter((c) => c.api_family === family);
    const select = $(`#${prefix}Connection`);
    const options = [];
    if (connections.length === 0) {
      options.push({ value: '', label: '新增連線' });
    } else {
      connections.forEach((c, i) => {
        const caps = [];
        if (c.validation?.asr) caps.push('辨識');
        if (c.validation?.translation) caps.push('翻譯');
        const capStr = caps.length ? `（${caps.join('＋')}）` : '';
        const name = connections.length === 1 ? 'Groq 主要連線' : `Groq 連線 ${i + 1}`;
        const auth = c.auth_source === 'none' ? '尚無金鑰' : '••••（已加密）';
        options.push({ value: c.id, label: `${name}${capStr} · ${auth}` });
      });
      options.push({ value: '', label: '＋ 新增其他連線…' });
    }
    fillSelect(select, options, cid);
    select.disabled = !!subtitleServiceBusy || subtitleProfileBusy;
    const connection = connections.find((c) => c.id === cid);
    const lines = ['asr', 'translation'];
    setText($(`#${prefix}Status`), [subtitleServiceMessages[key], ...lines.map((role) => subtitleCapabilityText(connection, role))].filter(Boolean).join('\n'));

    // Mask key placeholder if saved connection has a credential
    const keyInput = $(`#${prefix}Key`);
    if (keyInput) {
      const hasKey = connection && connection.auth_source && connection.auth_source !== 'none';
      keyInput.placeholder = cid && hasKey
        ? '••••••••••••••••（已加密儲存，留空保留原金鑰）'
        : '請貼上 Groq API 金鑰（gsk_...）';
      keyInput.disabled = !!subtitleServiceBusy;
    }

    const button = $(`#${prefix}Save`);
    button.disabled = !!subtitleServiceBusy || subtitleProfileBusy || !hasApi('test_save_subtitle_connection');
    button.textContent = subtitleServiceBusy === key ? '正在驗證…' : '驗證並儲存';
    $(`#${prefix}Form`).setAttribute('aria-busy', String(subtitleServiceBusy === key));
  }
  renderSubtitleGroqPool(profile);
  if (typeof renderSubtitleCustom === 'function' && typeof subtitleAsrChoices === 'function') {
    renderSubtitleCustom(profile, settings);
  }
}
function renderSubtitleCustom(profile, settings) {
  if (!profile || profile.active_mode !== 'custom') return;
  const custom = profile.profiles.custom;
  const asrVal = custom.asr || subtitleAsrValue(subtitlePrimaryFromSettings(settings));
  fillSelect($('#subtitleAsr'), subtitleAsrChoices(asrVal), asrVal);

  const transVal = custom.translation || subtitleTranslationValue(subtitlePrimaryFromSettings(settings));
  const directTransChoices = subtitleModelChoices('translation').map((m) => ({
    value: subtitleTranslationModelValue(m),
    label: `${subtitleModelLabel(m)}（${isCloudModel(m) ? '雲端' : '本機'} · ${subtitleModelStatus(m)}）`,
    disabled: m.status === 'missing'
  }));
  fillSelect($('#subtitleTranslation'), directTransChoices, transVal);

  const describePart = (id) => {
    if (!id) return { label: '待設定', cloud: false };
    if (id === SUBTITLE_DUAL_VALUE) return { label: '雙模型融合 (本機)', cloud: false };
    const m = subtitleModels.find((row) => row.id === id);
    if (!m) return { label: `${id}（未登錄）`, cloud: false };
    return {
      label: `${m.label}（${isCloudModel(m) ? '雲端' : '本機'} · ${subtitleModelStatus(m)}）`,
      cloud: isCloudModel(m)
    };
  };
  const asrP = describePart(custom.asr);
  const transP = describePart(custom.translation);
  const flow = `自訂資料流：${asrP.label} → ${transP.label}`;
  const usesCloud = asrP.cloud || transP.cloud;
  if (usesCloud && !settings.subtitle_cloud_enabled) {
    settings.subtitle_cloud_enabled = true;
    pushSetting('subtitle_cloud_enabled', true);
  }
  setText($('#subtitleCustomFlowSummary'), flow);
}
async function saveSubtitleService(key) {
  if (subtitleServiceBusy || subtitleProfileBusy) return;
  const profile = subtitleProfileValue();
  const settings = current && current.settings || {};
  if (!profile) return;
  if (!settings.subtitle_cloud_enabled) {
    settings.subtitle_cloud_enabled = true;
    pushSetting('subtitle_cloud_enabled', true);
  }
  const prefix = 'subtitleGroq';
  const spec = { api_family: 'groq' };
  const cid = $(`#${prefix}Connection`).value;
  if (cid) spec.id = cid;
  const poolUI = typeof $('#subtitleGroqPool')?.replaceChildren === 'function';
  if (poolUI) spec.label = $('#subtitleGroqName').value.trim();
  const input = $(`#${prefix}Key`);
  if (input.value.trim() && input.dataset.masked !== 'true' && !input.value.startsWith('•••')) {
    spec.api_key = input.value.trim();
  }
  if (poolUI && (!spec.label || (!cid && !spec.api_key))) {
    subtitleServiceMessages[key] = '請填寫名稱與 API 金鑰。'; renderSubtitleHome(settings); return;
  }
  const priority = poolUI ? Number($('#subtitleGroqPriority').value) : 0;
  const pool = poolUI ? subtitlePoolValue(profile) : null;
  const audioInput = $('#subtitleGroqAudio');
  spec.test_audio_path = (audioInput ? audioInput.value.trim() : '') || '__builtin__';
  subtitleServiceBusy = key;
  subtitleServiceMessages[key] = '正在驗證連線…';
  renderSubtitleHome(settings);
  try {
    const result = await api('test_save_subtitle_connection', spec);
    if (!result || !result.connection) {
      subtitleServiceMessages[key] = String(result && result.error || '驗證失敗，請重試。');
      return;
    }
    subtitleConnectionChoice[key] = result.connection.id;
    subtitleConnections = subtitleConnections.filter((c) => c.id !== result.connection.id).concat([result.connection]);
    // Only use bindings returned by this selected connection, never another account.
    const rows = result.models || [];
    const find = (kind) => rows.find((m) => m.kind === kind && m.remote_model === result.connection.validation[kind]?.remote_model);
    const cloud = profile.profiles.cloud;
    if (poolUI) {
      const existing = pool.connections.find((c) => c.id === result.connection.id);
      pool.connections = pool.connections.filter((c) => c.id !== result.connection.id);
      pool.connections.splice(priority, 0, existing || { id: result.connection.id, enabled: true });
      cloud.connection_pool = pool;
    } else {
      cloud.asr = find('asr')?.id || null;
      cloud.translation = find('translation')?.id || null;
    }
    subtitleServiceMessages[key] = result.ok ? '驗證成功，連線已保存。' : '部分能力尚未通過，連線已保存；請修正後重試。';
    const saved = await saveSubtitleProfile(profile);
    if (!saved) subtitleServiceMessages[key] += ' 字幕方案未能儲存，請重試。';
    else if (poolUI && ['asr', 'translation'].every((role) => ['verified', 'reachable'].includes(result.connection.validation[role]?.status))) subtitlePoolEditing = null;
    await loadSubtitleModels(true);
  } catch (_) {
    subtitleServiceMessages[key] = '連線儲存失敗，請重試。';
  } finally {
    if (input.dataset.masked !== 'true') input.value = '';
    input.type = 'password';
    if (poolUI) setText($('#subtitleGroqShowKey'), '顯示');
    subtitleServiceBusy = '';
    renderSubtitleHome(current.settings || {});
  }
}

function cloneConfig(value) {
  if (value === undefined) return undefined;
  if (value === null || typeof value !== 'object') return value;
  return JSON.parse(JSON.stringify(value));
}
function mergeConfig(base, value) {
  const out = cloneConfig(base);
  if (!value || typeof value !== 'object' || Array.isArray(value)) return out;
  for (const [key, item] of Object.entries(value)) {
    out[key] = item && typeof item === 'object' && !Array.isArray(item) && out[key] && typeof out[key] === 'object'
      ? mergeConfig(out[key], item) : cloneConfig(item);
  }
  return out;
}
function defaultSubtitlePrimary() {
  return {
    id: 'primary', label: '主字幕',
    asr: { mode: 'dual', model: 'whisper-ja-1.5B' },
    correction: { mode: 'none' },
    translation: { profile: DEFAULT_TRANSLATION_RECIPE },
    sentence: { sentence_mode: true, short_threshold: 1, short_policy: 'whitelist', show_interjections: false },
    japanese_track: false,
  };
}
function subtitlePrimaryFromSettings(settings) {
  let saved = settings && settings.subtitle_primary;
  if (settings && settings.subtitle_runtime_contract === 'cloud-first-1' && settings.subtitle_profile) {
    const profile = settings.subtitle_profile.profiles[settings.subtitle_profile.active_mode];
    saved = mergeConfig(defaultSubtitlePrimary(), profile.legacy_primary);
    if (profile.asr) saved.asr = { ...saved.asr, model: profile.asr, mode: profile.legacy_primary?.asr?.mode === 'dual' ? 'dual' : 'single' };
    if (profile.translation) saved.translation = { ...saved.translation, model_id: profile.translation,
      mode: 'standard' };
  }
  const primary = mergeConfig(defaultSubtitlePrimary(), saved);
  if (!saved || typeof saved !== 'object') {
    const legacyMode = settings && settings.subtitle_asr_mode;
    primary.asr.mode = legacyMode == null ? 'dual' : (legacyMode === 'dual' ? 'dual' : 'single');
    if (settings && settings.subtitle_asr) primary.asr.model = String(settings.subtitle_asr);
    if (settings && settings.subtitle_translation) primary.translation.profile = String(settings.subtitle_translation);
    primary.japanese_track = !!(settings && settings.subtitle_japanese);
    primary.sentence.show_interjections = !!(settings && settings.subtitle_interjections);
  } else if (saved.asr && saved.asr.mode !== undefined) {
    primary.asr.mode = saved.asr.mode === 'dual' ? 'dual' : 'single';
  } else if (saved.asr && saved.asr.model) {
    primary.asr.mode = 'single';
  }
  primary.id = 'primary';
  primary.label = '主字幕';
  if (!primary.asr || typeof primary.asr !== 'object') primary.asr = { mode: 'dual', model: 'whisper-ja-1.5B' };
  if (primary.asr.mode !== 'dual') primary.asr.mode = 'single';
  if (!primary.asr.model) primary.asr.model = 'whisper-ja-1.5B';
  if (!primary.translation || typeof primary.translation !== 'object') primary.translation = { profile: DEFAULT_TRANSLATION_RECIPE };
  if (!primary.translation.profile) primary.translation.profile = DEFAULT_TRANSLATION_RECIPE;
  if (!primary.sentence || typeof primary.sentence !== 'object') primary.sentence = {};
  if (!['whitelist', 'keep', 'drop'].includes(primary.sentence.short_policy)) primary.sentence.short_policy = 'whitelist';
  return primary;
}
function subtitleRowsFromSettings(settings) {
  const source = settings && Array.isArray(settings.subtitle_comparison_rows) ? settings.subtitle_comparison_rows : [];
  return source.filter((row) => row && typeof row === 'object').map((row, index) => {
    const out = mergeConfig(defaultSubtitlePrimary(), row);
    out.id = String(row.id || `comparison-${index + 1}`);
    out.label = String(row.label || row.name || `額外字幕軌 ${index + 1}`);
    if (!out.asr || typeof out.asr !== 'object') out.asr = { mode: 'single', model: 'whisper-ja-1.5B' };
    out.asr.mode = row.asr && row.asr.mode === 'dual' ? 'dual' : 'single';
    if (!out.asr.model) out.asr.model = 'whisper-ja-1.5B';
    if (!out.translation || typeof out.translation !== 'object') out.translation = { profile: DEFAULT_TRANSLATION_RECIPE };
    if (!out.translation.profile) out.translation.profile = DEFAULT_TRANSLATION_RECIPE;
    if (!out.sentence || typeof out.sentence !== 'object') out.sentence = {};
    if (!['whitelist', 'keep', 'drop'].includes(out.sentence.short_policy)) out.sentence.short_policy = 'whitelist';
    return out;
  });
}
const subtitleRecipeValue = (model) => String(model && (model.recipe || model.id) || '');
const subtitleRegisteredValue = (model) => String(model && model.id || '');
const subtitleModelLabel = (model) => {
  if (model && (model.id === SUBTITLE_DUAL_VALUE || model.recipe === SUBTITLE_DUAL_VALUE)) {
    return '雙模型融合 · whisper-ja-1.5B + anime-whisper';
  }
  if (model && model.provider === 'openai-compatible') {
    const label = String(model.label || '雲端模型');
    return model.remote_model && !label.includes(String(model.remote_model)) ? `${label} · ${model.remote_model}` : label;
  }
  return String(model && (model.label || model.id) || '');
};
function subtitleModelChoices(kind) {
  return subtitleModels.filter((model) => model.kind === kind && subtitleRegisteredValue(model));
}
function fillSelect(select, rows, value) {
  const normalized = rows.map((row) => ({ value: String(row.value), label: String(row.label || row.value), disabled: !!row.disabled }));
  if (value != null && !normalized.some((row) => row.value === String(value))) normalized.push({ value: String(value), label: `${value}（目前未登錄）`, disabled: true });
  const signature = JSON.stringify(normalized);
  if (select.dataset.options !== signature) {
    select.dataset.options = signature;
    select.replaceChildren();
    for (const row of normalized) {
      const option = el('option', null, row.label);
      option.value = row.value; option.disabled = row.disabled;
      select.appendChild(option);
    }
    if (!normalized.length) {
      const option = el('option', null, '沒有可用方案'); option.value = ''; option.disabled = true; select.appendChild(option);
    }
  }
  if (value != null) select.value = String(value);
}
function subtitleAsrValue(primary) { return primary.asr.mode === 'dual' ? SUBTITLE_DUAL_VALUE : String(primary.asr.model || ''); }
function subtitleTranslationModelValue(model) { return subtitleRegisteredValue(model); }
function subtitleTranslationValue(primary) {
  return String(primary.translation.model_id || primary.translation.profile || primary.translation.recipe || primary.translation.model || primary.translation.model_file || DEFAULT_TRANSLATION_RECIPE);
}
const SUBTITLE_DOWNLOAD_HINT = '可到 設定 → 字幕 → 本機 下載本機字幕模型';
function subtitleMissingNote(model) {
  if (!model || model.provider !== 'local' || model.status !== 'missing') return '';
  if (model.id === SUBTITLE_DUAL_VALUE) {
    const gone = ['whisper-ja-1.5B', 'anime-whisper'].filter((name) => {
      const one = subtitleModels.find((m) => m.kind === 'asr' && (m.id === name || m.recipe === name));
      return !one || one.status === 'missing';
    });
    return `（缺少 ${(gone.length ? gone : ['雙模型']).join('、')}，${SUBTITLE_DOWNLOAD_HINT}）`;
  }
  return `（缺少 ${model.recipe || model.label}，${SUBTITLE_DOWNLOAD_HINT}）`;
}
function subtitleAsrChoices(selected) {
  const dualModel = subtitleModels.find((m) => m.id === SUBTITLE_DUAL_VALUE);
  const rows = [{
    value: SUBTITLE_DUAL_VALUE,
    label: (dualModel ? subtitleModelLabel(dualModel) : '雙模型融合 · whisper-ja-1.5B + anime-whisper') + subtitleMissingNote(dualModel),
    disabled: dualModel ? dualModel.status === 'missing' : false
  }];
  for (const model of subtitleModelChoices('asr')) {
    const value = subtitleRegisteredValue(model);
    if (!rows.some((row) => row.value === value)) rows.push({ value, label: `單模型 · ${subtitleModelLabel(model)}${subtitleMissingNote(model)}`, disabled: model.status === 'missing' });
  }
  if (selected && selected !== SUBTITLE_DUAL_VALUE && !rows.some((row) => row.value === selected)) rows.push({ value: selected, label: `單模型 · ${selected}（未登錄）`, disabled: true });
  return rows;
}
function subtitleTranslationChoices(selected) {
  const models = subtitleModelChoices('translation');
  const preferred = models.find((model) => model.is_default === true || model.default === true || subtitleRecipeValue(model) === DEFAULT_TRANSLATION_RECIPE);
  const rows = [];
  const modelRows = models.map((model) => {
    const value = subtitleTranslationModelValue(model), isDefault = model === preferred || (model.builtin && subtitleRecipeValue(model) === DEFAULT_TRANSLATION_RECIPE);
    return { value, label: `${isDefault ? '預設 · ' : ''}${subtitleModelLabel(model)}${subtitleMissingNote(model)}`, disabled: model.status === 'missing' };
  });
  rows.push(...modelRows);
  if (!rows.length && selected) rows.push({ value: selected, label: `目前方案 · ${selected}`, disabled: true });
  if (!rows.length) rows.push({ value: DEFAULT_TRANSLATION_RECIPE, label: `預設 · ${DEFAULT_TRANSLATION_RECIPE}`, disabled: false });
  if (selected && !rows.some((row) => row.value === selected)) rows.push({ value: selected, label: `目前方案 · ${selected}（未登錄）`, disabled: true });
  return rows;
}
function setSubtitleSubview(name, focusTarget) {
  subtitleSubview = ['main', 'advanced', 'models'].includes(name) ? name : 'main';
  setHidden($('#subtitleMainView'), subtitleSubview !== 'main');
  setHidden($('#subtitleAdvancedView'), subtitleSubview !== 'advanced');
  setHidden($('#subtitleModelsView'), subtitleSubview !== 'models');
  if (subtitleSubview === 'models') loadSubtitleModels();
  if (focusTarget) {
    $(focusTarget).focus();
    $('.sdlg-pane', drawer).scrollTop = 0;
  }
}
$('#subtitleMode').addEventListener('change', async (event) => {
  const profile = subtitleProfileValue();
  if (!profile) return;
  const newMode = event.target.value;
  subtitleLocalPeek = false;
  profile.active_mode = newMode;
  if (newMode === 'cloud') {
    if (current && current.settings) current.settings.subtitle_cloud_enabled = true;
    pushSetting('subtitle_cloud_enabled', true);
  }
  await saveSubtitleProfile(profile);
  renderSubtitleSettings(current.settings || {});
});
for (const key of ['groq']) {
  const prefix = 'subtitleGroq';
  $(`#${prefix}Form`).addEventListener('submit', (event) => { event.preventDefault(); saveSubtitleService(key); });
  $(`#${prefix}Connection`).addEventListener('change', (event) => {
    const val = event.target.value;
    subtitleConnectionChoice[key] = val;
    const keyInput = $(`#${prefix}Key`);
    if (keyInput) keyInput.value = '';
    subtitleServiceMessages[key] = val ? '已選擇連線；按「驗證並儲存連線」即可套用。' : '請輸入 API 金鑰以新增連線。';
    renderSubtitleHome(current.settings || {});
  });
}
$('#subtitleGroqAdd')?.addEventListener('click', () => openSubtitlePoolEditor(''));
$('#subtitleGroqCancel')?.addEventListener('click', () => {
  if (subtitleServiceBusy || subtitleProfileBusy) return;
  subtitlePoolEditing = null; $('#subtitleGroqKey').value = ''; renderSubtitleHome(current.settings || {});
});
$('#subtitleGroqAuto')?.addEventListener('change', async (event) => {
  const profile = subtitleProfileValue();
  if (!profile) return;
  const pool = subtitlePoolValue(profile); pool.auto_switch = event.target.checked;
  try { await saveSubtitlePool(pool); } catch (_) { subtitleServiceMessages.groq = '連線池儲存失敗，請重試。'; renderSubtitleHome(current.settings || {}); }
});
$('#subtitleGroqDelete')?.addEventListener('click', deleteSubtitlePoolConnection);
$('#subtitleGroqShowKey')?.addEventListener('click', () => {
  const input = $('#subtitleGroqKey');
  input.type = input.type === 'password' ? 'text' : 'password';
  setText($('#subtitleGroqShowKey'), input.type === 'password' ? '顯示' : '隱藏');
});
$('#subtitleLocalRecommend').addEventListener('click', async () => {
  const profile = subtitleProfileValue();
  if (!profile) return;
  Object.assign(profile.profiles.local, { asr: 'whisper-ja-1.5B', translation: DEFAULT_TRANSLATION_RECIPE,
    legacy_primary: { ...defaultSubtitlePrimary(), asr: { mode: 'single', model: 'whisper-ja-1.5B' } } });
  await saveSubtitleProfile(profile);
});

/* ------------------------------------------------------------------ local subtitle model download card */
const MDL_GROUPS = [
  { key: 'asr', title: '語音辨識（ASR）', sub: '至少選 1 個；兩個都選可使用雙模型，品質最好。' },
  { key: 'translation', title: '翻譯模型', sub: '至少選 1 個。' },
  { key: 'runtime', title: '執行程式', sub: '必要，無法取消勾選。' },
];

const mdlItems = () => (mdl.data ? Object.values(mdl.data.groups).flat() : []);
// verifying: the job checks a finished file's SHA256 between downloads; still active (polling, cancel, progress)
const MDL_ACTIVE = ['queued', 'running', 'verifying'];
const mdlRunning = () => !!mdl.progress && MDL_ACTIVE.includes(mdl.progress.state);
function mdlEffectiveIds() {
  // Selected ids plus the mandatory runtime when it is not installed yet; installed items are never downloaded again.
  return mdlItems().filter((it) => !it.installed && (mdl.selected.has(it.id) || it.category === 'runtime')).map((it) => it.id);
}
function mdlRules() {
  const covered = (cat) => mdlItems().some((it) => it.category === cat && (it.installed || mdl.selected.has(it.id)));
  if (!covered('asr')) return '請至少選擇一個語音辨識模型。';
  if (!covered('translation')) return '請至少選擇一個翻譯模型。';
  return '';
}
function mdlTotal() {
  const ids = new Set(mdlEffectiveIds());
  return mdlItems().filter((it) => ids.has(it.id)).reduce((n, it) => n + num(it.remaining_bytes), 0);
}

function mdlLicenseLines() {
  const ids = new Set(mdlEffectiveIds());
  return mdlItems().filter((it) => ids.has(it.id) && it.license_label)
    .map((it) => `．${it.label.replace(/（.*$/, '')}：${it.license_label}`).join('\n');
}

async function loadModelDl() {
  if (mdl.loading || !hasApi('subtitle_model_downloads')) { renderModelDl(); return; }
  mdl.loading = true;
  try {
    const res = await backend.subtitle_model_downloads();
    if (res && res.groups) {
      const first = !mdl.data;
      mdl.data = res;
      if (first) mdl.selected = new Set(res.default_ids || []);
      if (res.progress) mdl.progress = res.progress;
      for (const it of mdlItems()) if (it.installed) mdl.selected.delete(it.id);
      if (mdlRunning()) startModelDlPoll();
    }
  } catch (e) { /* the card shows its unavailable hint */ }
  mdl.loading = false;
  renderModelDl();
  refreshModelDlPlan();
}
async function refreshModelDlPlan() {
  if (!mdl.data || !hasApi('plan_subtitle_model_download')) return;
  const seq = ++mdl.planSeq;
  mdl.plan = null;
  renderModelDl();
  const res = await backend.plan_subtitle_model_download(mdlEffectiveIds());
  if (seq !== mdl.planSeq) return;
  mdl.plan = res || null;
  renderModelDl();
}
function renderModelDl() {
  const box = $('#subtitleModelDlGroups');
  if (!box) return;
  const running = mdlRunning();
  if (!mdl.data) {
    box.replaceChildren();
    const unavailable = !hasApi('subtitle_model_downloads');
    setText($('#subtitleModelDlStatus'), unavailable ? '目前環境無法管理字幕模型。' : '正在讀取模型狀態…');
    setText($('#subtitleModelDlSummary'), '');
    $('#subtitleModelDlStart').disabled = true;
    setHidden($('#subtitleModelDlManage'), true);
    return;
  }
  const groups = [];
  for (const g of MDL_GROUPS) {
    const items = mdl.data.groups[g.key] || [];
    if (!items.length) continue;
    const wrap = el('section', 'mdl-group');
    wrap.append(el('h5', null, g.title), el('p', 'mdl-sub', g.sub));
    for (const it of items) {
      const forced = it.category === 'runtime' && !it.installed;
      const row = el('label', 'mdl-row' + (it.installed ? ' installed' : ''));
      row.dataset.id = it.id;
      const cb = document.createElement('input');
      cb.type = 'checkbox';
      cb.dataset.id = it.id;
      cb.checked = it.installed || forced || mdl.selected.has(it.id);
      cb.disabled = it.installed || forced || running;
      cb.addEventListener('change', () => {
        if (cb.checked) mdl.selected.add(it.id); else mdl.selected.delete(it.id);
        refreshModelDlPlan();
      });
      const partial = !it.installed && num(it.partial_bytes) > 0 ? `（已下載 ${fmtBytes(it.partial_bytes)}）` : '';
      row.append(cb, el('span', 'mdl-label', it.label), el('span', 'mdl-size', it.installed ? '已安裝' : fmtBytes(it.size) + partial));
      if (it.description) row.append(el('span', 'mdl-desc', it.description));
      if (it.license_label) {
        const lic = el('span', 'mdl-lic', '授權：' + it.license_label + '　');
        if (it.card_url) {
          const link = el('button', 'btn link mdl-card', '模型卡');
          link.type = 'button';
          link.addEventListener('click', (event) => { event.preventDefault(); api('open_model_card', it.id); });
          lic.append(link);
        }
        row.append(lic);
      }
      wrap.append(row);
    }
    groups.push(wrap);
  }
  box.replaceChildren(...groups);

  const p = mdl.progress;
  const total = mdlTotal();
  const rule = mdlRules();
  const free = mdl.plan ? mdl.plan.free_bytes : mdl.data.free_bytes;
  let reason = rule;
  if (!reason && mdl.plan && !mdl.plan.ok) reason = mdl.plan.message || mdl.plan.error || '無法下載所選項目。';
  if (!reason && total <= 0) reason = '所選項目都已安裝，不需要下載。';
  const pending = !reason && !mdl.plan;
  setText($('#subtitleModelDlSummary'), running ? '下載進行中；已下載部分會保留，隨時可取消。'
    : (reason ? reason + '\n' : '') + `預計下載 ${fmtBytes(total)}　磁碟可用 ${fmtBytes(free)}`);
  const startBtn = $('#subtitleModelDlStart');
  const covered = (cat) => mdlItems().some((it) => it.category === cat && it.installed);
  const ready = covered('asr') && covered('translation') && covered('runtime');
  const gb = fmtBytes(total);
  let status;
  if (running) status = '正在下載本機字幕模型；已下載部分會保留，隨時可取消。';
  else if (p && p.state === 'failed') status = '下載未完成，已下載部分會保留。';
  else if (ready && !(p && ['cancelled'].includes(p.state))) status = '已安裝，可以使用。';
  else if (reason && !ready) status = reason;
  else status = `尚未下載：需要語音辨識模型、翻譯模型和執行程式，共約 ${gb}。`;
  setText($('#subtitleModelDlStatus'), status);
  setHidden(startBtn, running || (ready && !(p && p.state === 'failed')));
  setHidden($('#subtitleModelDlManage'), false);
  setText($('#subtitleModelDlManage'), ready ? '管理已下載的模型' : '選擇下載項目');
  startBtn.disabled = running || pending || !!reason || !hasApi('start_subtitle_model_download');
  startBtn.title = reason && !running ? reason : '';
  setText(startBtn, p && p.state === 'failed' ? '重試下載' : (p && p.state === 'cancelled' ? '繼續下載' : '下載本機字幕模型'));
  setHidden($('#subtitleModelDlCancel'), !running);

  const show = !!p && [...MDL_ACTIVE, 'cancelled', 'done'].includes(p.state);
  setHidden($('#subtitleModelDlProgress'), !show);
  setHidden($('#subtitleModelDlError'), !(p && p.state === 'failed'));
  if (p && p.state === 'failed') setText($('#subtitleModelDlError'), (p.error || '下載失敗。') + '\n已下載部分會保留，按「重試」可從中斷處繼續。');
  if (show) {
    const pct = p.bytes_total > 0 ? Math.min(100, p.bytes_done / p.bytes_total * 100) : 0;
    $('#subtitleModelDlBar').style.width = pct.toFixed(1) + '%';
    let text;
    if (p.state === 'done') text = '下載完成。';
    else if (p.state === 'cancelled') text = '已取消。已下載部分會保留，按「繼續下載」會從中斷處接續。';
    else {
      text = `${p.item_label || '準備中…'}　${fmtBytes(p.bytes_done)} / ${fmtBytes(p.bytes_total)}（${pct.toFixed(0)}%）`;
      if (p.state === 'verifying') text += '　正在核對檔案…';
      else if (p.speed > 0) text += `　${fmtBytes(p.speed)}/秒`;
      if (p.eta != null && p.state !== 'verifying') text += `　剩餘約 ${fmtEtaZh(p.eta)}`;
    }
    setText($('#subtitleModelDlProgressText'), text);
  }
}
function startModelDlPoll() {
  if (mdl.timer) return;
  mdl.timer = setInterval(async () => {
    const res = await backend.subtitle_model_download_progress();
    const wasRunning = mdlRunning();
    mdl.progress = (res && res.progress) || mdl.progress;
    if (!mdlRunning()) {
      clearInterval(mdl.timer); mdl.timer = null;
      if (!wasRunning) { renderModelDl(); return; }
      const last = mdl.progress;
      mdl.data = null;
      await loadModelDl();
      mdl.progress = last;
      renderModelDl();
      if (last && last.state === 'done') {
        toast('字幕模型下載完成');
        loadSubtitleModels(true);
        refresh();
      }
      return;
    }
    renderModelDl();
  }, 1000);
}
function ensureModelDlLoaded() {
  if (typeof mdl === 'undefined' || mdl.data || mdl.loading || mdl.autoTried) return;
  mdl.autoTried = true;
  loadModelDl();
}
function setModelDlDetails(open) {
  mdl.detailsOpen = open;
  setHidden($('#subtitleModelDlDetails'), !open);
  $('#subtitleModelDlManage').setAttribute('aria-expanded', String(open));
}
$('#subtitleModelDlManage').addEventListener('click', () => setModelDlDetails(!mdl.detailsOpen));
// 前往下載: settings -> 字幕 -> main view, local card visible, download button focused.
function goToModelDownload() {
  if (!settingsOpen()) openSettings();
  selectSettingsCat('subtitles', false);
  setSubtitleSubview('main');
  const profile = subtitleProfileValue();
  subtitleLocalPeek = !!profile && profile.active_mode !== 'local';
  renderSubtitleHome((current && current.settings) || {});
  ensureModelDlLoaded();
  const btn = $('#subtitleModelDlStart').hidden ? $('#subtitleModelDlManage') : $('#subtitleModelDlStart');
  btn.scrollIntoView({ block: 'center' });
  btn.focus();
}
for (const goBtn of document.querySelectorAll('[data-go-model-dl]')) goBtn.addEventListener('click', goToModelDownload);
$('#subtitleModelDlStart').addEventListener('click', async () => {
  if (!mdl.data || mdlRunning() || mdlRules()) return;
  const ok = await confirmDlg({
    title: '下載本機字幕模型',
    body: `將從網路下載 ${mdlEffectiveIds().length} 個項目，共約 ${fmtBytes(mdlTotal())}。\n已下載的部分會保留，中途取消後可從中斷處繼續。\n\n模型授權（模型由原作者提供，本程式不附帶；詳見各模型卡）：\n` + mdlLicenseLines(),
    ok: '同意授權並開始下載',
  });
  if (!ok) return;
  const res = await api('start_subtitle_model_download', mdlEffectiveIds());
  if (res && res.progress) mdl.progress = res.progress;
  renderModelDl();
  if (res && res.ok) startModelDlPoll();
});
$('#subtitleModelDlCancel').addEventListener('click', async () => {
  await api('cancel_subtitle_model_download');
});

$('#subtitleOpenAdvanced').addEventListener('click', () => setSubtitleSubview('advanced', '#subtitleBackFromAdvanced'));
$('#subtitleOpenModels').addEventListener('click', () => { setSubtitleSubview('models', '#subtitleBackFromModels'); loadSubtitleModels(); });
$('#subtitleBackFromAdvanced').addEventListener('click', () => setSubtitleSubview('main', '#subtitleOpenAdvanced'));
$('#subtitleBackFromModels').addEventListener('click', () => setSubtitleSubview('main', '#subtitleOpenModels'));

async function saveSubtitleConfiguration(primary, enabled, rows) {
  if (subtitleProfileBusy || subtitleServiceBusy) return false;
  const config = { primary: cloneConfig(primary), enabled: !!enabled, rows: cloneConfig(rows || []) };
  const settings = current && current.settings;
  if (settings) {
    settings.subtitle_primary = config.primary;
    settings.subtitle_comparison_enabled = config.enabled;
    settings.subtitle_comparison_rows = config.rows;
  }
  if (!hasApi('set_subtitle_configuration')) {
    toast('目前版本尚未支援字幕設定儲存。', true);
    return false;
  }
  lastEdit.subtitle_configuration = Date.now();
  subtitleSaveChain = subtitleSaveChain.catch(() => {}).then(async () => {
    const result = await api('set_subtitle_configuration', config.primary, config.enabled, config.rows, null);
    if (result && result.ok === false) {
      lastEdit.subtitle_configuration = 0;
      await refresh();
      return false;
    }
    if (result && result.subtitle_profile && current && current.settings) current.settings.subtitle_profile = result.subtitle_profile;
    return true;
  });
  return subtitleSaveChain;
}
function updatePrimaryValue(field, value) {
  const settings = current && current.settings || {};
  const primary = subtitlePrimaryFromSettings(settings), rows = subtitleRowsFromSettings(settings);
  const profile = typeof subtitleProfileValue === 'function' ? subtitleProfileValue() : null;
  if (field === 'asr') {
    const model = subtitleModelChoices('asr').find((item) => subtitleRegisteredValue(item) === value);
    primary.asr = value === SUBTITLE_DUAL_VALUE
      ? { ...primary.asr, mode: 'dual', model: 'whisper-ja-1.5B', decode: null }
      : { ...primary.asr, mode: 'single', model: model && model.builtin ? subtitleRegisteredValue(model) : (model ? model.id : value), decode: null };
  } else if (field === 'translation') {
    {
      const model = subtitleModelChoices('translation').find((item) => subtitleTranslationModelValue(item) === value);
      if (model && !model.builtin) {
        primary.translation = { ...primary.translation, mode: 'standard', model_id: model.id, model_file: null, gpu_layers: null, backend: null, prompt_mode: 'official-numbered' };
        delete primary.translation.provider; delete primary.translation.remote_model;
        if (!primary.translation.profile) primary.translation.profile = DEFAULT_TRANSLATION_RECIPE;
      } else {
        primary.translation = { ...primary.translation, mode: 'standard', profile: model ? subtitleRecipeValue(model) : value, model_file: null, gpu_layers: null, backend: null, prompt_mode: 'official-numbered' };
        delete primary.translation.model_id; delete primary.translation.provider; delete primary.translation.remote_model;
      }
      if (profile && profile.active_mode === 'custom' && typeof saveSubtitleProfile === 'function') {
        profile.profiles.custom.translation = model ? model.id : value;
        saveSubtitleProfile(profile);
      }
    }
  } else if (field === 'japanese_track') primary.japanese_track = !!value;
  else if (field === 'show_interjections') primary.sentence.show_interjections = !!value;
  else if (field === 'short_policy') primary.sentence.short_policy = value;
  if (profile && profile.active_mode === 'custom' && field === 'asr' && typeof saveSubtitleProfile === 'function') {
    const model = subtitleModelChoices('asr').find((item) => subtitleRegisteredValue(item) === value);
    profile.profiles.custom.asr = value === SUBTITLE_DUAL_VALUE ? SUBTITLE_DUAL_VALUE : (model ? model.id : value);
    saveSubtitleProfile(profile);
  }
  if (settings) settings.subtitle_primary = primary;
  return saveSubtitleConfiguration(primary, settings.subtitle_comparison_enabled === true, rows);
}
$('#subtitleAsr').addEventListener('change', (event) => updatePrimaryValue('asr', event.target.value));
$('#subtitleTranslation').addEventListener('change', (event) => updatePrimaryValue('translation', event.target.value));
$('#subtitleJapanese').addEventListener('change', (event) => updatePrimaryValue('japanese_track', event.target.checked));
$('#subtitleCloudStrategy')?.addEventListener('change', (event) => {
  const val = event.target.value;
  const profile = subtitleProfileValue();
  if (profile) {
    profile.profiles[profile.active_mode].processing = val;
    saveSubtitleProfile(profile);
  } else pushSetting('subtitle_cloud_strategy', val);
  const hint = $('#subtitleCloudStrategyHint');
  if (hint) {
    hint.textContent = val === 'full_preload'
      ? '首段開播後，背景直接將整部影片從頭到尾處理完畢（消耗完整影片額度）。'
      : '隨播放與影像修復進度處理；跳轉至未處理段落時優先生成字幕，實際時間依服務與內容而異。';
  }
});

function newSubtitleRowId() {
  subtitleRowIdSequence++;
  return `comparison-${Date.now().toString(36)}-${subtitleRowIdSequence}`;
}
function makeSubtitleComparisonRow(primary, index) {
  const row = cloneConfig(primary);
  row.id = newSubtitleRowId();
  row.label = `額外字幕軌 ${index + 1}`;
  return row;
}
function renderRowSelect(kind, selected, field, index) {
  const select = el('select'); select.dataset.rowField = field; select.dataset.rowIndex = String(index);
  const rows = kind === 'asr' ? subtitleAsrChoices(selected) : subtitleTranslationChoices(selected);
  fillSelect(select, rows, selected);
  return select;
}
function comparisonRowElement(row, index, total) {
  const article = el('article', 'subtitle-comparison-row'); article.dataset.rowIndex = String(index);
  const head = el('div', 'subtitle-row-head');
  head.appendChild(el('strong', null, `額外字幕軌 ${index + 1}`));
  const actions = el('div', 'subtitle-row-actions');
  const actionButton = (action, label, aria, disabled = false) => {
    const button = el('button', 'btn sm ghost', label); button.type = 'button'; button.dataset.rowAction = action; button.dataset.rowIndex = String(index);
    button.setAttribute('aria-label', aria); button.disabled = disabled; return button;
  };
  actions.append(actionButton('up', '↑', `將 ${row.label} 上移`, index === 0), actionButton('down', '↓', `將 ${row.label} 下移`, index === total - 1), actionButton('clone', '複製', `複製 ${row.label}`), actionButton('remove', '移除', `移除 ${row.label}`));
  head.appendChild(actions); article.appendChild(head);
  const fields = el('div', 'subtitle-row-fields');
  const nameLabel = el('label', null, '字幕軌名稱');
  const name = el('input'); name.type = 'text'; name.maxLength = 80; name.value = String(row.label || ''); name.dataset.rowField = 'label'; name.dataset.rowIndex = String(index);
  nameLabel.appendChild(name);
  const asrLabel = el('label', null, 'ASR 模型'); asrLabel.appendChild(renderRowSelect('asr', subtitleAsrValue(row), 'asr', index));
  const translationLabel = el('label', null, '翻譯模型'); translationLabel.appendChild(renderRowSelect('translation', subtitleTranslationValue(row), 'translation', index));
  fields.append(nameLabel, asrLabel, translationLabel); article.appendChild(fields);
  const more = el('details', 'subtitle-row-more');
  more.appendChild(el('summary', null, '更多選項'));
  const japanese = el('label', 'subtitle-option-row'); japanese.dataset.rowIndex = String(index);
  japanese.appendChild(el('span', null, '另外輸出日文辨識字幕軌'));
  const japaneseInput = el('input'); japaneseInput.type = 'checkbox'; japaneseInput.checked = !!row.japanese_track;
  japaneseInput.dataset.rowField = 'japanese_track'; japaneseInput.dataset.rowIndex = String(index);
  japanese.appendChild(japaneseInput); more.appendChild(japanese);
  article.appendChild(more);
  return article;
}
function renderSubtitleComparisonRows(settings) {
  const box = $('#subtitleComparisonRows'), rows = subtitleRowsFromSettings(settings);
  const sig = JSON.stringify({ rows, models: subtitleModels.map((model) => [model.id, model.label, model.status]) });
  if (box.dataset.signature === sig) return;
  const active = document.activeElement;
  const focusRow = active && active.dataset ? active.dataset.rowIndex : null;
  const focusField = active && active.dataset ? (active.dataset.rowField || active.dataset.rowAction) : null;
  box.dataset.signature = sig;
  box.replaceChildren();
  rows.forEach((row, index) => box.appendChild(comparisonRowElement(row, index, rows.length)));
  if (focusRow != null && focusField) {
    const restore = box.querySelector(`[data-row-index="${focusRow}"][data-row-field="${focusField}"], [data-row-index="${focusRow}"][data-row-action="${focusField}"]`);
    if (restore) restore.focus();
  }
}
async function changeSubtitleRow(index, field, value) {
  const settings = current && current.settings || {}, primary = subtitlePrimaryFromSettings(settings), rows = subtitleRowsFromSettings(settings);
  const row = rows[index]; if (!row) return;
  if (field === 'label') row.label = String(value || '').trim() || `額外字幕軌 ${index + 1}`;
  else if (field === 'asr') {
    const model = subtitleModelChoices('asr').find((item) => subtitleRegisteredValue(item) === value);
    row.asr = value === SUBTITLE_DUAL_VALUE ? { ...row.asr, mode: 'dual', model: 'whisper-ja-1.5B', decode: null } : { ...row.asr, mode: 'single', model: model && model.builtin ? subtitleRegisteredValue(model) : (model ? model.id : value), decode: null };
  }
  else if (field === 'translation') {
    {
      const model = subtitleModelChoices('translation').find((item) => subtitleTranslationModelValue(item) === value);
      if (model && !model.builtin) {
        row.translation = { ...row.translation, mode: 'standard', model_id: model.id, model_file: null, gpu_layers: null, backend: null, prompt_mode: 'official-numbered' };
        delete row.translation.provider; delete row.translation.remote_model;
        if (!row.translation.profile) row.translation.profile = DEFAULT_TRANSLATION_RECIPE;
      } else {
        row.translation = { ...row.translation, mode: 'standard', profile: model ? subtitleRecipeValue(model) : value, model_file: null, gpu_layers: null, backend: null, prompt_mode: 'official-numbered' };
        delete row.translation.model_id; delete row.translation.provider; delete row.translation.remote_model;
      }
    }
  } else if (field === 'japanese_track') row.japanese_track = !!value;
  else if (field === 'show_interjections') row.sentence.show_interjections = !!value;
  else if (field === 'short_policy') row.sentence.short_policy = value;
  settings.subtitle_comparison_rows = rows;
  renderSubtitleComparisonRows(settings);
  await saveSubtitleConfiguration(primary, settings.subtitle_comparison_enabled === true, rows);
}
$('#subtitleComparisonRows').addEventListener('change', (event) => {
  const target = event.target;
  if (!target.dataset.rowField) return;
  const value = target.type === 'checkbox' ? target.checked : target.value;
  changeSubtitleRow(Number(target.dataset.rowIndex), target.dataset.rowField, value);
});
$('#subtitleComparisonRows').addEventListener('click', async (event) => {
  const target = event.target.closest('[data-row-action]');
  if (!target || target.disabled) return;
  const settings = current && current.settings || {}, primary = subtitlePrimaryFromSettings(settings), rows = subtitleRowsFromSettings(settings);
  const index = Number(target.dataset.rowIndex);
  if (target.dataset.rowAction === 'clone' && rows[index]) rows.splice(index + 1, 0, { ...cloneConfig(rows[index]), id: newSubtitleRowId(), label: `${rows[index].label} 副本` });
  else if (target.dataset.rowAction === 'remove') rows.splice(index, 1);
  else if (target.dataset.rowAction === 'up' && index > 0) [rows[index - 1], rows[index]] = [rows[index], rows[index - 1]];
  else if (target.dataset.rowAction === 'down' && index < rows.length - 1) [rows[index + 1], rows[index]] = [rows[index], rows[index + 1]];
  settings.subtitle_comparison_rows = rows;
  renderSubtitleComparisonRows(settings);
  await saveSubtitleConfiguration(primary, settings.subtitle_comparison_enabled === true, rows);
});
$('#subtitleAddComparison').addEventListener('click', async () => {
  const settings = current && current.settings || {}, primary = subtitlePrimaryFromSettings(settings), rows = subtitleRowsFromSettings(settings);
  rows.push(makeSubtitleComparisonRow(primary, rows.length));
  settings.subtitle_comparison_rows = rows;
  if (settings.subtitle_comparison_enabled !== true) $('#subtitleComparisonEnabled').checked = true;
  setHidden($('#subtitleComparisonEditor'), false);
  await saveSubtitleConfiguration(primary, true, rows);
  renderSubtitleComparisonRows(settings);
});
$('#subtitleComparisonEnabled').addEventListener('change', async (event) => {
  const settings = current && current.settings || {}, primary = subtitlePrimaryFromSettings(settings), rows = subtitleRowsFromSettings(settings);
  if (event.target.checked && rows.length === 0) rows.push(makeSubtitleComparisonRow(primary, 0));
  settings.subtitle_comparison_rows = rows;
  settings.subtitle_comparison_enabled = event.target.checked;
  setHidden($('#subtitleComparisonEditor'), !event.target.checked);
  renderSubtitleComparisonRows(settings);
  await saveSubtitleConfiguration(primary, event.target.checked, rows);
});

function isLoopbackBaseUrl(url) {
  try {
    const h = new URL(String(url || '')).hostname.toLowerCase().replace(/^\[|\]$/g, '');
    return h === 'localhost' || h === '127.0.0.1' || h === '::1';
  } catch (e) { return false; }
}
// Mirrors model_registry.is_cloud_model: local files and loopback servers (Ollama / LM Studio) are not cloud.
function isCloudModel(model) {
  return !!model && model.provider !== 'local' && !isLoopbackBaseUrl(model.base_url);
}
function subtitleModelStatus(model) {
  if (model.status_label) return model.status_label;
  if (model.status === 'ready' || model.status === 'available') return model.provider === 'openai-compatible' ? '已通過新增測試；目前連線未檢查' : '本機模型可用';
  if (model.status === 'missing') return '找不到本機模型檔案';
  if (model.status === 'testing') return '測試中';
  if (model.status === 'failed' || model.status === 'error') return '最近測試失敗';
  if (model.status === 'unverified') return '待驗證';
  return model.builtin ? '內建方案' : '已登錄';
}
function subtitleModelAuthBadge(model) {
  if (!model || model.provider !== 'openai-compatible') return null;
  if (model.auth_source === 'dpapi') return { text: '金鑰已加密儲存 (DPAPI)', warn: false };
  if (model.auth_source === 'keyring') return { text: '金鑰已加密儲存 (Keychain)', warn: false };
  if (model.auth_source === 'env') return { text: '使用系統環境變數金鑰', warn: false };
  if (model.auth_source === 'keyless') return { text: '本機服務，不需金鑰', warn: false };
  if (model.auth_source === 'none') return { text: '尚未設定金鑰', warn: true };
  return null;
}
const CLOUD_SUBTITLE_PRESETS = {
  groq: {
    api_family: 'groq',
    base_url: 'https://api.groq.com/openai/v1',
    asr: {
      default_model: 'whisper-large-v3-turbo',
      default_label: 'Groq Whisper Large v3 Turbo',
      models: ['whisper-large-v3-turbo', 'whisper-large-v3'],
    },
    translation: {
      default_model: 'qwen/qwen3.8-27b',
      default_label: 'Groq Qwen 3.8 27B',
      models: ['qwen/qwen3.8-27b', 'openai/gpt-oss-120b', 'openai/gpt-oss-20b'],
    },
  },
  azure: {
    api_family: 'azure-translator',
    base_url: 'https://api.cognitive.microsofttranslator.com',
    asr: {
      default_model: '',
      default_label: 'Azure 語音服務（尚未支援）',
      models: [],
    },
    translation: {
      default_model: 'azure-translator-lines',
      default_label: 'Azure Translator (MT 逐行繁中)',
      models: ['azure-translator-lines'],
    },
  },
  gemini: {
    api_family: 'gemini',
    base_url: 'https://generativelanguage.googleapis.com/v1beta/openai',
    asr: {
      default_model: '',
      default_label: 'Gemini 語音服務',
      models: [],
    },
    translation: {
      default_model: 'gemini-2.5-flash',
      default_label: 'Gemini 2.5 Flash',
      models: ['gemini-2.5-flash', 'gemini-2.5-flash-lite', 'gemini-2.5-pro'],
    },
  },
  openrouter: {
    api_family: 'openrouter',
    base_url: 'https://openrouter.ai/api/v1',
    asr: {
      default_model: '',
      default_label: 'OpenRouter 語音服務',
      models: [],
    },
    translation: {
      default_model: 'google/gemini-2.5-flash',
      default_label: 'OpenRouter Gemini 2.5 Flash',
      models: ['google/gemini-2.5-flash', 'deepseek/deepseek-chat-v3-0324:free', 'qwen/qwen-2.5-72b-instruct'],
    },
  },
  ollama: {
    api_family: 'openai-compatible',
    base_url: 'http://localhost:11434/v1',
    asr: { default_model: '', default_label: 'Ollama（不支援語音辨識）', models: [] },
    translation: {
      default_model: 'qwen2.5:7b',
      default_label: 'Ollama 本機翻譯',
      models: ['qwen2.5:7b', 'qwen2.5:14b', 'gemma2:9b'],
    },
  },
  lmstudio: {
    api_family: 'openai-compatible',
    base_url: 'http://localhost:1234/v1',
    asr: { default_model: '', default_label: 'LM Studio（不支援語音辨識）', models: [] },
    translation: {
      default_model: 'qwen2.5-7b-instruct',
      default_label: 'LM Studio 本機翻譯',
      models: ['qwen2.5-7b-instruct', 'gemma-2-9b-it'],
    },
  },
  custom: {
    api_family: 'openai-compatible',
    base_url: '',
    asr: { default_model: '', default_label: '', models: [] },
    translation: { default_model: '', default_label: '', models: [] },
  },
};
const CLOUD_PRESET_LABELS = new Set(
  Object.values(CLOUD_SUBTITLE_PRESETS).flatMap((p) => [p.asr.default_label, p.translation.default_label]).filter(Boolean)
);
function renderSubtitleModelList() {
  const list = $('#subtitleModelList');
  const rows = subtitleModels.filter((model) => model.kind === subtitleModelKind);
  list.setAttribute('aria-labelledby', subtitleModelKind === 'asr' ? 'subtitleModelTabAsr' : 'subtitleModelTabTranslation');
  list.replaceChildren();
  if (!subtitleModelsLoaded && !subtitleModelLoadBusy) {
    list.appendChild(el('p', 'snote', '模型清單尚未載入。'));
    return;
  }
  if (!rows.length) {
    list.appendChild(el('p', 'snote', '目前沒有這類型的已登錄模型。'));
    return;
  }
  for (const model of rows) {
    const card = el('article', `subtitle-model-card ${model.status || 'unknown'}`);
    const head = el('div', 'subtitle-model-card-head');
    const title = el('strong', null, model.label);
    const badge = el('span', 'subtitle-model-status', subtitleModelStatus(model));
    head.append(title, badge); card.appendChild(head);
    const details = el('div', 'subtitle-model-meta');
    details.appendChild(el('span', null, model.provider === 'openai-compatible' ? '雲端 · OpenAI 相容' : '本機'));
    if (model.recipe) details.appendChild(el('span', null, model.recipe));
    if (model.provider === 'openai-compatible' && model.remote_model) details.appendChild(el('span', null, model.remote_model));
    const authBadge = subtitleModelAuthBadge(model);
    if (authBadge) details.appendChild(el('span', `subtitle-model-auth${authBadge.warn ? ' warn' : ''}`, authBadge.text));
    card.appendChild(details);
    if (model.removable) {
      const actions = el('div', 'subtitle-card-actions');
      let keyEditor = null;
      if (model.provider === 'openai-compatible' && hasApi('update_subtitle_model_key')) {
        const editKeyBtn = el('button', 'btn sm ghost', '更新金鑰'); editKeyBtn.type = 'button';
        keyEditor = el('div', 'subtitle-key-editor'); keyEditor.hidden = true;
        const keyWrap = el('div', 'subtitle-key-wrap');
        const keyInput = el('input'); keyInput.type = 'password'; keyInput.autocomplete = 'new-password'; keyInput.spellcheck = false;
        keyInput.placeholder = '貼上新 API Key（將先測試連線再以系統加密覆蓋）';
        const toggleBtn = el('button', 'btn sm ghost', '顯示'); toggleBtn.type = 'button';
        toggleBtn.addEventListener('click', () => {
          const show = keyInput.type === 'password';
          keyInput.type = show ? 'text' : 'password';
          setText(toggleBtn, show ? '隱藏' : '顯示');
        });
        const saveKeyBtn = el('button', 'btn sm primary', '測試並更新'); saveKeyBtn.type = 'button';
        const cancelKeyBtn = el('button', 'btn sm ghost', '取消'); cancelKeyBtn.type = 'button';
        const keyStatus = el('span', null, ''); keyStatus.setAttribute('role', 'status');
        keyWrap.append(keyInput, toggleBtn, saveKeyBtn, cancelKeyBtn);
        keyEditor.append(keyWrap, keyStatus);
        editKeyBtn.addEventListener('click', () => {
          keyEditor.hidden = !keyEditor.hidden;
          keyInput.value = ''; keyInput.type = 'password'; setText(toggleBtn, '顯示'); setText(keyStatus, '');
          if (!keyEditor.hidden) keyInput.focus();
        });
        cancelKeyBtn.addEventListener('click', () => {
          keyInput.value = ''; keyInput.type = 'password'; setText(toggleBtn, '顯示'); setText(keyStatus, '');
          keyEditor.hidden = true;
        });
        saveKeyBtn.addEventListener('click', async () => {
          const nextKey = keyInput.value.trim();
          if (!nextKey) { setText(keyStatus, '請輸入新的 API 金鑰。'); return; }
          saveKeyBtn.disabled = true; setText(keyStatus, '正在驗證新金鑰並更新加密憑證…');
          try {
            const result = await api('update_subtitle_model_key', model.id, nextKey);
            keyInput.value = ''; keyInput.type = 'password'; setText(toggleBtn, '顯示');
            if (!result || result.ok === false) {
              setText(keyStatus, String((result && result.error) || '金鑰驗證失敗，保留原有設定。'));
              return;
            }
            keyEditor.hidden = true;
            toast(`已更新「${model.label}」的加密金鑰`);
            await loadSubtitleModels(true);
          } finally { saveKeyBtn.disabled = false; }
        });
        actions.appendChild(editKeyBtn);
      }
      const remove = el('button', 'btn sm ghost danger', '移除'); remove.type = 'button';
      remove.addEventListener('click', async () => {
        const ok = await confirmDlg({ title: '移除字幕模型', body: `要從清單移除「${model.label}」並清除其儲存的加密金鑰嗎？`, ok: '移除', danger: true });
        if (!ok) return;
        const result = await api('remove_subtitle_model', model.id);
        if (result && result.ok === false) return;
        await loadSubtitleModels(true);
      });
      actions.appendChild(remove);
      card.appendChild(actions);
      if (keyEditor) card.appendChild(keyEditor);
    }
    list.appendChild(card);
  }
}
function setSubtitleModelKind(kind, focus = false) {
  subtitleModelKind = ['asr', 'translation', 'connections'].includes(kind) ? kind : 'asr';
  for (const [value, id] of [['asr', 'subtitleModelTabAsr'], ['translation', 'subtitleModelTabTranslation'], ['connections', 'subtitleModelTabConnections']]) {
    const button = $(`#${id}`);
    if (!button) continue;
    const selected = value === subtitleModelKind;
    button.setAttribute('aria-selected', String(selected)); button.tabIndex = selected ? 0 : -1;
    if (selected && focus) button.focus();
  }
  const isConn = subtitleModelKind === 'connections';
  setHidden($('#subtitleModelList'), isConn);
  setHidden($('#subtitleModelForm'), isConn);
  setHidden($('#subtitleConnectionListPanel'), !isConn);
  if (isConn) {
    renderSubtitleConnectionList();
  } else {
    renderSubtitleModelList(); syncSubtitleModelProvider();
  }
}
$('#subtitleModelTabAsr').addEventListener('click', () => setSubtitleModelKind('asr'));
$('#subtitleModelTabTranslation').addEventListener('click', () => setSubtitleModelKind('translation'));
$('#subtitleModelTabConnections')?.addEventListener('click', () => setSubtitleModelKind('connections'));
$('.subtitle-model-tabs').addEventListener('keydown', (event) => {
  if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
  event.preventDefault();
  const tabs = ['asr', 'translation', 'connections'];
  const curIdx = tabs.indexOf(subtitleModelKind);
  let nextIdx = 0;
  if (event.key === 'Home') nextIdx = 0;
  else if (event.key === 'End') nextIdx = tabs.length - 1;
  else if (event.key === 'ArrowLeft') nextIdx = (curIdx - 1 + tabs.length) % tabs.length;
  else if (event.key === 'ArrowRight') nextIdx = (curIdx + 1) % tabs.length;
  setSubtitleModelKind(tabs[nextIdx], true);
});

function renderSubtitleConnectionList() {
  const box = $('#subtitleConnectionList');
  if (!box) return;
  box.replaceChildren();
  if (!subtitleConnections.length) {
    box.appendChild(el('p', 'snote', '目前尚未登錄任何雲端服務連線。可在雲端推薦主頁設定 Groq 連線，或於「新增模型」加入其他雲端服務。'));
    return;
  }
  subtitleConnections.forEach((conn) => {
    const card = el('article', 'subtitle-connection-card');
    card.id = `conn-card-${conn.id}`;
    const head = el('div', 'subtitle-connection-head');
    const title = el('strong', null, conn.label || (conn.api_family === 'groq' ? 'Groq 服務連線'
      : conn.api_family === 'azure-translator' ? 'Azure Translator 服務連線' : 'OpenAI 相容服務連線'));
    const idBadge = el('span', 'subtitle-model-status', conn.id.slice(0, 18) + '…');
    head.append(title, idBadge);
    card.appendChild(head);

    const body = el('div', 'subtitle-connection-body');
    const authText = conn.auth_source === 'dpapi' ? '金鑰已加密儲存 (DPAPI)'
      : conn.auth_source === 'keyring' ? '金鑰已加密儲存 (Keychain)'
      : conn.auth_source === 'env' ? '使用系統環境變數'
      : conn.auth_source === 'keyless' ? '本機服務，不需金鑰' : '尚未設定金鑰';
    body.appendChild(el('div', null, `API 端點：${conn.base_url || '預設端點'}${conn.azure_region ? ' · 區域：' + conn.azure_region : ''}`));
    body.appendChild(el('div', null, `憑證保護：${authText}`));
    if (conn.validation) {
      if (conn.validation.asr) body.appendChild(el('div', null, subtitleCapabilityText(conn, 'asr')));
      if (conn.validation.translation) body.appendChild(el('div', null, subtitleCapabilityText(conn, 'translation')));
    }
    card.appendChild(body);

    const actions = el('div', 'subtitle-connection-actions');
    const updateKeyBtn = el('button', 'btn sm ghost', '更新金鑰');
    const removeBtn = el('button', 'btn sm ghost danger', '移除連線');
    const keyEditor = el('div', 'subtitle-key-editor');
    keyEditor.hidden = true;
    const keyWrap = el('div', 'subtitle-key-wrap');
    const keyInput = el('input'); keyInput.type = 'password'; keyInput.placeholder = '貼上新金鑰進行原地覆蓋（輪替）';
    const toggleBtn = el('button', 'btn sm ghost', '顯示');
    toggleBtn.addEventListener('click', () => {
      const show = keyInput.type === 'password';
      keyInput.type = show ? 'text' : 'password';
      setText(toggleBtn, show ? '隱藏' : '顯示');
    });
    const saveKeyBtn = el('button', 'btn sm primary', '驗證並更新');
    const cancelKeyBtn = el('button', 'btn sm ghost', '取消');
    const statusNote = el('span', null, ''); statusNote.setAttribute('role', 'status');
    keyWrap.append(keyInput, toggleBtn, saveKeyBtn, cancelKeyBtn);
    keyEditor.append(keyWrap, statusNote);

    updateKeyBtn.addEventListener('click', () => {
      keyEditor.hidden = !keyEditor.hidden;
      keyInput.value = ''; setText(statusNote, '');
      if (!keyEditor.hidden) keyInput.focus();
    });
    cancelKeyBtn.addEventListener('click', () => {
      keyEditor.hidden = true; keyInput.value = ''; setText(statusNote, '');
    });
    saveKeyBtn.addEventListener('click', async () => {
      const keyVal = keyInput.value.trim();
      if (!keyVal) { setText(statusNote, '請輸入新的金鑰'); return; }
      saveKeyBtn.disabled = true; setText(statusNote, '正在驗證新金鑰…');
      try {
        const spec = { id: conn.id, api_key: keyVal, api_family: conn.api_family, base_url: conn.base_url, azure_region: conn.azure_region };
        const res = await api('test_save_subtitle_connection', spec);
        if (!res || !res.connection) {
          setText(statusNote, String(res && res.error || '驗證失敗，保留原設定'));
          return;
        }
        toast('已更新連線金鑰');
        await loadSubtitleConnections(true);
        await loadSubtitleModels(true);
      } finally { saveKeyBtn.disabled = false; }
    });

    const errorMsg = el('p', 'subtitle-disabled-note', '');
    errorMsg.hidden = true;
    removeBtn.addEventListener('click', async () => {
      errorMsg.hidden = true;
      const ok = await confirmDlg({ title: '移除服務連線', body: '確定要移除此服務連線嗎？被目前字幕設定或佇列引用的連線無法直接刪除。', ok: '移除', danger: true });
      if (!ok) return;
      removeBtn.disabled = true;
      try {
        const res = await api('remove_subtitle_connection', conn.id);
        if (!res || res.ok !== true) {
          errorMsg.textContent = `無法移除：${res && res.error || '連線仍被引用或不存在'}`;
          errorMsg.hidden = false;
          return;
        }
        toast('已移除服務連線');
        await loadSubtitleConnections(true);
        await loadSubtitleModels(true);
      } finally { removeBtn.disabled = false; }
    });

    actions.append(updateKeyBtn, removeBtn);
    card.append(actions, keyEditor, errorMsg);
    box.appendChild(card);
  });
}
function applySubtitleCloudPreset(forceFill = false) {
  const presetEl = $('#subtitleModelPreset');
  if (!presetEl) return;
  const cloud = $('#subtitleModelProvider').value === 'openai-compatible';
  const presetKey = presetEl.value in CLOUD_SUBTITLE_PRESETS ? presetEl.value : 'custom';
  const preset = CLOUD_SUBTITLE_PRESETS[presetKey];
  const kindCfg = preset[subtitleModelKind] || { default_model: '', default_label: '', models: [] };
  const datalist = $('#subtitleModelRemoteList');
  if (datalist) {
    datalist.replaceChildren();
    for (const modelName of kindCfg.models) {
      const opt = el('option'); opt.value = modelName; datalist.appendChild(opt);
    }
  }
  const pairWrap = $('#subtitleModelPairGroqWrap');
  if (pairWrap) setHidden(pairWrap, !(cloud && presetKey === 'groq'));
  const azureWrap = $('#subtitleModelAzureRegionWrap');
  if (azureWrap) setHidden(azureWrap, !(cloud && presetKey === 'azure'));
  if (!cloud) return;
  const baseUrlEl = $('#subtitleModelBaseUrl'), remoteEl = $('#subtitleModelRemote'), labelEl = $('#subtitleModelLabel');
  if (presetKey === 'azure') {
    const regionEl = $('#subtitleModelAzureRegion');
    if (regionEl && (forceFill || !regionEl.value.trim())) regionEl.value = 'eastasia';
  }
  if (presetKey !== 'custom' && baseUrlEl && (forceFill || !baseUrlEl.value.trim())) {
    baseUrlEl.value = preset.base_url;
  }
  if (remoteEl && (forceFill || !remoteEl.value.trim())) {
    remoteEl.value = kindCfg.default_model;
  }
  if (labelEl && kindCfg.default_label && (forceFill || !labelEl.value.trim() || CLOUD_PRESET_LABELS.has(labelEl.value.trim()))) {
    labelEl.value = kindCfg.default_label;
  }
}
function resetSubtitleApiKeyToggle() {
  const keyEl = $('#subtitleModelApiKey'), toggleEl = $('#subtitleModelKeyToggle');
  if (keyEl) keyEl.type = 'password';
  if (toggleEl) setText(toggleEl, '顯示');
}
function syncSubtitleModelProvider() {
  const cloud = $('#subtitleModelProvider').value === 'openai-compatible';
  setText($('#subtitleModelPathLabel'), subtitleModelKind === 'asr' ? '本機模型資料夾' : '本機模型檔案（GGUF）');
  $('#subtitleModelPath').placeholder = subtitleModelKind === 'asr' ? '輸入模型資料夾路徑' : '輸入 .gguf 模型檔案路徑';
  const audioLabel = $('#subtitleModelAudioPathLabel'), audioInput = $('#subtitleModelAudioPath');
  if (audioLabel) setText(audioLabel, cloud ? 'ASR 測試音訊路徑（選填，留空使用內建 1 秒語音樣本）' : 'ASR 測試音訊路徑');
  if (audioInput) audioInput.placeholder = cloud ? '留空即使用內建 1 秒語音樣本測試' : '輸入本機語音樣本路徑';
  for (const node of document.querySelectorAll('.subtitle-cloud-field')) setHidden(node, !cloud);
  for (const node of document.querySelectorAll('.subtitle-local-field')) setHidden(node, cloud);
  for (const node of document.querySelectorAll('.subtitle-audio-field')) setHidden(node, subtitleModelKind !== 'asr');
  resetSubtitleApiKeyToggle();
  renderSubtitleModelRecipes();
  applySubtitleCloudPreset(false);
}
function renderSubtitleModelRecipes() {
  const select = $('#subtitleModelRecipe');
  if (!select) return;
  const kind = subtitleModelKind, provider = $('#subtitleModelProvider').value;
  const models = subtitleModels.filter((model) => model.kind === kind && model.builtin && model.provider === provider && subtitleRecipeValue(model));
  const values = [...new Set(models.map(subtitleRecipeValue))];
  if (provider === 'openai-compatible' && !values.length) values.push('openai-compatible');
  const rows = values.map((value) => {
    const model = models.find((item) => subtitleRecipeValue(item) === value);
    return { value, label: model ? subtitleModelLabel(model) : (kind === 'asr' ? 'OpenAI 相容 ASR' : 'OpenAI 相容翻譯') };
  });
  const previous = select.value;
  fillSelect(select, rows, rows.some((row) => row.value === previous) ? previous : rows[0] && rows[0].value);
  select.disabled = !rows.length;
  $('#subtitleModelTestAdd').disabled = !rows.length;
}
$('#subtitleModelProvider').addEventListener('change', () => {
  syncSubtitleModelProvider();
  if ($('#subtitleModelProvider').value === 'openai-compatible') applySubtitleCloudPreset(true);
});
if ($('#subtitleModelPreset')) {
  $('#subtitleModelPreset').addEventListener('change', () => applySubtitleCloudPreset(true));
}
if ($('#subtitleModelKeyToggle')) {
  $('#subtitleModelKeyToggle').addEventListener('click', () => {
    const keyEl = $('#subtitleModelApiKey'), toggleEl = $('#subtitleModelKeyToggle');
    if (!keyEl || !toggleEl) return;
    const show = keyEl.type === 'password';
    keyEl.type = show ? 'text' : 'password';
    setText(toggleEl, show ? '隱藏' : '顯示');
  });
}
$('#subtitleModelForm').addEventListener('submit', async (event) => {
  event.preventDefault();
  const kind = subtitleModelKind, provider = $('#subtitleModelProvider').value;
  const label = $('#subtitleModelLabel').value.trim(), recipe = $('#subtitleModelRecipe').value;
  const spec = { kind, provider, label, recipe };
  if (!label || !recipe) { setText($('#subtitleModelFormStatus'), '請填寫名稱並選擇模型方案。'); return; }
  const presetKey = ($('#subtitleModelPreset') && $('#subtitleModelPreset').value) || 'custom';
  const presetCfg = CLOUD_SUBTITLE_PRESETS[presetKey] || CLOUD_SUBTITLE_PRESETS.custom;
  if (provider === 'local') {
    spec.model_path = $('#subtitleModelPath').value.trim();
    if (!spec.model_path) { setText($('#subtitleModelFormStatus'), '請填寫本機模型檔案路徑。'); return; }
  } else {
    spec.base_url = $('#subtitleModelBaseUrl').value.trim();
    spec.remote_model = $('#subtitleModelRemote').value.trim();
    spec.api_key = $('#subtitleModelApiKey').value;
    spec.api_family = presetCfg.api_family || 'openai-compatible';
    if (presetKey === 'azure' || spec.api_family === 'azure-translator') {
      const reg = ($('#subtitleModelAzureRegion') && $('#subtitleModelAzureRegion').value.trim()) || 'eastasia';
      spec.azure_region = reg;
    }
    if (!spec.base_url || !spec.remote_model) { setText($('#subtitleModelFormStatus'), '請填寫服務網址與遠端模型名稱。'); return; }
    if (kind === 'asr' && (presetKey === 'ollama' || presetKey === 'lmstudio')) { setText($('#subtitleModelFormStatus'), 'Ollama / LM Studio 僅支援翻譯，不提供語音辨識。'); return; }
  }
  if (kind === 'asr') {
    const file = $('#subtitleModelAudio').files && $('#subtitleModelAudio').files[0];
    spec.test_audio_path = $('#subtitleModelAudioPath').value.trim() || (file && (file.pywebviewFullPath || file.path)) || '';
    if (!spec.test_audio_path && provider === 'openai-compatible') {
      spec.test_audio_path = '__builtin__';
    }
    if (!spec.test_audio_path) { setText($('#subtitleModelFormStatus'), '請選擇可供真實推論測試的音訊檔。'); return; }
  }
  const button = $('#subtitleModelTestAdd'); button.disabled = true;
  setText($('#subtitleModelFormStatus'), '正在執行真實推論測試…');
  try {
    const result = await api('test_add_subtitle_model', spec);
    if (!result || result.ok === false) { setText($('#subtitleModelFormStatus'), String((result && result.error) || '測試失敗，模型未加入。')); return; }
    let pairAdded = false;
    const pairGroq = provider === 'openai-compatible' && presetKey === 'groq' && $('#subtitleModelPairGroq') && $('#subtitleModelPairGroq').checked;
    if (pairGroq) {
      const otherKind = kind === 'asr' ? 'translation' : 'asr';
      const otherPreset = CLOUD_SUBTITLE_PRESETS.groq[otherKind];
      const alreadyExists = subtitleModels.some((m) => m.kind === otherKind && m.provider === 'openai-compatible' && m.base_url === spec.base_url && m.remote_model === otherPreset.default_model);
      if (!alreadyExists && otherPreset && otherPreset.default_model) {
        setText($('#subtitleModelFormStatus'), `主模型已加入，正在同步測試並啟用 ${otherPreset.default_label}…`);
        const pairSpec = {
          kind: otherKind, provider: 'openai-compatible', label: otherPreset.default_label, recipe: 'openai-compatible',
          base_url: spec.base_url, remote_model: otherPreset.default_model, api_key: spec.api_key, api_family: 'groq',
          ...(otherKind === 'asr' ? { test_audio_path: '__builtin__' } : {}),
        };
        const pairRes = await api('test_add_subtitle_model', pairSpec);
        pairAdded = !!(pairRes && pairRes.ok !== false);
      }
    }
    $('#subtitleModelApiKey').value = '';
    resetSubtitleApiKeyToggle();
    $('#subtitleModelAudio').value = '';
    $('#subtitleModelAudioPath').value = '';
    $('#subtitleModelForm').reset();
    await loadSubtitleModels(true);
    syncSubtitleModelProvider();
    setText($('#subtitleModelFormStatus'), pairAdded ? '推論測試成功，Groq 辨識與翻譯模型皆已加密儲存並加入。' : '推論測試成功，模型已加入。');
  } finally { button.disabled = false; }
});

function renderSubtitleSettings(settings) {
  renderSubtitleHome(settings);
  loadSubtitleConnections();
  if (fresh('subtitle_cloud_strategy') && $('#subtitleCloudStrategy')) {
    const savedProfile = subtitleProfileValue();
    const strat = savedProfile ? savedProfile.profiles[savedProfile.active_mode].processing : settings.subtitle_cloud_strategy || 'smart_horizon';
    $('#subtitleCloudStrategy').value = strat;
    $('#subtitleCloudStrategy').disabled = subtitleProfileBusy || !!subtitleServiceBusy;
    const hint = $('#subtitleCloudStrategyHint');
    if (hint) {
      hint.textContent = strat === 'full_preload'
        ? '首段開播後，背景直接將整部影片從頭到尾處理完畢（消耗完整影片額度）。'
        : '隨播放與影像修復進度處理；跳轉至未處理段落時優先生成字幕，實際時間依服務與內容而異。';
    }
  }
  const env = current && current.subtitle_environment ? current.subtitle_environment : {};
  const available = subtitleRuntimeAvailable(env);
  setText($('#subtitleEnvironment'), available ? '字幕處理已就緒' : subtitleRuntimeReason(env));
  $('#subtitleEnvironment').classList.toggle('unavailable', !available);

  const primary = subtitlePrimaryFromSettings(settings), rows = subtitleRowsFromSettings(settings);
  fillSelect($('#subtitleAsr'), subtitleAsrChoices(subtitleAsrValue(primary)), subtitleAsrValue(primary));
  fillSelect($('#subtitleTranslation'), subtitleTranslationChoices(subtitleTranslationValue(primary)), subtitleTranslationValue(primary));
  $('#subtitleAsr').disabled = subtitleProfileBusy || !!subtitleServiceBusy || !hasApi('set_subtitle_configuration');
  $('#subtitleTranslation').disabled = subtitleProfileBusy || !!subtitleServiceBusy || !hasApi('set_subtitle_configuration');
  $('#subtitleOpenAdvanced').disabled = !hasApi('set_subtitle_configuration');
  $('#subtitleOpenModels').disabled = !hasApi('list_subtitle_models');
  if (fresh('subtitle_configuration')) {
    $('#subtitleJapanese').checked = primary.japanese_track === true;
    $('#subtitleComparisonEnabled').checked = settings.subtitle_comparison_enabled === true;
    setHidden($('#subtitleComparisonEditor'), settings.subtitle_comparison_enabled !== true);
  }
  renderSubtitleComparisonRows(settings);
  renderSubtitleFallback();
  if (subtitleModelKind === 'connections') {
    renderSubtitleConnectionList();
  }
  const whisperPolicySel = $('#subtitleWhisperPolicy');
  if (whisperPolicySel) {
    const prof = subtitleProfileValue();
    const wp = (prof && prof.profiles && prof.profiles.cloud && prof.profiles.cloud.whisper_policy) || 'auto';
    whisperPolicySel.value = wp;
  }
  const whisperSec = $('#subtitleWhisperPolicySection');
  if (whisperSec) {
    const prof = subtitleProfileValue();
    const activeMode = prof && prof.active_mode;
    const active = prof && prof.profiles && prof.profiles[activeMode];
    const asrId = active && active.asr;
    const fallbackAsrs = (active && active.fallback && active.fallback.asr) || [];
    const usesGroqAsr = activeMode === 'cloud' ||
      asrId === 'groq-whisper-1' ||
      fallbackAsrs.some((id) => String(id).toLowerCase().includes('groq') || String(id).toLowerCase().includes('whisper-large'));
    setHidden(whisperSec, !usesGroqAsr);
  }

  renderSubtitleModelList(); renderSubtitleModelRecipes(); syncSubtitleModelProvider();
  setSubtitleSubview(subtitleSubview);
}

let subtitleFallbackFocus = null;   // keyboard users keep focus on the moved item's button across the re-render
const subtitleFallbackButtons = new Map();   // data-fallback-focus key -> button of the current render
function restoreSubtitleFallbackFocus() {
  const key = subtitleFallbackFocus;
  if (!key || subtitleProfileBusy) return;      // buttons are disabled while saving: retry on the next render
  subtitleFallbackFocus = null;
  let target = key.startsWith('#') ? $(key) : subtitleFallbackButtons.get(key);
  if (target && target.disabled) {
    // moved to the top/bottom: that arrow is now disabled, use the other arrow of the same item
    const other = key.endsWith(':up') ? key.replace(/:up$/, ':down') : key.replace(/:down$/, ':up');
    target = subtitleFallbackButtons.get(other);
  }
  if (target && !target.disabled && typeof target.focus === 'function') target.focus();
}
function renderSubtitleFallback() {
  const profile = subtitleProfileValue();
  if (!profile) return;
  // state pushes re-render this list often: keep a focused row button focused across the rebuild
  const activeEl = typeof document !== 'undefined' ? document.activeElement : null;
  const focused = activeEl && activeEl.dataset && activeEl.dataset.fallbackFocus;
  if (focused && !subtitleFallbackFocus) subtitleFallbackFocus = focused;
  subtitleFallbackButtons.clear();
  const active = profile.profiles[profile.active_mode];
  const fb = active.fallback || { asr: [], translation: [], max_attempts: 4, retries: 1, cooldown_seconds: 60 };

  const maxAttEl = $('#subtitleFallbackMaxAttempts');
  if (maxAttEl && fresh('subtitle_fallback_params')) maxAttEl.value = fb.max_attempts ?? 4;
  const retriesEl = $('#subtitleFallbackRetries');
  if (retriesEl && fresh('subtitle_fallback_params')) retriesEl.value = fb.retries ?? 1;
  const cdEl = $('#subtitleFallbackCooldown');
  if (cdEl && fresh('subtitle_fallback_params')) cdEl.value = fb.cooldown_seconds ?? 60;

  const asrBox = $('#subtitleFallbackAsrList');
  if (asrBox) {
    asrBox.replaceChildren();
    if (!fb.asr.length) {
      asrBox.appendChild(el('p', 'snote', '尚未設定辨識備援模型（不備援）'));
    } else {
      fb.asr.forEach((mid, idx) => {
        const model = subtitleModels.find((m) => m.id === mid);
        const item = el('div', 'subtitle-fallback-item');
        const meta = el('div', 'subtitle-fallback-meta');
        meta.appendChild(el('span', 'subtitle-fallback-order', `${idx + 1}.`));
        meta.appendChild(el('strong', null, model ? subtitleModelLabel(model) : (mid === SUBTITLE_DUAL_VALUE ? '雙模型融合 · whisper-ja-1.5B + anime-whisper' : mid)));
        meta.appendChild(el('span', 'subtitle-model-status', model ? subtitleModelStatus(model) : (mid === SUBTITLE_DUAL_VALUE ? '本機模型可用' : '未知模型')));
        item.appendChild(meta);

        const actions = el('div', 'subtitle-fallback-actions');
        const upBtn = el('button', 'btn sm ghost', '↑');
        upBtn.disabled = idx === 0 || subtitleProfileBusy;
        upBtn.setAttribute('aria-label', `將 ${model ? model.label : mid} 上移`);
        upBtn.dataset.fallbackFocus = `asr:${mid}:up`;
        subtitleFallbackButtons.set(`asr:${mid}:up`, upBtn);
        upBtn.addEventListener('click', async () => {
          subtitleFallbackFocus = `asr:${mid}:up`;
          const temp = fb.asr[idx - 1]; fb.asr[idx - 1] = fb.asr[idx]; fb.asr[idx] = temp;
          await saveSubtitleProfile(profile);
          renderSubtitleSettings(current.settings || {});
        });
        const downBtn = el('button', 'btn sm ghost', '↓');
        downBtn.disabled = idx === fb.asr.length - 1 || subtitleProfileBusy;
        downBtn.setAttribute('aria-label', `將 ${model ? model.label : mid} 下移`);
        downBtn.dataset.fallbackFocus = `asr:${mid}:down`;
        subtitleFallbackButtons.set(`asr:${mid}:down`, downBtn);
        downBtn.addEventListener('click', async () => {
          subtitleFallbackFocus = `asr:${mid}:down`;
          const temp = fb.asr[idx + 1]; fb.asr[idx + 1] = fb.asr[idx]; fb.asr[idx] = temp;
          await saveSubtitleProfile(profile);
          renderSubtitleSettings(current.settings || {});
        });
        const removeBtn = el('button', 'btn sm ghost danger', '✕');
        removeBtn.disabled = subtitleProfileBusy;
        removeBtn.setAttribute('aria-label', `移除 ${model ? model.label : mid}`);
        removeBtn.dataset.fallbackFocus = `asr:${mid}:remove`;
        subtitleFallbackButtons.set(`asr:${mid}:remove`, removeBtn);
        removeBtn.addEventListener('click', async () => {
          subtitleFallbackFocus = '#subtitleFallbackAsrAdd';
          fb.asr.splice(idx, 1);
          await saveSubtitleProfile(profile);
          renderSubtitleSettings(current.settings || {});
        });
        actions.append(upBtn, downBtn, removeBtn);
        item.appendChild(actions);
        asrBox.appendChild(item);
      });
    }
    const asrSelect = $('#subtitleFallbackAsrSelect');
    if (asrSelect) {
      const avail = subtitleModels.filter((m) => m.kind === 'asr' && m.id !== active.asr && !fb.asr.includes(m.id));
      if (!avail.some((m) => m.id === SUBTITLE_DUAL_VALUE) && active.asr !== SUBTITLE_DUAL_VALUE && !fb.asr.includes(SUBTITLE_DUAL_VALUE)) {
        avail.unshift({
          id: SUBTITLE_DUAL_VALUE,
          label: '雙模型融合 · whisper-ja-1.5B + anime-whisper',
          kind: 'asr',
          provider: 'local',
          status: 'ready',
          builtin: true
        });
      }
      fillSelect(asrSelect, avail.map((m) => ({ value: m.id, label: `${subtitleModelLabel(m)}（${isCloudModel(m) ? '雲端' : '本機'} · ${subtitleModelStatus(m)}）` })));
      const addBtn = $('#subtitleFallbackAsrAdd');
      if (addBtn) addBtn.disabled = !avail.length || subtitleProfileBusy;
    }
  }

  const transBox = $('#subtitleFallbackTranslationList');
  if (transBox) {
    transBox.replaceChildren();
    if (!fb.translation.length) {
      transBox.appendChild(el('p', 'snote', '尚未設定主要翻譯備援模型（不備援）'));
    } else {
      fb.translation.forEach((mid, idx) => {
        const model = subtitleModels.find((m) => m.id === mid);
        const item = el('div', 'subtitle-fallback-item');
        const meta = el('div', 'subtitle-fallback-meta');
        meta.appendChild(el('span', 'subtitle-fallback-order', `${idx + 1}.`));
        meta.appendChild(el('strong', null, model ? subtitleModelLabel(model) : mid));
        meta.appendChild(el('span', 'subtitle-model-status', model ? subtitleModelStatus(model) : '未知模型'));
        item.appendChild(meta);

        const actions = el('div', 'subtitle-fallback-actions');
        const upBtn = el('button', 'btn sm ghost', '↑');
        upBtn.disabled = idx === 0 || subtitleProfileBusy;
        upBtn.setAttribute('aria-label', `將 ${model ? model.label : mid} 上移`);
        upBtn.dataset.fallbackFocus = `translation:${mid}:up`;
        subtitleFallbackButtons.set(`translation:${mid}:up`, upBtn);
        upBtn.addEventListener('click', async () => {
          subtitleFallbackFocus = `translation:${mid}:up`;
          const temp = fb.translation[idx - 1]; fb.translation[idx - 1] = fb.translation[idx]; fb.translation[idx] = temp;
          await saveSubtitleProfile(profile);
          renderSubtitleSettings(current.settings || {});
        });
        const downBtn = el('button', 'btn sm ghost', '↓');
        downBtn.disabled = idx === fb.translation.length - 1 || subtitleProfileBusy;
        downBtn.setAttribute('aria-label', `將 ${model ? model.label : mid} 下移`);
        downBtn.dataset.fallbackFocus = `translation:${mid}:down`;
        subtitleFallbackButtons.set(`translation:${mid}:down`, downBtn);
        downBtn.addEventListener('click', async () => {
          subtitleFallbackFocus = `translation:${mid}:down`;
          const temp = fb.translation[idx + 1]; fb.translation[idx + 1] = fb.translation[idx]; fb.translation[idx] = temp;
          await saveSubtitleProfile(profile);
          renderSubtitleSettings(current.settings || {});
        });
        const removeBtn = el('button', 'btn sm ghost danger', '✕');
        removeBtn.disabled = subtitleProfileBusy;
        removeBtn.setAttribute('aria-label', `移除 ${model ? model.label : mid}`);
        removeBtn.dataset.fallbackFocus = `translation:${mid}:remove`;
        subtitleFallbackButtons.set(`translation:${mid}:remove`, removeBtn);
        removeBtn.addEventListener('click', async () => {
          subtitleFallbackFocus = '#subtitleFallbackTranslationAdd';
          fb.translation.splice(idx, 1);
          await saveSubtitleProfile(profile);
          renderSubtitleSettings(current.settings || {});
        });
        actions.append(upBtn, downBtn, removeBtn);
        item.appendChild(actions);
        transBox.appendChild(item);
      });
    }
    const transSelect = $('#subtitleFallbackTranslationSelect');
    if (transSelect) {
      const avail = subtitleModels.filter((m) => m.kind === 'translation' && m.id !== active.translation && !fb.translation.includes(m.id));
      fillSelect(transSelect, avail.map((m) => ({ value: m.id, label: `${subtitleModelLabel(m)}（${isCloudModel(m) ? '雲端' : '本機'} · ${subtitleModelStatus(m)}）` })));
      const addBtn = $('#subtitleFallbackTranslationAdd');
      if (addBtn) addBtn.disabled = !avail.length || subtitleProfileBusy;
    }
  }
  restoreSubtitleFallbackFocus();
}

async function addSubtitleFallback(role) {
  const profile = subtitleProfileValue();
  if (!profile) return;
  const select = role === 'asr' ? $('#subtitleFallbackAsrSelect') : $('#subtitleFallbackTranslationSelect');
  if (!select) return;
  const mid = select.value;
  if (!mid) return;
  const model = subtitleModels.find((m) => m.id === mid) || (mid === SUBTITLE_DUAL_VALUE ? { id: SUBTITLE_DUAL_VALUE, label: '雙模型融合 · whisper-ja-1.5B + anime-whisper', kind: 'asr', provider: 'local', status: 'ready', builtin: true } : null);
  if (!model) return;

  if (profile.active_mode === 'local' && isCloudModel(model)) {
    profile.active_mode = 'custom';
    Object.assign(profile.profiles.custom, cloneConfig(profile.profiles.local));
    toast('已自動切換至「自訂」模式以支援雲端備援模型。');
  }

  const target = profile.profiles[profile.active_mode];
  if (!target.fallback) target.fallback = { asr: [], translation: [], max_attempts: 4, retries: 1, cooldown_seconds: 60 };
  if (!target.fallback[role].includes(mid)) {
    target.fallback[role].push(mid);
  }
  await saveSubtitleProfile(profile);
  renderSubtitleSettings(current.settings || {});
}

$('#subtitleFallbackAsrAdd')?.addEventListener('click', () => addSubtitleFallback('asr'));
$('#subtitleFallbackTranslationAdd')?.addEventListener('click', () => addSubtitleFallback('translation'));

$('#subtitleFallbackMaxAttempts')?.addEventListener('change', async (event) => {
  const profile = subtitleProfileValue();
  if (!profile) return;
  const val = Number(event.target.value);
  if (!Number.isInteger(val) || val < 1 || val > 12) return;
  const target = profile.profiles[profile.active_mode];
  if (!target.fallback) target.fallback = { asr: [], translation: [], max_attempts: 4, retries: 1, cooldown_seconds: 60 };
  target.fallback.max_attempts = val;
  await saveSubtitleProfile(profile);
});
$('#subtitleFallbackRetries')?.addEventListener('change', async (event) => {
  const profile = subtitleProfileValue();
  if (!profile) return;
  const val = Number(event.target.value);
  if (!Number.isInteger(val) || val < 0 || val > 3) return;
  const target = profile.profiles[profile.active_mode];
  if (!target.fallback) target.fallback = { asr: [], translation: [], max_attempts: 4, retries: 1, cooldown_seconds: 60 };
  target.fallback.retries = val;
  await saveSubtitleProfile(profile);
});
$('#subtitleFallbackCooldown')?.addEventListener('change', async (event) => {
  const profile = subtitleProfileValue();
  if (!profile) return;
  const val = Number(event.target.value);
  if (!Number.isInteger(val) || val < 1 || val > 3600) return;
  const target = profile.profiles[profile.active_mode];
  if (!target.fallback) target.fallback = { asr: [], translation: [], max_attempts: 4, retries: 1, cooldown_seconds: 60 };
  target.fallback.cooldown_seconds = val;
  await saveSubtitleProfile(profile);
});

$('#subtitleWhisperPolicy')?.addEventListener('change', async (event) => {
  const profile = subtitleProfileValue();
  if (!profile) return;
  profile.profiles.cloud.whisper_policy = event.target.value;
  if (event.target.value === 'fixed') {
    profile.profiles.cloud.whisper_models = ['whisper-large-v3-turbo'];
  } else {
    profile.profiles.cloud.whisper_models = ['whisper-large-v3-turbo', 'whisper-large-v3'];
  }
  await saveSubtitleProfile(profile);
  renderSubtitleSettings(current.settings || {});
});

$('#subtitleOpenFallbackFromCloud')?.addEventListener('click', () => {
  setSubtitleSubview('advanced', '#subtitleFallbackSection');
});
$('#subtitleOpenFallbackFromLocal')?.addEventListener('click', () => {
  setSubtitleSubview('advanced', '#subtitleFallbackSection');
});

/* ------------------------------------------------------------------ global keys */

document.addEventListener('keydown', (e) => {
  if (e.key !== 'Escape') return;
  if (!$('#reprocessModal').hidden) closeReprocessDialog();
  else if (!$('#modal').hidden) closeModal(false);
  else if (settingsOpen()) closeSettings();
});
$('#modalCancel').addEventListener('click', () => closeModal(false));
$('#modalOk').addEventListener('click', () => closeModal(true));
$('#modal').addEventListener('mousedown', (e) => { if (e.target === $('#modal')) closeModal(false); });
document.addEventListener('contextmenu', (e) => {
  if (!(e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement)) e.preventDefault();
});

/* ------------------------------------------------------------------ GPU panel (off by default) */

const GPU_KEY = 'ohStreamer.queue.gpuPanel';
const GPU_POINTS = 120;            // 1 sample / s -> last 2 minutes
const gpuHist = { util: [], mem: [], temp: [], power: [], enc: [], dec: [] };
let gpuOpen = false, gpuTimer = null, gpuLast = null;

function lsGet(k) { try { return localStorage.getItem(k); } catch (e) { return null; } }
function lsSet(k, v) { try { localStorage.setItem(k, v); } catch (e) { /* storage unavailable */ } }

function drawChart(canvas, series, yMax) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth, h = canvas.clientHeight;
  if (!w || !h) return;
  if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
    canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr);
  }
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  const cs = getComputedStyle(document.documentElement);
  const grid = cs.getPropertyValue('--chart-grid').trim() || 'rgba(128,128,128,.2)';
  const tick = cs.getPropertyValue('--chart-tick').trim() || '#888';
  const padL = 30, padB = 4, padT = 6, pw = w - padL - 4, ph = h - padB - padT;
  // pixel skin: Silkscreen at 8px (its native grid) for the numeric ticks; legacy skin keeps the UI font
  ctx.font = document.documentElement.dataset.skin === 'pixel'
    ? '8px ' + (cs.getPropertyValue('--px').trim() || 'monospace')
    : '11px ' + (cs.getPropertyValue('--font') || 'sans-serif');
  ctx.textBaseline = 'middle';
  for (let i = 0; i <= 4; i++) {
    const y = padT + ph * i / 4;
    ctx.strokeStyle = grid; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(padL, y + .5); ctx.lineTo(w - 4, y + .5); ctx.stroke();
    ctx.fillStyle = tick; ctx.textAlign = 'right';
    ctx.fillText(String(Math.round(yMax * (1 - i / 4))), padL - 6, y);
  }
  for (const s of series) {
    const d = s.data;
    if (d.length < 2) continue;
    ctx.strokeStyle = s.color; ctx.lineWidth = 2; ctx.lineJoin = ctx.lineCap = 'round';
    ctx.setLineDash(s.dash || []);
    ctx.beginPath();
    d.forEach((v, i) => {
      const x = padL + pw * (i + (GPU_POINTS - d.length)) / (GPU_POINTS - 1);
      const y = padT + ph * (1 - Math.max(0, Math.min(1, v / yMax)));
      if (i) ctx.lineTo(x, y); else ctx.moveTo(x, y);
    });
    ctx.stroke();
  }
  ctx.setLineDash([]);
}

function pushHist(key, v) {
  const a = gpuHist[key];
  a.push(typeof v === 'number' && isFinite(v) ? v : (a.length ? a[a.length - 1] : 0));
  if (a.length > GPU_POINTS) a.shift();
}

function gpuClass(node, v, warn, danger) {
  node.classList.toggle('warn', v >= warn && v < danger);
  node.classList.toggle('danger', v >= danger);
}

// stat tile value: big number + small unit suffix ("–" alone when there's no reading); rebuilt only on change
function setStat(node, v, unit, digits = 0) {
  const has = typeof v === 'number' && isFinite(v);
  const sig = has ? v.toFixed(digits) + unit : '–';
  if (node.dataset.sig === sig) return;
  node.dataset.sig = sig;
  node.replaceChildren(has ? v.toFixed(digits) : '–');
  if (has && unit) node.appendChild(el('span', 'u', unit));
}

// pixel skin: 10-block mini meter under a stat tile (pct 0..100; any non-zero reading lights at least one block)
function setGbar(node, pct, warn = Infinity, danger = Infinity) {
  if (!node.childElementCount) for (let i = 0; i < 10; i++) node.appendChild(document.createElement('i'));
  const p = typeof pct === 'number' && isFinite(pct) ? Math.min(100, Math.max(0, pct)) : 0;
  const on = p > 0 ? Math.max(1, Math.round(p / 10)) : 0;
  [...node.children].forEach((b, i) => b.classList.toggle('on', i < on));
  gpuClass(node, p, warn, danger);
}

// series colours come from the skin's --chart-1 ... --chart-6 tokens (fallbacks = the legacy palette)
const CHART_FALLBACK = ['#4cc2ff', '#ff9a3c', '#ff6b6b', '#6ccb5f', '#c08cff', '#f5d04b'];
function chartColors() {
  const cs = getComputedStyle(document.documentElement);
  return CHART_FALLBACK.map((fb, i) => cs.getPropertyValue('--chart-' + (i + 1)).trim() || fb);
}

function drawGpuCharts() {
  const cc = chartColors();
  drawChart($('#chUtil'), [
    { data: gpuHist.util, color: cc[0] },
    { data: gpuHist.mem, color: cc[1], dash: [5, 3] },
  ], 100);
  const pMax = Math.max(100, Math.ceil(Math.max(...gpuHist.power, 1) / 25) * 25);
  drawChart($('#chTemp'), [
    { data: gpuHist.temp.map((t) => t * pMax / 100), color: cc[2] },
    { data: gpuHist.power, color: cc[3], dash: [5, 3] },
  ], pMax);
  drawChart($('#chCodec'), [
    { data: gpuHist.enc, color: cc[4] },
    { data: gpuHist.dec, color: cc[5], dash: [5, 3] },
  ], 100);
}

function renderGpu(g) {
  gpuLast = g;
  const ok = g && typeof g.util === 'number';
  const note = $('#gpuNote'), mini = $('#gpuMini');
  if (!ok) {
    setText(note, g && g.backend === null ? '找不到 NVIDIA 驅動（NVML / nvidia-smi）' : '讀取中…');
    setHidden(mini, true);
    return;
  }
  const memPct = g.mem_total_gib ? g.mem_used_gib / g.mem_total_gib * 100 : 0;
  pushHist('util', g.util); pushHist('mem', memPct); pushHist('temp', g.temp); pushHist('power', g.power_w);
  pushHist('enc', g.enc_util); pushHist('dec', g.dec_util);
  setText($('#gpuName'), g.name || 'GPU');
  setText(note, '最近 2 分鐘 · 每秒更新（溫度以功耗軸等比例繪製）');
  setStat($('#gUtil'), g.util, '%');
  setStat($('#gMem'), g.mem_used_gib, 'GB', 1);
  setText($('#gMemTot'), g.mem_total_gib ? ` / ${num(g.mem_total_gib).toFixed(1)} GB` : '');
  setStat($('#gTemp'), g.temp, '°C');
  setStat($('#gPower'), g.power_w, 'W');
  setStat($('#gEnc'), g.enc_util, '%');
  setStat($('#gDec'), g.dec_util, '%');
  gpuClass($('#gMem'), memPct, 80, 92);
  gpuClass($('#gTemp'), num(g.temp), 80, 87);
  // power is scaled to the driver's enforced limit; without one, to the chart's power axis
  const pCap = g.power_limit_w > 0 ? g.power_limit_w : Math.max(100, Math.ceil(Math.max(...gpuHist.power, 1) / 25) * 25);
  $('#gPowerBox').title = g.power_limit_w > 0 ? `方塊條：功耗上限 ${Math.round(g.power_limit_w)} W` : '';
  setGbar($('#gUtilBar'), g.util);
  setGbar($('#gMemBar'), memPct, 80, 92);
  setGbar($('#gTempBar'), g.temp == null ? 0 : (g.temp - 30) / 60 * 100, (80 - 30) / 60 * 100, (87 - 30) / 60 * 100);
  setGbar($('#gPowerBar'), g.power_w == null ? 0 : g.power_w / pCap * 100);
  setGbar($('#gEncBar'), g.enc_util);
  setGbar($('#gDecBar'), g.dec_util);
  drawGpuCharts();
  setText(mini, `${Math.round(g.util)}% · ${num(g.mem_used_gib).toFixed(1)}G`);
  setHidden(mini, false);
}

function pollGpu() {
  if (!gpuOpen) return;
  Promise.resolve().then(() => backend.gpu(true)).then(renderGpu).catch(() => {});
}

function setGpuOpen(open, persist = true) {
  gpuOpen = open;
  const btn = $('#btnGpu');
  setAttr(btn, 'aria-expanded', open);
  btn.title = open ? '隱藏 GPU 用量' : '顯示 GPU 用量';
  setHidden($('#gpuPanel'), !open);
  if (persist) lsSet(GPU_KEY, open ? '1' : '0');
  clearInterval(gpuTimer); gpuTimer = null;
  if (open) { pollGpu(); gpuTimer = setInterval(pollGpu, 1000); }
  else { Promise.resolve().then(() => backend.gpu(false)).catch(() => {}); setHidden($('#gpuMini'), true); }
}

$('#btnGpu').addEventListener('click', () => setGpuOpen(!gpuOpen));
window.addEventListener('resize', () => { if (gpuOpen && gpuHist.util.length) drawGpuCharts(); });

/* ------------------------------------------------------------------ skin switcher (trial period) */

// app.css = "pixel" (default), app-modern.css = the frozen legacy look. Ctrl+Shift+U toggles; choice kept in localStorage;
// ?skin=modern forces the legacy skin (without saving it).
const SKIN_KEY = 'ohStreamer.skin';
const SKIN_CSS = { pixel: 'app.css', modern: 'app-modern.css' };

function currentSkin() { return document.documentElement.dataset.skin === 'modern' ? 'modern' : 'pixel'; }

function applySkin(skin, persist) {
  skin = skin === 'modern' ? 'modern' : 'pixel';
  const link = $('#skinCss');
  document.documentElement.dataset.skin = skin;
  if (persist) lsSet(SKIN_KEY, skin);
  if (!link) return;
  const done = () => {
    if (gpuOpen && gpuHist.util.length) drawGpuCharts();   // chart colours come from the skin's CSS tokens
    if (skin === 'pixel') drawLogo();
  };
  if (link.getAttribute('href') === SKIN_CSS[skin]) { done(); return; }
  // freeze transitions/animations (rule lives in index.html's inline <style>) until the new sheet has been applied
  const root = document.documentElement;
  root.classList.add('skin-switching');
  let finished = false;
  const finish = () => {
    if (finished) return;
    finished = true;
    link.removeEventListener('load', finish);
    link.removeEventListener('error', finish);
    done();
    // two frames: the first runs before the new styles are computed, the second after
    requestAnimationFrame(() => requestAnimationFrame(() => root.classList.remove('skin-switching')));
  };
  link.addEventListener('load', finish);
  link.addEventListener('error', finish);
  setTimeout(finish, 4000);
  link.setAttribute('href', SKIN_CSS[skin]);
}

document.addEventListener('keydown', (e) => {
  if (e.ctrlKey && e.shiftKey && !e.altKey && !e.metaKey && (e.key === 'U' || e.key === 'u')) {
    e.preventDefault();
    applySkin(currentSkin() === 'pixel' ? 'modern' : 'pixel', true);
  }
});

/* ------------------------------------------------------------------ branding (from branding.js) */

const BRAND_NAME = (window.BRAND && window.BRAND.name) || 'Streamer';
const BRAND_LOGO = (window.BRAND && window.BRAND.logo) || BRAND_NAME;
{
  document.title = BRAND_NAME;
  const h1 = document.querySelector('h1.modern-only');
  if (h1) h1.textContent = BRAND_NAME;
  const lg = document.getElementById('pxLogo');
  if (lg) lg.title = BRAND_NAME;
}

/* ------------------------------------------------------------------ pixel logo (pixel skin only) */

// BRAND.logo in an RGB-offset triad, drawn through a shrinking pixelation so it resolves over ~1 s; click replays.
// After the intro a state loop runs (30 fps cap): processing = bar flows + faster breathing + micro-glitch;
// unwatched finished items (state done, seen === false; missing `seen` counts as seen) = sparkles (burst, then 25%);
// idle = very slow breathing. The sparkle layer is independent of processing (drawn on top of either). Reduced motion: only the final static frame.
const LOGO_H = 50;
const LOGO_FONT_PX = 46;
const LOGO_PAD = 6;
const LOGO_FPS = 30;
const LOGO_INTRO = [22, 16, 12, 9, 7, 5, 4, 3, 2, 1], LOGO_INTRO_MS = 95;
const LOGO_FLOW_MS = 2400;          // the r/g/b bar moves one bar width per this while processing
const LOGO_COLORS = ['#ff4766', '#36db86', '#4a86ff'];
const LOGO_FONT = (px) => `900 ${px}px "Arial Black", "Segoe UI Black", "Helvetica Neue", Arial, sans-serif`;
const reducedMotion = window.matchMedia ? window.matchMedia('(prefers-reduced-motion: reduce)') : { matches: false };
const lrand = (a, b) => a + Math.random() * (b - a);
const lpick = (arr) => arr[Math.floor(Math.random() * arr.length)];
const mkCanvas = (w, h) => { const c = document.createElement('canvas'); c.width = w; c.height = h; return c; };

const logoState = { proc: false, unseen: 0 };
let logoL = null;       // built renderer (masks etc.)
let logoRaf = 0, logoLast = 0, logoT0 = 0, logoBurstT0 = -1e9, logoPrevUnseen = 0;
let logoRun = null;     // per-run animation state

function buildLogo() {
  const cv = $('#pxLogo');
  if (!cv) return null;
  const dpr = window.devicePixelRatio || 1;
  const L = { cv, g: cv.getContext('2d'), dpr };
  const m = mkCanvas(1, 1).getContext('2d'); m.font = LOGO_FONT(LOGO_FONT_PX);
  L.wCss = Math.ceil(m.measureText(BRAND_LOGO).width) + LOGO_PAD * 2;
  L.W = Math.round(L.wCss * dpr); L.H = Math.round(LOGO_H * dpr);
  cv.width = L.W; cv.height = L.H;
  cv.style.width = L.wCss + 'px'; cv.style.height = LOGO_H + 'px';
  const bar = document.querySelector('.px-rgb');
  if (bar) bar.style.width = L.wCss + 'px';
  L.y = L.H - Math.round(6 * dpr); L.x0 = Math.round(LOGO_PAD * dpr);
  L.base = Math.max(1, Math.round(1.6 * dpr));
  const mask = (color) => {
    const c = mkCanvas(L.W, L.H), g = c.getContext('2d');
    g.font = LOGO_FONT(Math.round(LOGO_FONT_PX * dpr)); g.textBaseline = 'alphabetic'; g.fillStyle = color; g.fillText(BRAND_LOGO, L.x0, L.y);
    return c;
  };
  L.masks = LOGO_COLORS.map(mask);
  L.white = mask('#fff');
  L.buf = mkCanvas(L.W, L.H); L.bg = L.buf.getContext('2d'); L.tmp = mkCanvas(1, 1);
  // glyph cells (sparkle positions): inside cells + edge cells
  const s = Math.max(2, Math.round(3 * dpr)); L.cell = s;
  const d = L.white.getContext('2d').getImageData(0, 0, L.W, L.H).data;
  const on = (xx, yy) => xx >= 0 && yy >= 0 && xx < L.W && yy < L.H && d[((yy + (s >> 1)) * L.W + xx + (s >> 1)) * 4 + 3] > 160;
  L.cells = []; L.edges = [];
  for (let yy = 0; yy + s <= L.H; yy += s) for (let xx = 0; xx + s <= L.W; xx += s) {
    if (!on(xx, yy)) continue;
    L.cells.push([xx, yy]);
    if (!on(xx - s, yy) || !on(xx + s, yy) || !on(xx, yy - s) || !on(xx, yy + s)) L.edges.push([xx, yy]);
  }
  L.sizeBar = () => { if (bar) { L.barW = bar.offsetWidth || L.wCss; bar.style.backgroundSize = L.barW + 'px 100%'; } };
  L.sizeBar();
  // The r/g/b bar scrolls one bar width per period while processing. The position accumulates from 0 (no jump when it
  // starts) and, once processing stops (period null), runs on to the next whole width before resting (no snap back).
  L.flowPos = 0; L.flowT = null;
  L.flow = (t, period) => {
    if (!bar) return;
    if (t == null) { L.flowPos = 0; L.flowT = null; bar.style.backgroundPosition = '0 0'; return; }
    const dt = L.flowT == null ? 0 : t - L.flowT;
    L.flowT = t;
    if (period == null && L.flowPos === 0) return;
    L.flowPos += dt / (period || LOGO_FLOW_MS) * L.barW;
    if (period == null && L.flowPos >= L.barW) L.flowPos = 0; else L.flowPos %= L.barW;
    bar.style.backgroundPosition = Math.round(L.flowPos / 2) * 2 + 'px 0';
  };
  L.rgb = (g, off = L.base, dy = 0, layers = L.masks) => {
    g.clearRect(0, 0, L.W, L.H);
    g.globalCompositeOperation = 'lighter';
    g.drawImage(layers[0], -off, -dy); g.drawImage(layers[1], 0, 0); g.drawImage(layers[2], off, dy);
    g.globalCompositeOperation = 'source-over';
  };
  L.pixelate = (g, src, b) => {
    g.clearRect(0, 0, L.W, L.H);
    if (b <= 1) { g.drawImage(src, 0, 0); return; }
    const tw = Math.ceil(L.W / b), th = Math.ceil(L.H / b);
    L.tmp.width = tw; L.tmp.height = th; L.tmp.getContext('2d').drawImage(src, 0, 0, tw, th);
    g.imageSmoothingEnabled = false; g.drawImage(L.tmp, 0, 0, tw, th, 0, 0, tw * b, th * b); g.imageSmoothingEnabled = true;
  };
  L.slices = (g, src, bands) => {
    g.clearRect(0, 0, L.W, L.H);
    let y = 0;
    for (const [h, dx] of bands) { g.drawImage(src, 0, y, L.W, h, dx, y, L.W, h); y += h; }
  };
  L.bands = (maxShift, prob) => {
    const out = []; let y = 0;
    while (y < L.H) { const h = Math.round(lrand(3, 12) * dpr); out.push([h, Math.random() < prob ? Math.round(lrand(-maxShift, maxShift) * dpr) : 0]); y += h; }
    return out;
  };
  L.sparks = [];
  L.addSpark = (t, kind) => {
    const src = kind === 'star' ? L.edges : L.cells;
    if (!src.length) return;
    const [x, y] = lpick(src);
    L.sparks.push({ x, y, t0: t, life: kind === 'star' ? lrand(500, 800) : lrand(350, 650), kind, color: lpick(LOGO_COLORS) });
  };
  L.drawSparks = (g, t) => {
    const c = L.cell;
    L.sparks = L.sparks.filter((p) => t - p.t0 < p.life);
    for (const p of L.sparks) {
      const a = 1 - (t - p.t0) / p.life;
      g.globalAlpha = a;
      if (p.kind === 'star') {
        g.globalCompositeOperation = 'lighter'; g.fillStyle = '#fff';
        g.fillRect(p.x, p.y, c, c);
        g.globalAlpha = a * .75;
        g.fillRect(p.x - c, p.y, c, c); g.fillRect(p.x + c, p.y, c, c); g.fillRect(p.x, p.y - c, c, c); g.fillRect(p.x, p.y + c, c, c);
        if (a > .5) { g.globalAlpha = a - .5; g.fillRect(p.x - 2 * c, p.y, c, c); g.fillRect(p.x + 2 * c, p.y, c, c); g.fillRect(p.x, p.y - 2 * c, c, c); g.fillRect(p.x, p.y + 2 * c, c, c); }
      } else {
        g.globalCompositeOperation = 'source-atop'; g.fillStyle = p.color; g.fillRect(p.x, p.y, c, c);
      }
    }
    g.globalAlpha = 1; g.globalCompositeOperation = 'source-over';
  };
  return L;
}

// The phase starts at 0 when the intro ends (offset = base, the intro's last frame) and accumulates, so neither the
// end of the intro nor a processing/idle switch (different period) makes the logo jump.
function logoBreathe(L, t, period, amp) {
  L.ph = (L.ph || 0) + (L.pt == null ? 0 : (t - L.pt) / period * Math.PI * 2);
  L.pt = t;
  L.rgb(L.g, Math.max(0, L.base + Math.round(Math.sin(L.ph) * amp * L.dpr * 2) / 2));
}

function logoStateFrame(L, t) {
  const R = logoRun, g = L.g;
  if (logoState.proc) {
    if (t >= R.next && t > R.end) { R.end = t + lrand(90, 150); R.next = R.end + lrand(2200, 4500); }
    if (t < R.end) {
      if (t - R.lastB > 45) { R.lastB = t; R.b = L.bands(3, .35); }
      L.rgb(L.bg, L.base + Math.round(2 * L.dpr)); L.slices(g, L.buf, R.b);
    } else logoBreathe(L, t, 2400, 1);
  } else logoBreathe(L, t, 7000, .7);
  // unwatched sparkle layer: independent of processing
  if (logoState.unseen > 0 && t - R.lastSp > 50) {
    R.lastSp = t;
    const rate = t - logoBurstT0 < 1100 ? 1 : .25;   // completion burst, then the steady "unwatched" level
    R.acc += 2 * rate;
    while (R.acc >= 1) { L.addSpark(t, 'pix'); R.acc -= 1; }
    if (Math.random() < .5 * rate) L.addSpark(t, 'star');
  }
  L.drawSparks(g, t);
}

function logoTick(t) {
  logoRaf = requestAnimationFrame(logoTick);
  if (t - logoLast < 1000 / LOGO_FPS - 2) return;
  logoLast = t;
  const L = logoL;
  if (!L) return;
  // the first rAF timestamp can be a little earlier than performance.now() at scheduling time: clamp to frame 0
  const i = Math.max(0, Math.floor((t - logoT0) / LOGO_INTRO_MS));
  L.flow(t, logoState.proc ? LOGO_FLOW_MS : null);              // the bar runs during the intro too
  if (i < LOGO_INTRO.length) {
    L.rgb(L.bg); L.pixelate(L.g, L.buf, Math.max(1, Math.round(LOGO_INTRO[i] * L.dpr)));
    return;
  }
  logoStateFrame(L, t);
}

function stopLogoLoop() { if (logoRaf) cancelAnimationFrame(logoRaf); logoRaf = 0; }

// (re)build the masks and replay the intro, then run the state loop
function drawLogo() {
  stopLogoLoop();
  if (currentSkin() !== 'pixel') return;
  const L = buildLogo();
  if (!L) return;
  logoL = L;
  if (reducedMotion.matches) { L.rgb(L.g); L.flow(null); return; }
  const t0 = performance.now();
  logoT0 = t0;
  logoRun = { next: t0 + LOGO_INTRO.length * LOGO_INTRO_MS + 1500, end: 0, lastB: 0, lastSp: 0, acc: 0, b: [] };
  logoRaf = requestAnimationFrame(logoTick);
}

// called from render() with the current items
function updateLogoState(items) {
  let proc = false, unseen = 0;
  for (const it of items || []) {
    if (it.state === 'processing') proc = true;
    if (it.state === 'done' && it.seen === false) unseen++;
  }
  if (unseen > logoPrevUnseen) logoBurstT0 = performance.now();
  logoPrevUnseen = unseen;
  logoState.proc = proc; logoState.unseen = unseen;
  const cv = $('#pxLogo');
  if (cv) cv.title = unseen > 0 ? `有 ${unseen} 部完成未看 · 點一下全部標為已看` : BRAND_NAME;
}

function onLogoClick() {
  if (logoState.unseen > 0 && backend && typeof backend.mark_all_seen === 'function') {
    Promise.resolve().then(() => backend.mark_all_seen()).catch(() => {});
  }
  drawLogo();
}

{
  const logo = $('#pxLogo');
  if (logo) logo.addEventListener('click', onLogoClick);
  window.addEventListener('resize', () => { if (logoL) logoL.sizeBar(); });
  // wait for fonts so the glyphs are the final ones; drawing is harmless while the canvas is hidden (legacy skin)
  (document.fonts && document.fonts.ready ? document.fonts.ready : Promise.resolve()).then(drawLogo, drawLogo);
}

(function initSkin() {
  let skin = lsGet(SKIN_KEY) === 'modern' ? 'modern' : 'pixel';
  try { if (new URLSearchParams(location.search).get('skin') === 'modern') skin = 'modern'; } catch (e) { /* keep saved skin */ }
  applySkin(skin, false);
})();

/* ------------------------------------------------------------------ performance check panel */

// one table: 30 fps off / auto / full; 60 fps at its own rate (above 40 fps never interpolated, so only "off") and
// restored at 30 (the player's 修復幀率「最高 30fps」, matrix key 60r30) with auto interpolation back to 60
const PERF_GROUPS = [['30 fps 影片', 3], ['60 fps 影片', 2]];
const PERF_COLS = [['30', 'off', '不補幀'], ['30', 'auto', '自動補幀'], ['30', 'full', '全補幀'], ['60', 'off', '原始幀率'],
                   ['60r30', 'auto', '降到 30']];
const PERF_HEIGHTS = [1080, 720, 480];
let perfOpen = false, perfCtlSig = '', perfResSig = '', perfLast = null;

function fmtRatio(v) { return v == null || !isFinite(+v) ? '—' : (+v).toFixed(1) + 'x'; }

function perfCell(row, half) {
  if (!row || typeof row !== 'object') { const e = el('div', 'perf-cell empty'); e.appendChild(el('b', null, '—')); return e; }
  const level = row.level === 'live' || row.level === 'prebuffer' ? row.level : 'queue';
  const c = el('div', 'perf-cell ' + level);
  // auto only spends spare speed on interpolation, so its speed is "off"'s: show how much it can interpolate instead
  const frac = row.mode === 'auto' && row.interp_fraction != null && isFinite(+row.interp_fraction)
    ? Math.round(+row.interp_fraction * 100) : null;
  c.appendChild(el('b', null, frac == null ? fmtRatio(row.ratio_typical) : frac > 0 ? `補 ${frac}%` : '不補'));
  const pre = num(row.prebuffer_s);
  c.appendChild(el('span', 'lvl', level === 'live' ? '可即時' : level === 'prebuffer' ? (pre > 0 ? `先緩衝 ${Math.round(pre)} 秒` : '先緩衝') : '建議佇列'));
  c.title = frac == null
    ? `典型 ${fmtRatio(row.ratio_typical)}　吃重場景 ${fmtRatio(row.ratio_heavy)}\n（倍率 ≥ 1.0x 才追得上即時播放）`
    : `只用多出來的速度補幀，不會拖慢處理：約 ${frac}% 的格數會補上\n（處理速度同不補幀：典型 ${fmtRatio(row.ratio_typical)}）`;
  if (half) c.title += '\n播放器「修復幀率：最高 30fps」：60 fps 影片先降到 30 fps 再修復，自動補幀補回 60 fps（佇列不支援）';
  return c;
}

function perfBtn(label, cls, fn, title) {
  const b = el('button', 'btn ' + cls, label);
  b.type = 'button';
  if (title) b.title = title;
  b.addEventListener('click', fn);
  return b;
}

const PERF_KINDS = [['quick', '快速', '約 1.5 分鐘'], ['standard', '標準', '約 3.5 分鐘，結果較準']];
let perfKind = null;   // chosen in the panel; defaults to the last run's kind, else standard

function perfStart(kind) { act('perf_start', kind || perfKind || 'standard'); }

function renderPerfCtl(perf) {
  const status = perf.status || 'none';
  const running = status === 'running';
  const box = $('#perfCtl');
  if (!perfKind) perfKind = perf.kind === 'quick' ? 'quick' : 'standard';
  // progress is updated in place so the bar animates and nothing is rebuilt per tick
  const sig = JSON.stringify([status, perf.kind, perf.can_start, perf.blocked_reason, perf.error, !!perf.profile, perfKind]);
  if (sig !== perfCtlSig) {
    perfCtlSig = sig;
    box.replaceChildren();
    if (running) {
      const wrap = el('div', 'perf-run');
      const top = el('div', 'perf-run-top');
      top.append(el('span', 'perf-stage'), el('span', 'perf-pct'), perfBtn('取消', 'sm', () => act('perf_cancel')));
      const bar = el('div', 'bar'); bar.appendChild(el('i'));
      bar.setAttribute('role', 'progressbar'); bar.setAttribute('aria-valuemin', '0'); bar.setAttribute('aria-valuemax', '100');
      wrap.append(top, bar);
      box.appendChild(wrap);
    } else {
      if (status === 'failed' || status === 'cancelled') {
        const f = el('div', 'perf-fail' + (status === 'cancelled' ? ' neutral' : ''));
        f.appendChild(svgEl('alert'));
        f.appendChild(el('span', null, status === 'cancelled' ? '檢測已取消。' : `檢測失敗：${perf.error || '未知錯誤'}`));
        box.appendChild(f);
      }
      // one start button; quick / standard is a switch next to it
      const row = el('div', 'perf-btns');
      const blocked = perf.can_start === false;
      const tip = blocked ? (perf.blocked_reason || '目前無法開始檢測') : '';
      const seg = el('div', 'seg perf-kind');
      seg.setAttribute('role', 'radiogroup'); seg.setAttribute('aria-label', '檢測方式');
      for (const [value, label, hint] of PERF_KINDS) {
        const l = el('label'); l.title = hint;
        const input = el('input'); input.type = 'radio'; input.name = 'perfKind'; input.value = value;
        input.checked = value === perfKind;
        input.addEventListener('change', () => { perfKind = value; perfCtlSig = ''; renderPerfCtl(perfLast || perf); });
        l.append(input, el('span', null, label));
        seg.appendChild(l);
      }
      const retry = status === 'failed' || status === 'cancelled';
      const label = retry ? '重試檢測' : perf.profile ? '重新檢測' : '開始檢測';
      const go = perfBtn(label, perf.profile && !retry ? '' : 'primary', () => perfStart(perfKind), tip);
      go.disabled = blocked;
      row.append(seg, go, el('span', 'perf-kind-hint', PERF_KINDS.find((k) => k[0] === perfKind)[2]));
      if (blocked) row.title = tip;
      box.appendChild(row);
      if (blocked && perf.blocked_reason) box.appendChild(el('p', 'perf-block', String(perf.blocked_reason)));
    }
  }
  if (running) {
    const p = Math.max(0, Math.min(1, num(perf.progress)));
    setText($('.perf-stage', box), perf.stage || '準備中…');
    const left = perf.remaining_s == null ? null : num(perf.remaining_s);
    const leftText = left == null ? '' : (left < 60 ? ' · 剩不到 1 分' : ` · 約剩 ${Math.round(left / 60)} 分`);
    setText($('.perf-pct', box), Math.round(p * 100) + '%' + leftText);
    const bar = $('.bar', box);
    $('i', bar).style.width = (p * 100).toFixed(1) + '%';
    setAttr(bar, 'aria-valuenow', Math.round(p * 100));
  }
}

// the check measures every detection model; the table shows one at a time, the one in use first
const PERF_DET_HINT = { 'v4-fast': '速度較快', 'v4-accurate': '偵測較準' };
const PERF_LEVEL = { live: '可即時', prebuffer: '需先緩衝', queue: '建議佇列' };
let perfDet = null;   // detector the table shows (null: the one in use)

function perfEntries(prof) {
  const list = (Array.isArray(prof.by_detector) ? prof.by_detector : [])
    .filter((e) => e && e.detector && e.matrix && typeof e.matrix === 'object');
  return list.length ? list : [{ detector: prof.detector || '', matrix: prof.matrix || {}, recommend: prof.recommend }];
}

function perfRow(entry, fps, h, mode) {
  const rows = Array.isArray(entry.matrix[fps]) ? entry.matrix[fps] : [];
  return rows.find((r) => r && num(r.height) === h && r.mode === mode) || null;
}

// How the two models differ on this computer: the speed gap, and the cells whose verdict changes.
function perfCompare(a, b, advice) {
  const cells = [];
  for (const h of PERF_HEIGHTS) {
    for (const [fps, mode, name] of [['30', 'off', '不補幀'], ['30', 'full', '全補幀'], ['60', 'off', '60 fps 影片'],
      ['60r30', 'auto', '60 fps 降到 30']]) {
      const ra = perfRow(a, fps, h, mode), rb = perfRow(b, fps, h, mode);
      if (ra && rb && num(ra.ratio_typical) > 0 && num(rb.ratio_typical) > 0) cells.push({ h, name, ra, rb });
    }
  }
  if (!cells.length) return null;
  const mean = (k) => cells.reduce((s, c) => s + num(c[k].ratio_typical), 0) / cells.length;
  const [fast, slow, fk, sk] = mean('ra') >= mean('rb') ? [a, b, 'ra', 'rb'] : [b, a, 'rb', 'ra'];
  const pcts = cells.map((c) => Math.round((1 - num(c[sk].ratio_typical) / num(c[fk].ratio_typical)) * 100));
  const lo = Math.max(0, Math.min(...pcts)), hi = Math.max(0, Math.max(...pcts));
  const fn = modelLabel('detection_model', fast.detector), sn = modelLabel('detection_model', slow.detector);
  let text = `${sn} 比 ${fn} 慢 ${lo === hi ? `約 ${lo}` : `${lo}–${hi}`}%。`;
  const diff = cells.filter((c) => c[fk].level !== c[sk].level);
  if (!diff.length) {
    text += '兩者在每種情況的判定（可即時／先緩衝／佇列）都相同，速度差異不影響能否播放'
      + (advice ? `，建議用偵測較準的 ${modelLabel('detection_model', advice)}。` : '。');
  } else {
    const shown = diff.slice(0, 4).map((c) => `${c.h}p ${c.name}（${fn} ${PERF_LEVEL[c[fk].level] || '—'}、${sn} ${PERF_LEVEL[c[sk].level] || '—'}）`);
    text += `判定不同的情況：${shown.join('；')}${diff.length > 4 ? ` 等 ${diff.length} 項` : ''}。`
      + `常看這些影片又想即時播放就選 ${fn}，否則可選 ${sn}。`;
  }
  return text;
}

const PERF_SETTING_VALUE = { interpolation: { off: '關', auto: '自動' }, restoration_fps: { original: '原始', 30: '最高 30fps' } };

function perfSettingValue(key, v) {
  if (key === 'detection_model') return modelLabel('detection_model', v);
  return (PERF_SETTING_VALUE[key] || {})[v] || String(v);
}

// "套用建議設定", spelled out: every player setting it writes, from -> to, and why
function perfPlan(plan, running) {
  const wrap = el('div', 'perf-plan');
  const changes = plan.changes.filter((c) => c && c.key);
  const todo = changes.filter((c) => c.from !== c.to);
  wrap.appendChild(el('p', 'perf-plan-head', todo.length ? '套用建議設定會改播放器的這些設定：' : '播放器設定已經符合建議：'));
  const ul = el('ul', 'perf-plan-list');
  for (const c of changes) {
    const li = el('li', c.from !== c.to ? 'chg' : 'same');
    li.appendChild(el('b', null, String(c.label || c.key) + '：'));
    li.append(c.from !== c.to
      ? `${perfSettingValue(c.key, c.from)} → ${perfSettingValue(c.key, c.to)}`
      : `${perfSettingValue(c.key, c.to)}（維持）`);
    if (c.reason) li.appendChild(el('span', 'why', String(c.reason)));
    ul.appendChild(li);
  }
  wrap.appendChild(ul);
  if (plan.error) wrap.appendChild(el('p', 'perf-block', String(plan.error)));
  const foot = el('div', 'perf-plan-foot');
  foot.appendChild(el('span', 'perf-sub', '佇列的設定不會改；播放器下次開啟時生效。'));
  const ap = perfBtn(todo.length ? '套用建議設定' : '已是建議設定', 'primary sm', () => act('perf_apply'));
  ap.disabled = running || !todo.length;
  foot.appendChild(ap);
  wrap.appendChild(foot);
  return wrap;
}

function renderPerfResult(perf) {
  const box = $('#perfResult');
  const prof = perf.profile;
  const running = perf.status === 'running';
  if (!prof || typeof prof !== 'object') { perfResSig = ''; setHidden(box, true); return; }
  const entries = perfEntries(prof);
  const inUse = entries.find((e) => e.detector === prof.detector) || entries[0];
  const entry = entries.find((e) => e.detector === perfDet) || inUse;
  const plan = perf.apply_plan && Array.isArray(perf.apply_plan.changes) ? perf.apply_plan : null;
  const sig = JSON.stringify([prof, plan, running, entry.detector, current && current.model_options]);
  if (sig === perfResSig) return;
  perfResSig = sig;
  box.replaceChildren();
  box.style.opacity = running ? '.5' : '';

  const meta = el('div', 'perf-meta');
  meta.appendChild(el('span', 'perf-gpu', prof.gpu || '未知 GPU'));
  const when = prof.created ? new Date(prof.created) : null;
  const bits = [];
  if (when && !isNaN(when)) bits.push(when.toLocaleString('zh-TW', { hour12: false, year: 'numeric', month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit' }));
  if (num(prof.duration_s) > 0) bits.push(`耗時 ${fmtDur(prof.duration_s)}`);
  if (bits.length) meta.appendChild(el('span', 'perf-sub', bits.join(' · ')));
  box.appendChild(meta);

  const chips = el('div', 'perf-chips');
  const chip = (text, title) => { const c = el('span', 'chip perf-warn', text); if (title) c.title = title; chips.appendChild(c); };
  if (prof.on_ac_power === false) chip('未接電源，結果可能偏低');
  for (const w of Array.isArray(prof.warnings) ? prof.warnings : []) chip(String(w));
  if (Array.isArray(prof.stale) && prof.stale.length) chip('硬體或版本已變更，建議重測', prof.stale.join('、'));
  if (Array.isArray(prof.settings_mismatch) && prof.settings_mismatch.length) chip('目前設定與檢測時不同：' + prof.settings_mismatch.join('、'));
  box.appendChild(chips);

  // which detection model the table shows
  const det = el('div', 'perf-det');
  det.appendChild(el('span', 'perf-det-label', '偵測模型'));
  if (entries.length > 1) {
    const seg = el('div', 'seg');
    seg.setAttribute('role', 'radiogroup'); seg.setAttribute('aria-label', '表格顯示的偵測模型');
    for (const e of entries) {
      const l = el('label');
      const input = el('input'); input.type = 'radio'; input.name = 'perfDet'; input.value = e.detector;
      input.checked = e === entry;
      input.addEventListener('change', () => { perfDet = e.detector; renderPerfResult(perfLast || perf); });
      // the player and the queue each have their own detection model setting
      const users = [];
      if (plan && plan.player && plan.player.detection_model === e.detector) users.push('播放器');
      if (e.detector === prof.current_detector) users.push('佇列');
      const name = modelLabel('detection_model', e.detector) + (users.length ? `（${users.join('、')}使用中）` : '');
      l.append(input, el('span', null, name));
      if (PERF_DET_HINT[e.detector]) l.title = PERF_DET_HINT[e.detector];
      seg.appendChild(l);
    }
    det.appendChild(seg);
  } else {
    det.appendChild(el('span', 'perf-det-one', modelLabel('detection_model', entry.detector) || '—'));
  }
  box.appendChild(det);
  if (entries.length > 1) {
    const hints = entries.filter((e) => PERF_DET_HINT[e.detector])
      .map((e) => `${modelLabel('detection_model', e.detector)} ${PERF_DET_HINT[e.detector]}`);
    const cmp = perfCompare(entries[0], entries[1], prof.detector_advice && prof.detector_advice.detector);
    if (cmp) box.appendChild(el('p', 'perf-cmp', (hints.length ? hints.join('，') + '。' : '') + cmp));
  }

  const table = el('div', 'perf-table'); table.setAttribute('role', 'table');
  table.appendChild(el('span'));
  for (const [label, span] of PERF_GROUPS) {
    const g = el('span', 'perf-tg', label);
    g.style.gridColumn = `span ${span}`;
    table.appendChild(g);
  }
  table.appendChild(el('span'));
  for (const [, , label] of PERF_COLS) table.appendChild(el('span', 'perf-th', label));
  for (const h of PERF_HEIGHTS) {
    table.appendChild(el('span', 'perf-rh', h + 'p'));
    for (const [fps, mode] of PERF_COLS) table.appendChild(perfCell(perfRow(entry, fps, h, mode), fps === '60r30'));
  }
  box.appendChild(table);

  const lg = el('div', 'perf-legend');
  for (const [cls, text] of [['l-live', '可即時：直接邊修復邊看'], ['l-pre', '先緩衝：等數秒再開始看'], ['l-q', '建議佇列：先背景處理完再看']]) {
    const s = el('span'); s.appendChild(el('i', 'sw ' + cls)); s.append(text); lg.appendChild(s);
  }
  box.appendChild(lg);
  box.appendChild(el('p', 'perf-note', '數字為處理速度倍率（相對影片實際長度），≥ 1.0x 才追得上即時播放。'
    + '自動補幀只用多出來的速度補，顯示能補上的格數比例。60 fps 影片用原始幀率時不補幀；'
    + '「降到 30」是播放器的「修復幀率：最高 30fps」，先降幀再修復，自動補幀補回 60 fps（佇列不支援）。'));

  const summary = entry.recommend && entry.recommend.summary;
  if (summary || plan) {
    const rec = el('div', 'perf-rec');
    if (summary) {
      rec.appendChild(el('p', 'perf-rec-sum', entry === inUse ? String(summary)
        : `${modelLabel('detection_model', entry.detector)} 的結果：${summary}`));
    }
    if (plan) rec.appendChild(perfPlan(plan, running));
    box.appendChild(rec);
  }
  setHidden(box, false);
}

function setPerfOpen(open) {
  perfOpen = open;
  const btn = $('#btnPerf');
  setAttr(btn, 'aria-expanded', open);
  btn.title = open ? '隱藏效能檢測' : '電腦效能檢測';
  setHidden($('#perfPanel'), !open);
  if (perfLast) renderPerf(perfLast);
}

function renderPerf(perf) {
  const has = !!perf && typeof perf === 'object';
  perfLast = has ? perf : null;
  setHidden($('#btnPerf'), !has);
  const banner = $('#perfBanner');
  if (!has) { setHidden(banner, true); setHidden($('#perfPanel'), true); return; }
  setHidden(banner, !(perf.prompt && perf.status !== 'running' && !perfOpen));
  if (!perfOpen) return;
  setHidden($('#perfPanel'), false);
  renderPerfCtl(perf);
  renderPerfResult(perf);
}

$('#btnPerf').addEventListener('click', () => setPerfOpen(!perfOpen));
$('#perfClose').addEventListener('click', () => setPerfOpen(false));
$('#perfBannerGo').addEventListener('click', () => {
  setPerfOpen(true);
  if (perfLast && perfLast.can_start !== false && perfLast.status !== 'running') perfStart();
});
$('#perfBannerDismiss').addEventListener('click', () => { setHidden($('#perfBanner'), true); act('perf_dismiss_prompt'); });

/* ------------------------------------------------------------------ mock backend (browser preview) */

// Preview-only: a v2 `subtitle.view` as the scheduler would publish it.
function mockSubtitleView(state, extra) {
  return { v: 2, state, progress: 0, covered_s: 0, slices_done: 0, slices_total: 4, open: true, running: null, wait: null, stuck: null,
    failures_elsewhere: 0, partial_stuck: false, retryable: state === 'stuck', ...(extra || {}) };
}
// preview data of the 串流中 card: ?stream=sub (default) | nosub | behind | stuck | none; elapsed = seconds since the page opened
function mockStreaming(kind, elapsed) {
  if (kind === 'none') return null;
  const pos = 1345 + elapsed;
  const live = { title: '【4K】夜景城市散步 Tokyo Night Walk 2024 完整版', pos, duration: 7083, playing: true, buffering: false,
    ahead_s: Math.round((45 + 3 * Math.sin(elapsed / 5)) * 10) / 10, speed: 0.7, restore_done: false,
    stream_item: kind === 'nosub' ? null : 'stream01', subtitle: null };
  if (kind === 'nosub') return live;
  const sub = (view, lead) => ({ status: 'queued', view: mockSubtitleView(view.state, { covered_s: pos + lead, ...view }), detail: '', error: '', lead_s: lead });
  if (kind === 'behind') live.subtitle = sub({ state: 'running', progress: 0.3, running: { executor: 'local', slice: 4, stage: 'mt', candidate_name: 'Sakura 7B' } }, -8);
  else if (kind === 'stuck') live.subtitle = sub({ state: 'stuck', retryable: true, stuck: { slice: 5, stage: 'asr', kind: 'rate_limit', candidates: [{ id: 'groq', name: 'Groq', kind: 'rate_limit' }] } }, 95);
  else live.subtitle = sub({ state: 'running', progress: 0.42, running: { executor: 'cloud', slice: 2, stage: 'asr', candidate_name: 'Groq' } }, Math.round(720 + 20 * Math.sin(elapsed / 9)));
  return live;
}
function makeMock() {
  const GB = 1073741824, MB = 1048576;
  const now = Date.now() / 1000;
  let seq = 100;
  const q = new URLSearchParams(location.search);
  const mockPrimary = {
    id: 'primary', label: '主字幕', asr: { mode: 'dual', model: 'whisper-ja-1.5B' }, correction: { mode: 'none' },
    translation: { profile: DEFAULT_TRANSLATION_RECIPE },
    sentence: { sentence_mode: true, short_threshold: 1, short_policy: 'whitelist', show_interjections: false }, japanese_track: false,
  };
  const mockComparisonRows = [{
    id: 'comparison-demo', label: 'Anime Whisper · Sakura 7B', asr: { mode: 'single', model: 'anime-whisper' }, correction: { mode: 'none' },
    translation: { profile: DEFAULT_TRANSLATION_RECIPE },
    sentence: { sentence_mode: true, short_threshold: 1, short_policy: 'whitelist', show_interjections: false }, japanese_track: false,
  }];
  const mockMissing = new Set((q.get('missing') || '').split(',').filter(Boolean));   // ?missing=anime-whisper,...
  const mockMissingApply = (rows) => rows.map((m) => (mockMissing.has(m.id) ? { ...m, status: 'missing', status_label: '找不到模型檔案' } : m));
  const mockModelDl = (() => {
    const installed = new Set((q.get('mdlinstalled') || '').split(',').filter(Boolean));
    const freeBytes = (q.get('mdlfree') ? +q.get('mdlfree') : 200) * GB;
    const speed = (q.get('mdlspeed') ? +q.get('mdlspeed') : 400) * MB;
    const verifyTicks = +(q.get('mdlverify') || 0);   // ?mdlverify=N: N x 200 ms of `verifying` after the first item
    const cat = [
      { id: 'whisper-ja-1.5B', label: '日文語音辨識模型 whisper-ja-1.5B', category: 'asr', size: 3.1 * GB, default: true, description: '日文語音辨識，通用、穩定。', license_id: 'unspecified', license_label: '模型卡未標示授權，請自行確認使用條件', card_url: 'https://huggingface.co/efwkjn/whisper-ja-1.5B' },
      { id: 'anime-whisper', label: '日文語音辨識模型 anime-whisper', category: 'asr', size: 3.1 * GB, default: true, description: '動畫／口語風格的日文語音辨識。', license_id: 'MIT', license_label: 'MIT', card_url: 'https://huggingface.co/litagin/anime-whisper' },
      { id: 'sakura-galtransl-7b-v3.7', label: '日翻中模型 Sakura-GalTransl 7B v3.7（建議）', category: 'translation', size: 4.0 * GB, default: true, description: '預設翻譯模型，速度與品質平衡。', license_id: 'CC-BY-NC-SA-4.0', license_label: 'CC BY-NC-SA 4.0（僅限非商業用途）；模型卡另禁止付費翻譯服務與付費字幕／補丁', card_url: 'https://huggingface.co/SakuraLLM/Sakura-GalTransl-7B-v3.7' },
      { id: 'sakura-galtransl-14b-v3.8', label: '日翻中模型 Sakura-GalTransl 14B v3.8', category: 'translation', size: 7.6 * GB, default: false, description: '較準，需要較多顯示記憶體。', license_id: 'CC-BY-NC-SA-4.0', license_label: 'CC BY-NC-SA 4.0（僅限非商業用途）', card_url: 'https://huggingface.co/SakuraLLM/Sakura-GalTransl-14B-v3.8' },
      { id: 'sakura-7b-qwen2.5-v1.0', label: '日翻中模型 Sakura 7B Qwen2.5 v1.0', category: 'translation', size: 4.0 * GB, default: false, description: '舊版。', license_id: 'CC-BY-NC-SA-4.0', license_label: 'CC BY-NC-SA 4.0（僅限非商業用途）', card_url: 'https://huggingface.co/SakuraLLM/Sakura-7B-Qwen2.5-v1.0-GGUF' },
      { id: 'llama-server', label: '翻譯執行程式 llama-server（NVIDIA CUDA 12.4，Windows）', category: 'runtime', size: 0.63 * GB, default: true, description: '執行翻譯模型用的程式，必要。', license_id: 'MIT', license_label: 'MIT（llama.cpp）；內含 NVIDIA CUDA 執行階段，適用 NVIDIA 授權條款', card_url: 'https://github.com/ggml-org/llama.cpp' },
    ].map((r) => ({ ...r, size: Math.round(r.size) }));
    let job = null, timer = null;
    const row = (r) => ({ ...r, installed: installed.has(r.id), partial_bytes: 0, remaining_bytes: installed.has(r.id) ? 0 : r.size, revision: 'mock', note: '' });
    const clear = () => { if (timer) { clearInterval(timer); timer = null; } };
    return {
      view() {
        const groups = { asr: [], translation: [], runtime: [] };
        for (const r of cat) groups[r.category].push(row(r));
        return { ok: true, models_dir: 'C:/mock/models', groups, default_ids: cat.filter((r) => r.default).map((r) => r.id), free_bytes: freeBytes, progress: this.snapshot() };
      },
      plan(ids) {
        const chosen = new Set(ids || []);
        for (const r of cat) if (r.category === 'runtime' && !installed.has(r.id)) chosen.add(r.id);
        const covered = (c) => cat.some((r) => r.category === c && (installed.has(r.id) || chosen.has(r.id)));
        const need = cat.filter((r) => chosen.has(r.id) && !installed.has(r.id)).reduce((n, r) => n + r.size, 0);
        let message = '';
        if (!covered('asr')) message = '請至少選擇一個語音辨識模型。';
        else if (!covered('translation')) message = '請至少選擇一個翻譯模型。';
        else if (q.get('mdlinuse') === '1' && chosen.has('llama-server') && !installed.has('llama-server')) message = '本機字幕正在使用 llama-server，請等字幕處理完成或暫停後再安裝。';
        else if (freeBytes < need + 512 * MB) message = `磁碟空間不足：需要約 ${fmtBytes(need + 512 * MB)}，目前只剩 ${fmtBytes(freeBytes)}。請清出空間後再試。`;
        return { ok: !message, message, ids: [...chosen], download_bytes: need, free_bytes: freeBytes };
      },
      start(ids) {
        if (job && ['queued', 'running', 'verifying'].includes(job.state)) return { ok: false, error: '已有模型正在下載。' };
        const plan = this.plan(ids);
        if (!plan.ok) return { ok: false, error: plan.message };
        const list = cat.filter((r) => plan.ids.includes(r.id) && !installed.has(r.id));
        job = { state: 'running', done: job && job.state === 'cancelled' ? job.done : 0, total: plan.download_bytes, list, error: '', last: Date.now() };
        clear();
        timer = setInterval(() => {
          if (job.state === 'verifying') { if (--job.verifyLeft <= 0) job.state = 'running'; return; }
          if (job.state !== 'running') return clear();
          job.done = Math.min(job.total, job.done + speed * 0.2);
          if (verifyTicks && !job.verified && job.done >= list[0].size) { job.verified = true; job.done = list[0].size; job.state = 'verifying'; job.verifyLeft = verifyTicks; return; }
          if (q.get('mdlfail') === '1' && job.done > job.total * 0.3) { job.state = 'failed'; job.error = '下載失敗：網路連線中斷。'; return clear(); }
          if (job.done >= job.total) { job.state = 'done'; list.forEach((r) => installed.add(r.id)); clear(); }
        }, 200);
        return { ok: true, progress: this.snapshot() };
      },
      cancel() { if (job && ['running', 'verifying'].includes(job.state)) { job.state = 'cancelled'; clear(); } return { ok: true }; },
      snapshot() {
        if (!job) return null;
        let acc = 0, label = '';
        for (const r of job.list) { acc += r.size; if (job.done < acc) { label = r.label; break; } }
        const running = job.state === 'running';
        return { state: job.state, item: '', item_label: running || job.state === 'verifying' ? label || job.list[0].label : '', bytes_done: job.done, bytes_total: job.total, speed: running ? speed : 0, eta: running ? (job.total - job.done) / speed : null, error: job.error };
      },
    };
  })();
  const st = {
    items: [], paused_all: false, player_busy: q.get('busy') === '1', notice: q.get('notice') || '',
    subtitle_environment: { runtime_available: true, runtime_reason: '', available: true, reason: '' },
    subtitle_models: [
      { id: '__dual__', label: '雙模型融合 · whisper-ja-1.5B + anime-whisper', kind: 'asr', provider: 'local', recipe: '__dual__', builtin: true, removable: false, status: 'ready', status_label: '內建方案' },
      { id: 'whisper-ja-1.5B', label: 'Whisper ja 1.5B', kind: 'asr', provider: 'local', recipe: 'whisper-ja-1.5B', builtin: true, removable: false, status: 'ready', status_label: '內建方案' },
      { id: 'anime-whisper', label: 'Anime Whisper', kind: 'asr', provider: 'local', recipe: 'anime-whisper', builtin: true, removable: false, status: 'ready', status_label: '內建方案' },
      { id: 'whisper-large-v3-turbo', label: 'Whisper large-v3-turbo', kind: 'asr', provider: 'local', recipe: 'whisper-large-v3-turbo', builtin: true, removable: false, status: 'ready', status_label: '內建方案' },
      { id: 'sakura-galtransl-7b-v3.7', label: 'Sakura-GalTransl 7B v3.7', kind: 'translation', provider: 'local', recipe: 'sakura-galtransl-7b-v3.7', builtin: true, removable: false, status: 'ready', status_label: '內建方案', is_default: true },
      { id: 'sakura-v1', label: 'Sakura v1', kind: 'translation', provider: 'local', recipe: 'sakura-v1', builtin: true, removable: false, status: 'ready', status_label: '內建方案' },
      { id: 'sakura-galtransl-14b-v3.8', label: 'Sakura-GalTransl 14B v3.8', kind: 'translation', provider: 'local', recipe: 'sakura-galtransl-14b-v3.8', builtin: true, removable: false, status: 'ready', status_label: '內建方案' },
      { id: 'qwen3', label: 'Qwen3', kind: 'translation', provider: 'local', recipe: 'qwen3', builtin: true, removable: false, status: 'ready', status_label: '內建方案' },
      { id: 'mock-cloud-translation', label: '翻譯服務範例', kind: 'translation', provider: 'openai-compatible', recipe: 'openai-compatible', builtin: false, removable: true, status: 'ready', status_label: '已通過新增測試；目前連線未檢查', base_url: 'https://example.invalid/v1', remote_model: 'example-translation' },
    ],
    engine_options: q.get('engines') === '2' ? [{ value: 'lada', label: 'Lada' }, { value: 'demo', label: 'Demo' }] : [{ value: 'lada', label: 'Lada' }],
    model_options: {
      detection_model: [{ value: 'v4-fast', label: 'V4 Fast' }, { value: 'v4-accurate', label: 'V4 Accurate' }],
      restoration_model: [{ value: 'basicvsrpp', label: 'BasicVSR++' }, { value: 'basicvsrpp-fast', label: 'BasicVSR++ Fast' }],
    },
    settings: { cap_gb: 50, cache_dir: '', encoder: 'hevc', interpolation: 'full', save_compare: false, compare_original: 'copy', compare_quality: 'small',
                prevent_sleep: true, shutdown_when_done: false, show_thumbnails: true, detection_model: 'v4-fast', restoration_model: 'basicvsrpp',
                subtitle_primary: mockPrimary, subtitle_comparison_enabled: false, subtitle_comparison_rows: mockComparisonRows },
    usage_bytes: 0, cap_bytes: 50 * GB,
  };
  // ?perf=none|prompt|running|failed|cancelled|blocked (default: finished profile)
  const pq = q.get('perf') || 'done';
  const PR = (height, mode, level, rt, rh, pre, frac) => ({ height, mode, level, ratio_typical: rt, ratio_heavy: rh, prebuffer_s: pre, interp_fraction: frac });
  const profile = () => {
    const fast = withHalf(mockProfile());
    // v4-accurate: ~10% slower, which moves 480p 60 fps from prebuffer to queue
    const acc = JSON.parse(JSON.stringify(fast.matrix));
    for (const rows of Object.values(acc)) for (const r of rows) { r.ratio_typical = +(r.ratio_typical * 0.9).toFixed(2); r.ratio_heavy = +(r.ratio_heavy * 0.9).toFixed(2); }
    for (const r of acc['60']) if (r.height === 480) Object.assign(r, { level: 'queue', prebuffer_s: null });
    const recAcc = { summary: '即時播放最高 480p；自動補幀可用（約補 70% 的格數）；全補幀請用佇列。' };
    const det = st.settings.detection_model;
    return Object.assign(fast, {
      detector: det, current_detector: det,
      by_detector: [{ detector: 'v4-fast', matrix: fast.matrix, recommend: fast.recommend }, { detector: 'v4-accurate', matrix: acc, recommend: recAcc }],
      detector_advice: { detector: null, differs: [{ src_fps: 60, height: 480, mode: 'off', levels: { 'v4-accurate': 'queue', 'v4-fast': 'prebuffer' } }] },
    }, det === 'v4-accurate' ? { matrix: acc, recommend: recAcc } : {});
  };
  const mockProfile = () => ({
    created: new Date(Date.now() - 3600e3).toISOString(), gpu: 'NVIDIA GeForce RTX 5060 Laptop GPU', duration_s: 71, on_ac_power: false,
    warnings: [], stale: pq === 'stale' ? ['驅動程式版本已變更'] : [], settings_mismatch: pq === 'stale' ? ['插幀'] : [],
    matrix: {
      '30': [
        PR(480, 'off', 'live', 1.8, 1.5, null, null), PR(480, 'auto', 'live', 1.8, 1.4, null, 0.88), PR(480, 'full', 'queue', 0.9, 0.7, null, null),
        PR(720, 'off', 'prebuffer', 1.7, 1.0, 10, null), PR(720, 'auto', 'prebuffer', 1.7, 1.0, 10, 0.33), PR(720, 'full', 'queue', 0.5, 0.4, null, null),
        PR(1080, 'off', 'queue', 1.0, 0.7, null, null), PR(1080, 'auto', 'queue', 1.0, 0.7, null, 0), PR(1080, 'full', 'queue', 0.4, 0.3, null, null),
      ],
      '60': [   // never interpolated (> 40 fps): auto / full rows equal off
        PR(480, 'off', 'prebuffer', 1.05, 0.8, 15, null), PR(480, 'auto', 'prebuffer', 1.05, 0.8, 15, 0), PR(480, 'full', 'prebuffer', 1.05, 0.8, 15, 0),
        PR(720, 'off', 'queue', 0.85, 0.6, null, null), PR(720, 'auto', 'queue', 0.85, 0.6, null, 0), PR(720, 'full', 'queue', 0.85, 0.6, null, 0),
        PR(1080, 'off', 'queue', 0.5, 0.35, null, null), PR(1080, 'auto', 'queue', 0.5, 0.35, null, 0), PR(1080, 'full', 'queue', 0.5, 0.35, null, 0),
      ],
    },
    recommend: { summary: '即時播放最高 480p；自動補幀可用（約補 88% 的格數）；全補幀請用佇列。' },
  });
  const withHalf = (p) => { p.matrix['60r30'] = JSON.parse(JSON.stringify(p.matrix['30'])); return p; };   // 60 restored at 30 = a 30 fps source
  st.perf = { status: 'none', kind: null, stage: '', progress: 0, error: null, can_start: true, blocked_reason: null, prompt: pq === 'prompt', profile: pq === 'done' || pq === 'stale' ? profile() : null };
  const mockPlan = (interp) => ({ player: { detection_model: 'v4-fast', interpolation: interp }, error: null, changes: [
    { key: 'interpolation', label: '播放器補幀', from: interp, to: 'auto', reason: '只用多出來的速度補幀，不會拖慢；這台電腦約可補 88% 的格數' }] });
  st.perf.apply_plan = st.perf.profile ? mockPlan('off') : null;
  if (pq === 'running') Object.assign(st.perf, { status: 'running', kind: 'quick', stage: '測試 720p 自動補幀…', progress: 0.42, remaining_s: 95, can_start: false });
  if (pq === 'failed') Object.assign(st.perf, { status: 'failed', error: '找不到測試用影片（預覽）' });
  if (pq === 'cancelled') Object.assign(st.perf, { status: 'cancelled' });
  if (pq === 'blocked') Object.assign(st.perf, { can_start: false, blocked_reason: '播放器正在使用 GPU，請先關閉播放器' });
  let perfT = null;
  const S = (o) => Object.assign({ encoder: 'hevc', interpolation: 'full', save_compare: false, subtitles: false,
    detection_model: st.settings.detection_model, restoration_model: st.settings.restoration_model }, o);
  const mk = (o) => Object.assign({
    id: 'm' + (seq++), title: '', target: '', kind: 'url', headers: {}, quality: null, settings: S(), cache_key: '', state: 'waiting',
    order: 0, created: now, last_used: now, watched: false, seen: true, keep: false, error: '', source_path: '', output_path: '', size_bytes: 0, progress: {},
    has_compare: false, compare_original: null, subtitle: {},
  }, o);
  st.items = [
    mk({ title: '【4K】夜景城市散步 Tokyo Night Walk 2024 完整版', target: 'https://www.youtube.com/watch?v=abcd1234', quality: 1080, state: 'processing', headers: { Cookie: 'x' },
         size_bytes: 1.4 * GB, progress: { frame: 41230, frames: 129600, fps: 42.6, eta: 2080 } }),
    mk({ title: 'Sample Clip Final v3 (HLS)', target: 'https://media.sample-host.tv/v3/final.mp4', quality: 720, state: 'downloading',
         progress: { downloaded_bytes: 612 * MB, total_bytes: 1.9 * GB, speed: 7.8 * MB, eta: 172 }, headers: { Authorization: 'Bearer x' } }),
    mk({ title: '', target: 'https://cdn.example-video.net/stream/master.m3u8?token=abc', quality: 'best', state: 'queued', size_bytes: 880 * MB }),
    mk({ title: 'Interview — Part 2 (留言區分享的連結)', target: 'https://vimeo.com/765432198', quality: 1080, state: 'waiting' }),
    mk({ title: '已暫停的長片 Lecture Series Ep.07', target: 'https://www.bilibili.com/video/BV1xx411c7mD', quality: 720, state: 'paused',
         progress: { frame: 20000, frames: 86400 }, size_bytes: 2.1 * GB, headers: { Cookie: 'SESSDATA=…' } }),
    mk({ title: '假日旅行紀錄 Okinawa Trip', target: 'E:\\Camera\\2024\\okinawa_trip_01.mkv', kind: 'file', quality: 1080, state: 'done', watched: true, keep: true,
         size_bytes: 3.2 * GB, settings: S({ save_compare: true }), has_compare: true, compare_original: 'copy' }),
    mk({ title: '貓咪日常合集 (未看)', target: 'https://www.youtube.com/watch?v=zzz987', quality: 720, state: 'done', seen: false, size_bytes: 640 * MB,
      settings: S({ encoder: 'h264', interpolation: 'off', subtitles: true }), subtitle: { status: 'failed', error: '預覽字幕錯誤範例', view: mockSubtitleView('stuck', { stuck: { slice: 1, stage: 'mt', kind: 'error', candidates: [{ id: 'm1', name: '預覽翻譯模型', kind: 'error' }] } }) } }),
    mk({ title: '字幕暫停等待額度（預覽）', target: 'https://www.youtube.com/watch?v=blocked01', quality: 720, state: 'downloading',
         progress: { downloaded_bytes: 300 * MB, total_bytes: 1.2 * GB, speed: 6 * MB, eta: 150 },
         settings: S({ subtitles: true }),
         subtitle: { status: 'queued', view: mockSubtitleView('waiting_quota', { covered_s: 600, wait: { kind: 'quota', slice: 2, stage: 'mt', retry_at: Date.now() / 1000 + 10 } }) } }),
    mk({ title: '字幕設定已變更（預覽，可改用新翻譯重做）', target: 'https://www.youtube.com/watch?v=pend-mt', quality: 1080, state: 'processing',
         size_bytes: 1.0 * GB, progress: { frame: 30000, frames: 129600, fps: 40, eta: 1500 },
         settings: S({ subtitles: true, subtitle_overrides: { subtitle_comparison_enabled: true } }),
         subtitle: { status: 'translate', view: mockSubtitleView('running', { progress: 0.5, running: { executor: 'local', slice: 1, stage: 'mt' } }), settings_pending: { asr_changed: false } } }),
    mk({ title: '字幕設定已變更（預覽，語音辨識也變了）', target: 'https://www.youtube.com/watch?v=pend-asr', quality: 720, state: 'processing',
         size_bytes: 800 * MB, progress: { frame: 20000, frames: 86400, fps: 40, eta: 1200 },
         settings: S({ subtitles: true, subtitle_hold: true }),
         subtitle: { status: 'asr', view: mockSubtitleView('running', { progress: 0.2, running: { executor: 'local', slice: 0, stage: 'asr' } }), settings_pending: { asr_changed: true } } }),
    // subtitle combo previews: ?combos=none hides the whole feature (older backend)
    mk({ title: '字幕組合：進行中 3 組（1 完成、1 進行、1 等待）', target: 'https://www.youtube.com/watch?v=combo-run', quality: 1080, state: 'processing',
         size_bytes: 1.1 * GB, progress: { frame: 90000, frames: 129600, fps: 40, eta: 900 },
         settings: S({ subtitles: true }),
         subtitle: { status: 'asr', view: mockSubtitleView('running', { progress: 0.43, running: { executor: 'local', slice: 1, stage: 'asr' } }), detail: '第 2 組：辨識中 43%', combos: { selected: ['anime-sakura1', 'ja15b-sakura1', 'ja15b-gt14b'], total: 3, done: 1, failed: 0, items: [
           { id: 'anime-sakura1', label: 'Anime Whisper + Sakura v1', status: 'done', cue_count: 143 },
           { id: 'ja15b-sakura1', label: 'Whisper ja 1.5B + Sakura v1', status: 'running' },
           { id: 'ja15b-gt14b', label: 'Whisper ja 1.5B + GT 14B', status: 'pending' }] } } }),
    mk({ title: '字幕組合：全部完成 2 組', target: 'https://www.youtube.com/watch?v=combo-done', quality: 720, state: 'done', size_bytes: 700 * MB,
         settings: S({ subtitles: true }),
         subtitle: { status: 'done', percent: 100, combos: { selected: ['anime-sakura1', 'ja15b-sakura1'], total: 2, done: 2, failed: 0, items: [
           { id: 'anime-sakura1', label: 'Anime Whisper + Sakura v1', status: 'done', cue_count: 143 },
           { id: 'ja15b-sakura1', label: 'Whisper ja 1.5B + Sakura v1', status: 'done', cue_count: 134 }] } } }),
    mk({ title: '字幕組合：部分失敗（影片仍完成）', target: 'https://www.youtube.com/watch?v=combo-part', quality: 720, state: 'done', size_bytes: 820 * MB, seen: false,
         settings: S({ subtitles: true }),
         subtitle: { status: 'done', percent: 100, combos: { selected: ['anime-sakura1', 'ja15b-sakura1', 'ja15b-gt14b'], total: 3, done: 2, failed: 1, items: [
           { id: 'anime-sakura1', label: 'Anime Whisper + Sakura v1', status: 'done', cue_count: 143 },
           { id: 'ja15b-sakura1', label: 'Whisper ja 1.5B + Sakura v1', status: 'done', cue_count: 134 },
           { id: 'ja15b-gt14b', label: 'Whisper ja 1.5B + GT 14B', status: 'failed', failed_stage: 'translation', error: 'llama-server 啟動失敗：顯示記憶體不足' }] } } }),
    mk({ title: '字幕組合：排隊中已選 2 組（可調整）', target: 'https://vimeo.com/combo-wait', quality: 1080, state: 'queued',
         settings: S({ subtitles: true }), subtitle: {} }),
    mk({ title: '失效的連結範例', target: 'https://video.example.org/file/9981', quality: 1080, state: 'failed', headers: { Cookie: 'expired' },
         error: 'HTTP 403：網址已過期，重新解析後仍無法下載（請重新貼上最新的連結或 cURL）。' }),
  ];

  // ---- synthetic film-strip data (never real video frames): canvas-drawn 320x180 frames, restored ones are crisper and carry an "R" ----
  const FRAMES = 12, mockImgs = new Map();
  const mockImg = (i, restored) => {
    const key = i + (restored ? 'r' : 'u');
    if (mockImgs.has(key)) return mockImgs.get(key);
    const c = document.createElement('canvas'); c.width = 320; c.height = 180;
    const x = c.getContext('2d'), hue = (i * 47 + 200) % 360, rn = (a) => { let t = (a * 2654435761) >>> 0; return () => ((t = (Math.imul(t, 1664525) + 1013904223) >>> 0) / 4294967296); }, r = rn(i + 3);
    const g = x.createLinearGradient(0, 0, 320, 180);
    g.addColorStop(0, `hsl(${hue},${restored ? 70 : 45}%,${restored ? 42 : 34}%)`); g.addColorStop(1, `hsl(${(hue + 60) % 360},${restored ? 75 : 50}%,${restored ? 22 : 18}%)`);
    x.fillStyle = g; x.fillRect(0, 0, 320, 180);
    for (let k = 0; k < 7; k++) {
      x.fillStyle = `hsla(${(hue + r() * 120) % 360},70%,${50 + r() * 25}%,.55)`;
      if (k % 2) { x.beginPath(); x.arc(r() * 320, r() * 180, 14 + r() * 40, 0, 7); x.fill(); } else x.fillRect(r() * 280, r() * 140, 20 + r() * 70, 14 + r() * 50);
    }
    if (restored) { x.strokeStyle = 'rgba(255,255,255,.35)'; x.lineWidth = 2; for (let k = -180; k < 320; k += 16) { x.beginPath(); x.moveTo(k, 180); x.lineTo(k + 180, 0); x.stroke(); } }
    x.fillStyle = 'rgba(255,255,255,.92)'; x.font = 'bold 96px Consolas, monospace'; x.textAlign = 'center'; x.textBaseline = 'middle';
    x.fillText(String(i + 1), 160, 92);
    if (restored) { x.font = 'bold 22px Consolas, monospace'; x.textAlign = 'left'; x.fillText('R', 10, 22); }
    const url = c.toDataURL('image/jpeg', 0.82); mockImgs.set(key, url); return url;
  };
  // per-item thumb simulation: processing advances p (processed) and dl (download front) so the strip moves; others are static
  const TH = new Map();   // id -> {duration, p, dl, ranges (static items), worker}
  const thOf = (it) => { let t = TH.get(it.id); if (!t) { t = { duration: 5400, p: 0.2, dl: 0.45, ranges: [], worker: null }; TH.set(it.id, t); } return t; };
  const decorateThumbs = (it) => {
    const t = TH.get(it.id);
    if (!t) { it.thumbs = null; return; }
    let dlf = 1, ranges = [], worker = null;
    if (it.state === 'downloading') { const p = it.progress || {}; dlf = p.total_bytes > 0 ? Math.min(1, p.downloaded_bytes / p.total_bytes) : 0; }
    else if (it.state === 'processing') { dlf = t.dl; ranges = [[0, t.p]]; worker = t.p; }
    else if (it.state === 'done') { ranges = [[0, 1]]; }
    else { ranges = t.ranges; worker = t.worker; }
    const nfr = it.state === 'downloading' ? (dlf < 0.03 ? 0 : Math.min(FRAMES, 1 + Math.floor(dlf * FRAMES))) : FRAMES;
    const frames = [];
    for (let i = 0; i < nfr; i++) {
      const at = (i + 0.5) / FRAMES;
      const restored = it.state === 'done' || ranges.some((r) => at >= r[0] && at <= r[1]);
      frames.push({ at: at * t.duration, url: mockImg(i, restored), restored, w: 320, h: 180 });
    }
    it.thumbs = nfr ? { duration: t.duration, frames } : null;
    it.processed_ranges = ranges; it.download_fraction = dlf; it.worker_fraction = worker;
  };
  const advanceThumbs = () => {
    for (const it of st.items) {
      if (it.state !== 'processing') continue;
      const t = thOf(it);
      t.dl = Math.min(1, t.dl + 0.003); t.p = Math.min(t.dl, t.p + 0.004);
      if (t.p >= 1) { t.p = 0.1; t.dl = 0.5; }   // loop so the preview keeps moving
    }
  };

  const mockStart = Date.now();
  const emit = () => {
    st.streaming = mockStreaming(q.get('stream') || 'sub', (Date.now() - mockStart) / 1000);
    st.items.forEach((it, i) => { it.order = i; decorateThumbs(it); });
    st.usage_bytes = st.items.reduce((a, it) => a + (it.size_bytes || 0), 0);
    st.cap_bytes = st.settings.cap_gb * GB;
    if (window.ohStreamer && window.ohStreamer.onState) window.ohStreamer.onState(JSON.parse(JSON.stringify(st)));
  };
  const find = (id) => st.items.find((x) => x.id === id);
  {   // which mock items have thumbnails (the interview / failed / second finished item deliberately have none)
    const by = (i) => st.items[i];
    Object.assign(thOf(by(0)), { p: 0.2, dl: 0.45 });                                   // processing: strip moves
    thOf(by(1));                                                                         // downloading: frames arrive, blue light moves
    Object.assign(thOf(by(2)), { ranges: [], worker: null });                            // queued: downloaded, nothing processed
    Object.assign(thOf(by(4)), { ranges: [[0, 0.22], [0.45, 0.5], [0.62, 0.8]], worker: 0.8, duration: 7200 });   // paused: non-contiguous
    thOf(by(5));                                                                         // done: fully processed, all restored
  }
  // ---- intake panel mock (?intake=noperf: no performance profile, ?intake=samedet: stream and queue detector are equal) ----
  const iq = q.get('intake') || '';
  const saved = {
    stream: { detection_model: 'v4-fast', restoration_model: 'basicvsrpp', interpolation: 'auto', restoration_fps: 'original' },
    queue: { detection_model: iq === 'samedet' ? 'v4-fast' : 'v4-accurate', restoration_model: 'basicvsrpp', interpolation: 'auto', encoder: 'hevc', save_compare: false, subtitles: false },
  };
  const calls = (window.__mockCalls = []);                     // what the panel sent (inspect from the console)
  const infos = new Map();                                      // token -> mock Info
  let tokSeq = 1, fileDialogN = 0;
  const URL_A = { kind: 'url', title: '示範影片標題：解析後取得的標題會顯示在這裡，太長的時候最多換兩行，之後以省略號結尾，這一段只是用來測試長標題的換行與省略', site: 'video.example.com',
    duration: 7083, fps: 30, vcodec: 'H.264', acodec: 'AAC', hls: false, headers_count: 0,
    heights: [1080, 720, 480, 360].map((h, i) => ({ height: h, fps: 30, label: h + 'p', size_bytes: [2.4, 1.3, 0.7, 0.45][i] * GB })), default_height: 1080 };
  const URL_B = { kind: 'url', title: '另一部 60 fps 的示範影片（cURL 貼上，帶標頭）', site: 'cdn.example.net', duration: 1520, fps: 60, vcodec: 'H.264', acodec: 'AAC', hls: true, headers_count: 3,
    heights: [1080, 720, 480].map((h, i) => ({ height: h, fps: 60, label: h + 'p60', size_bytes: [1.6, 0.9, 0.5][i] * GB })), default_height: 1080 };
  const mockInfo = (text) => {
    const t = text.trim();
    const target = /^curl\s/i.test(t) ? ((t.match(/https?:\/\/[^\s'"]+/) || [''])[0]) : t.split(/\r?\n/)[0].trim();
    const file = !/^https?:\/\//i.test(target);
    let base;
    if (file) {
      const hi = /60/.test(target);
      base = { kind: 'file', title: basename(target), site: target.replace(/[\\/][^\\/]*$/, '') || 'D:\\Videos', duration: 1450 + (target.length % 7) * 400, fps: hi ? 60 : 29.97,
        vcodec: 'HEVC', acodec: 'AAC', hls: false, headers_count: 0, heights: [{ height: 1080, fps: hi ? 60 : 29.97, label: '1080p', size_bytes: 1.1 * GB }], default_height: 1080 };
    } else if (/m3u8|^curl\s|60fps/i.test(t)) base = Object.assign({}, URL_B);
    else {
      base = Object.assign({}, URL_A);
      if (/def|second/.test(target)) Object.assign(base, { title: '同站的第二部影片（畫質選項相同）', duration: 2710 });
      if (/single/.test(target)) { base.heights = base.heights.slice(0, 1); base.default_height = 1080; base.title = '只有單一畫質的示範影片（串流字幕會多下載一份）'; }
      if (/hd/.test(target)) { base.heights = base.heights.slice(0, 2); base.default_height = 720; base.title = '只有 1080 / 720 的另一部影片（畫質選項不同）'; }
    }
    const token = 'tok' + (tokSeq++);
    const info = Object.assign({}, base, { token, target, thumb: mockImg((target.length * 7 + tokSeq) % FRAMES, false), dupe: null });
    if (/dupe/i.test(target)) info.dupe = { id: st.items[0].id, state: 'processing', progress: 0.42 };
    if (/nothumb/i.test(target)) info.thumb = null;
    infos.set(token, info);
    return info;
  };
  const OFF = { 'v4-fast': { 1080: 68, 720: 70, 480: 72, 360: 74 }, 'v4-accurate': { 1080: 60, 720: 63, 480: 66, 360: 68 } };
  const FULL = { 'v4-fast': { 1080: 30, 720: 35, 480: 38, 360: 40 }, 'v4-accurate': { 1080: 29, 720: 33, 480: 36, 360: 38 } };
  const lookup = (tbl, h) => tbl[h] || tbl[h > 1080 ? 1080 : 360];
  const streamEst = (h, det, fin, interp) => {
    const off = lookup(OFF[det] || OFF['v4-fast'], h), ratio = off / fin;
    const verdict = ratio >= 1.15 ? 'live' : ratio >= 0.9 ? 'prebuffer' : 'queue';
    const share = interp ? Math.max(0, Math.min(1, (lookup(FULL[det] || FULL['v4-fast'], h) / fin - 0.55) * 1.6)) : null;
    return { verdict, restore_fps: off, need_fps: fin, ratio, prebuffer_s: verdict === 'prebuffer' ? Math.round((1.15 - ratio) * 80) + 6 : null, interp_share: share, detector: det };
  };

  const mockSleep = (ms) => new Promise((r) => setTimeout(r, ms));

  let gpuT = 0;
  const ok = (message) => ({ ok: true, message: message ? '（預覽模式）' + message : undefined });

  setInterval(() => {
    if (st.paused_all) return;
    const dt = 0.5;
    const dl = st.items.find((i) => i.state === 'downloading');
    if (dl) {
      const p = dl.progress;
      p.speed = (5 + Math.random() * 4) * MB;
      p.downloaded_bytes = Math.min(p.total_bytes, p.downloaded_bytes + p.speed * dt * 3);
      p.eta = (p.total_bytes - p.downloaded_bytes) / p.speed;
      if (p.downloaded_bytes >= p.total_bytes) { dl.state = 'queued'; dl.size_bytes = p.total_bytes; dl.progress = {}; }
    } else {
      const w = st.items.find((i) => i.state === 'waiting');
      if (w) {
        w.state = 'downloading';
        if (!w.progress.total_bytes) w.progress = { downloaded_bytes: 0, total_bytes: (400 + Math.random() * 900) * MB, speed: 0, eta: 0 };
      }
    }
    const pr = st.items.find((i) => i.state === 'processing');
    if (pr) {
      const p = pr.progress;
      p.fps = 38 + Math.random() * 8;
      p.frame = Math.min(p.frames, p.frame + p.fps * dt * 6);
      p.eta = (p.frames - p.frame) / p.fps;
      if (p.frame >= p.frames) {
        pr.state = 'done'; pr.progress = {}; pr.size_bytes = Math.round((pr.size_bytes || 800 * MB) * 1.2);
        pr.has_compare = !!pr.settings.save_compare; pr.compare_original = pr.has_compare ? st.settings.compare_original : null;
      }
    } else if (!st.player_busy) {
      const n = st.items.find((i) => i.state === 'queued');
      if (n) { if (TH.has(n.id)) Object.assign(TH.get(n.id), { p: 0, dl: 1 }); n.state = 'processing'; if (!n.progress.frames) n.progress = { frame: 0, frames: 60000 + Math.round(Math.random() * 60000), fps: 0, eta: 0 }; }
    }
    advanceThumbs();
    emit();
  }, 500);

  const previewProfile = (local) => ({ asr: local ? 'whisper-ja-1.5B' : null,
    translation: local ? DEFAULT_TRANSLATION_RECIPE : null,
    processing: 'smart_horizon', whisper_policy: 'auto', whisper_models: ['whisper-large-v3-turbo', 'whisper-large-v3'],
    fallback: { asr: [], translation: [], max_attempts: 4, retries: 1, cooldown_seconds: 60 }, legacy_primary: local ? cloneConfig(mockPrimary) : {} });
  st.settings.subtitle_profile = { schema_version: 1, plan_version: 'cloud-first-1', active_mode: 'cloud',
    profiles: { cloud: previewProfile(false), local: previewProfile(true), custom: previewProfile(true) }, migration: {} };
  st.settings.subtitle_cloud_enabled = true;
  const mockApi = {
    get_state() { return JSON.parse(JSON.stringify((emit(), st))); },
    stream() { return { ok: false, error: `串流播放僅能在 ${BRAND_NAME} 桌面視窗中使用。` }; },
    parse_inputs(text) {
      text = String(text || '').trim();
      if (!text) return { ok: false, error: '內容是空的' };
      const nh = (t) => (t.match(/(^|\s)-H\s/g) || []).length;
      if (/^curl\s/i.test(text) || /^Invoke-WebRequest/i.test(text)) {
        const u = (text.match(/https?:\/\/[^\s'"]+/) || [''])[0];
        return { ok: true, entries: [{ text, kind: 'url', target: u, headers_count: nh(text) }], errors: [] };
      }
      const lines = text.split(/\r?\n/).map((l) => l.trim()).filter(Boolean);
      const isHeader = (l) => /^[A-Za-z][\w-]*:\s/.test(l);
      if (lines.length >= 2 && lines.every((l) => !isHeader(l) && (/^https?:\/\/\S+$/i.test(l) || /^([a-z]:[\\/]|\/|\\\\)/i.test(l)))) {
        return { ok: true, entries: lines.map((l) => ({ text: l, kind: /^https?:/i.test(l) ? 'url' : 'file', target: l, headers_count: 0 })), errors: [] };
      }
      const first = lines[0];
      const kind = /^https?:\/\//i.test(first) ? 'url' : 'file';
      return { ok: true, entries: [{ text, kind, target: first, headers_count: lines.filter(isHeader).length }], errors: [] };
    },
    async resolve(text) {
      await mockSleep(900 + (String(text).length * 37) % 900);
      if (/fail/i.test(text)) return { ok: false, error: '網站拒絕存取，可能需要登入或標頭', detail: 'ERROR: [generic] Unable to download webpage: HTTP Error 403: Forbidden' };
      return { ok: true, info: mockInfo(String(text)) };
    },
    defaults() {
      return {
        stream: Object.assign({}, saved.stream), queue: Object.assign({}, saved.queue),
        choices: { detection_model: st.model_options.detection_model.map((o) => Object.assign({}, o)), restoration_model: st.model_options.restoration_model.map((o) => Object.assign({}, o)) },
        subtitle_environment: JSON.parse(JSON.stringify(st.subtitle_environment)),
        has_profile: iq !== 'noperf',
      };
    },
    async estimate(tokens, ov) {
      await mockSleep(60);
      calls.push({ estimate: { tokens, ov } });
      ov = ov || {};
      const prof = iq !== 'noperf';
      const items = tokens.map((token) => {
        const info = infos.get(token);
        if (!info || !prof) return { token, stream: null, stream_by_height: {}, queue: null };
        const high = info.fps > 31, h = ov.height || info.default_height;
        const sDet = ov.detection_model || saved.stream.detection_model, qDet = ov.detection_model || saved.queue.detection_model;
        const rfps = ov.restoration_fps || saved.stream.restoration_fps, r30 = high && rfps === '30';
        const sIn = r30 ? 30 : info.fps;
        const sInterp = high ? r30 : (ov.interpolation || saved.stream.interpolation) === 'auto';
        const out = { token, stream: null, stream_by_height: {}, queue: null };
        if (tokens.length === 1) {
          const e = streamEst(h, sDet, sIn, sInterp);
          let suggest = null;
          if (e.verdict !== 'live') {
            if (high && !r30 && streamEst(h, sDet, 30, true).verdict === 'live') suggest = { restoration_fps: '30' };
            else {
              const lower = info.heights.map((x) => x.height).find((x) => x < h && streamEst(x, sDet, sIn, sInterp && !high).verdict === 'live');
              if (lower && info.kind === 'url') suggest = { height: lower };
            }
          }
          out.stream = Object.assign(e, { ratio: +e.ratio.toFixed(2), suggest });
        }
        for (const x of info.heights) out.stream_by_height[String(x.height)] = streamEst(x.height, sDet, sIn, sInterp).verdict;
        const full = !high && (ov.interpolation || saved.queue.interpolation) === 'auto';
        const fps = full ? lookup(FULL[qDet] || FULL['v4-fast'], h) : lookup(OFF[qDet] || OFF['v4-fast'], h);
        out.queue = { fps, eta_s: info.duration * info.fps / fps };
        return out;
      });
      return { has_profile: prof, queue_position: st.items.filter((i) => i.state !== 'done' && i.state !== 'failed').length + 1, queue_ahead_s: prof ? 2400 : null, items };
    },
    stream_resolved(token, ov) {
      calls.push({ stream_resolved: { token, ov } });
      const info = infos.get(token);
      if (!info) return { ok: false, error: '找不到解析結果，請重新解析' };
      if (/streamfail/i.test(info.target)) return { ok: false, error: '無法啟動播放器（預覽）' };
      return { ok: true, message: '（預覽模式）已送至播放器：' + (info.title || info.target) + (ov && ov.subtitles === true ? '（含字幕）' : ''),
        ...(ov && ov.subtitles === true ? { stream_item: 'stream-preview' } : {}) };
    },
    add_resolved(tokens, ov) {
      calls.push({ add_resolved: { tokens, ov } });
      ov = ov || {};
      const ids = [], errors = [];
      for (const token of tokens) {
        const info = infos.get(token);
        if (!info) { errors.push('找不到解析結果，請重新解析'); continue; }
        const interp = ov.interpolation || saved.queue.interpolation;
        const comparisonEnabled = ov.subtitle_comparison_enabled !== undefined
          ? !!ov.subtitle_comparison_enabled : st.settings.subtitle_comparison_enabled === true;
        const it = mk({ target: info.target, kind: info.kind, title: info.title, quality: info.kind === 'url' ? (ov.height || info.default_height) : null,
          settings: S({ encoder: ov.encoder || saved.queue.encoder, interpolation: interp === 'auto' ? 'full' : interp, save_compare: ov.save_compare != null ? ov.save_compare : saved.queue.save_compare,
            subtitles: ov.subtitles === true,
            ...(ov.subtitles === true ? {
              subtitle_primary: JSON.parse(JSON.stringify(st.settings.subtitle_primary)),
              subtitle_comparison_enabled: comparisonEnabled,
              subtitle_comparison_rows: JSON.parse(JSON.stringify(st.settings.subtitle_comparison_rows || [])),
              subtitle_configs: [JSON.parse(JSON.stringify(st.settings.subtitle_primary)), ...(comparisonEnabled ? JSON.parse(JSON.stringify(st.settings.subtitle_comparison_rows || [])) : [])],
            } : {}),
            detection_model: ov.detection_model || saved.queue.detection_model, restoration_model: ov.restoration_model || saved.queue.restoration_model }) });
        st.items.push(it); ids.push(it.id);
      }
      emit();
      return { ok: ids.length > 0, ids, errors, message: ids.length ? `（預覽模式）已加入 ${ids.length} 部到佇列` : '' };
    },
    save_defaults(ov) {
      calls.push({ save_defaults: ov });
      for (const [k, v] of Object.entries(ov || {})) {
        if (k === 'height') continue;
        if (k in saved.stream) saved.stream[k] = v;
        if (k in saved.queue) saved.queue[k] = v;
      }
      return { ok: true, message: '已設為預設' };
    },
    check_source(text) {
      const target = /^curl\s/i.test(text) ? ((text.match(/https?:\/\/[^\s'"]+/) || [''])[0] || '') : text.trim();
      if (!/^https?:\/\//i.test(target)) return { status: 'file', message: '本機檔案路徑；解析後檢查檔案' };
      try { if (!new URL(target).hostname) throw new Error('host'); } catch (e) { return { status: 'invalid', message: '網址格式不正確' }; }
      if (/\.(mp4|mkv|webm|ts|mov|flv|m3u8)(\?|#|$)/i.test(target))
        return { status: 'recognized', message: '已辨識媒體網址；尚未驗證影片' };
      return { status: 'unknown', message: '無法預先判定此網址；仍可嘗試解析' };
    },
    add(text) {
      text = String(text || '').trim();
      if (!text) return { error: '內容是空的' };
      const isUrl = /^https?:\/\//i.test(text) || /^curl\s/i.test(text);
      const target = /^curl\s/i.test(text) ? ((text.match(/https?:\/\/[^\s'"]+/) || [''])[0] || text) : text;
      st.items.push(mk({ target, kind: isUrl ? 'url' : 'file', title: isUrl ? '' : basename(target), quality: isUrl ? 1080 : null,
        settings: S({ encoder: st.settings.encoder, interpolation: st.settings.interpolation, save_compare: st.settings.save_compare }) }));
      emit(); return ok();
    },
    add_files() {   // alternates: one file, then two files (the panel opens either way)
      const n = ++fileDialogN % 2 ? 1 : 2;
      return { ok: true, paths: Array.from({ length: n }, (_, i) => 'C:\\Users\\demo\\Videos\\new_file_' + (fileDialogN * 10 + i) + '.mp4') };
    },
    remove(id) { st.items = st.items.filter((x) => x.id !== id); emit(); return true; },
    move(id, index) {
      const it = find(id); if (!it) return;
      st.items.splice(st.items.indexOf(it), 1);
      st.items.splice(Math.max(0, Math.min(index, st.items.length)), 0, it); emit();
    },
    mark_all_seen() { st.items.forEach((it) => { if (it.state === 'done') it.seen = true; }); emit(); return ok(); },
    pause(id) {
      const it = find(id);
      if (it && it.state === 'done' && it.subtitle && it.subtitle.only) it.subtitle = { ...it.subtitle, status: 'paused' };
      else if (it) it.state = 'paused';
      emit();
    },
    cancel_subtitles(id) { const it = find(id); if (it) it.subtitle = { status: 'cancelled', percent: 0 }; emit(); return ok(); },
    resume(id) {
      const it = find(id);
      if (it && it.state === 'done' && it.subtitle && it.subtitle.status === 'paused') { it.subtitle = { status: 'queued', only: !!it.subtitle.only, view: mockSubtitleView('queued') }; emit(); return; }
      if (it) it.state = it.progress && it.progress.frames ? 'queued' : 'waiting'; emit(); },
    pause_all() { st.paused_all = true; emit(); },
    resume_all() { st.paused_all = false; emit(); },
    list_subtitle_models() {
      const rows = mockMissingApply(JSON.parse(JSON.stringify(st.subtitle_models)));
      const dual = rows.find((m) => m.id === '__dual__');
      if (dual && rows.some((m) => m.kind === 'asr' && ['whisper-ja-1.5B', 'anime-whisper'].includes(m.id) && m.status === 'missing')) { dual.status = 'missing'; dual.status_label = '找不到模型檔案'; }
      return rows;
    },
    subtitle_combo_fallback() {
      const gone = ['whisper-ja-1.5B', 'anime-whisper'].filter((n) => mockMissing.has(n));
      if (!gone.length || gone.length > 1) return { ok: true, fallback: null };
      const to = gone[0] === 'anime-whisper' ? 'whisper-ja-1.5B＋GalTransl 7B' : 'anime-whisper＋GalTransl 7B v3.7';
      return { ok: true, fallback: { to_label: to, missing: gone, message: `預設組合缺少模型（${gone[0]}），暫時改用 ${to}` } };
    },
    /* --- mock local subtitle model downloads (?mdlinstalled=a,b  ?mdlfree=<GB>  ?mdlfail=1  ?mdlspeed=<MB/s>  ?mdlinuse=1) --- */
    subtitle_model_downloads() { return mockModelDl.view(); },
    plan_subtitle_model_download(ids) { return mockModelDl.plan(ids); },
    start_subtitle_model_download(ids) { return mockModelDl.start(ids); },
    cancel_subtitle_model_download() { return mockModelDl.cancel(); },
    subtitle_model_download_progress() { return { ok: true, progress: mockModelDl.snapshot() }; },
    list_subtitle_connections() { return cloneConfig(st.subtitle_connections || []); },
    /* --- mock subtitle_usage (stage B) --- */
    subtitle_usage() {
      const now = Math.floor(Date.now() / 1000), hour = 3600, top = now - (now % hour);
      const points = Array.from({ length: 48 }, (_, i) => {
        const t = top - (47 - i) * hour, w = 1 + Math.sin(i / 3.5);
        return { t, tokens: Math.round(9000 * w + 1500), audio_s: Math.round(1200 * w * 10) / 10, requests: Math.round(14 * w + 2) };
      });
      const scope = (limit, remaining, reset) => ({ limit, remaining, reset_at: now + reset });
      return {
        version: 1, generated_at: now, warning: 'low',
        providers: [{
          provider: 'groq', connection_id: 'preview-groq',
          models: [
            { model: 'whisper-large-v3-turbo', kind: 'asr', warning: 'low', updated_at: now - 30,
              requests: scope(2000, 1850, 1200), tokens: scope(null, null, 0),
              requests_minute: { used: 3, limit: 20 }, tokens_day: { used: 0, limit: null, reset_in_s: null },
              audio: { hour: { used_s: 6100, limit_s: 7200 }, day: { used_s: 21000, limit_s: 28800 }, limits_verified: true },
              limits_source: 'mixed' },
            { model: 'llama-3.3-70b-versatile', kind: 'chat', warning: 'ok', updated_at: now - 30,
              requests: scope(1000, 940, 600), tokens: scope(8000, 6800, 40),
              requests_minute: { used: 2, limit: 30 }, tokens_day: { used: 12500, limit: 200000, reset_in_s: null },
              audio: { hour: { used_s: 0, limit_s: null }, day: { used_s: 0, limit_s: null }, limits_verified: false },
              limits_source: 'mixed' },
          ],
          series: { granularity: 'hour', points },
        }],
      };
    },
    /* --- end mock subtitle_usage --- */
    set_subtitle_profile(profile) {
      profile = cloneConfig(profile);
      const cloud = profile.profiles.cloud, pool = cloud.connection_pool;
      if (pool) {
        const enabled = pool.connections.filter((c) => c.enabled);
        const selected = pool.auto_switch ? enabled : enabled.slice(0, 1);
        for (const role of ['asr', 'translation']) {
          const ids = selected.map((c) => st.subtitle_models.find((m) => m.connection_id === c.id && m.kind === role)?.id).filter(Boolean);
          cloud[role] = ids[0] || null; cloud.fallback[role] = ids.slice(1);
        }
      }
      st.settings.subtitle_profile = cloneConfig(profile);
      st.settings.subtitle_runtime_contract = 'cloud-first-1';
      emit();
      return { ok: true, profile: cloneConfig(profile), runtime_contract: 'cloud-first-1' };
    },
    test_save_subtitle_connection(spec) {
      st.settings.subtitle_cloud_enabled = true;
      const family = spec.api_family, preset = family === 'groq' ? CLOUD_SUBTITLE_PRESETS.groq : CLOUD_SUBTITLE_PRESETS.azure;
      st.subtitle_connections ||= [];
      const id = spec.id || `preview-connection-${st.subtitle_connections.length + 1}`;
      const connection = { id, label: spec.label || (st.subtitle_connections.find((c) => c.id === id)?.label), api_family: family, auth_source: 'dpapi', azure_region: spec.azure_region || '', validation: {} };
      for (const kind of family === 'groq' ? ['asr', 'translation'] : ['translation']) {
        const remote = preset[kind].default_model;
        connection.validation[kind] = { status: kind === 'asr' && spec.test_audio_path === '__builtin__' ? 'reachable' : 'verified', remote_model: remote };
        const mid = `${id}-${kind}`;
        if (!st.subtitle_models.some((m) => m.id === mid)) st.subtitle_models.push({ id: mid, kind, label: preset[kind].default_label,
          provider: 'openai-compatible', remote_model: remote, connection_id: id, status: 'tested', auth_source: 'dpapi' });
      }
      st.subtitle_connections = st.subtitle_connections.filter((c) => c.id !== id).concat([connection]);
      const verified = Object.values(connection.validation).every((v) => v.status === 'verified');
      return { ok: verified, code: verified ? 'saved' : 'partial_validation', connection: cloneConfig(connection),
        models: cloneConfig(st.subtitle_models.filter((m) => m.connection_id === id)) };
    },
    remove_subtitle_pool_connection(id) {
      const profile = cloneConfig(st.settings.subtitle_profile);
      if (!profile.profiles.cloud.connection_pool) return { ok: false, error: '請先儲存連線池設定' };
      profile.profiles.cloud.connection_pool.connections = profile.profiles.cloud.connection_pool.connections.filter((c) => c.id !== id);
      st.subtitle_connections = (st.subtitle_connections || []).filter((c) => c.id !== id);
      st.subtitle_models = st.subtitle_models.filter((m) => m.connection_id !== id);
      const result = this.set_subtitle_profile(profile);
      return { ok: true, profile: result.profile };
    },
    test_add_subtitle_model(spec) {
      spec = spec || {};
      calls.push({ test_add_subtitle_model: { kind: spec.kind, provider: spec.provider, label: spec.label, recipe: spec.recipe, api_family: spec.api_family } });
      if (!spec.label || !spec.recipe || !['asr', 'translation'].includes(spec.kind)) return { ok: false, error: '請提供模型名稱與方案。' };
      if (spec.provider === 'local' && !spec.model_path) return { ok: false, error: '請提供本機模型檔案路徑。' };
      if (spec.provider === 'openai-compatible' && (!spec.base_url || !spec.remote_model)) return { ok: false, error: '請提供服務網址與遠端模型名稱。' };
      if (spec.kind === 'asr' && !spec.test_audio_path) return { ok: false, error: '請提供可執行推論測試的音訊樣本。' };
      const model = {
        id: `user-${Date.now().toString(36)}-${st.subtitle_models.length}`,
        label: String(spec.label), kind: spec.kind, provider: spec.provider || 'local', recipe: String(spec.recipe),
        builtin: false, removable: true, status: 'ready',
        status_label: spec.provider === 'openai-compatible' ? '已通過新增測試；目前連線未檢查' : '本機模型可用',
        auth_source: spec.provider === 'openai-compatible' ? (spec.api_key ? 'dpapi' : 'env') : 'local',
        ...(spec.provider === 'local' ? { model_path: String(spec.model_path) } : { base_url: String(spec.base_url), remote_model: String(spec.remote_model), api_family: String(spec.api_family || 'openai-compatible') }),
      };
      st.subtitle_models.push(model); emit(); return { ok: true, model: JSON.parse(JSON.stringify(model)) };
    },
    update_subtitle_model_key(id, apiKey) {
      calls.push({ update_subtitle_model_key: { id } });
      const model = st.subtitle_models.find((item) => item.id === id);
      if (!model || !model.removable || model.provider !== 'openai-compatible') return { ok: false, error: '找不到可更新金鑰的雲端模型。' };
      if (!String(apiKey || '').trim()) return { ok: false, error: '請提供新的 API 金鑰。' };
      model.auth_source = 'dpapi';
      model.status = 'ready';
      model.status_label = '已通過新增測試；目前連線未檢查';
      emit();
      return { ok: true, model: JSON.parse(JSON.stringify(model)) };
    },
    remove_subtitle_model(id) {
      const model = st.subtitle_models.find((item) => item.id === id);
      if (!model || !model.removable) return { ok: false, error: '此模型不能移除。' };
      st.subtitle_models = st.subtitle_models.filter((item) => item.id !== id); emit(); return { ok: true };
    },
    set_subtitle_configuration(primary, enabled, rows, itemId = null) {
      calls.push({ set_subtitle_configuration: { itemId, enabled: !!enabled, rowCount: (rows || []).length } });
      if (itemId != null) {
        const item = find(itemId);
        if (!item || !SUBTITLE_EDIT_STATES.has(item.state)) return { ok: false, error: '此工作已開始處理，無法變更字幕設定' };
        Object.assign(item.settings, { subtitle_primary: JSON.parse(JSON.stringify(primary)), subtitle_comparison_enabled: !!enabled, subtitle_comparison_rows: JSON.parse(JSON.stringify(rows || [])) });
      } else {
        Object.assign(st.settings, { subtitle_primary: JSON.parse(JSON.stringify(primary)), subtitle_comparison_enabled: !!enabled, subtitle_comparison_rows: JSON.parse(JSON.stringify(rows || [])) });
      }
      emit(); return { ok: true };
    },
    set_subtitles(id, enabled) {
      const it = find(id);
      if (!it || !SUBTITLE_EDIT_STATES.has(it.state)) return { ok: false, error: '此工作已開始處理，無法變更字幕設定' };
      if (!subtitleRuntimeAvailable(st.subtitle_environment)) return { ok: false, error: subtitleRuntimeReason(st.subtitle_environment) };
      it.settings.subtitles = !!enabled;
      if (!enabled) it.subtitle = { status: 'skipped', percent: 0 };
      else it.subtitle = {};
      emit(); return { ok: true };
    },
    rerun_subtitles(id) {
      const it = find(id);
      if (!it || it.state !== 'done') return { ok: false, error: '只有已完成的影片可以重跑字幕' };
      if (!subtitleRuntimeAvailable(st.subtitle_environment)) return { ok: false, error: subtitleRuntimeReason(st.subtitle_environment) };
      it.subtitle = { status: 'queued', only: true, view: mockSubtitleView('queued') };
      emit(); return { ok: true };
    },
    retry_subtitles(id) {
      const it = find(id);
      if (!it || !it.subtitle || !it.subtitle.status === 'failed' || (it.subtitle.view && it.subtitle.view.retryable)) return { ok: false, error: '字幕沒有等待重試' };
      it.subtitle = { status: 'queued', only: !!it.subtitle.only, view: mockSubtitleView('queued') };
      emit(); return { ok: true };
    },
    apply_subtitle_settings(id, choice) {
      calls.push({ apply_subtitle_settings: { id, choice } });
      const it = find(id);
      if (!it || !it.subtitle || !it.subtitle.settings_pending) return { ok: false, error: '這個項目沒有待套用的字幕設定' };
      if (choice === 'redo_mt' && it.subtitle.settings_pending.asr_changed) return { ok: false, error: '語音辨識設定已變更，無法只重做翻譯' };
      delete it.subtitle.settings_pending;
      it.settings = { ...(it.settings || {}), subtitle_hold: choice === 'keep' };
      emit(); return { ok: true, error: '' };
    },
    apply_subtitle_settings_all(choice) {
      calls.push({ apply_subtitle_settings_all: { choice } });
      let count = 0;
      for (const it of st.items) {
        if (!it.subtitle || !it.subtitle.settings_pending) continue;
        delete it.subtitle.settings_pending; count++;
        it.settings = { ...(it.settings || {}), subtitle_hold: choice === 'keep' };
      }
      emit(); return { ok: true, count };
    },
    restart_subtitles(id) {
      const it = find(id);
      if (!it || !it.settings || !it.settings.subtitles) return { ok: false, error: '此項目沒有字幕' };
      it.subtitle = { status: 'queued', only: it.state === 'done', view: mockSubtitleView('queued') };
      emit(); return { ok: true };
    },
    set_keep(id, v) { const it = find(id); if (it) it.keep = !!v; emit(); },
    cancel_play_after_subtitles(id) { const it = find(id); if (it) it.play_after_subtitles = false; emit(); return ok(); },
    play(id, start, waitForSubtitles = false) { const it = find(id); if (it && subtitlePlayPrompt(it)) { if (waitForSubtitles) { it.play_after_subtitles = true; emit(); return { ok: true, pending: true }; } it.play_after_subtitles = false; it.subtitle = { status: 'skipped' }; emit(); } if (it && it.state === 'done') { it.watched = true; it.seen = true; emit(); } return ok('播放：' + (it && (it.title || it.target)) + (start != null ? ' @ ' + start + ' 秒' : '')); },
    export(id, includeCompare) { return ok('匯出：' + id + (includeCompare ? '（含比對軌）' : '')); },
    remove_compare(id) {
      const it = find(id);
      if (!it || !it.has_compare) return { ok: false, error: '這個項目沒有比對軌' };
      return new Promise((resolve) => setTimeout(() => {   // slow on purpose, to preview the busy state
        it.has_compare = false; it.compare_original = null;
        it.size_bytes = Math.round((it.size_bytes || 0) * 0.55);
        emit(); resolve(ok('已移除比對軌'));
      }, 1500));
    },
    clear_headers(id) { const it = find(id); if (it) it.headers = {}; emit(); },
    retry(id) { const it = find(id); if (it) { it.state = 'waiting'; it.error = ''; it.progress = {}; } emit(); },
    set_settings(d) { Object.assign(st.settings, d); emit(); return st.settings; },
    apply_model_settings(id) {
      const it = find(id);
      if (!it || !MODEL_EDIT_STATES.has(it.state)) return { error: '此工作目前無法變更模型' };
      for (const key of MODEL_KEYS) it.settings[key] = st.settings[key];
      it.progress = {}; it.cache_key = ''; emit();
      return ok('已套用目前模型');
    },
    reprocess_options(id) {
      const it = find(id);
      if (!it) return { ok: false, error: '找不到這個工作' };
      const isUrl = it.kind !== 'file' && /^https?:\/\//i.test(String(it.target || ''));
      const missing = it.kind === 'file' && /missing/i.test(String(it.target || ''));
      const source = missing
        ? { mode: 'missing', message: '找不到本機原始檔案；請重新指定目前可用的檔案路徑。', lossy: false }
        : isUrl
          ? { mode: 'download', message: '網址來源可用。更改輸出畫質時會重新下載來源。', lossy: false }
          : { mode: 'file', message: '本機原始檔案可用。', lossy: false };
      const hasProgress = Object.values(it.progress || {}).some((value) => num(value) > 0);
      const editInPlace = it.state === 'waiting' && !it.source_path && !hasProgress;
      return { ok: true, settings: Object.assign({}, it.settings), quality: it.quality, source,
        state: it.state, target: it.target, kind: it.kind, edit_in_place: editInPlace };
    },
    reprocess(id, changes, options) {
      const it = find(id);
      if (!it) return { ok: false, error: '找不到這個工作' };
      changes = changes || {}; options = options || {};
      calls.push({ reprocess: { id, changes: Object.assign({}, changes), options: Object.assign({}, options) } });
      const idx = st.items.indexOf(it);
      const sourcePath = options.source_path || it.source_path || (it.kind === 'file' ? it.target : '');
      const hasProgress = Object.values(it.progress || {}).some((value) => num(value) > 0);
      if (it.state === 'waiting' && !it.source_path && !hasProgress) {
        if (options.source_path) it.target = options.source_path;
        it.quality = changes.quality !== undefined ? changes.quality : it.quality;
        it.settings = S(Object.assign({}, it.settings || {}, changes));
        it.cache_key = ''; it.error = ''; it.progress = {};
        emit();
        return { ok: true, id: it.id, message: '（預覽模式）已更新此工作設定' };
      }
      if (options.stop_current && (it.state === 'downloading' || it.state === 'processing')) it.state = 'paused';
      const next = mk({
        title: `${it.title || basename(it.target) || '未命名工作'}（重新處理）`,
        target: it.target,
        kind: it.kind,
        headers: Object.assign({}, it.headers || {}),
        quality: changes.quality !== undefined ? changes.quality : it.quality,
        settings: S(Object.assign({}, it.settings || {}, changes)),
        state: options.redownload || (it.kind !== 'file' && !sourcePath) ? 'waiting' : 'queued',
        source_path: options.source_path || sourcePath,
        parent_id: id,
      });
      st.items.splice(idx + 1, 0, next);
      emit();
      return { ok: true, id: next.id, message: `（預覽模式）已建立重新處理版本 ${next.id}` };
    },
    open_folder(id) { return ok('開啟資料夾：' + (id || '快取資料夾')); },
    runtime_info() {   // ?resetup=dev previews a development checkout
      return new URLSearchParams(location.search).get('resetup') === 'dev'
        ? { available: false, flavor: null, label: '', verified: false, note: '開發環境不提供（只有安裝版可以重新偵測或切換版本）' }
        : { available: true, flavor: 'nvidia', label: 'NVIDIA（CUDA 12.8）', verified: true, note: '' };
    },
    resetup() { return ok('不會真的關閉並開啟設定畫面'); },
    app_version() { return '0.1.0'; },
    check_update() {   // ?update=update|latest|no_release|error picks the simulated result
      const k = new URLSearchParams(location.search).get('update') || 'update';
      const base = { current: '0.1.0', latest: '', notes: '', url: '' };
      const R = {
        update: { status: 'update', latest: '0.2.0', message: '有新版本 0.2.0 可以下載。',
          notes: '- 新增檢查更新\n- 修正字幕佇列問題\n<b>這行不應被當成 HTML</b>', url: 'https://github.com/life-hack-lab/oh-streamer/releases/tag/v0.2.0' },
        latest: { status: 'latest', latest: '0.1.0', message: '目前已是最新版本（0.1.0）。' },
        no_release: { status: 'no_release', message: '目前還沒有公開的版本可以下載。' },
        error: { status: 'error', message: '無法連線，請確認網路後再試。' },
      };
      return new Promise((res) => setTimeout(() => res(Object.assign(base, R[k] || R.error)), 150));
    },
    open_release_page(url) { return ok('（預覽模式）開啟 ' + url); },
    open_model_card(id) { (window.__modelCardOpened = window.__modelCardOpened || []).push(id); return ok('（預覽模式）開啟模型卡 ' + id); },
    perf_start(kind) {
      if (st.perf.status === 'running') return { ok: false, error: '檢測進行中' };
      Object.assign(st.perf, { status: 'running', kind, stage: '準備測試影片…', progress: 0, error: null, can_start: false, prompt: false });
      const stages = ['準備測試影片…', '測試 480p', '測試 720p', '測試 1080p', '測試補幀', '整理結果…'];
      const total = kind === 'standard' ? 24 : 10;
      let n = 0;
      clearInterval(perfT);
      perfT = setInterval(() => {
        n++;
        st.perf.progress = Math.min(1, n / total);
        st.perf.remaining_s = n >= 3 ? (total - n) * 12 : null;
        st.perf.stage = stages[Math.min(stages.length - 1, Math.floor(n / total * stages.length))];
        if (n >= total) { clearInterval(perfT); Object.assign(st.perf, { status: 'done', stage: '', progress: 1, can_start: true, profile: profile(), apply_plan: mockPlan('off') }); }
        emit();
      }, 500);
      emit(); return ok('已開始檢測');
    },
    perf_cancel() { clearInterval(perfT); Object.assign(st.perf, { status: 'cancelled', can_start: true, progress: 0 }); emit(); return ok('已取消檢測'); },
    perf_apply() { st.perf.apply_plan = mockPlan('auto'); emit(); return ok('已套用：播放器補幀「自動」。開著的播放器也會立即更新。'); },
    perf_dismiss_prompt() { st.perf.prompt = false; emit(); return { ok: true }; },
    gpu(active) {
      if (!active) return {};
      gpuT += 1;
      const busy = st.items.some((i) => i.state === 'processing') && !st.paused_all;
      const util = busy ? 80 + 12 * Math.sin(gpuT / 5) + Math.random() * 4 : 3 + Math.random() * 3;
      return { name: 'NVIDIA GeForce RTX 5060 Laptop GPU（預覽）', util, mem_used_gib: busy ? 5.6 + 0.4 * Math.sin(gpuT / 17) : 0.9,
               mem_total_gib: 8, temp: busy ? 62 + Math.min(14, gpuT / 6) : 48, power_w: busy ? 78 + 6 * Math.sin(gpuT / 4) : 9, power_limit_w: 115, sm_clock: 2400, backend: 'nvml',
               enc_util: busy ? 55 + 10 * Math.sin(gpuT / 6) : 0, dec_util: busy ? 28 + 6 * Math.sin(gpuT / 7) : 0 };
    },
  };  // ?combos=none: behave like an older backend without the combo methods
  return mockApi;
}

/* ------------------------------------------------------------------ boot */

let booted = false;
function boot(b, mock) {
  if (booted) return;
  booted = true;
  backend = b; isMock = mock;
  if (mock) document.title = `${BRAND_NAME}（預覽模式）`;
  refresh();
  loadSubtitleModels();
  if (lsGet(GPU_KEY) === '1' || new URLSearchParams(location.search).has('gpu')) setGpuOpen(true, false);
}

function realApiReady() {
  const a = window.pywebview && window.pywebview.api;
  return !!(a && typeof a.get_state === 'function');
}

(function start() {
  if (new URLSearchParams(location.search).has('mock')) { boot(makeMock(), true); return; }
  if (realApiReady()) { boot(window.pywebview.api, false); return; }
  window.addEventListener('pywebviewready', () => boot(window.pywebview.api, false));
  if (!window.pywebview) {
    // plain browser: pywebview never shows up -> use the mock after a short grace period
    setTimeout(() => {
      if (booted) return;
      if (realApiReady()) boot(window.pywebview.api, false);
      else if (!window.pywebview) boot(makeMock(), true);
    }, 400);
  }
})();

})();
