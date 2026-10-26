// Run with node tests/test_queue_subtitle_view_ui.js (no browser dependencies).
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const source = fs.readFileSync(path.join(__dirname, '../src/oh_streamer/resources/queue_ui/app.js'), 'utf8');
const helperStart = source.indexOf('const SUBTITLE_EDIT_STATES =');
const helperEnd = source.indexOf('\nfunction modelLabel(', helperStart);
assert(helperStart >= 0 && helperEnd > helperStart, 'subtitle state helpers are present');
const context = {};
vm.createContext(context);
vm.runInContext(source.slice(helperStart, helperEnd), context);

const NOW = 1000;
const base = { v: 2, progress: 0, covered_s: 0, slices_done: 0, slices_total: 4, failures_elsewhere: 0, partial_stuck: false, retryable: false };
const show = (view, extra) => context.subtitleDisplay({ status: 'queued', view: { ...base, ...view }, ...extra }, NOW);

// complete
let d = show({ state: 'complete', progress: 1, covered_s: 600 });
assert.equal(d.label, '字幕：完成');
assert.equal(d.active, false);
assert.equal(d.retryable, false);

// stuck, retryable
d = show({ state: 'stuck', retryable: true, stuck: { slice: 2, stage: 'asr', kind: 'rate_limit', exhausted: true,
  candidates: [{ id: 'groq', name: 'Groq', kind: 'rate_limit' }, { id: 'local', name: '本地', kind: 'unsupported' }] } });
assert.equal(d.label, '字幕：語音辨識卡住：Groq（被限流）、本地（不支援或環境不符）');
assert.equal(d.retryable, true);
assert.equal(show({ state: 'stuck', covered_s: 125, stuck: { stage: 'mt', candidates: [{ name: 'Gemini', kind: 'timeout' }] } }).label,
  '字幕已到 02:05 · 翻譯卡住：Gemini（逾時）');

assert.equal(show({ state: 'stuck', stuck: { stage: 'asr', candidates: [{ name: '本地', kind: 'error' }, { name: 'Groq', kind: 'oom' }, { name: 'X', kind: 'exhausted' }] } }).label,
  '字幕：語音辨識卡住：本地（執行錯誤）、Groq（GPU 記憶體不足）、X（所有候選都無法使用）');

// running
d = show({ state: 'running', progress: 0.42, running: { executor: 'cloud', slice: 2, stage: 'asr', candidate_name: 'Groq' } });
assert.equal(d.label, '字幕：語音辨識中（雲端，第 3 片）');
assert.equal(d.percent, 42);
assert.equal(d.showPercent, true);
assert.equal(d.showTrack, true);
assert.equal(d.indeterminate, false);
assert.equal(d.cat, 'run');
assert.equal(d.glyph, '');
assert.equal(d.fill, 42);
d = show({ state: 'running', covered_s: 65, progress: 0.5, running: { executor: 'local', slice: 0, stage: 'mt' } });
assert.equal(d.label, '字幕已到 01:05 · 翻譯中（本地，第 1 片）');

// waiting_quota: seconds follow the clock, not remaining_s
const quota = { state: 'waiting_quota', progress: 0.3, wait: { kind: 'quota', slice: 1, stage: 'asr', retry_at: 1030, remaining_s: 99 } };
d = context.subtitleDisplay({ status: 'queued', view: { ...base, ...quota } }, 1000);
assert.equal(d.label, '字幕：等待額度中');
assert.equal(d.detail, '約 30 秒後重試');
assert.equal(d.indeterminate, false);
assert.equal(d.cat, 'wait');
assert.equal(d.glyph, '⧗');
assert.equal(d.showTrack, true);
assert.equal(d.fill, 30, 'bar is coverage, not a sliding animation');
assert.equal(d.showPercent, false);
assert.equal(d.percent, null);
assert.equal(context.subtitleDisplay({ status: 'queued', view: { ...base, ...quota } }, 1025).detail, '約 10 秒後重試');
assert.equal(context.subtitleDisplay({ status: 'queued', view: { ...base, ...quota } }, 2000).detail, '即將重試');
assert.equal(context.subtitleDisplay({ status: 'queued', view: { ...base, ...quota } }, 1030).detail, '即將重試');
assert.equal(context.subtitleDisplay({ status: 'queued', view: { ...base, ...quota } }, 1030 - 125).detail, '約 2 分鐘後重試');
assert.equal(context.subtitleDisplay({ status: 'queued', view: { ...base, ...quota } }, 1001).detail, context.subtitleDisplay({ status: 'queued', view: { ...base, ...quota } }, 1002).detail, 'detail is stable between seconds');
assert.equal(context.subtitleDisplay({ status: 'queued', view: { ...base, ...quota } }, 1029.5).detail, '約 10 秒後重試');
d = show({ state: 'waiting_quota', wait: { ...quota.wait, also_waiting_download: true } });
assert.equal(d.label, '字幕：等待下載');
assert.equal(d.glyph, '↓');

// retrying: local error / network backoff is not a quota wait
d = show({ state: 'retrying', wait: { kind: 'retry', slice: 0, stage: 'mt', reason: 'error', retry_at: 1030 } });
assert.equal(d.label, '字幕：重試中（執行錯誤）');
assert.equal(d.detail, '約 30 秒後重試');
assert.equal(d.indeterminate, false);
assert.equal(d.active, true);
assert.equal(show({ state: 'retrying', wait: { kind: 'retry', reason: 'network', retry_at: 1004 } }).detail, '約 10 秒後重試');
assert.equal(show({ state: 'retrying', wait: { kind: 'retry', reason: 'network', retry_at: 900 } }).detail, '即將重試');
assert.equal(show({ state: 'running', running: { executor: 'cloud', slice: 1, stage: 'asr', quota_wait: true, quota_retry_at: 999 } }).detail, '部分等待額度（即將重試）');
assert.equal(show({ state: 'retrying', wait: { kind: 'retry', retry_at: null } }).detail, '稍後重試');

// running while part of the work waits for quota: main state stays running, detail says so
d = show({ state: 'running', running: { executor: 'cloud', slice: 1, stage: 'asr', quota_wait: true, quota_retry_at: 1020 } });
assert.equal(d.label, '字幕：語音辨識中（雲端，第 2 片）');
assert.equal(d.detail, '部分等待額度（約 20 秒後重試）');
assert.equal(show({ state: 'running', running: { executor: 'cloud', slice: 1, stage: 'asr', quota_wait: true } }).detail, '部分等待額度');

// waiting_gpu / waiting_executor / waiting_download / queued
d = show({ state: 'waiting_gpu', wait: { kind: 'gpu', executor: 'local' } });
assert.equal(d.label, '字幕：等待 GPU');
assert.equal(d.detail, '修復中，稍後輪到字幕');
assert.equal(d.indeterminate, false);
assert.equal(show({ state: 'waiting_executor', wait: { kind: 'executor', executor: 'local', reclaim_at: null } }).label,
  '字幕：已交給本地 GPU 處理，等待開始');
assert.equal(show({ state: 'waiting_executor', wait: { kind: 'executor', executor: 'local', reclaim_at: 1045 } }).label,
  '字幕：已交給本地 GPU 處理，等待開始');
assert.equal(show({ state: 'waiting_executor', wait: { kind: 'executor', executor: 'local', reclaim_at: 1045 } }).detail, '雲端約 50 秒後恢復');
assert.equal(show({ state: 'waiting_executor', wait: { kind: 'executor', executor: 'local', reclaim_at: 1120 } }).detail, '雲端約 2 分鐘後恢復');
assert.equal(show({ state: 'waiting_executor', wait: { kind: 'executor', executor: 'cloud' } }).label, '字幕：已交給雲端處理，等待開始');
assert.equal(show({ state: 'waiting_download', wait: { kind: 'download' } }).label, '字幕：等待下載');
d = show({ state: 'queued', wait: null });
assert.equal(d.label, '字幕：排隊中');
assert.equal(d.showTrack, true);
assert.equal(d.cat, 'wait');
assert.equal(d.retryable, false);

// stale events must never leak: view says queued, old exhausted fallback / cloud_mode / retry_note ignored
d = show({ state: 'queued' }, {
  fallback: { state: 'exhausted', role: 'asr', primary_error: 'rate_limit' }, backup_status: '本地備援失敗',
  cloud_mode: true, block_mode: true, retry_note: '服務限流（429），預計 10:00 自動重試', phase: 'waiting_download',
});
assert.equal(d.label, '字幕：排隊中');
assert.doesNotMatch(d.label, /失敗|無法使用|卡住/);
assert.equal(d.retryable, false);

// partial_stuck: another position failed, playback not blocked
d = show({ state: 'running', partial_stuck: true, failures_elsewhere: 2, retryable: true, progress: 0.2,
  running: { executor: 'cloud', slice: 0, stage: 'asr' } });
assert.equal(d.retryable, true);
assert.match(d.detail, /另有 2 段失敗，可重試/);
assert.equal(d.label, '字幕：語音辨識中（雲端，第 1 片）');

// without a v2 view only the item-level status decides; scheduler phases read as queued
const plain = (status, extra) => context.subtitleDisplay({ status, ...extra });
for (const status of ['queued', 'waiting_gpu', 'asr', 'translate', 'partial', 'combos', 'blocked']) {
  assert.equal(plain(status).label, '字幕：排隊中', `${status} without a view is just queued`);
}
assert.equal(context.subtitleDisplay({ status: 'queued', view: { v: 1, state: 'complete' } }).label, '字幕：排隊中');
assert.equal(plain('done').label, '字幕：完成');
assert.equal(plain('done').restartable, true);
assert.equal(plain('mux').label, '字幕：封裝中');
assert.equal(plain('ready').label, '字幕：等待封裝');
assert.equal(plain('mux_pending').label, '字幕：等關閉播放器後封裝');
assert.equal(plain('skipped').label, '字幕：略過');
assert.equal(plain('cancelled').label, '字幕：已取消');
assert.equal(plain('paused').label, '字幕：已暫停');
const failed = plain('failed', { error: 'boom' });
assert.equal(failed.label, '字幕：失敗');
assert.equal(failed.retryable, true);
assert.equal(failed.error, 'boom');
assert.equal(context.subtitleDisplay({}), null);
assert.equal(context.subtitleDisplay(null), null);

// a finished slice table followed by mux: the item status wins over view.complete
d = show({ state: 'complete', progress: 1, covered_s: 600 }, { status: 'mux_pending' });
assert.equal(d.label, '字幕：等關閉播放器後封裝');
assert.equal(d.restartable, false);

// play prompt reads the view: covered_s > 0 plays directly; nothing covered yet asks
const prompt = (view, status = 'queued') => context.subtitlePlayPrompt({ subtitle: { status, view: { ...base, ...view } } });
assert.equal(prompt({ state: 'running', covered_s: 30, running: { executor: 'local', slice: 0, stage: 'asr' } }), null);
assert.equal(prompt({ state: 'queued' }), '字幕：排隊中');
assert.equal(prompt({ state: 'waiting_download' }), '字幕：等待下載 字幕已追上下載進度');
assert.equal(show({ state: 'waiting_cut', wait: { kind: 'cut' } }).label, '字幕：準備切片');
assert.equal(show({ state: 'waiting_cut' }).active, true);
assert.equal(prompt({ state: 'waiting_cut' }), '字幕：準備切片');
assert.equal(prompt({ state: 'complete', progress: 1 }), null);
assert.equal(prompt({ state: 'stuck', stuck: { stage: 'mt', candidates: [] } }), null);
assert.equal(context.subtitlePlayPrompt({ subtitle: { status: 'queued' } }), '字幕：排隊中');
assert.equal(context.subtitlePlayPrompt({ subtitle: { status: 'done' } }), null);
assert.equal(context.subtitlePlayPrompt({ subtitle: { status: 'mux' } }), null);

console.log('Subtitle view display: passed');

// ---- quiet progress: halt / pause overrides (subtitleHalt) ----
{
  const item = (extra, view) => ({ state: 'processing', subtitle: { status: 'queued', view: { ...base, ...view }, ...extra } });
  const disp = (it, state) => context.subtitleDisplay(it.subtitle, NOW, { item: it, state });
  const wait = { state: 'waiting_gpu', covered_s: 65, progress: 0.3 };
  assert.equal(context.subtitleHalt(item({}, wait), {}), null);
  const run = item({}, { state: 'running', running: { executor: 'cloud', stage: 'asr' } });
  assert.equal(context.subtitleHalt(run, { paused_all: true }), null);
  assert.equal(disp(run, { paused_all: true }).cat, 'run');
  let it = item({}, wait); it.state = 'paused';
  let d = disp(it, { paused_all: true });
  assert.equal(d.cat, 'paused'); assert.equal(d.glyph, '⏸'); assert.equal(d.label, '字幕已到 01:05 · 已暫停'); assert.equal(d.fill, 30);
  d = disp(item({}, wait), { paused_all: true, perf: { status: 'running' }, player_busy: true });
  assert.equal(d.cat, 'halt'); assert.equal(d.glyph, '⏸'); assert.equal(d.label, '字幕已到 01:05 · 全部暫停中'); assert.equal(d.detail, '繼續後自動恢復');
  d = disp(item({}, wait), { perf: { status: 'running' }, player_busy: true });
  assert.equal(d.label, '字幕已到 01:05 · 效能檢測中'); assert.equal(d.glyph, '⇄'); assert.equal(d.detail, '檢測完成後自動繼續');
  d = disp(item({}, { ...wait, covered_s: 0 }), { player_busy: true });
  assert.equal(d.label, '字幕：讓給播放器'); assert.equal(d.detail, '播放結束後自動繼續');
  d = disp(item({}, wait), { notice: '空間不足，佇列已暫停' });
  assert.equal(d.cat, 'attn'); assert.equal(d.tone, 'warn'); assert.equal(d.glyph, '!'); assert.equal(d.detail, '釋放空間後自動繼續');
  assert.equal(d.label, '字幕已到 01:05 · 磁碟空間不足');
  assert.equal(disp(item({}, wait), { notice: '其他提示' }).cat, 'wait');
  assert.equal(disp(item({}, { state: 'complete', progress: 1 }), { paused_all: true }).cat, 'done');
  assert.equal(disp(item({}, { state: 'complete', progress: 1 }), { paused_all: true }).fill, 100);
  d = disp(item({}, { state: 'stuck', stuck: { stage: 'asr' } }), { paused_all: true });
  assert.equal(d.cat, 'attn'); assert.equal(d.tone, '');
  d = disp(item({ status: 'mux', percent: null }, { state: 'complete', progress: 1 }), { paused_all: true });
  assert.equal(d.cat, 'run'); assert.equal(d.dim, true); assert.equal(d.fill, 100);
  d = disp(item({ status: 'mux_pending' }, { state: 'complete', progress: 1 }), { player_busy: true });
  assert.equal(d.cat, 'wait');
  const old = { state: 'processing', subtitle: { status: 'queued' } };
  assert.equal(context.subtitleDisplay(old.subtitle, NOW, { item: old, state: { paused_all: true } }).cat, 'halt');
  assert.equal(context.subtitleDisplay({ status: 'skipped' }, NOW).cat, 'plain');
}
