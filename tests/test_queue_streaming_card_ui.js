// Run with node tests/test_queue_streaming_card_ui.js (no browser dependencies).
// T3.5: the 「串流中」 card. streamingCardModel is pure (get_state().streaming -> texts); renderStreaming writes it into the static card.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const root = path.join(__dirname, '../src/oh_streamer/resources/queue_ui');
const source = fs.readFileSync(path.join(root, 'app.js'), 'utf8');
const html = fs.readFileSync(path.join(root, 'index.html'), 'utf8');
const css = fs.readFileSync(path.join(root, 'app.css'), 'utf8');

function slice(from, to) {
  const a = source.indexOf(from);
  const b = source.indexOf(to, a);
  assert(a >= 0 && b > a, `${from} .. ${to} is present`);
  return source.slice(a, b);
}

// minimal DOM: nodes are looked up by selector; only what setText/setHidden/setAttr and renderStreaming touch
function fakeDom() {
  const nodes = new Map();
  const make = () => ({
    textContent: '', hidden: false, attrs: {}, style: {},
    getAttribute(n) { return n in this.attrs ? this.attrs[n] : null; },
    setAttribute(n, v) { this.attrs[n] = String(v); },
    removeAttribute(n) { delete this.attrs[n]; },
  });
  return { nodes, $: (sel) => { if (!nodes.has(sel)) nodes.set(sel, make()); return nodes.get(sel); } };
}

const dom = fakeDom();
const context = { console, $: dom.$ };
vm.createContext(context);
vm.runInContext(slice('function setText(', 'const num ='), context);
vm.runInContext(slice('const num =', '\nfunction basename('), context);
vm.runInContext(slice('const SUBTITLE_EDIT_STATES =', '\nfunction modelLabel('), context);
vm.runInContext(slice('/* ---- streaming card (T3.5)', '/* ---- end streaming card ---- */'), context);

const NOW = 1000;
const model = (st) => context.streamingCardModel(st, NOW);
const view = (state, extra) => ({ v: 2, state, progress: 0, covered_s: 0, slices_done: 0, slices_total: 4, failures_elsewhere: 0,
  partial_stuck: false, retryable: false, ...extra });
const live = (extra) => ({ title: '示範影片', pos: 125.5, duration: 3600, playing: true, buffering: false, ahead_s: 45, speed: 0.7,
  restore_done: false, stream_item: null, subtitle: null, ...extra });

// -- no stream, no card
assert.equal(model(null), null);
assert.equal(model(undefined), null);

// -- stream without subtitles: playback info only, no subtitle row
let m = model(live());
assert.equal(m.title, '示範影片');
assert.equal(m.state, '播放中');
assert.equal(m.timeText, '0:02:06 / 1:00:00');
assert.equal(m.restore, '修復領先 45 秒・0.7x');
assert.equal(m.restoreLevel, 'warn');
assert.equal(m.subtitle, false);
assert.ok(Math.abs(m.playedPct - 125.5 / 36) < 1e-9);
assert.ok(Math.abs(m.aheadPct - 170.5 / 36) < 1e-9);

// -- stream with subtitles: lead in minutes, state of the work as the note
m = model(live({ subtitle: { status: 'queued', lead_s: 720, view: view('running', { progress: 0.4, covered_s: 845.5,
  running: { executor: 'cloud', slice: 2, stage: 'asr', candidate_name: 'Groq' } }) } }));
assert.equal(m.subtitle, true);
assert.equal(m.subLabel, '字幕：已領先 12 分鐘');
assert.equal(m.subLevel, 'good');
assert.equal(m.subNote, '語音辨識中（雲端，第 3 片）・Groq');

// -- short leads read in seconds, a small lead is a warning
m = model(live({ subtitle: { status: 'queued', lead_s: 20, view: view('running', { covered_s: 145 }) } }));
assert.equal(m.subLabel, '字幕：已領先 20 秒');
assert.equal(m.subLevel, 'warn');
m = model(live({ subtitle: { status: 'queued', lead_s: 90, view: view('running', { covered_s: 215 }) } }));
assert.equal(m.subLabel, '字幕：已領先 1 分 30 秒');

// -- subtitles that do not cover the viewer yet
m = model(live({ subtitle: { status: 'queued', lead_s: -8, view: view('running', { covered_s: 117.5,
  running: { executor: 'local', slice: 4, stage: 'mt' } }) } }));
assert.equal(m.subLabel, '字幕：尚未跟上播放');
assert.equal(m.subLevel, 'warn');
assert.equal(m.subNote, '翻譯中（本地，第 5 片）');

// -- nothing covered yet: the plain subtitle label of the queue (same wording as an item card)
m = model(live({ subtitle: { status: 'queued', lead_s: null, view: view('waiting_quota', { wait: { kind: 'quota', retry_at: NOW + 12 } }) } }));
assert.equal(m.subLabel, '字幕：等待額度中（12 秒後重試）');
m = model(live({ subtitle: { status: 'queued', lead_s: null, view: null } }));
assert.equal(m.subLabel, '字幕：排隊中');
m = model(live({ subtitle: { status: null, view: null } }));
assert.equal(m.subLabel, '字幕：準備中');

// -- complete / stuck
m = model(live({ subtitle: { status: 'queued', lead_s: 2000, view: view('complete', { covered_s: 3600 }) } }));
assert.equal(m.subLabel, '字幕：完成');
assert.equal(m.subLevel, 'good');
m = model(live({ subtitle: { status: 'queued', lead_s: 95, view: view('stuck', { covered_s: 220, retryable: true,
  stuck: { slice: 5, stage: 'asr', candidates: [{ id: 'g', name: 'Groq', kind: 'rate_limit' }] } }) } }));
assert.equal(m.subLabel, '字幕已到 03:40 · 語音辨識卡住：Groq（被限流）');
assert.equal(m.subLevel, 'bad');

// -- restore states
m = model(live({ speed: 1.4, ahead_s: 30 }));
assert.equal(m.restore, '修復領先 30 秒・1.4x');
assert.equal(m.restoreLevel, 'good');
m = model(live({ speed: 0.6, ahead_s: 2 }));
assert.equal(m.restoreLevel, 'bad');
m = model(live({ speed: null, ahead_s: null }));
assert.equal(m.restore, '修復準備中');
assert.equal(m.restoreLevel, '');
m = model(live({ restore_done: true, speed: null, ahead_s: 600 }));
assert.equal(m.restore, '修復完成');
m = model(live({ ahead_s: 125, speed: 1.0 }));
assert.equal(m.restore, '修復領先 2 分 5 秒・1.0x');

// -- player states, unknown position / duration
assert.equal(model(live({ playing: false })).state, '已暫停');
assert.equal(model(live({ buffering: true, playing: false })).state, '緩衝中');
m = model(live({ pos: null, ahead_s: null, buffering: true, playing: false }));
assert.equal(m.timeText, '載入中…');
assert.equal(m.playedPct, null);
m = model(live({ duration: null }));
assert.equal(m.timeText, '2:06');
assert.equal(m.playedPct, null);
assert.equal(model(live({ title: '' })).title, '串流播放');
assert.equal(model(live({ pos: 99999 })).playedPct, 100, 'a position past the end is clamped');

// -- spans
assert.equal(context.fmtSpan(0), '0 秒');
assert.equal(context.fmtSpan(59.6), '1 分鐘');
assert.equal(context.fmtSpan(600), '10 分鐘');
assert.equal(context.fmtSpan(655), '10 分鐘');
assert.equal(context.fmtSpan(7380), '2 小時 3 分');

// -- DOM: hidden without a stream, filled with one
context.renderStreaming(null);
assert.equal(dom.$('#streamCard').hidden, true);
context.renderStreaming(live({ subtitle: { status: 'queued', lead_s: 720, view: view('running', { covered_s: 845.5,
  running: { executor: 'cloud', slice: 2, stage: 'asr' } }) } }));
assert.equal(dom.$('#streamCard').hidden, false);
assert.equal(dom.$('#streamTitle').textContent, '示範影片');
assert.equal(dom.$('#streamState').textContent, '播放中');
assert.equal(dom.$('#streamTime').textContent, '0:02:06 / 1:00:00');
assert.equal(dom.$('#streamRestore').textContent, '修復領先 45 秒・0.7x');
assert.equal(dom.$('#streamRestore').attrs['data-level'], 'warn');
assert.equal(dom.$('#streamSubRow').hidden, false);
assert.equal(dom.$('#streamSub').textContent, '字幕：已領先 12 分鐘');
assert.equal(dom.$('#streamSubNote').hidden, false);
assert.match(dom.$('#streamPlayed').style.width, /^3\.48/);
assert.equal(dom.$('#streamBar').attrs['aria-valuenow'], '3');
// without subtitles the subtitle row disappears again; after the stream ends the card goes
context.renderStreaming(live());
assert.equal(dom.$('#streamSubRow').hidden, true);
context.renderStreaming(null);
assert.equal(dom.$('#streamCard').hidden, true);

// -- wiring: ids of the render exist in index.html, render() calls it, the card is styled and separate from the list, mock has data
for (const id of ['streamCard', 'streamTitle', 'streamState', 'streamTime', 'streamBar', 'streamAhead', 'streamPlayed', 'streamRestore',
  'streamSubRow', 'streamSub', 'streamSubNote']) {
  assert.match(html, new RegExp(`id="${id}"`), `index.html has #${id}`);
}
assert.ok(html.indexOf('id="streamCard"') < html.indexOf('id="list"'), 'the card sits above the queue list');
assert.match(source, /renderStreaming\(state\.streaming\)/, 'render() draws the card');
assert.match(css, /\.stream-card\s*\{/);
assert.match(source, /function mockStreaming\(/);
assert.match(source, /st\.streaming = mockStreaming\(/, 'the preview backend sends a stream');
assert.doesNotMatch(source.slice(source.indexOf('/* ---- streaming card (T3.5)'), source.indexOf('/* ---- end streaming card ---- */')), /Oh Streamer/, 'no hard-coded product name');

// mock data: every kind builds a model
for (const kind of ['sub', 'nosub', 'behind', 'stuck']) {
  const ctx2 = { console, $: dom.$ };
  vm.createContext(ctx2);
  vm.runInContext(slice('const num =', '\nfunction basename('), ctx2);
  vm.runInContext(slice('const SUBTITLE_EDIT_STATES =', '\nfunction modelLabel('), ctx2);
  vm.runInContext(slice('/* ---- streaming card (T3.5)', '/* ---- end streaming card ---- */'), ctx2);
  vm.runInContext(slice('function mockSubtitleView(', 'function makeMock()'), ctx2);
  const st = ctx2.mockStreaming(kind, 3);
  const mm = ctx2.streamingCardModel(st, NOW);
  assert.ok(mm && mm.restore.startsWith('修復領先'), kind);
  assert.equal(mm.subtitle, kind !== 'nosub', kind);
}
assert.equal(context.streamingCardModel(null), null);

console.log('ok test_queue_streaming_card_ui');
