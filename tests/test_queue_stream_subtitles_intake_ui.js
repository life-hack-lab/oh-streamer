// Run with node tests/test_queue_stream_subtitles_intake_ui.js (no browser dependencies).
// T3.4: the intake panel's 串流播放 sends the subtitle switch like 加入佇列; the extra comparison tracks stay queue-only.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const source = fs.readFileSync(path.join(__dirname, '../src/oh_streamer/resources/queue_ui/app.js'), 'utf8');

const start = source.indexOf('async function ipStream()');
const end = source.indexOf('\nasync function ipQueue()', start);
assert(start >= 0 && end > start, 'ipStream is present');
const fnSource = source.slice(start, end);

async function runStream(overrides, apiResult) {
  const calls = [], statuses = [];
  const p = { phase: 'ready', streamBusy: false, sentStream: false };
  const context = {
    ip: p,
    isSingle: () => true,
    okEntries: () => [{ info: { token: 'tok1' } }],
    renderReady: () => {},
    setEntryStatus: (text, kind) => statuses.push([text, kind]),
    sendOverrides: () => ({ ...overrides }),
    api: async (name, ...args) => { calls.push([name, ...args]); return apiResult || { ok: true, message: 'opened' }; },
  };
  vm.createContext(context);
  vm.runInContext(`${fnSource}\nthis.ipStream = ipStream;`, context);
  await context.ipStream();
  return { calls, statuses, p };
}

async function testSubtitlesAreSent() {
  const on = await runStream({ height: 720, subtitles: true });
  assert.equal(on.calls.length, 1);
  assert.deepEqual(on.calls[0].slice(0, 2), ['stream_resolved', 'tok1']);
  assert.equal(on.calls[0][2].subtitles, true, 'subtitles flag reaches stream_resolved');
  assert.equal(on.calls[0][2].height, 720);
  const off = await runStream({ subtitles: false });
  assert.equal(off.calls[0][2].subtitles, false, 'an explicit off is sent too');
}

async function testComparisonIsNeverSent() {
  const r = await runStream({ subtitles: true, subtitle_comparison_enabled: true });
  assert.equal('subtitle_comparison_enabled' in r.calls[0][2], false, 'streaming has no extra subtitle tracks');
  assert.equal(r.calls[0][2].subtitles, true);
}

async function testSubtitleErrorIsShown() {
  const r = await runStream({ subtitles: true }, { ok: false, error: '字幕設定尚未就緒' });
  assert.deepEqual(r.statuses[r.statuses.length - 1], ['字幕設定尚未就緒', 'error'], 'backend refusal reaches the status line');
  assert.equal(r.p.sentStream, false);
}

function testPanelWiring() {
  assert.doesNotMatch(fnSource, /delete overrides\.subtitles\b/, 'subtitles are no longer stripped from the stream call');
  assert.match(fnSource, /delete overrides\.subtitle_comparison_enabled/, 'comparison stays out of the stream call');
  assert.match(source, /single \? '串流與加入佇列都產生字幕' : '加入佇列時產生字幕'/, 'the switch label covers streaming for a single entry');
  assert.match(source, /\$\{single \? '<span class="tagonly">僅佇列<\/span>' : ''\}<span class="note">\$\{esc\(subtitleComparisonSummary/, 'comparison row is tagged queue-only');
}

// T3.1-A (D5): a single entry whose saved main subtitle model is local gets a short note that streaming waits for GPU sharing.
function loadLocalProbe(models) {
  const s = source.indexOf('function subtitlePrimaryIsLocal(');
  const e = source.indexOf('\nconst hasLow', s);
  assert(s >= 0 && e > s, 'subtitlePrimaryIsLocal is present');
  const context = { subtitleModels: models || [], SUBTITLE_DUAL_VALUE: '__dual__',
    isCloudModel: (m) => !!m && m.provider !== 'local' && !/^https?:\/\/(localhost|127\.0\.0\.1)/.test(m.base_url || '') };
  vm.createContext(context);
  vm.runInContext(`${source.slice(s, e)}\nthis.probe = subtitlePrimaryIsLocal;`, context);
  return context.probe;
}

function profileOf(mode, active) {
  return { subtitle_profile: { active_mode: mode, profiles: { cloud: { asr: null, translation: null }, local: { asr: 'w', translation: 't' }, custom: active || {} } } };
}

function testLocalPrimaryNote() {
  const models = [{ id: 'groq-asr', provider: 'openai-compatible', base_url: 'https://api.example.org/v1' },
    { id: 'loc-asr', provider: 'local' }, { id: 'lm-mt', provider: 'openai-compatible', base_url: 'http://127.0.0.1:1234/v1' }];
  const probe = loadLocalProbe(models);
  assert.equal(probe(profileOf('cloud')), false, 'cloud mode');
  assert.equal(probe(profileOf('local')), true, 'local mode');
  assert.equal(probe(profileOf('custom', { asr: 'groq-asr', translation: 'groq-asr' })), false, 'custom, all cloud');
  assert.equal(probe(profileOf('custom', { asr: 'loc-asr', translation: 'groq-asr' })), true, 'custom, local ASR');
  assert.equal(probe(profileOf('custom', { asr: 'groq-asr', translation: 'lm-mt' })), true, 'custom, loopback server counts as local');
  assert.equal(probe(profileOf('custom', { asr: '__dual__', translation: null })), true, 'custom, dual-model ASR is local');
  assert.equal(probe(profileOf('custom', { asr: 'unknown', translation: null })), false, 'unknown id: no warning');
  assert.equal(probe({}), false, 'no profile');
  assert.equal(probe(null), false, 'no settings');
  assert.match(source, /const STREAM_LOCAL_SUBTITLE_NOTE = '主字幕模型為本機時，串流播放中字幕會和修復輪流使用 GPU，影片或字幕可能跟不上'/, 'note text');
  assert.match(source, /single && subtitleAvailable && subtitleEnabled && subtitlePrimaryIsLocal\(current && current\.settings\) && !subtitleGpuOf\(it0\) \? `<span class="note traffic-note" role="note" id="ipStreamLocalNote">\$\{esc\(STREAM_LOCAL_SUBTITLE_NOTE\)\}/,
    'shown on the 產生字幕 row only for a single entry with subtitles on');
  assert.doesNotMatch(source, /Oh Streamer/, 'no hard-coded product name');
}

(async () => {
  testLocalPrimaryNote();
  await testSubtitlesAreSent();
  await testComparisonIsNeverSent();
  await testSubtitleErrorIsShown();
  testPanelWiring();
  console.log('stream subtitle intake ui tests ok');
})().catch((e) => { console.error(e); process.exit(1); });
