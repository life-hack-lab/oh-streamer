// Run with node tests/test_queue_stream_subtitle_gpu_notice_ui.js (no browser dependencies).
// T4.3: the intake panel tells how local subtitles change the stream verdict (stream.subtitle_gpu from estimate).
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const source = fs.readFileSync(path.join(__dirname, '../src/oh_streamer/resources/queue_ui/app.js'), 'utf8');
const start = source.indexOf('const SUBTITLE_GPU_VRAM_NOTE =');
const end = source.indexOf('\nconst hasLow =', start);
assert(start >= 0 && end > start, 'subtitle gpu notice helpers are present');
const context = { ip: null, estOf: () => null };
vm.createContext(context);
vm.runInContext(`${source.slice(start, end)}\nthis.notice = subtitleGpuNotice; this.text = subtitleGpuNoticeText; this.of = subtitleGpuOf;`, context);

const g = (over) => Object.assign({ verdict: 'queue', ratio: 0.7, ratio_heavy: 0.5, prebuffer_s: null, vram_ok: true,
  suggest: [{ kind: 'queue' }] }, over);

function testSlowComputerSuggestsQueue() {
  assert.equal(context.text(g()), '開本機字幕後，這台電腦邊看邊修復約 0.7x 即時，建議：加入佇列');
}
function testAllSuggestions() {
  const t = context.text(g({ verdict: 'prebuffer', suggest: [{ kind: 'queue' }, { kind: 'prebuffer', seconds: 120.4 }, { kind: 'cloud' }] }));
  assert.equal(t, '開本機字幕後，這台電腦邊看邊修復約 0.7x 即時，建議：加入佇列／先緩衝約 120 秒／改用雲端字幕');
  assert.deepEqual(context.notice(g({ suggest: [{ kind: 'queue' }, { kind: 'cloud' }] })).actions.map((a) => a.kind), ['queue', 'cloud']);
}
function testNotEnoughVram() {
  assert.equal(context.text(g({ vram_ok: false })), '顯示卡記憶體不足以同時修復與產生本機字幕，建議：加入佇列');
  assert.equal(context.text(g({ vram_ok: false, suggest: [] })), '顯示卡記憶體不足以同時修復與產生本機字幕');
}
function testLiveStaysQuiet() {
  assert.equal(context.text(g({ verdict: 'live', ratio: 1.6, suggest: [] })), '開本機字幕後，這台電腦邊看邊修復約 1.6x 即時，仍可即時播放');
}
function testMissingOrBroken() {
  for (const bad of [null, undefined, 5, {}, { vram_ok: true, ratio: null }]) assert.equal(context.text(bad), '');
  assert.equal(context.notice(g({ suggest: [{ kind: 'bogus' }, null, { kind: 'prebuffer', seconds: 0 }] })).actions.length, 0);
}
function testPanelWiring() {
  assert.match(source, /subtitleGpuNotice\(s\.subtitle_gpu\)/, 'rendered under the stream verdict');
  assert.match(source, /id="ipSubtitleGpuNote"/);
  assert.match(source, /\) && !subtitleGpuOf\(it0\) \?/, 'the stale local-subtitle note gives way to the verdict');
  assert.match(source, /o\.subtitles = !!\(subtitleRuntimeAvailable\(env\) && pick\('subtitles', 'queue'\)\)/, 'estimate overrides carry the toggle');
  assert.match(source, /t\.dataset\.k === 'subtitles'\) cur\.subtitles = t\.checked;\r?\n[\s\S]*?settingsChanged\(\)/, 'toggling re-estimates');
}

testSlowComputerSuggestsQueue();
testAllSuggestions();
testNotEnoughVram();
testLiveStaysQuiet();
testMissingOrBroken();
testPanelWiring();
console.log('stream subtitle gpu notice ui tests ok');
