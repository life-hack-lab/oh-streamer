// Run with node tests/test_queue_stream_subtitle_notice_ui.js (no browser dependencies).
// T2.6: the intake panel warns when streaming with subtitles has to download a second copy of a single-quality source.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const source = fs.readFileSync(path.join(__dirname, '../src/oh_streamer/resources/queue_ui/app.js'), 'utf8');

const start = source.indexOf('const SINGLE_QUALITY_NOTE =');
const end = source.indexOf('\nconst hasLow =', start);
assert(start >= 0 && end > start, 'single-quality notice helper is present');
const heightsLine = source.split('\n').find((l) => l.startsWith('const heightsOf ='));
assert(heightsLine, 'heightsOf helper is present');
const context = {};
vm.createContext(context);
vm.runInContext(`${heightsLine}\n${source.slice(start, end)}\nthis.doubles = subtitleStreamDoubles; this.note = SINGLE_QUALITY_NOTE;`, context);

const q = (n) => Array.from({ length: n }, (_, i) => ({ height: 1080 - i * 360, fps: 30 }));

function testFallbackCountsQualities() {   // no subtitle_audio field yet (T2.3 not wired)
  assert.equal(context.doubles({ kind: 'url', heights: q(1) }), true, 'one quality shows the note');
  assert.equal(context.doubles({ kind: 'url', heights: [] }), true, 'no listed quality counts as single');
  assert.equal(context.doubles({ kind: 'url' }), true, 'missing heights counts as single');
  assert.equal(context.doubles({ kind: 'url', heights: q(2) }), false, 'two qualities: lowest one is used instead');
  assert.equal(context.doubles({ kind: 'url', heights: q(4) }), false);
}

function testLocalFilesNeverShow() {
  assert.equal(context.doubles({ kind: 'file', heights: q(1) }), false);
  assert.equal(context.doubles({ kind: 'file', heights: q(1), subtitle_audio: 'same' }), false);
  assert.equal(context.doubles(null), false);
}

function testSubtitleAudioFieldWins() {
  assert.equal(context.doubles({ kind: 'url', heights: q(1), subtitle_audio: 'same' }), true);
  assert.equal(context.doubles({ kind: 'url', heights: q(3), subtitle_audio: 'same' }), true, 'field overrides the quality count');
  assert.equal(context.doubles({ kind: 'url', heights: q(1), subtitle_audio: 'audio_only' }), false);
  assert.equal(context.doubles({ kind: 'url', heights: q(1), subtitle_audio: 'lowest' }), false);
  assert.equal(context.doubles({ kind: 'url', heights: q(1), subtitle_audio: 'file' }), false);
  assert.equal(context.doubles({ kind: 'url', heights: q(1), subtitle_audio: null }), true, 'null falls back to the count');
}

function testPanelWiring() {
  assert.equal(context.note, '此來源只有一種畫質，串流時產生字幕需要另外下載一份（流量約加倍）');
  assert.match(source, /single && subtitleAvailable && subtitleEnabled && subtitleStreamDoubles\(it0\)/, 'shown only for a single entry with subtitles on');
}

testFallbackCountsQualities();
testLocalFilesNeverShow();
testSubtitleAudioFieldWins();
testPanelWiring();
console.log('stream subtitle notice ui tests ok');
