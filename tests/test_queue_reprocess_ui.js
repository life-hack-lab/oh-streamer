// Run with node tests/test_queue_reprocess_ui.js (no browser dependencies).
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const source = fs.readFileSync(path.join(__dirname, '../src/oh_streamer/resources/queue_ui/app.js'), 'utf8');
const qualityStart = source.indexOf('function normalizeQuality(');
const normalizeEnd = source.indexOf('\nfunction renderReprocessQuality(', qualityStart);
const syncStart = source.indexOf('function syncReprocessQualityNotice(');
const qualityEnd = source.indexOf('\nfunction syncReprocessCompare()', syncStart);
const dialogStart = source.indexOf('function fillReprocessDialog(');
const dialogEnd = source.indexOf('\nfunction restoreReprocessFocus(', dialogStart);
assert(qualityStart >= 0 && normalizeEnd > qualityStart && qualityEnd > syncStart, 'quality and source helpers are present');
assert(dialogStart >= 0 && dialogEnd > dialogStart, 'dialog preload helper is present');

const nodes = Object.fromEntries([
  'reprocessQualityNotice', 'reprocess_quality', 'reprocessQualityLocalHint',
  'reprocess_redownload', 'reprocessSource', 'reprocessLossy', 'reprocess_save_compare',
  'reprocess_compare_settings', 'reprocess_compare_quality_row', 'reprocessSourcePathRow',
  'reprocessRedownloadRow',
  'reprocess_source_path', 'reprocess_stop_current', 'reprocessStopRow',
  'reprocessStateNote', 'reprocessError', 'reprocessSubmit',
].map((id) => [id, {
  id, hidden: false, disabled: false, checked: false, required: false,
  value: '', textContent: '', placeholder: '',
} ]));

const context = {
  nodes,
  current: { settings: {} },
  MODEL_KEYS: [],
  STATE_LABEL: { waiting: '等待中', processing: '處理中' },
  reprocessDialogState: null,
  $: (selector) => nodes[selector.slice(1)],
  setHidden: (node, hidden) => { node.hidden = !!hidden; },
  setText: (node, value) => { node.textContent = String(value); },
  setCheckedValue() {},
  populateReprocessModel() {},
  renderReprocessQuality() { nodes.reprocess_quality.value = 'keep'; },
  syncReprocessCompare() {},
  qualityLabel: () => '',
};
vm.createContext(context);
vm.runInContext(source.slice(qualityStart, normalizeEnd), context);
vm.runInContext(source.slice(syncStart, qualityEnd), context);
vm.runInContext(source.slice(dialogStart, dialogEnd), context);

function resetQuality({ isUrl = true, mode = 'cache', quality = 720, lossy = true } = {}) {
  for (const node of Object.values(nodes)) Object.assign(node, {
    hidden: false, disabled: false, checked: false, required: false,
    value: '', textContent: '', placeholder: '',
  });
  nodes.reprocess_quality.value = 'keep';
  context.reprocessDialogState = {
    isUrl,
    redownloadRequired: mode === 'download',
    redownloadManuallyChosen: false,
    qualityForcedRedownload: false,
    data: { quality, source: { mode, message: '重用既有來源。', lossy } },
    item: { quality },
  };
}

function testQualityRedownloadBehavior() {
  resetQuality();
  nodes.reprocess_quality.value = 'height:1080';
  context.syncReprocessQualityNotice();
  assert.equal(nodes.reprocess_redownload.checked, true, 'quality changes automatically request a URL redownload');
  assert.equal(nodes.reprocess_redownload.disabled, true, 'quality changes require that redownload');
  assert.match(nodes.reprocessSource.textContent, /重新下載/, 'source text reflects the pending redownload');
  assert.equal(nodes.reprocessLossy.hidden, true, 'the old lossy-source warning is hidden during redownload');

  nodes.reprocess_quality.value = 'keep';
  context.syncReprocessQualityNotice();
  assert.equal(nodes.reprocess_redownload.checked, false, 'reverting quality clears the automatic redownload');
  assert.equal(nodes.reprocess_redownload.disabled, false, 'reverting quality enables the redownload checkbox');
  assert.equal(nodes.reprocessSource.textContent, '重用既有來源。', 'reverting quality restores the source message');
  assert.equal(nodes.reprocessLossy.hidden, false, 'reverting quality restores the lossy-source warning');
}

function testManualAndMandatoryRedownload() {
  resetQuality();
  nodes.reprocess_redownload.checked = true;
  context.reprocessDialogState.redownloadManuallyChosen = true;
  nodes.reprocess_quality.value = 'height:1080';
  context.syncReprocessQualityNotice();
  nodes.reprocess_quality.value = 'keep';
  context.syncReprocessQualityNotice();
  assert.equal(nodes.reprocess_redownload.checked, true, 'manual redownload choice survives a quality revert');

  resetQuality({ mode: 'download', lossy: false });
  context.syncReprocessQualityNotice();
  assert.equal(nodes.reprocess_redownload.checked, true, 'download-mode source requires redownloading');
  assert.equal(nodes.reprocess_redownload.disabled, true, 'required redownload cannot be unchecked');
  assert.match(nodes.reprocessSource.textContent, /重新下載/, 'required redownload is reflected in the source message');
}

function testLocalQualityAndInPlaceDialog() {
  resetQuality({ isUrl: false, mode: 'file' });
  context.syncReprocessQualityNotice();
  assert.equal(nodes.reprocess_quality.disabled, true, 'local-file quality selection is disabled');
  assert.equal(nodes.reprocessQualityLocalHint.hidden, false, 'local-file quality limitation is explained');

  context.fillReprocessDialog({
    ok: true,
    settings: {},
    quality: 720,
    source: { mode: 'file', message: '本機原始檔案可用。', lossy: false },
    state: 'waiting',
    target: 'C:\\videos\\clip.mp4',
    kind: 'file',
    edit_in_place: true,
  }, { id: 'item', kind: 'file', target: 'C:\\videos\\clip.mp4', state: 'waiting', settings: {} });
  assert.equal(nodes.reprocessStopRow.hidden, true, 'an in-place edit does not offer stop-current');
  assert.equal(nodes.reprocessSubmit.textContent, '更新設定', 'an in-place edit labels the submit action as an update');
  assert.match(nodes.reprocessStateNote.textContent, /直接更新設定並保留排序/, 'waiting note explains in-place update behavior');

  context.fillReprocessDialog({
    ok: true,
    settings: {},
    quality: 720,
    source: { mode: 'cache', message: '使用既有來源。', lossy: false },
    state: 'processing',
    target: 'https://example.test/clip',
    kind: 'url',
    edit_in_place: false,
  }, { id: 'active', kind: 'url', target: 'https://example.test/clip', state: 'processing', settings: {} });
  assert.equal(nodes.reprocessStopRow.hidden, false, 'processing item offers stop-current');
  assert.equal(nodes.reprocessSubmit.textContent, '開始重新處理', 'a new version uses the reprocess action label');
  assert.match(nodes.reprocessStateNote.textContent, /建立重新處理版本/, 'new-version note explains the behavior');
}

testQualityRedownloadBehavior();
testManualAndMandatoryRedownload();
testLocalQualityAndInPlaceDialog();
console.log('Queue reprocess UI: 3 scenarios passed');
