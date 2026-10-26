const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');

const source = fs.readFileSync(path.join(__dirname, '../src/oh_streamer/resources/queue_ui/app.js'), 'utf8');

const helpersStart = source.indexOf('const num =');
const helpersEnd = source.indexOf('\nfunction basename(', helpersStart);

const progStart = source.indexOf('function progressInfo(');
const progEnd = source.indexOf('\nconst comboOpen =', progStart);

assert(helpersStart >= 0 && helpersEnd > helpersStart, 'helpers must be present');
assert(progStart >= 0 && progEnd > progStart, 'progressInfo must be present');

const context = {
  console,
  current: { settings: {} },
};
vm.createContext(context);
vm.runInContext(source.slice(helpersStart, helpersEnd), context);
vm.runInContext(source.slice(progStart, progEnd), context);

// Test 1: fmtTimeProgress formatting
assert.equal(context.fmtTimeProgress(120, 1800), '2:00 / 30:00');
assert.equal(context.fmtTimeProgress(1374, 4320), '0:22:54 / 1:12:00');
assert.equal(context.fmtTimeProgress(3661, 7200), '1:01:01 / 2:00:00');

// Test 2: pure downloading state with speed
const dlItem = {
  state: 'downloading',
  progress: {
    downloaded_bytes: 100 * 1024 * 1024,
    total_bytes: 500 * 1024 * 1024,
    speed: 5.5 * 1024 * 1024,
    eta: 72,
  },
};
const dlInfo = context.progressInfo(dlItem);
assert.ok(dlInfo, 'progressInfo should not be null for downloading');
assert.equal(dlInfo.rate, '5.50 MB/s');
assert.equal(dlInfo.eta, '剩餘 1:12');
assert.ok(dlInfo.detail.includes('100 MB / 500 MB'));

// Test 3: processing state with duration - frames converted to time
const procItemWithDur = {
  state: 'processing',
  progress: {
    frame: 1800,
    frames: 54000,
    fps: 40.0,
    eta: 1300,
    duration: 1800, // 30 minutes
  },
};
const procInfo1 = context.progressInfo(procItemWithDur);
assert.ok(procInfo1);
assert.equal(procInfo1.detail, '1:00 / 30:00');
assert.equal(procInfo1.detailTitle, '1,800 / 54,000 格');
assert.equal(procInfo1.rate, '40.0 fps');

// Test 4: processing state over 1 hour with source_fps
const procItemOverHour = {
  state: 'processing',
  duration: 4320, // 1 hr 12 min
  progress: {
    frame: 41230,
    frames: 129600,
    source_fps: 30.0,
    fps: 42.6,
  },
};
const procInfo2 = context.progressInfo(procItemOverHour);
assert.ok(procInfo2);
assert.equal(procInfo2.detail, '0:22:54 / 1:12:00');
assert.equal(procInfo2.detailTitle, '41,230 / 129,600 格');

// Test 5: overlapping state (downloading and processing simultaneously)
const overlappingItem = {
  state: 'processing',
  source_status: 'downloading',
  duration: 3600,
  progress: {
    frame: 900,
    frames: 108000,
    fps: 35.0,
  },
  download_progress: {
    downloaded_bytes: 612 * 1024 * 1024,
    total_bytes: 1900 * 1024 * 1024,
    speed: 7.8 * 1024 * 1024,
  },
};
const overlapInfo = context.progressInfo(overlappingItem);
assert.ok(overlapInfo);
assert.equal(overlapInfo.detail, '0:00:30 / 1:00:00');
assert.ok(overlapInfo.frag.includes('7.80 MB/s'), 'frag should contain download speed: ' + overlapInfo.frag);
assert.ok(overlapInfo.frag.includes('612 MB / 1.86 GB'), 'frag should contain downloaded bytes: ' + overlapInfo.frag);

// Test 6: fallback to frames when neither duration nor fps is known
const procItemNoDur = {
  state: 'processing',
  progress: {
    frame: 50,
    frames: 200,
    fps: 10.0,
  },
};
const procInfoFallback = context.progressInfo(procItemNoDur);
assert.ok(procInfoFallback);
assert.equal(procInfoFallback.detail, '50 / 200 格');

console.log('All queue progress UI tests passed!');
