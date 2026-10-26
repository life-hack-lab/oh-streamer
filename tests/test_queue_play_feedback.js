// Run with node streamer/tests/test_queue_play_feedback.js (no browser dependencies).
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const source = fs.readFileSync(path.join(__dirname, '../src/oh_streamer/resources/queue_ui/app.js'), 'utf8');
const start = source.indexOf('async function openPlayer(');
const end = source.indexOf('\nfunction playFrom(', start);

async function scenario(result, statuses = [], mock = false) {
  let now = 0;
  const calls = [];
  const context = {
    openingIds: new Set(), playErrors: new Map(), current: { items: [] }, isMock: mock, subtitlePlayPrompt: () => null,
    render() {}, refresh: async () => {},
    Date: { now: () => now },
    setTimeout(callback) { now += 10000; callback(); },
    api: async (name) => {
      calls.push(name);
      if (name === 'play') return result;
      return { ready: statuses.shift() || false };
    },
  };
  vm.createContext(context);
  vm.runInContext(source.slice(start, end), context);
  const request = context.openPlayer('item');
  assert(context.openingIds.has('item'), 'feedback appears synchronously');
  await context.openPlayer('item');
  assert.equal(calls.filter((x) => x === 'play').length, 1, 'repeat click is ignored');
  await request;
  assert.equal(context.openingIds.size, 0, 'opening state always clears');
  return { context, calls };
}

(async () => {
  let test = await scenario({ ok: true, launched: true }, [false, true]);
  assert.equal(test.context.playErrors.size, 0);
  assert.equal(test.calls.filter((x) => x === 'player_window_ready').length, 2);
  test = await scenario({ ok: false, error: 'missing source' });
  assert(test.context.playErrors.has('item'), 'failure exposes retry');
  test = await scenario({ ok: true }, []);
  assert(test.context.playErrors.has('item'), 'timeout exposes retry');
  test = await scenario({ ok: true, pending: true });
  assert.equal(test.context.playErrors.size, 0, 'pending source is not an opening failure');
  assert.deepEqual(test.calls, ['play']);
  test = await scenario({ ok: true }, [], true);
  assert.equal(test.context.playErrors.size, 0);
  console.log('Queue playback feedback: 5 scenarios passed');
})().catch((error) => { console.error(error); process.exitCode = 1; });
