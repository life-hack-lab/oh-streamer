// Run with node tests/test_queue_paste_to_add_box_ui.js (no browser dependencies).
// P6: Ctrl/Cmd+V outside editable fields and dialogs fills the add box and starts the parse flow.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const source = fs.readFileSync(path.join(__dirname, '../src/oh_streamer/resources/queue_ui/app.js'), 'utf8');
const start = source.indexOf('function pasteTarget(');
const end = source.indexOf("document.addEventListener('paste', pasteToAddBox);", start);
assert(start >= 0 && end > start, 'paste handler is present');
assert.match(source, /document.addEventListener\('paste', pasteToAddBox\)/);

function setup({ active = null, dialogOpen = false } = {}) {
  const calls = [];
  const context = {
    document: { activeElement: active, querySelector: () => (dialogOpen ? {} : null) },
    openText: (t) => calls.push(t),
  };
  vm.createContext(context);
  vm.runInContext(`${source.slice(start, end)}\nthis.paste = pasteToAddBox;`, context);
  return { context, calls };
}
const editable = { closest: (sel) => (/textarea/.test(sel) ? {} : null) };
function ev(text, target = null) {
  const e = { target, defaultPrevented: false, prevented: false, preventDefault() { this.prevented = true; } };
  e.clipboardData = { getData: (t) => (t === 'text/plain' ? text : '') };
  return e;
}

{ // no focus: intercepted
  const { context, calls } = setup();
  const e = ev('  https://example.com/v  ', { closest: () => null });
  context.paste(e);
  assert.deepEqual(calls, ['https://example.com/v']);
  assert.equal(e.prevented, true);
}
{ // focus in an input / textarea (event target or activeElement): left alone
  const { context, calls } = setup();
  const e = ev('abc', editable);
  context.paste(e);
  assert.deepEqual(calls, []); assert.equal(e.prevented, false);
  const s2 = setup({ active: editable });
  s2.context.paste(ev('abc', null));
  assert.deepEqual(s2.calls, []);
}
{ // dialog open: left alone
  const { context, calls } = setup({ dialogOpen: true });
  const e = ev('abc');
  context.paste(e);
  assert.deepEqual(calls, []); assert.equal(e.prevented, false);
}
{ // empty text (files only): nothing
  const { context, calls } = setup();
  const e = ev('   ');
  context.paste(e);
  assert.deepEqual(calls, []); assert.equal(e.prevented, false);
  const e2 = { target: null, preventDefault() { throw new Error('no'); } };
  context.paste(e2);
  assert.deepEqual(calls, []);
}
console.log('paste to add box ui tests ok');
