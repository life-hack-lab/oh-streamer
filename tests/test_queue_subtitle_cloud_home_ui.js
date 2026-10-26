// Run with node tests/test_queue_subtitle_cloud_home_ui.js (no browser or provider calls).
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const root = path.join(__dirname, '../src/oh_streamer');
const uiRoot = path.join(root, 'resources/queue_ui');
const source = fs.readFileSync(path.join(uiRoot, 'app.js'), 'utf8');
const html = fs.readFileSync(path.join(uiRoot, 'index.html'), 'utf8');

function plain(value) {
  return value == null ? value : JSON.parse(JSON.stringify(value));
}

function profile() {
  const makeMode = (asr, translation, draft = null) => ({
    asr, translation, draft, processing: 'smart_horizon',
    whisper_policy: 'auto', whisper_models: ['whisper-large-v3-turbo', 'whisper-large-v3'],
    fallback: { asr: [], translation: [], max_attempts: 4, retries: 1, cooldown_seconds: 60 },
    legacy_primary: { marker: `${asr || 'unset'}:${translation || 'unset'}` },
  });
  return {
    schema_version: 1,
    plan_version: 'cloud-first-1',
    active_mode: 'cloud',
    profiles: {
      cloud: makeMode('groq-asr-old', 'groq-translation-old'),
      local: makeMode('local-asr-kept', 'local-translation-kept'),
      custom: makeMode('custom-asr-kept', 'custom-translation-kept', 'custom-draft-kept'),
    },
    migration: { source: 'test' },
  };
}

function connection(id, family, validation, azureRegion = '') {
  return { id, api_family: family, auth_source: 'dpapi', azure_region: azureRegion, validation };
}

function makeHarness({ cloudEnabled = true, initialProfile = profile() } = {}) {
  const nodes = new Map();
  const apiCalls = [];
  const apiQueues = new Map();
  const loadModelCalls = [];
  const settingsProfile = plain(initialProfile);

  function node(selector) {
    if (!nodes.has(selector)) {
      nodes.set(selector, {
        value: '', textContent: '', hidden: false, disabled: false, checked: false,
        type: 'text', dataset: {}, attributes: {}, listeners: {},
        addEventListener(event, callback) { this.listeners[event] = callback; },
        setAttribute(name, value) { this.attributes[name] = String(value); },
      });
    }
    return nodes.get(selector);
  }

  const current = {
    settings: { subtitle_profile: settingsProfile, subtitle_cloud_enabled: cloudEnabled },
  };
  const context = {
    current,
    pushSetting(key, value) {
      current.settings[key] = value;
      apiCalls.push(['set_settings', { [key]: value }]);
    },
    subtitleModels: [],
    hasApi: () => true,
    api: async (...args) => {
      apiCalls.push(plain(args));
      const queue = apiQueues.get(args[0]);
      if (queue && queue.length) return plain(queue.shift());
      if (args[0] === 'set_subtitle_profile') {
        return { ok: true, profile: plain(args[1]), runtime_contract: 'cloud-first-1' };
      }
      return undefined;
    },
    $: node,
    cloneConfig: plain,
    setText(element, value) { element.textContent = String(value == null ? '' : value); },
    setHidden(element, hidden) { element.hidden = !!hidden; },
    fillSelect(element, rows, selected) {
      element.options = plain(rows);
      element.value = rows.some((row) => row.value === selected) ? selected : (rows[0] && rows[0].value) || '';
    },
    subtitleModelStatus(model) { return model.status_label || model.status || '已登錄'; },
    loadSubtitleModels: async (...args) => { loadModelCalls.push(args); },
    renderSubtitleSettings() {},
  };

  vm.createContext(context);
  const homeStart = source.indexOf('const SUBTITLE_DUAL_VALUE =');
  const homeEnd = source.indexOf('\nfunction cloneConfig(', homeStart);
  assert(homeStart >= 0 && homeEnd > homeStart, 'S4 profile and cloud home functions exist');
  vm.runInContext(`${source.slice(homeStart, homeEnd)}\n` +
    'globalThis.__cloudHome = { renderSubtitleHome, saveSubtitleProfile, saveSubtitleService, ' +
    'get connections() { return subtitleConnections; }, set connections(value) { subtitleConnections = value; }, ' +
    'get connectionsLoaded() { return subtitleConnectionsLoaded; }, set connectionsLoaded(value) { subtitleConnectionsLoaded = value; }, ' +
    'get serviceMessages() { return subtitleServiceMessages; }, ' +
    'get profileBusy() { return subtitleProfileBusy; }, ' +
    'get saveFunction() { return saveSubtitleService; }, set saveFunction(value) { saveSubtitleService = value; } };', context);

  const eventStart = source.indexOf("$('#subtitleMode').addEventListener('change'");
  const eventEnd = source.indexOf("\n$('#subtitleLocalRecommend')", eventStart);
  assert(eventStart >= 0 && eventEnd > eventStart, 'S4 profile and service event handlers exist');
  vm.runInContext(source.slice(eventStart, eventEnd), context);

  return {
    context, current, nodes, apiCalls, apiQueues, loadModelCalls,
    node,
    queueApi(name, ...results) { apiQueues.set(name, results); },
    async submit(key) {
      const form = node('#subtitleGroqForm');
      assert.equal(typeof form.listeners.submit, 'function', `${key} form submit handler is wired`);
      // Capture the async task because the production submit listener does not return it.
      const home = context.__cloudHome;
      const original = home.saveFunction;
      let pending;
      home.saveFunction = (serviceKey) => { pending = original(serviceKey); };
      let prevented = false;
      form.listeners.submit({ preventDefault() { prevented = true; } });
      assert.equal(prevented, true, `${key} form prevents browser navigation`);
      await pending;
      home.saveFunction = original;
    },
    async changeMode(mode) {
      const select = node('#subtitleMode');
      select.value = mode;
      await select.listeners.change({ target: { value: mode } });
    },
  };
}

function successfulGroq(id, { partial = false } = {}) {
  const validation = {
    asr: { status: partial ? 'failed' : 'verified', remote_model: 'whisper-large-v3-turbo', ...(partial ? { error: 'sample rejected' } : {}) },
    translation: { status: 'verified', remote_model: 'qwen/qwen3.8-27b' },
  };
  const models = [
    ...(!partial ? [{ id: 'groq-asr-new', kind: 'asr', remote_model: 'whisper-large-v3-turbo' }] : []),
    { id: 'groq-translation-new', kind: 'translation', remote_model: 'qwen/qwen3.8-27b' },
  ];
  return { ok: !partial, connection: connection(id, 'groq', validation), models };
}

async function testRenderDoesNotValidateConnection() {
  const h = makeHarness();
  h.context.__cloudHome.renderSubtitleHome(h.current.settings);
  assert.deepEqual(h.apiCalls, [], 'opening/rendering the cloud home makes no provider or bridge calls');
  assert.equal(h.node('#subtitleGroqSave').disabled, false);
  assert.match(h.node('#subtitleProfileStatus').textContent, /目前流程：/);
  assert.match(html, /id="subtitleMode"/);
  assert.match(html, /id="subtitleGroqKey"/);
  assert.doesNotMatch(html, /id="subtitleGroqRemoteModel"|id="subtitleGroqModelName"/,
    'the first-run Groq form does not ask users to enter a model name');
}

async function testModeSwitchesSaveWithoutReplacingOtherProfiles() {
  const h = makeHarness();
  const before = plain(h.current.settings.subtitle_profile.profiles);
  h.context.__cloudHome.renderSubtitleHome(h.current.settings);

  await h.changeMode('local');
  await h.changeMode('custom');
  await h.changeMode('cloud');

  const saved = h.apiCalls.filter(([name]) => name === 'set_subtitle_profile').map(([, value]) => value);
  assert.deepEqual(saved.map((value) => value.active_mode), ['local', 'custom', 'cloud']);
  for (const value of saved) {
    assert.deepEqual(plain(value.profiles), before, 'mode selection preserves all three saved mode configurations');
  }
  assert.equal(h.current.settings.subtitle_profile.active_mode, 'cloud');
}

async function testGroqPartialRetryRotationAndStableConnectionId() {
  const h = makeHarness();
  const id = 'groq-connection-stable';
  h.node('#subtitleGroqKey').value = 'first-secret';
  h.queueApi('test_save_subtitle_connection', successfulGroq(id, { partial: true }));
  await h.submit('groq');

  const first = h.apiCalls.find(([name]) => name === 'test_save_subtitle_connection')[1];
  assert.deepEqual(first, { api_family: 'groq', api_key: 'first-secret', test_audio_path: '__builtin__' });
  assert.equal('models' in first, false, 'Groq model defaults are selected by the service, not typed by the user');
  assert.equal('remote_model' in first, false);
  assert.equal(h.node('#subtitleGroqKey').value, '', 'key field clears after partial validation');
  assert.equal(h.node('#subtitleGroqKey').type, 'password');
  assert.equal(h.current.settings.subtitle_profile.profiles.cloud.asr, null, 'failed ASR binding is not activated');
  assert.equal(h.current.settings.subtitle_profile.profiles.cloud.translation, 'groq-translation-new');
  assert.match(h.node('#subtitleGroqStatus').textContent, /部分能力尚未通過/);
  assert.equal(h.node('#subtitleGroqConnection').value, id, 'partial result retains its stable connection ID for retry');

  h.node('#subtitleGroqKey').value = 'rotated-secret';
  h.queueApi('test_save_subtitle_connection', successfulGroq(id));
  await h.submit('groq');

  const specs = h.apiCalls.filter(([name]) => name === 'test_save_subtitle_connection').map(([, spec]) => spec);
  assert.equal(specs.length, 2);
  assert.deepEqual(specs[1], { api_family: 'groq', id, api_key: 'rotated-secret', test_audio_path: '__builtin__' },
    'retry and key rotation update the existing connection without requiring model IDs');
  assert.equal(h.node('#subtitleGroqKey').value, '', 'rotated key also clears after validation');
  assert.equal(h.node('#subtitleGroqKey').type, 'password');
  assert.equal(h.current.settings.subtitle_profile.profiles.cloud.asr, 'groq-asr-new');
  assert.equal(h.current.settings.subtitle_profile.profiles.cloud.translation, 'groq-translation-new');
  assert.equal(h.context.__cloudHome.connections.length, 1);
  assert.equal(h.context.__cloudHome.connections[0].id, id);
  assert.doesNotMatch(h.node('#subtitleGroqStatus').textContent, /first-secret|rotated-secret/);
  assert(h.loadModelCalls.some(([force]) => force === true), 'saved bindings refresh the registered model list');
}

async function testCloudModeEnablesValidation() {
  const h = makeHarness({ cloudEnabled: false });
  h.context.__cloudHome.renderSubtitleHome(h.current.settings);
  assert.equal(h.node('#subtitleGroqSave').disabled, false, 'cloud mode enables connection validation');
  assert.equal(h.current.settings.subtitle_cloud_enabled, true);
  assert.equal(h.apiCalls.some(([name]) => name === 'test_save_subtitle_connection'), false,
    'enabling cloud mode alone does not contact a provider');
  h.node('#subtitleGroqKey').value = 'fake-key';
  h.queueApi('test_save_subtitle_connection', successfulGroq('groq-enabled'));
  await h.submit('groq');
  assert.equal(h.apiCalls.some(([name]) => name === 'test_save_subtitle_connection'), true);
}

async function testCloudModeReplacesSeparateEgressSwitch() {
  const h = makeHarness({ cloudEnabled: false });
  h.context.__cloudHome.renderSubtitleHome(h.current.settings);
  assert.doesNotMatch(html, /id="subtitleCloudEnabled(?:Row)?"/, 'no separate cloud permission switch');
  assert.match(h.node('#subtitleCloudNotice').textContent, /HTTPS.*音訊與字幕文字/);
  assert.doesNotMatch(h.node('#subtitleProfileStatus').textContent, /雲端外連尚未允許/);
  assert.equal(h.current.settings.subtitle_cloud_enabled, true);
  await h.changeMode('local');
  h.context.__cloudHome.renderSubtitleHome(h.current.settings);
  assert.equal(h.current.settings.subtitle_cloud_enabled, true, 'local mode preserves saved cloud connections and state');
  h.current.settings.subtitle_cloud_enabled = false;
  await h.changeMode('cloud');
  h.context.__cloudHome.renderSubtitleHome(h.current.settings);
  assert.equal(h.current.settings.subtitle_cloud_enabled, true, 'selecting cloud enables external access');
  assert.equal(h.node('#subtitleGroqSave').disabled, false);
  assert.doesNotMatch(html, /每月 200 萬字元免費|免費額度推薦/, 'no hard-coded free quota promises in the UI');
}

async function testProfileSaveFailureIsVisible() {
  const h = makeHarness();
  h.node('#subtitleGroqKey').value = 'temporary-secret';
  h.queueApi('test_save_subtitle_connection', successfulGroq('groq-connection-save-failure'));
  h.queueApi('set_subtitle_profile', { ok: false, error: 'profile write failed' });
  await h.submit('groq');

  assert.equal(h.current.settings.subtitle_profile.profiles.cloud.asr, 'groq-asr-old',
    'a failed profile save does not claim the new binding is persisted');
  assert.match(h.node('#subtitleGroqStatus').textContent, /字幕方案未能儲存/);
  assert.equal(h.node('#subtitleGroqKey').value, '', 'key is cleared even when the profile save fails');
}

async function testNoAzureSpecificUiAndDirectFlow() {
  const h = makeHarness();
  const p = h.current.settings.subtitle_profile;
  p.profiles.cloud.asr = 'groq-asr';
  p.profiles.cloud.translation = 'groq-translation';
  h.context.subtitleModels = [
    { id: 'groq-asr', connection_id: 'groq-connection', kind: 'asr', label: 'Groq ASR', status: 'verified' },
    { id: 'groq-translation', connection_id: 'groq-connection', kind: 'translation', label: 'Groq 翻譯', status: 'verified' },
  ];
  h.context.__cloudHome.renderSubtitleHome(h.current.settings);
  const status = h.node('#subtitleProfileStatus').textContent;
  assert.match(status, /Groq ASR.*→.*Groq 翻譯/, 'flow is ASR -> translation');
  assert.doesNotMatch(status, /初稿|精修|翻譯一次/, 'no draft/refine wording in flow');
}

async function main() {
  await testRenderDoesNotValidateConnection();
  await testModeSwitchesSaveWithoutReplacingOtherProfiles();
  await testGroqPartialRetryRotationAndStableConnectionId();
  await testCloudModeEnablesValidation();
  await testCloudModeReplacesSeparateEgressSwitch();
  await testProfileSaveFailureIsVisible();
  await testNoAzureSpecificUiAndDirectFlow();
  console.log('Queue subtitle cloud home UI: 7 scenarios passed');
}

main().catch((error) => { console.error(error); process.exitCode = 1; });
