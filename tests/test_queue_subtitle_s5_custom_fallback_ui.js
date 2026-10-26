// Run with node tests/test_queue_subtitle_s5_custom_fallback_ui.js
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const root = path.join(__dirname, '../src/oh_streamer');
const uiRoot = path.join(root, 'resources/queue_ui');
const source = fs.readFileSync(path.join(uiRoot, 'app.js'), 'utf8');

function plain(value) {
  return value == null ? value : JSON.parse(JSON.stringify(value));
}

function makeProfile(activeMode = 'custom') {
  const makeMode = (asr, translation, draft = null) => ({
    asr, translation, draft, processing: 'smart_horizon',
    whisper_policy: 'auto', whisper_models: ['whisper-large-v3-turbo', 'whisper-large-v3'],
    fallback: { asr: [], translation: [], max_attempts: 4, retries: 1, cooldown_seconds: 60 },
    legacy_primary: { marker: `${asr || 'unset'}:${translation || 'unset'}` },
  });
  return {
    schema_version: 1,
    plan_version: 'cloud-first-1',
    active_mode: activeMode,
    profiles: {
      cloud: makeMode('groq-whisper-1', 'groq-qwen-1'),
      local: makeMode('whisper-ja-1.5B', 'sakura-galtransl-7b-v3.7'),
      custom: makeMode('whisper-ja-1.5B', 'sakura-galtransl-7b-v3.7', null),
    },
    migration: { source: 'test' },
  };
}

function makeHarness({ cloudEnabled = true, initialProfile = makeProfile() } = {}) {
  const nodes = new Map();
  const apiCalls = [];
  const pushed = [];
  const apiQueues = new Map();

  function node(selector) {
    if (!nodes.has(selector)) {
      const e = {
        id: selector.replace(/^[#.]/, ''),
        value: '', textContent: '', hidden: false, disabled: false, checked: false,
        type: 'text', dataset: {}, attributes: {}, listeners: {}, children: [],
        addEventListener(event, callback) { this.listeners[event] = callback; },
        setAttribute(name, value) { this.attributes[name] = String(value); },
        replaceChildren(...newChildren) { this.children = newChildren; },
        appendChild(child) { this.children.push(child); return child; },
        classList: { toggle() {}, add() {}, remove() {} },
        get options() { return this.children.filter(c => c.tagName === 'OPTION'); },
      };
      nodes.set(selector, e);
    }
    return nodes.get(selector);
  }

  const current = {
    settings: {
      subtitle_profile: plain(initialProfile),
      subtitle_cloud_enabled: cloudEnabled,
      subtitle_primary: { asr: { mode: 'single', model: 'whisper-ja-1.5B' }, translation: { profile: 'sakura-galtransl-7b-v3.7' } },
    },
  };

  const sampleModels = [
    { id: 'whisper-ja-1.5B', label: 'Whisper JA 1.5B', kind: 'asr', provider: 'local', status: 'ready', builtin: true },
    { id: 'kotoba-whisper-v2.0', label: 'Kotoba Whisper', kind: 'asr', provider: 'local', status: 'missing', builtin: true },
    { id: 'groq-whisper-1', label: 'Groq Whisper Turbo', kind: 'asr', provider: 'openai-compatible', remote_model: 'whisper-large-v3-turbo', status: 'tested', connection_id: 'conn-groq' },
    { id: 'sakura-galtransl-7b-v3.7', label: 'Sakura 7B v3.7', kind: 'translation', provider: 'local', status: 'ready', builtin: true },
    { id: 'sakura-galtransl-14b-v3.8', label: 'Sakura 14B v3.8', kind: 'translation', provider: 'local', status: 'missing', builtin: true },
    { id: 'groq-qwen-1', label: 'Groq Qwen 3.8 27B', kind: 'translation', provider: 'openai-compatible', remote_model: 'qwen/qwen3.8-27b', status: 'tested', connection_id: 'conn-groq' },
    { id: 'azure-mt-1', label: 'Azure Translator Lines', kind: 'translation', provider: 'openai-compatible', remote_model: 'azure-translator-lines', status: 'unverified', connection_id: 'conn-azure' },
  ];

  const sampleConnections = [
    { id: 'conn-groq', api_family: 'groq', base_url: 'https://api.groq.com/openai/v1', auth_source: 'dpapi', validation: { asr: { status: 'verified' }, translation: { status: 'verified' } } },
    { id: 'conn-azure', api_family: 'azure-translator', base_url: 'https://api.cognitive.microsofttranslator.com', auth_source: 'dpapi', azure_region: 'eastasia', validation: { translation: { status: 'unverified' } } },
  ];

  const context = {
    current,
    subtitleModels: sampleModels,
    subtitleConnections: sampleConnections,
    hasApi: () => true,
    api: async (...args) => {
      apiCalls.push(plain(args));
      const queue = apiQueues.get(args[0]);
      if (queue && queue.length) return plain(queue.shift());
      if (args[0] === 'list_subtitle_connections') {
        return plain(sampleConnections);
      }
      if (args[0] === 'set_subtitle_profile') {
        current.settings.subtitle_profile = plain(args[1]);
        return { ok: true, profile: plain(args[1]), runtime_contract: 'cloud-first-1' };
      }
      if (args[0] === 'remove_subtitle_connection') {
        const cid = args[1];
        // Check if connection is referenced
        const prof = current.settings.subtitle_profile;
        const active = prof.profiles[prof.active_mode];
        const fbModels = [...active.fallback.asr, ...active.fallback.translation];
        const activeModels = [active.asr, active.translation, active.draft].filter(Boolean);
        const refModels = [...activeModels, ...fbModels];
        const connModels = sampleModels.filter(m => m.connection_id === cid).map(m => m.id);
        if (connModels.some(m => refModels.includes(m))) {
          return { ok: false, code: 'connection_in_use', error: '服務連線仍被字幕設定引用' };
        }
        return { ok: true };
      }
      return { ok: true };
    },
    $: node,
    cloneConfig: plain,
    fresh: () => true,
    setText(element, value) { element.textContent = String(value == null ? '' : value); },
    setHidden(element, hidden) { element.hidden = !!hidden; },
    el(tag, className, text) {
      const e = {
        tagName: tag.toUpperCase(),
        className: className || '',
        _textContent: text || '',
        get textContent() {
          if (this._textContent) return this._textContent;
          return this.children.map(c => typeof c === 'string' ? c : (c.textContent || '')).join(' ');
        },
        set textContent(val) { this._textContent = String(val == null ? '' : val); },
        hidden: false,
        disabled: false,
        children: [],
        listeners: {},
        attributes: {},
        dataset: {},
        addEventListener(event, callback) { this.listeners[event] = callback; },
        setAttribute(name, value) { this.attributes[name] = String(value); },
        append(...items) { this.children.push(...items); },
        appendChild(child) { this.children.push(child); return child; },
        replaceChildren(...newChildren) { this.children = newChildren; },
      };
      return e;
    },
    subtitleModelStatus(model) { return model.status_label || model.status || '已登錄'; },
    subtitleRuntimeAvailable() { return true; },
    subtitleRuntimeReason() { return ''; },
    loadSubtitleModels: async () => {},
    loadSubtitleConnections: async () => {},
    toast: () => {},
    pushSetting: (k, v) => { pushed.push([k, v]); },
    confirmDlg: async () => true,
    lastEdit: {},
    subtitleModelsLoaded: true,
    subtitleModelLoadBusy: false,
    document: { activeElement: null, querySelectorAll: () => [] },
  };

  vm.createContext(context);

  // Load relevant functions from app.js
  const displayStart = source.indexOf('const SUBTITLE_ACTIVE_STATES =');
  const displayEnd = source.indexOf('\nfunction subtitlePlayPrompt(', displayStart);
  assert(displayStart >= 0 && displayEnd > displayStart);
  vm.runInContext(source.slice(displayStart, displayEnd), context);

  const comboStart = source.indexOf('const COMBO_STATUS =');
  const comboEnd = source.indexOf('\nfunction modelLabel(', comboStart);
  assert(comboStart >= 0 && comboEnd > comboStart);
  vm.runInContext(source.slice(comboStart, comboEnd), context);

  const helpersStart = source.indexOf('const SUBTITLE_DUAL_VALUE =');
  const helpersEnd = source.indexOf('\n/* ------------------------------------------------------------------ global keys */', helpersStart);
  assert(helpersStart >= 0 && helpersEnd > helpersStart);
  vm.runInContext(`${source.slice(helpersStart, helpersEnd)}\n` +
    'globalThis.__s5Hook = { setConnections(c) { subtitleConnections = c; subtitleConnectionsLoaded = true; }, getConnections() { return subtitleConnections; } };', context);
  context.__s5Hook.setConnections(sampleConnections);

  return {
    context, current, nodes, apiCalls, node, pushed,
    queueApi(name, ...results) {
      if (!apiQueues.has(name)) apiQueues.set(name, []);
      apiQueues.get(name).push(...results);
    },
  };
}

async function testCustomModeSelectionAndFlowSummary() {
  const h = makeHarness({ cloudEnabled: true });
  h.context.renderSubtitleSettings(h.current.settings);

  assert.ok(!h.nodes.has('#subtitleDraft') || !h.nodes.get('#subtitleDraft').options?.length, 'No draft dropdown is rendered');
  const transSelect = h.node('#subtitleTranslation');
  assert.ok(transSelect.options.some(o => o.value === 'azure-mt-1'), 'Azure MT is an ordinary translation choice');
  assert.ok(!transSelect.options.some(o => o.value === 'tiered'), 'No tiered choice');

  // Verify custom flow summary text
  const flowSummary = h.node('#subtitleCustomFlowSummary');
  assert.ok(flowSummary.textContent.includes('自訂資料流：'), 'Renders custom flow prefix');
  assert.ok(flowSummary.textContent.includes('Whisper JA 1.5B'), 'Includes ASR model');
  assert.ok(flowSummary.textContent.includes('Sakura 7B v3.7'), 'Includes translation model');

  assert.ok(!flowSummary.textContent.includes('初稿'), 'Flow summary has no draft stage');
  assert.ok(!flowSummary.textContent.includes('精修'), 'Flow summary has no refine stage');
}

async function testCustomModeCloudAutoEnables() {
  const h = makeHarness({ cloudEnabled: false });
  // Custom profile uses groq whisper (cloud): cloud egress is auto-enabled (commit 1040121)
  h.current.settings.subtitle_profile.profiles.custom.asr = 'groq-whisper-1';
  h.context.renderSubtitleSettings(h.current.settings);

  assert.strictEqual(h.current.settings.subtitle_cloud_enabled, true, 'cloud egress auto-enabled');
  assert.ok(h.pushed.some(([k, v]) => k === 'subtitle_cloud_enabled' && v === true), 'pushSetting called');
  const flowSummary = h.node('#subtitleCustomFlowSummary');
  assert.ok(!flowSummary.textContent.includes('雲端外連總開關已關閉'), 'old warning removed');
}

async function testFallbackListRenderingAndReordering() {
  const h = makeHarness();
  const profile = h.current.settings.subtitle_profile;
  const custom = profile.profiles.custom;
  custom.fallback.translation = ['groq-qwen-1', 'azure-mt-1'];

  h.context.renderSubtitleSettings(h.current.settings);

  const transList = h.node('#subtitleFallbackTranslationList');
  assert.equal(transList.children.length, 2, '2 translation fallback items rendered');

  // First candidate should have groq-qwen-1
  assert.ok(transList.children[0].textContent.includes('Groq Qwen 3.8 27B'));
  // Second candidate should have azure-mt-1 and its unverified status
  assert.ok(transList.children[1].textContent.includes('Azure Translator Lines'));
  assert.ok(transList.children[1].textContent.includes('待驗證') || transList.children[1].textContent.includes('unverified'));

  // Test reordering: move azure-mt-1 up (click up on second item)
  const upButton = transList.children[1].querySelector ? transList.children[1].querySelector('[aria-label*="上移"]')
    : transList.children[1].children[1].children[0];
  await upButton.listeners.click();

  assert.deepEqual(h.current.settings.subtitle_profile.profiles.custom.fallback.translation, ['azure-mt-1', 'groq-qwen-1'], 'Items reordered in fallback list');

  // Test remove: remove azure-mt-1
  const removeButton = transList.children[0].children[1].children[2];
  await removeButton.listeners.click();
  assert.deepEqual(h.current.settings.subtitle_profile.profiles.custom.fallback.translation, ['groq-qwen-1'], 'Item removed from fallback list');
}

async function testLocalModeAddingCloudCandidateSwitchesToCustom() {
  const p = makeProfile('local');
  const h = makeHarness({ initialProfile: p });
  h.context.renderSubtitleSettings(h.current.settings);

  assert.equal(h.current.settings.subtitle_profile.active_mode, 'local', 'Starts in local mode');

  // Select a cloud candidate for translation fallback
  const transSelect = h.node('#subtitleFallbackTranslationSelect');
  transSelect.value = 'groq-qwen-1'; // cloud model

  const addBtn = h.node('#subtitleFallbackTranslationAdd');
  await addBtn.listeners.click();

  // Mode should have switched to custom
  assert.equal(h.current.settings.subtitle_profile.active_mode, 'custom', 'Switched to custom mode');
  assert.ok(h.current.settings.subtitle_profile.profiles.custom.fallback.translation.includes('groq-qwen-1'), 'Cloud model added to custom fallback');
}

async function testConnectionManagementAndDeletionPrevention() {
  const h = makeHarness();
  // Set custom mode referencing conn-groq via groq-qwen-1
  h.current.settings.subtitle_profile.profiles.custom.translation = 'groq-qwen-1';

  h.context.setSubtitleModelKind('connections');

  const list = h.node('#subtitleConnectionList');
  assert.equal(list.children.length, 2, 'Rendered 2 connection cards');

  // Try to remove conn-groq (which is referenced)
  const groqCard = list.children[0];
  const removeBtn = groqCard.children[2].children[1]; // Remove button
  const errorMsg = groqCard.children[4];

  await removeBtn.listeners.click();
  assert.ok(errorMsg.textContent.includes('無法移除：服務連線仍被字幕設定引用'), 'Displays error preventing removal of referenced connection');
  assert.equal(errorMsg.hidden, false, 'Error message is visible');
}

async function testWhisperPolicyUpdate() {
  const h = makeHarness();
  h.context.renderSubtitleSettings(h.current.settings);

  const policySel = h.node('#subtitleWhisperPolicy');
  assert.equal(policySel.value, 'auto');

  policySel.value = 'fixed';
  await policySel.listeners.change({ target: { value: 'fixed' } });

  const prof = h.current.settings.subtitle_profile.profiles.cloud;
  assert.equal(prof.whisper_policy, 'fixed');
  assert.deepEqual(prof.whisper_models, ['whisper-large-v3-turbo']);
}

async function testSubtitleProgressPhasesAndFallbackEvents() {
  const h = makeHarness();
  // M7: the label is derived from subtitle.view only (old phase / fallback event fields are gone).
  const base = { v: 2, state: 'queued', progress: 0, covered_s: 0, slices_done: 0, slices_total: 4, running: null, wait: null, stuck: null,
    failures_elsewhere: 0, partial_stuck: false, retryable: false };
  const show = (view, now) => h.context.subtitleDisplay({ status: 'queued', view: { ...base, ...view } }, now);

  // 1. Translation running shows the candidate in use, with a real percent
  const running = show({ state: 'running', progress: 0.45, running: { executor: 'cloud', slice: 0, stage: 'mt', candidate_name: 'sakura-7b' } });
  assert.equal(running.label, '字幕：翻譯中（雲端，第 1 片）');
  assert.equal(running.percent, 45);
  assert.equal(running.detail, 'sakura-7b');

  // 2. Quota cooldown
  const waitDisp = show({ state: 'waiting_quota', wait: { kind: 'quota', slice: 0, stage: 'mt', retry_at: 1012.3 } }, 1000);
  assert.equal(waitDisp.label, '字幕：等待額度中', 'Displays quota wait');
  assert.equal(waitDisp.detail, '約 20 秒後重試', 'Rounded retry time lives in the detail');
  assert.equal(waitDisp.indeterminate, false, 'Never slides during quota wait');
  assert.equal(waitDisp.showPercent, false, 'Does not show fake countdown ETA');

  // 3. Handed to a backup executor
  const handoff = show({ state: 'waiting_executor', wait: { kind: 'executor', slice: 0, stage: 'asr', executor: 'local', reclaim_at: null } });
  assert.ok(handoff.label.includes('已交給本地 GPU 處理，等待開始'), 'Shows the hand-off to the local backup');

  // 4. Every candidate failed
  const stuck = show({ state: 'stuck', retryable: true, stuck: { slice: 1, stage: 'mt', kind: 'rate_limit', exhausted: true,
    candidates: [{ id: 'a', name: '主翻譯', kind: 'rate_limit' }, { id: 'b', name: '備援翻譯', kind: 'bad_gateway' }] } });
  assert.ok(stuck.label.includes('翻譯卡住：主翻譯（被限流）、備援翻譯'), 'Lists every candidate and why it failed');
  assert.equal(stuck.retryable, true);

  // 6. Combos comparison track does not pollute main subtitle
  const comboSummary = h.context.subtitleComboSummary({
    combos: { total: 2, done: 1, failed: 1, items: [{ id: 'c1', status: 'done' }, { id: 'c2', status: 'failed', error: 'out_of_mem' }] }
  });
  assert.equal(comboSummary.failed, 1, 'Combo tracks failure recorded in combos');
  const mainDisp = show({ state: 'running', progress: 0.5, running: { executor: 'local', slice: 0, stage: 'asr' } });
  assert.equal(mainDisp.status, 'running', 'Main subtitle remains active despite combo failure');
  assert.equal(mainDisp.error, '', 'Main subtitle error is empty');
}

async function testDualModelAsrFallbackSelectionAndRendering() {
  const h = makeHarness({ initialProfile: makeProfile('cloud') });
  h.context.renderSubtitleSettings(h.current.settings);

  // Fallback ASR select should contain dual model option
  const asrSelect = h.node('#subtitleFallbackAsrSelect');
  assert.ok(asrSelect.options.some(o => o.value === '__dual__'), 'Fallback ASR dropdown includes dual model');
  const dualOption = asrSelect.options.find(o => o.value === '__dual__');
  assert.ok((dualOption.label || dualOption.textContent).includes('雙模型融合'), 'Dual model label contains 雙模型融合');
  assert.ok((dualOption.label || dualOption.textContent).includes('本機'), 'Dual model label indicates 本機');

  // Select dual model and add to fallback
  asrSelect.value = '__dual__';
  const addBtn = h.node('#subtitleFallbackAsrAdd');
  await addBtn.listeners.click();

  const profile = h.current.settings.subtitle_profile;
  assert.ok(profile.profiles.cloud.fallback.asr.includes('__dual__'), 'Added __dual__ to cloud fallback asr list');

  // Render should show dual model in fallback list
  const asrList = h.node('#subtitleFallbackAsrList');
  assert.equal(asrList.children.length, 1, '1 ASR fallback item rendered');
  assert.ok(asrList.children[0].textContent.includes('雙模型融合'), 'Rendered dual model label in fallback list');
  assert.ok(asrList.children[0].textContent.includes('本機模型可用'), 'Rendered local model ready status');

  // Asr select should now NOT contain __dual__ since it is already added
  assert.ok(!asrSelect.options.some(o => o.value === '__dual__'), 'Dual model removed from dropdown once added');

  // Remove dual model from fallback list
  const removeButton = asrList.children[0].children[1].children[2];
  await removeButton.listeners.click();
  assert.equal(h.current.settings.subtitle_profile.profiles.cloud.fallback.asr.length, 0, 'Dual model removed from fallback');
}

async function run() {
  await testCustomModeSelectionAndFlowSummary();
  await testCustomModeCloudAutoEnables();
  await testFallbackListRenderingAndReordering();
  await testLocalModeAddingCloudCandidateSwitchesToCustom();
  await testConnectionManagementAndDeletionPrevention();
  await testWhisperPolicyUpdate();
  await testSubtitleProgressPhasesAndFallbackEvents();
  await testDualModelAsrFallbackSelectionAndRendering();
  console.log('Queue subtitle S5 UI & Progress tests: 8 scenarios passed');
}

run();
