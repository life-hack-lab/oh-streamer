// Run with node tests/test_queue_subtitles_ui.js (no browser dependencies).
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const root = path.join(__dirname, '../src/oh_streamer');
const uiRoot = path.join(root, 'resources/queue_ui');
const source = fs.readFileSync(path.join(uiRoot, 'app.js'), 'utf8');
const html = fs.readFileSync(path.join(uiRoot, 'index.html'), 'utf8');
const bridge = fs.readFileSync(path.join(root, 'queue/queue_app.py'), 'utf8');

assert.doesNotMatch(html, /subtitleTranslationIgpu|subtitle_translation_igpu/, 'the retired iGPU switch is gone');
assert.doesNotMatch(html, /subtitleHeadMinutes/, 'the head-block minutes field is gone (D5)');
assert.doesNotMatch(source, /subtitle_head_minutes|subtitleHeadMinutes/, 'the head-block minutes setting is no longer read or written');

const helperStart = source.indexOf('const SUBTITLE_EDIT_STATES =');
const helperEnd = source.indexOf('\nfunction modelLabel(', helperStart);
assert(helperStart >= 0 && helperEnd > helperStart, 'subtitle state helpers are present');
const context = { esc: (s) => String(s == null ? '' : s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c])) };
vm.createContext(context);
vm.runInContext(source.slice(helperStart, helperEnd), context);

const VIEW = { v: 2, state: 'queued', progress: 0, covered_s: 0, slices_done: 0, slices_total: 4, running: null, wait: null, stuck: null,
  failures_elsewhere: 0, partial_stuck: false, retryable: false };
const withView = (view, extra) => ({ status: 'queued', view: { ...VIEW, ...view }, ...extra });

function testSubtitleWaitingLabels() {
  assert.equal(context.subtitleDisplay(withView({ state: 'waiting_download' })).label, '字幕：等待下載');
  assert.equal(context.subtitleDisplay(withView({ state: 'queued' })).label, '字幕：排隊中');
  assert.equal(context.subtitleDisplay(withView({ state: 'waiting_gpu' })).label, '字幕：等待 GPU');
  // old event fields never reach the label
  assert.equal(context.subtitleDisplay(withView({ state: 'queued' }, { cloud_mode: true, retry_note: '服務限流', phase: 'waiting_download' })).label, '字幕：排隊中');
}

function testSubtitleProgressStatuses() {
  const partial = context.subtitleDisplay(withView({ state: 'running', covered_s: 612.9, slices_done: 1, running: { executor: 'cloud', slice: 1, stage: 'asr' } }));
  assert.equal(partial.label, '字幕已到 10:12 · 語音辨識中（雲端，第 2 片）');
  assert.equal(partial.active, true, 'a running subtitle keeps pause/cancel controls');
  // item-level statuses without a view
  const cases = [
    ['queued', '字幕：排隊中', true],
    ['waiting_gpu', '字幕：排隊中', true],
    ['asr', '字幕：排隊中', true],
    ['translate', '字幕：排隊中', true],
    ['partial', '字幕：排隊中', true],
    ['ready', '字幕：等待封裝', true],
    ['mux', '字幕：封裝中', true],
    ['mux_pending', '字幕：等關閉播放器後封裝', true],
    ['done', '字幕：完成', false],
    ['failed', '字幕：失敗', false],
    ['skipped', '字幕：略過', false],
    ['cancelled', '字幕：已取消', false],
    ['paused', '字幕：已暫停', false],
  ];
  for (const [status, label, active] of cases) {
    const result = context.subtitleDisplay({ status, percent: 125, error: status === 'failed' ? 'ASR failed' : '' });
    assert.equal(result.label, label, `${status} status label`);
    assert.equal(result.active, active, `${status} activity`);
    assert.equal(result.showPercent, status === 'mux', `${status} percentage visibility`);
    if (status === 'mux') assert.equal(result.percent, 100, 'mux percentage is clamped');
    else assert.equal(result.percent, null, `${status} percentage is suppressed`);
    assert.equal(Object.hasOwn(result, 'eta'), false, 'subtitle progress does not expose an ETA');
  }
  const muxStatic = context.subtitleDisplay({ status: 'mux', percent: null });
  assert.equal(muxStatic.indeterminate, false, 'no sliding animation');
  assert.equal(muxStatic.fill, 100);
  assert.equal(muxStatic.dim, true);
  assert.equal(muxStatic.cat, 'run');
  assert.equal(context.subtitleDisplay({ status: 'failed', error: 'ASR failed' }).error, 'ASR failed');
  assert.equal(context.subtitleDisplay({ status: 'unrecognized' }).label, '字幕：排隊中');
  assert.equal(context.subtitleDisplay({}), null);
  assert.equal(context.subtitleDisplay(null), null);
}

function testViewProgress() {
  const running = context.subtitleDisplay(withView({ state: 'running', progress: 0.4, running: { executor: 'local', slice: 0, stage: 'mt' } }));
  assert.equal(running.label, '字幕：翻譯中（本地，第 1 片）');
  assert.equal(running.percent, 40);
  assert.equal(running.showPercent, true);
  const unknown = context.subtitleDisplay(withView({ state: 'running', progress: null, running: { executor: 'local', slice: 0, stage: 'asr' } }));
  assert.equal(unknown.indeterminate, false, 'unknown progress never slides');
  assert.equal(unknown.fill, 0);
  const queued = context.subtitleDisplay(withView({ state: 'queued', progress: 0.5 }));
  assert.equal(queued.percent, null, 'queued work never displays a percent');
  assert.equal(queued.showTrack, true, 'quiet bar always shows coverage');
  assert.equal(context.subtitleDisplay(withView({ state: 'complete', progress: 1 })).label, '字幕：完成');
  const stuck = context.subtitleDisplay(withView({ state: 'stuck', retryable: true, stuck: { stage: 'asr', candidates: [] } }, { error: 'boom' }));
  assert.equal(stuck.retryable, true);
  assert.equal(stuck.restartable, true);
  assert.equal(stuck.error, 'boom');
}

function testToggleAvailabilityAndQueueStates() {
  const enabled = { available: true };
  const unavailable = { available: false, reason: '找不到 ASR runtime' };
  let model = context.subtitleToggleState({ state: 'waiting', settings: {} }, enabled);
  assert.equal(model.visible, true);
  assert.equal(model.checked, false, 'subtitle checkbox defaults off');
  assert.equal(model.disabled, false);

  model = context.subtitleToggleState({ state: 'queued', settings: { subtitles: true } }, enabled);
  assert.equal(model.checked, true);
  assert.equal(model.disabled, false);

  model = context.subtitleToggleState({ state: 'queued', settings: {} }, unavailable);
  assert.equal(model.visible, true);
  assert.equal(model.disabled, true, 'missing runtime disables the item checkbox');
  assert.equal(model.reason, '找不到 ASR runtime');

  for (const state of ['downloading', 'paused']) {
    assert.equal(context.subtitleToggleState({ state, settings: {} }, enabled).visible, true, `${state} remains editable`);
  }
  assert.equal(context.subtitleToggleState({ state: 'processing', settings: {} }, enabled).visible, false);
  assert.equal(context.subtitleToggleState({ state: 'done', settings: {} }, enabled).visible, false);
  const cloudReady = { runtime_available: true, runtime_reason: '', available: false, reason: 'legacy local models missing' };
  assert.equal(context.subtitleToggleState({ state: 'queued', settings: {} }, cloudReady).disabled, false,
    'unified runtime availability can support cloud even when the legacy local-model gate is false');
  const runtimeMissing = { runtime_available: false, runtime_reason: '字幕 runtime 尚未安裝', available: true };
  model = context.subtitleToggleState({ state: 'queued', settings: {} }, runtimeMissing);
  assert.equal(model.disabled, true);
  assert.equal(model.reason, '字幕 runtime 尚未安裝');
}

function testUiAndBridgeWiring() {
  assert.match(html, /data-cat="subtitles"/, 'global subtitle settings pane exists');
  for (const id of ['subtitleMainView', 'subtitleAsr', 'subtitleTranslation', 'subtitleJapanese', 'subtitleAdvancedView', 'subtitleModelsView', 'subtitleComparisonEnabled', 'subtitleComparisonRows', 'subtitleModelList', 'subtitleModelForm', 'subtitleModelTabAsr', 'subtitleModelTabTranslation']) {
    assert.match(html, new RegExp(`id="${id}"`), `${id} exists`);
  }
  for (const oldId of ['subtitleAsrMode', 'subtitlePython', 'subtitleServer', 'subtitleInterjections', 'subtitleShortPolicy', 'subtitleComboDefault']) {
    assert.doesNotMatch(html, new RegExp(`id="${oldId}"`), `${oldId} legacy setting is removed`);
  }
  assert.match(html, /每次都會產生精簡中文與完整中文/, 'output tracks are explained as fixed behavior');
  assert.match(html, /多字幕比較/, 'comparison is presented as additional tracks');
  assert.match(html, /主字幕仍使用上方設定，預設播放不會被取代/, 'extra rows cannot replace primary/default playback');
  assert.match(html, /服務供應商可能依使用量收費/, 'cloud disclosure includes possible provider charges');
  assert.match(html, /雲端辨識可能會傳送測試音訊/, 'cloud disclosure warns about audio transfer');
  assert.match(html, /雲端翻譯可能會傳送字幕文字/, 'cloud disclosure warns about subtitle text transfer');
  assert.doesNotMatch(html, /自動判定/, 'translation UI never offers automatic model selection');
  assert.match(source, /data-k="subtitles"/, 'intake exposes the per-item subtitle checkbox');
  assert.match(source, /o\.subtitles = !!\(subtitleRuntimeAvailable\(env\) && pick\('subtitles', 'queue'\)\)/, 'intake sends an explicit subtitle boolean using unified runtime availability');
  assert.match(source, /data-k="subtitle_comparison_enabled"/, 'intake only exposes the comparison on/off override');
  assert.match(source, /set_subtitle_configuration/, 'settings use the unified subtitle configuration API');
  assert.match(source, /list_subtitle_models/, 'model management reads the safe model registry');
  assert.match(source, /test_add_subtitle_model/, 'model management tests inference before registration');
  assert.match(source, /remove_subtitle_model/, 'user models can be removed');
  assert.doesNotMatch(source, /set_subtitle_combos|list_subtitle_combos|subtitle_combos_default/, 'old preset-combo configuration is gone');
  assert.doesNotMatch(source, /loadComboCatalog/, 'mock startup does not call the removed combo loader');
  assert.match(source, /loadSubtitleModels\(\);/, 'boot loads the model registry');
  assert.match(source, /act\('set_subtitles', id, input\.checked\)/, 'pending item checkbox calls set_subtitles');
  assert.match(source, /act\('rerun_subtitles', id\)/, 'completed item action calls rerun_subtitles');
  assert.match(source, /buildBtn\('暫停字幕', 'pause', '', \(\) => act\('pause', id\)/, 'active subtitle reruns can be paused');
  assert.match(source, /buildBtn\('繼續字幕', 'play', 'primary', \(\) => act\('resume', id\)/, 'paused subtitle reruns can resume');
  assert.match(source, /buildBtn\('取消字幕', 'pause', '', \(\) => act\('cancel_subtitles', id\)/, 'subtitle reruns can be cancelled');
  assert.match(source, /subtitle_environment/, 'UI reads subtitle runtime availability');
  assert.match(source, /runtime_available != null \? environment\.runtime_available === true : environment\.available === true/, 'new runtime gate falls back only for older backends');
  assert.match(source, /subtitleDetail/, 'item card exposes the subtitle phase detail');
  assert.doesNotMatch(source, /classList\.toggle\('indeterminate', subtitle\.indeterminate\)/, 'subtitle bar never slides');
  assert.match(source, /setAttr\(c\.subtitleProgress, 'data-cat'/, 'subtitle info carries its category');
  assert.match(source, /removeAttribute\('aria-valuenow'\)/, 'unknown progress does not expose a fake zero');
  assert.doesNotMatch(fs.readFileSync(path.join(uiRoot, 'app.css'), 'utf8'), /\.subtitle-track\.indeterminate/, 'app.css has no sliding subtitle bar');
  for (const stylesheet of ['app.css', 'app-modern.css']) {
    const css = fs.readFileSync(path.join(uiRoot, stylesheet), 'utf8');
    assert.match(css, /\.subtitle-detail\s*\{/, `${stylesheet} styles phase detail unobtrusively`);
  }
  assert.match(bridge, /def set_subtitles\(self, id, enabled\)/, 'Python bridge exposes set_subtitles');
  assert.match(bridge, /def rerun_subtitles\(self, id\)/, 'Python bridge exposes rerun_subtitles');
  assert.match(bridge, /def set_subtitle_configuration\(/, 'Python bridge accepts complete row configuration');
  assert.match(bridge, /def list_subtitle_models\(/, 'Python bridge exposes safe registered models');
  assert.match(bridge, /def test_add_subtitle_model\(/, 'Python bridge runs a real add-model test');
  assert.match(bridge, /def update_subtitle_model_key\(/, 'Python bridge supports write-only API key rotation');
  assert.match(bridge, /def remove_subtitle_model\(/, 'Python bridge removes registered user models');
  assert.doesNotMatch(html, /id="subtitleCloudEnabled"/, "standalone cloud-egress switch was removed (commit 1040121); cloud is auto-enabled");
  for (const id of ['subtitleModelPreset', 'subtitleModelRemoteList', 'subtitleModelKeyToggle', 'subtitleModelPairGroq']) {
    assert.match(html, new RegExp(`id="${id}"`), `${id} cloud key management control exists`);
  }
  assert.match(source, /update_subtitle_model_key/, 'UI supports updating stored cloud model API keys');
  assert.match(source, /__builtin__/, 'cloud ASR defaults to built-in synthetic audio sample when no local file is chosen');
}

function testSubtitleTrackStatusSummary() {
  const running = context.subtitleDisplay({ status: 'combos', detail: '第 2 組', percent: null });
  assert.equal(running.label, '字幕：排隊中');
  assert.equal(running.active, true);
  assert.equal(context.subtitleComboSummary({ status: 'done' }), null, 'no combos -> no chip');
  const sum = context.subtitleComboSummary({ status: 'done', combos: { total: 3, done: 2, failed: 1, items: [
    { id: 'a', label: 'A', status: 'done', cue_count: 10 },
    { id: 'b', label: 'B', status: 'done', cue_count: 0 },
    { id: 'c', label: 'C', status: 'failed', failed_stage: 'translation', error: 'boom' }] } });
  assert.equal(sum.text, '字幕 2/3 組');
  assert.equal(sum.failText, '1 組失敗');
  assert.equal(sum.tone, 'warn');
  assert.equal(sum.items[0].cues, 10);
  assert.equal(sum.items[2].failure, '失敗於翻譯：boom');
  assert.equal(sum.running, false, 'a failed combo does not make the subtitle step running');
  assert.equal(context.subtitleComboSummary({ status: 'combos', combos: { total: 1, done: 0, failed: 0, items: [{ id: 'a', status: 'running' }] } }).running, true);
}

async function testSubtitleConfigRoundTripAndChoices() {
  const start = source.indexOf('const SUBTITLE_DUAL_VALUE =');
  const end = source.indexOf('\nfunction setSubtitleSubview(', start);
  const modelData = [
    { id: 'whisper-ja-1.5B', label: 'Whisper ja 1.5B', kind: 'asr', provider: 'local', recipe: 'whisper-ja-1.5B', builtin: true, status: 'ready' },
    { id: 'anime-whisper', label: 'Anime Whisper', kind: 'asr', provider: 'local', recipe: 'anime-whisper', builtin: true, status: 'ready' },
    { id: 'whisper-large-v3-turbo', label: 'Whisper large-v3-turbo', kind: 'asr', provider: 'local', recipe: 'whisper-large-v3-turbo', builtin: true, status: 'ready' },
    { id: 'sakura-galtransl-7b-v3.7', label: 'Sakura-GalTransl 7B v3.7', kind: 'translation', provider: 'local', recipe: 'sakura-galtransl-7b-v3.7', builtin: true, status: 'ready', is_default: true },
    { id: 'sakura-v1', label: 'Sakura v1', kind: 'translation', provider: 'local', recipe: 'sakura-v1', builtin: true, status: 'ready' },
    { id: 'local-translation-4c81', label: '我的本機翻譯', kind: 'translation', provider: 'local', recipe: 'sakura-galtransl-7b-v3.7', builtin: false, status: 'ready' },
    { id: 'cloud-translation-4c81', label: '雲端翻譯', kind: 'translation', provider: 'openai-compatible', remote_model: 'remote-model-v2', builtin: false, status: 'ready' },
  ];
  const config = { subtitleModels: modelData };
  vm.createContext(config);
  vm.runInContext(source.slice(start, end), config);
  const settings = {
    subtitle_primary: {
      id: 'primary', label: '主字幕', asr: { mode: 'diagnostic', model: 'anime-whisper', decode: { no_repeat_ngram_size: 2 } },
      translation: { profile: 'sakura-v1', model_id: 'local-translation-4c81', model_file: null, gpu_layers: null },
      japanese_track: true, sentence: { short_policy: 'keep', show_interjections: true },
    },
    subtitle_comparison_enabled: false,
    subtitle_comparison_rows: [{ id: 'extra-1', label: 'Saved extra', asr: { model: 'anime-whisper', mode: 'diagnostic' },
      translation: { profile: 'sakura-v1', model_id: 'cloud-translation-4c81', model_file: null, gpu_layers: null }, japanese_track: true,
      sentence: { short_policy: 'drop', show_interjections: false } }],
  };
  const primary = config.subtitlePrimaryFromSettings(settings);
  const rows = config.subtitleRowsFromSettings(settings);
  assert.equal(primary.asr.mode, 'single', 'legacy diagnostic primary model stays single');
  assert.equal(primary.translation.profile, 'sakura-v1');
  assert.equal(primary.translation.model_id, 'local-translation-4c81');
  assert.equal(primary.japanese_track, true);
  assert.equal(primary.sentence.short_policy, 'keep');
  assert.equal(rows[0].asr.mode, 'single', 'legacy diagnostic extra rows stay single');
  assert.equal(rows[0].translation.model_id, 'cloud-translation-4c81');
  assert.equal(rows[0].japanese_track, true);
  assert.equal(config.subtitleTranslationValue(primary), 'local-translation-4c81', 'registered translation ID survives normalization');

  const asrChoices = config.subtitleAsrChoices(config.subtitleAsrValue(primary));
  assert.equal(asrChoices[0].value, '__dual__');
  assert.match(asrChoices[0].label, /whisper-ja-1\.5B \+ anime-whisper/);
  assert(asrChoices.some((choice) => choice.label === '單模型 · Whisper large-v3-turbo'));
  const translations = config.subtitleTranslationChoices(config.subtitleTranslationValue(primary));
  assert(translations.some((choice) => choice.value === 'local-translation-4c81' && choice.label === '我的本機翻譯'));
  assert(translations.some((choice) => choice.value === 'cloud-translation-4c81' && choice.label.includes('remote-model-v2')));
  assert(translations.some((choice) => choice.label === '預設 · Sakura-GalTransl 7B v3.7'));
  assert.doesNotMatch(translations.find((choice) => choice.value === 'cloud-translation-4c81').label, /cloud-translation-4c81/,
    'opaque registered IDs stay out of user-facing labels');

  const applySourceStart = source.indexOf('function updatePrimaryValue(');
  const applySourceEnd = source.indexOf("\n$('#subtitleAsr').addEventListener", applySourceStart);
  const calls = [];
  const update = {
    current: { settings: JSON.parse(JSON.stringify(settings)) },
    SUBTITLE_DUAL_VALUE: '__dual__', DEFAULT_TRANSLATION_RECIPE: 'sakura-galtransl-7b-v3.7',
    subtitlePrimaryFromSettings: config.subtitlePrimaryFromSettings,
    subtitleRowsFromSettings: config.subtitleRowsFromSettings,
    subtitleModelChoices: config.subtitleModelChoices,
    subtitleTranslationModelValue: config.subtitleTranslationModelValue,
    subtitleRegisteredValue: vm.runInContext('subtitleRegisteredValue', config),
    subtitleRecipeValue: vm.runInContext('subtitleRecipeValue', config),
    cloneConfig: config.cloneConfig,
    saveSubtitleConfiguration(primaryValue, enabled, rowsValue) { calls.push({ primaryValue, enabled, rowsValue }); return Promise.resolve(true); },
  };
  vm.createContext(update);
  vm.runInContext(source.slice(applySourceStart, applySourceEnd), update);
  await update.updatePrimaryValue('translation', 'cloud-translation-4c81');
  let saved = update.current.settings.subtitle_primary.translation;
  assert.equal(saved.model_id, 'cloud-translation-4c81');
  assert.equal(saved.profile, 'sakura-v1', 'custom translation retains the validated profile');
  assert.equal(saved.provider, undefined, 'provider metadata is not sent in the combo schema');
  assert.equal(saved.remote_model, undefined, 'remote model metadata is not sent in the combo schema');
  assert.equal(saved.model_file, null, 'registered models do not inherit stale local paths');
  assert.equal(saved.gpu_layers, null, 'translation model changes recalculate offload defaults');
  await update.updatePrimaryValue('translation', 'sakura-v1');
  saved = update.current.settings.subtitle_primary.translation;
  assert.equal(saved.profile, 'sakura-v1');
  assert.equal(saved.model_id, undefined, 'built-in translation uses its profile');
  assert.equal(saved.model_file, null, 'built-in changes clear a previously selected GGUF path');
  await update.updatePrimaryValue('asr', 'whisper-large-v3-turbo');
  assert.deepEqual(JSON.parse(JSON.stringify(update.current.settings.subtitle_primary.asr)),
    { mode: 'single', model: 'whisper-large-v3-turbo', decode: null }, 'ASR model changes reset model-specific decode defaults');
  await update.updatePrimaryValue('asr', '__dual__');
  assert.equal(update.current.settings.subtitle_primary.asr.mode, 'dual');
  assert.equal(update.current.settings.subtitle_primary.asr.decode, null);
  assert.equal(calls.length, 4);
}

function testRetryAndRestartButtons() {
  const quota = context.subtitleDisplay(withView({ state: 'waiting_quota', covered_s: 600, wait: { kind: 'quota', slice: 2, stage: 'mt', retry_at: 1030 } }), 1000);
  assert.equal(quota.retryable, false, 'a scheduled wait retries by itself');
  assert.equal(quota.blocked, false);
  assert.equal(quota.showTrack, true);
  const running = context.subtitleDisplay(withView({ state: 'running', running: { executor: 'local', slice: 0, stage: 'asr' } }));
  assert.equal(running.retryable, false);
  assert.equal(running.restartable, false, 'nothing is thrown away while a run is active');
  const partialStuck = context.subtitleDisplay(withView({ state: 'running', partial_stuck: true, failures_elsewhere: 1, retryable: true, running: { executor: 'local', slice: 0, stage: 'asr' } }));
  assert.equal(partialStuck.retryable, true);
  assert.equal(context.subtitleDisplay({ status: 'failed' }).retryable, true);
  assert.equal(context.subtitleDisplay({ status: 'done' }).restartable, true);
  assert.match(bridge, /def retry_subtitles\(self, id\)/);
  assert.match(bridge, /def restart_subtitles\(self, id\)/);
}

testRetryAndRestartButtons();
testSubtitleWaitingLabels();
testSubtitleProgressStatuses();
testViewProgress();
testToggleAvailabilityAndQueueStates();
testUiAndBridgeWiring();
testSubtitleTrackStatusSummary();
testSubtitleConfigRoundTripAndChoices().catch((error) => { console.error(error); process.exitCode = 1; });
console.log('Queue subtitles UI: 6 scenarios passed');

async function testSubtitlePlayChoices() {
  for (const state of ['queued', 'running', 'waiting_quota', 'waiting_gpu', 'waiting_executor', 'waiting_download']) {
    assert(context.subtitlePlayPrompt({ subtitle: withView({ state }) }), `${state} prompts before playback`);
  }
  for (const state of ['complete', 'stuck']) {
    assert.equal(context.subtitlePlayPrompt({ subtitle: withView({ state }) }), null, `${state} can play directly`);
  }
  assert.equal(context.subtitlePlayPrompt({ subtitle: withView({ state: 'running', covered_s: 30 }) }), null, 'covered subtitles never block playback');
  for (const status of ['ready', 'mux', 'mux_pending', 'done', 'failed', 'skipped']) {
    assert.equal(context.subtitlePlayPrompt({ subtitle: { status } }), null, `${status} can play directly`);
  }
  assert.equal(context.subtitlePlayPrompt({ subtitle: { status: 'queued' } }), '字幕：排隊中');
  assert.equal(context.subtitlePlayPrompt({ subtitle: withView({ state: 'running', progress: 0.43, running: { executor: 'local', slice: 0, stage: 'asr' } }) }),
    '字幕：語音辨識中（本地，第 1 片） 43%');
  for (const text of ['字幕處理中', '等字幕完成再播', '先看，不要字幕']) assert(html.includes(text));
  assert(source.includes('字幕完成後自動播放'));
  assert(source.includes("act('cancel_play_after_subtitles', id)"));
  assert.match(bridge, /def cancel_play_after_subtitles\(self, id\)/);
  assert.match(bridge, /wait_for_subtitles=True/);
  const start = source.indexOf('async function openPlayer(');
  const end = source.indexOf('\nfunction playFrom(', start);
  for (const choice of ['wait', 'skip', null]) {
    const calls = [];
    const ctx = { current: { items: [{ id: 'a', subtitle: withView({ state: 'running' }) }] }, openingIds: new Set(), playErrors: new Map(), isMock: true,
      subtitlePlayPrompt: context.subtitlePlayPrompt, chooseSubtitlePlayback: async () => choice,
      api: async (...args) => { calls.push(args); return { ok: true }; }, render() {}, refresh: async () => {} };
    vm.createContext(ctx);
    vm.runInContext(source.slice(start, end), ctx);
    await ctx.openPlayer('a', 12);
    assert.equal(calls.length, choice ? 1 : 0, 'dismiss never starts playback');
    if (choice) assert.deepEqual(calls[0], choice === 'wait' ? ['play', 'a', 12, true] : ['play', 'a', 12]);
    assert.equal(ctx.openingIds.size, 0);
    assert.equal(ctx.playErrors.size, 0);
  }

  const nodes = new Map();
  const node = (selector) => {
    if (!nodes.has(selector)) nodes.set(selector, { textContent: '', listeners: {}, addEventListener(event, fn) { this.listeners[event] = fn; }, showModal() { this.open = true; }, close() { this.open = false; }, focus() {} });
    return nodes.get(selector);
  };
  const dialogContext = { current: { items: [{ id: 'a', subtitle: withView({ state: 'running', progress: 0.43, running: { executor: 'local', slice: 0, stage: 'asr' } }) }] }, document: { activeElement: null },
    $: node, setText: (element, value) => { element.textContent = value; }, subtitlePlayPrompt: context.subtitlePlayPrompt };
  vm.createContext(dialogContext);
  vm.runInContext(source.slice(source.indexOf('let subtitlePlayChoice = null;'), source.indexOf('async function openPlayer(')), dialogContext);
  const pendingChoice = dialogContext.chooseSubtitlePlayback('a');
  assert.equal(node('#subtitlePlayDialog').open, true);
  assert.equal(node('#subtitlePlayProgress').textContent, '字幕：語音辨識中（本地，第 1 片） 43%');
  dialogContext.current.items[0].subtitle.view.progress = 0.65;
  dialogContext.updateSubtitlePlayDialog();
  assert.equal(node('#subtitlePlayProgress').textContent, '字幕：語音辨識中（本地，第 1 片） 65%', 'open dialog tracks progress');
  dialogContext.current.items[0].subtitle.view.slices_total = 4;
  dialogContext.updateSubtitlePlayDialog();
  assert.equal(node('#subtitlePlayWait').textContent, '等前段字幕完成再播');
  node('#subtitlePlayWait').listeners.click();
  assert.equal(await pendingChoice, 'wait');
  assert.equal(node('#subtitlePlayDialog').open, false);
  console.log('Subtitle playback choices: passed');
}
testSubtitlePlayChoices().catch((error) => { console.error(error); process.exitCode = 1; });
