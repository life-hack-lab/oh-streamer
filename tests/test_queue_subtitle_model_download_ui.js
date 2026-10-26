// Browser regression for the local-subtitle model download card (mock bridge, no network).
// PLAYWRIGHT_MODULE and PLAYWRIGHT_EXECUTABLE can select installed tooling. SHOT_DIR (optional) saves screenshots.
const assert = require('node:assert/strict');
const path = require('node:path');
const { pathToFileURL } = require('node:url');
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');

const page_url = (query) => pathToFileURL(path.resolve(__dirname, '../src/oh_streamer/resources/queue_ui/index.html')).href + '?mock&' + query;

async function open(browser, query) {
  const page = await browser.newPage({ viewport: { width: 1100, height: 1100 } });
  const errors = [];
  page.on('pageerror', (e) => { if (!/drawImage/.test(e.message)) errors.push(e.message); });
  await page.goto(page_url(query));
  await page.locator('#btnSettings').click();
  await page.locator('#stab-subtitles').click();
  await page.locator('#subtitleMode').selectOption('local');
  await page.locator('#subtitleModelDlManage').waitFor({ state: 'visible' });
  await page.locator('#subtitleModelDlManage').click();
  await page.locator('.mdl-row').first().waitFor({ state: 'visible' });
  await page.waitForFunction(() => !document.getElementById('subtitleModelDlStart').disabled || /請至少|不足|已安裝|正在使用/.test(document.getElementById('subtitleModelDlSummary').textContent));
  return { page, errors };
}
const box = (page, id) => page.locator(`#subtitleModelDlGroups input[data-id="${id}"]`);

(async () => {
  const browser = await chromium.launch({ headless: true,
    ...(process.env.PLAYWRIGHT_EXECUTABLE ? { executablePath: process.env.PLAYWRIGHT_EXECUTABLE } : {}) });
  try {
    // --- rules, locked runtime, confirm, progress, cancel, resume
    let { page, errors } = await open(browser, 'mdlspeed=800');
    const headings = await page.locator('#subtitleModelDlGroups h5').allInnerTexts();
    assert.deepEqual(headings, ['語音辨識（ASR）', '翻譯模型', '執行程式']);
    assert.equal(await box(page, 'llama-server').isChecked(), true);
    assert.equal(await box(page, 'llama-server').isDisabled(), true, 'runtime locked');
    assert.equal(await box(page, 'sakura-galtransl-7b-v3.7').isChecked(), true, '7B default');
    assert.equal(await box(page, 'sakura-galtransl-14b-v3.8').isChecked(), false);
    assert.match(await page.locator('#subtitleModelDlSummary').innerText(), /預計下載 1[0-9.]+ GB/);
    assert.match(await page.locator('#subtitleModelDlStatus').innerText(), /尚未下載：需要語音辨識模型、翻譯模型和執行程式，共約 1[0-9.]+ GB/);
    assert.equal(await page.locator('#subtitleModelDlStart').innerText(), '下載本機字幕模型');
    assert.equal(await page.locator('#subtitleModelDlStart').isDisabled(), false);
    if (process.env.SHOT_DIR) { await page.locator('#subtitleModelDlSummary').scrollIntoViewIfNeeded(); await page.screenshot({ path: path.join(process.env.SHOT_DIR, 'S3a-card-default.png') }); }
    // license + model card are visible before download
    assert.match(await page.locator('.mdl-row[data-id="sakura-galtransl-7b-v3.7"] .mdl-lic').innerText(), /CC BY-NC-SA 4\.0（僅限非商業用途）/);
    assert.match(await page.locator('.mdl-row[data-id="whisper-ja-1.5B"] .mdl-lic').innerText(), /模型卡未標示授權/);
    await page.locator('.mdl-row[data-id="anime-whisper"] .mdl-card').click();
    assert.deepEqual(await page.evaluate(() => window.__modelCardOpened), ['anime-whisper']);
    // ASR rule
    await box(page, 'whisper-ja-1.5B').uncheck();
    assert.equal(await page.locator('#subtitleModelDlStart').isDisabled(), false, 'one ASR is enough');
    await box(page, 'anime-whisper').uncheck();
    assert.equal(await page.locator('#subtitleModelDlStart').isDisabled(), true);
    assert.match(await page.locator('#subtitleModelDlSummary').innerText(), /至少選擇一個語音辨識/);
    await box(page, 'anime-whisper').check();
    await box(page, 'whisper-ja-1.5B').check();
    // translation rule
    await box(page, 'sakura-galtransl-7b-v3.7').uncheck();
    assert.equal(await page.locator('#subtitleModelDlStart').isDisabled(), true);
    assert.match(await page.locator('#subtitleModelDlSummary').innerText(), /至少選擇一個翻譯/);
    await box(page, 'sakura-galtransl-7b-v3.7').check();
    await page.waitForFunction(() => !document.getElementById('subtitleModelDlStart').disabled);
    // confirm dialog: cancel first, nothing starts
    await page.locator('#subtitleModelDlStart').click();
    assert.match(await page.locator('#modalBody').innerText(), /共約 1[0-9.]+ GB/);
    assert.match(await page.locator('#modalBody').innerText(), /模型授權[\s\S]*Sakura-GalTransl 7B v3\.7：CC BY-NC-SA 4\.0/);
    await page.locator('#modalCancel').click();
    assert.equal(await page.locator('#subtitleModelDlCancel').isVisible(), false);
    assert.equal(await page.locator('#subtitleModelDlProgress').isVisible(), false);
    // confirm for real
    await page.locator('#subtitleModelDlStart').click();
    await page.locator('#modalOk').click();
    await page.locator('#subtitleModelDlCancel').waitFor({ state: 'visible' });
    await page.waitForFunction(() => /GB|MB/.test(document.getElementById('subtitleModelDlProgressText').textContent) && !/準備中/.test(document.getElementById('subtitleModelDlProgressText').textContent));
    assert.equal(await box(page, 'anime-whisper').isDisabled(), true, 'selection locked while running');
    if (process.env.SHOT_DIR) { await page.locator('#subtitleModelDlSummary').scrollIntoViewIfNeeded(); await page.screenshot({ path: path.join(process.env.SHOT_DIR, 'S3a-card-progress.png') }); }
    await page.locator('#subtitleModelDlCancel').click();
    await page.waitForFunction(() => /已取消/.test(document.getElementById('subtitleModelDlProgressText').textContent));
    assert.match(await page.locator('#subtitleModelDlProgressText').innerText(), /已下載部分會保留/);
    assert.equal(await page.locator('#subtitleModelDlStart').innerText(), '繼續下載');
    await page.close();
    assert.deepEqual(errors, []);

    // --- installed items: no re-download, count toward the rules
    ({ page, errors } = await open(browser, 'mdlinstalled=anime-whisper,llama-server'));
    assert.match(await page.locator('.mdl-row[data-id="anime-whisper"]').innerText(), /已安裝/);
    assert.equal(await box(page, 'anime-whisper').isDisabled(), true);
    assert.equal(await box(page, 'llama-server').isDisabled(), true);
    assert.match(await page.locator('#subtitleModelDlStatus').innerText(), /尚未下載/);
    await box(page, 'whisper-ja-1.5B').uncheck();            // installed anime-whisper still covers ASR
    assert.equal(await page.locator('#subtitleModelDlStart').isDisabled(), false);
    await box(page, 'sakura-galtransl-7b-v3.7').uncheck();
    assert.equal(await page.locator('#subtitleModelDlStart').isDisabled(), true);
    await page.close();
    assert.deepEqual(errors, []);

    // --- not enough disk
    ({ page, errors } = await open(browser, 'mdlfree=5'));
    assert.equal(await page.locator('#subtitleModelDlStart').isDisabled(), true);
    assert.match(await page.locator('#subtitleModelDlSummary').innerText(), /磁碟空間不足/);
    await page.close();

    // --- llama-server in use: download disabled with the reason; an installed runtime is not blocked
    ({ page, errors } = await open(browser, 'mdlinuse=1'));
    assert.equal(await page.locator('#subtitleModelDlStart').isDisabled(), true);
    assert.match(await page.locator('#subtitleModelDlSummary').innerText(), /本機字幕正在使用 llama-server/);
    await page.close();
    ({ page, errors } = await open(browser, 'mdlinuse=1&mdlinstalled=llama-server'));
    assert.equal(await page.locator('#subtitleModelDlStart').isDisabled(), false);
    await page.close();
    assert.deepEqual(errors, []);

    // --- failure shows the error and a retry button
    ({ page, errors } = await open(browser, 'mdlfail=1&mdlspeed=2000'));
    await page.locator('#subtitleModelDlStart').click();
    await page.locator('#modalOk').click();
    await page.locator('#subtitleModelDlError').waitFor({ state: 'visible', timeout: 15000 });
    assert.match(await page.locator('#subtitleModelDlError').innerText(), /網路連線中斷/);
    assert.equal(await page.locator('#subtitleModelDlStart').innerText(), '重試下載');
    assert.equal(await page.locator('#subtitleModelDlStart').isDisabled(), false);
    await page.close();
    assert.deepEqual(errors, []);

    // --- verifying (SHA256 check between files) stays "running": progress shown, polling continues to done
    ({ page, errors } = await open(browser, 'mdlverify=10&mdlspeed=4000'));
    await page.locator('#subtitleModelDlStart').click();
    await page.locator('#modalOk').click();
    await page.waitForFunction(() => /正在核對檔案/.test(document.getElementById('subtitleModelDlProgressText').textContent));
    assert.equal(await page.locator('#subtitleModelDlProgress').isVisible(), true);
    assert.equal(await page.locator('#subtitleModelDlCancel').isVisible(), true);
    assert.equal(await page.locator('#subtitleModelDlStart').isVisible(), false);
    await page.waitForFunction(() => /\/秒/.test(document.getElementById('subtitleModelDlProgressText').textContent));
    await page.waitForFunction(() => /下載完成/.test(document.getElementById('subtitleModelDlProgressText').textContent), null, { timeout: 15000 });
    await page.close();
    assert.deepEqual(errors, []);

    // --- fully installed: status + manage link, no download button
    ({ page, errors } = await open(browser, 'mdlinstalled=anime-whisper,whisper-ja-1.5B,sakura-galtransl-7b-v3.7,llama-server'));
    assert.match(await page.locator('#subtitleModelDlStatus').innerText(), /已安裝，可以使用/);
    assert.equal(await page.locator('#subtitleModelDlStart').isVisible(), false);
    assert.equal(await page.locator('#subtitleModelDlManage').innerText(), '管理已下載的模型');
    assert.equal(await page.locator('#subtitleModelDlGroups').isVisible(), true);
    await page.close();
    assert.deepEqual(errors, []);

    // --- 前往下載 from the combo-fallback note opens the local card and focuses the download button
    ({ page, errors } = await open(browser, 'missing=anime-whisper'));
    await page.locator('#subtitleMode').selectOption('cloud');
    assert.equal(await page.locator('#subtitleLocalHome').isVisible(), false);
    await page.locator('#subtitleComboFallbackGo button').click();
    assert.equal(await page.locator('#subtitleLocalHome').isVisible(), true);
    assert.equal(await page.evaluate(() => document.activeElement.id), 'subtitleModelDlStart');
    await page.close();
    assert.deepEqual(errors, []);
    console.log('model download card ui: ok');
  } finally {
    await browser.close();
  }
})().catch((e) => { console.error(e); process.exit(1); });
