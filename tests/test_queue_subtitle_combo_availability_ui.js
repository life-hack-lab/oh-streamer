// Browser regression: pickers gray out combos whose local models are missing, with a download hint,
// and the settings page says when a fallback combo is in effect (mock bridge, ?missing=<ids>).
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
  await page.waitForTimeout(400);
  return { page, errors };
}
const options = (page, sel) => page.locator(`${sel} option`).evaluateAll((os) => os.map((o) => ({ value: o.value, text: o.textContent, disabled: o.disabled })));

(async () => {
  const browser = await chromium.launch({ headless: true,
    ...(process.env.PLAYWRIGHT_EXECUTABLE ? { executablePath: process.env.PLAYWRIGHT_EXECUTABLE } : {}) });
  try {
    // everything installed: nothing grayed, no fallback note
    let { page, errors } = await open(browser, 'x=1');
    let rows = await options(page, '#subtitleAsr');
    assert.ok(rows.length >= 2 && rows.every((r) => !r.disabled), 'all enabled');
    assert.equal(await page.locator('#subtitleComboFallback').isVisible(), false);
    assert.deepEqual(errors, []);
    await page.close();

    // anime-whisper missing: dual + anime single grayed with reason, fallback note shown
    ({ page, errors } = await open(browser, 'missing=anime-whisper'));
    rows = await options(page, '#subtitleAsr');
    const dual = rows.find((r) => r.value === '__dual__');
    assert.equal(dual.disabled, true);
    assert.match(dual.text, /缺少 anime-whisper，可到 設定 → 字幕 → 本機 下載本機字幕模型/);
    const anime = rows.find((r) => r.value === 'anime-whisper');
    assert.equal(anime.disabled, true);
    assert.match(anime.text, /缺少 anime-whisper/);
    assert.equal(rows.find((r) => r.value === 'whisper-ja-1.5B').disabled, false);
    assert.equal(await page.locator('#subtitleComboFallback').isVisible(), true);
    assert.match(await page.locator('#subtitleComboFallback').innerText(), /預設組合缺少模型.*暫時改用/);
    if (process.env.SHOT_DIR) await page.screenshot({ path: path.join(process.env.SHOT_DIR, 'S3b-combo-unavailable.png') });
    assert.deepEqual(errors, []);
    await page.close();

    // a missing translation model is grayed in the translation picker too
    ({ page, errors } = await open(browser, 'missing=sakura-galtransl-14b-v3.8'));
    rows = await options(page, '#subtitleTranslation');
    const big = rows.find((r) => r.value === 'sakura-galtransl-14b-v3.8');
    assert.equal(big.disabled, true);
    assert.match(big.text, /設定 → 字幕 → 本機 下載本機字幕模型/);
    assert.equal(await page.locator('#subtitleComboFallback').isVisible(), false);
    assert.deepEqual(errors, []);
    console.log('ok');
  } finally {
    await browser.close();
  }
})().catch((e) => { console.error(e); process.exit(1); });
