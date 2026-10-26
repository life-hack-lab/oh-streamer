// Browser regression: pending subtitle-settings prompt rows, the queue-wide summary line and the 本片自訂 / 沿用舊設定 tags (mock bridge).
const assert = require('node:assert/strict');
const path = require('node:path');
const { pathToFileURL } = require('node:url');
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');

const url = pathToFileURL(path.resolve(__dirname, '../src/oh_streamer/resources/queue_ui/index.html')).href + '?mock';

async function open(browser) {
  const page = await browser.newPage({ viewport: { width: 1100, height: 1400 } });
  const errors = [];
  page.on('pageerror', (e) => { if (!/drawImage/.test(e.message)) errors.push(e.message); });
  await page.goto(url);
  await page.waitForSelector('.subtitle-settings-row:not([hidden])');
  return { page, errors };
}
const cardOf = (page, text) => page.locator('.card', { hasText: text });
const btnLabels = (loc) => loc.locator('button').evaluateAll((bs) => bs.map((b) => b.textContent.trim()));

(async () => {
  const browser = await chromium.launch({ headless: true,
    ...(process.env.PLAYWRIGHT_EXECUTABLE ? { executablePath: process.env.PLAYWRIGHT_EXECUTABLE } : {}) });
  try {
    const { page, errors } = await open(browser);
    const mt = cardOf(page, '可改用新翻譯重做');
    const asr = cardOf(page, '語音辨識也變了');

    // 3 buttons when only the translation changed, 2 when ASR changed
    assert.deepEqual(await btnLabels(mt.locator('.subtitle-settings-row')), ['剩下的用新設定', '維持原設定', '已完成的也用新翻譯重做']);
    assert.deepEqual(await btnLabels(asr.locator('.subtitle-settings-row')), ['剩下的用新設定', '維持原設定']);
    assert.match(await mt.locator('.subtitle-settings-row').innerText(), /字幕設定已變更/);

    // summary line
    const bar = page.locator('#subtitlePending');
    assert.match(await bar.innerText(), /2 個處理中的項目設定已變更/);
    assert.deepEqual(await btnLabels(bar), ['全部套用', '全部維持']);

    // tags
    const custom = mt.locator('.chip.custom');
    assert.equal(await custom.innerText(), '本片自訂');
    assert.match(await custom.getAttribute('title'), /多字幕比較：開/);
    assert.equal(await asr.locator('.chip.hold').innerText(), '沿用舊設定');
    assert.equal(await asr.locator('.chip.custom').count(), 0);
    if (process.env.C_SHOT) { await mt.screenshot({ path: process.env.C_SHOT }); }
    const shotCard = await mt.boundingBox();
    assert.ok(shotCard);

    // per-item buttons call the right method/choice
    await mt.locator('button[data-choice="redo_mt"]').click();
    await page.waitForFunction(() => window.__mockCalls.length === 1);
    const first = await page.evaluate(() => window.__mockCalls[0].apply_subtitle_settings);
    assert.equal(first.choice, 'redo_mt');
    assert.ok(first.id);
    assert.equal(await mt.locator('.subtitle-settings-row:not([hidden])').count(), 0, 'row disappears once applied');
    assert.match(await bar.innerText(), /1 個處理中的項目設定已變更/);

    await asr.locator('button[data-choice="keep"]').click();
    await page.waitForFunction(() => window.__mockCalls.length === 2);
    assert.equal(await page.evaluate(() => window.__mockCalls[1].apply_subtitle_settings.choice), 'keep');
    assert.equal(await bar.isVisible(), false, 'summary hides when nothing is pending');

    // queue-wide buttons
    await page.reload();
    await page.waitForSelector('.subtitle-settings-row:not([hidden])');
    await page.locator('#subtitlePending button[data-choice="keep"]').click();
    await page.waitForFunction(() => window.__mockCalls.length === 1);
    assert.deepEqual(await page.evaluate(() => window.__mockCalls[0]), { apply_subtitle_settings_all: { choice: 'keep' } });
    assert.equal(await page.locator('.subtitle-settings-row:not([hidden])').count(), 0);

    assert.deepEqual(errors, []);
    await page.close();
  } finally {
    await browser.close();
  }
  console.log('subtitle settings prompt UI: ok');
})().catch((e) => { console.error(e); process.exit(1); });
