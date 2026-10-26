// Browser regression: PLAYWRIGHT_MODULE and PLAYWRIGHT_EXECUTABLE can select installed tooling.
const assert = require('node:assert/strict');
const path = require('node:path');
const { pathToFileURL } = require('node:url');
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');

(async () => {
  const browser = await chromium.launch({ headless: true,
    ...(process.env.PLAYWRIGHT_EXECUTABLE ? { executablePath: process.env.PLAYWRIGHT_EXECUTABLE } : {}) });
  try {
    const page = await browser.newPage({ viewport: { width: 1100, height: 1000 } });
    const errors = [];
    page.on('pageerror', (error) => {
      // The existing animated mock thumbnails can briefly have a 0px canvas during skin swaps.
      if (error.message === "Failed to execute 'drawImage' on 'CanvasRenderingContext2D': The image argument is a canvas element with a width or height of 0.") return;
      errors.push(error.message);
    });
    await page.goto(pathToFileURL(path.resolve(__dirname, '../src/oh_streamer/resources/queue_ui/index.html')).href + '?mock&skin=modern');
    await page.locator('#btnSettings').click();
    await page.locator('#stab-subtitles').click();
    await page.locator('#subtitleGroqAdd').waitFor({ state: 'visible' });
    await page.waitForFunction(() => !document.getElementById('subtitleGroqAdd').disabled);
    assert.equal(await page.locator('#subtitleGroqForm').isVisible(), false, 'editor starts closed');
    for (const name of ['主要金鑰', '備援金鑰']) {
      await page.locator('#subtitleGroqAdd').click();
      await page.locator('#subtitleGroqName').fill(name);
      await page.locator('#subtitleGroqKey').fill('fake-demo-' + name);
      await page.locator('#subtitleGroqSave').click();
      await page.waitForFunction(() => document.getElementById('subtitleGroqForm').hidden);
    }
    assert.equal(await page.locator('.subtitle-pool-item').count(), 2);
    assert.match(await page.locator('.subtitle-pool-name').last().innerText(), /備援金鑰/);
    assert.equal(await page.locator('#subtitleGroqKey').inputValue(), '');
    assert.equal(await page.locator('#subtitleGroqAuto').isChecked(), true);
    await page.locator('.subtitle-pool-item').last().getByRole('button', { name: '編輯' }).click();
    assert.equal(await page.locator('#subtitleGroqKey').inputValue(), '', 'saved keys never returned');
    await page.locator('#subtitleGroqName').focus();
    // Mock queue updates every 500ms: an unrelated progress update must not detach the form.
    await page.waitForTimeout(650);
    assert.equal(await page.locator('#subtitleGroqName').evaluate((el) => document.activeElement === el), true);
    await page.locator('#subtitleGroqName').fill('備援已改名');
    await page.locator('#subtitleGroqPriority').selectOption('0');
    await page.locator('#subtitleGroqSave').click();
    await page.waitForFunction(() => document.getElementById('subtitleGroqForm').hidden);
    assert.match(await page.locator('.subtitle-pool-name').first().innerText(), /備援已改名/);
    await page.locator('#subtitleGroqAuto').uncheck();
    await page.waitForFunction(() => !document.getElementById('subtitleGroqAuto').disabled);
    await page.locator('#subtitleGroqAuto').check();
    await page.waitForFunction(() => !document.getElementById('subtitleGroqAuto').disabled);
    const enabled = page.locator('.subtitle-pool-row input[type=checkbox]').first();
    await enabled.uncheck();
    await page.waitForFunction(() => !document.getElementById('subtitleGroqAuto').disabled);
    assert.match(await page.locator('.subtitle-pool-state').first().innerText(), /已停用/);
    await page.locator('.subtitle-pool-item').first().getByRole('button', { name: '編輯' }).click();
    await page.locator('#subtitleGroqKey').fill('temporary-typed-secret');
    await page.locator('#subtitleGroqShowKey').click();
    assert.equal(await page.locator('#subtitleGroqKey').getAttribute('type'), 'text');
    await page.locator('#subtitleGroqCancel').click();
    assert.equal(await page.locator('#subtitleGroqKey').inputValue(), '');
    await page.locator('.subtitle-pool-item').first().getByRole('button', { name: '編輯' }).click();
    assert.equal(await page.locator('#subtitleGroqKey').getAttribute('type'), 'password');
    for (const skin of ['pixel', 'modern']) {
      if (await page.locator('html').getAttribute('data-skin') !== skin) await page.keyboard.press('Control+Shift+U');
      await page.waitForFunction((skin) => document.documentElement.dataset.skin === skin && !document.documentElement.classList.contains('skin-switching'), skin);
      for (const width of [1100, 390]) {
        await page.setViewportSize({ width, height: 1000 });
        const bounds = await page.locator('#subtitleGroqForm').boundingBox();
        assert(bounds && bounds.x >= 0 && bounds.x + bounds.width <= width + 1, `${skin} ${width}: form fits`);
        assert(await page.locator('#subtitleGroqSave').isVisible());
      }
    }
    await page.setViewportSize({ width: 1100, height: 1000 });
    await page.locator('#subtitleGroqDelete').click();
    await page.locator('#modalOk').click();
    await page.waitForFunction(() => document.querySelectorAll('.subtitle-pool-item').length === 1);
    assert.doesNotMatch(await page.locator('#subtitleCloudHome').innerText(), /temporary-typed-secret|組織/);
    assert.deepEqual(errors, []);
    console.log('PASS: pool add/edit/order/toggle/delete, keys stay masked, both skins and mobile');
  } finally { await browser.close(); }
})().catch((error) => { console.error(error); process.exitCode = 1; });
