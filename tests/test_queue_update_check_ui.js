// Browser regression for the manual "check for updates" row (mock bridge, no network).
// SHOT_DIR (optional) saves screenshots.
const assert = require('node:assert/strict');
const path = require('node:path');
const { pathToFileURL } = require('node:url');
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');

const url = (q) => pathToFileURL(path.resolve(__dirname, '../src/oh_streamer/resources/queue_ui/index.html')).href + '?mock&' + q;

(async () => {
  const browser = await chromium.launch({ headless: true,
    ...(process.env.PLAYWRIGHT_EXECUTABLE ? { executablePath: process.env.PLAYWRIGHT_EXECUTABLE } : {}) });
  try {
    for (const kind of ['update', 'latest', 'no_release', 'error']) {
      const page = await browser.newPage({ viewport: { width: 1100, height: 900 } });
      const errors = [];
      page.on('pageerror', (e) => { if (!/drawImage/.test(e.message)) errors.push(e.message); });
      await page.goto(url('update=' + kind));
      await page.locator('#btnSettings').click();
      await page.locator('#stab-gen').click();
      await page.locator('#updateRow').waitFor({ state: 'visible' });
      assert.match(await page.locator('#updateVersion').innerText(), /0\.1\.0/);
      assert.equal(await page.locator('#updateResult').isVisible(), false);
      await page.locator('#btnCheckUpdate').click();
      assert.equal(await page.locator('#btnCheckUpdate').isDisabled(), true, 'disabled while checking');
      await page.waitForFunction(() => !document.getElementById('btnCheckUpdate').disabled);
      const msg = await page.locator('#updateResult').innerText();
      const go = await page.locator('#btnOpenRelease').isVisible();
      const notes = await page.locator('#updateNotes').isVisible();
      if (kind === 'update') {
        assert.match(msg, /0\.2\.0/);
        assert.equal(go, true);
        assert.equal(notes, true);
        assert.match(await page.locator('#updateNotes').innerText(), /<b>這行不應被當成 HTML<\/b>/);
        assert.equal(await page.locator('#updateNotes b').count(), 0, 'notes are plain text');
      } else {
        assert.equal(go, false); assert.equal(notes, false);
        assert.match(msg, { latest: /最新/, no_release: /還沒有/, error: /無法連線/ }[kind]);
      }
      if (process.env.SHOT_DIR && (kind === 'update' || kind === 'error')) {
        await page.locator('#updateRow').scrollIntoViewIfNeeded();
        await page.screenshot({ path: path.join(process.env.SHOT_DIR, kind === 'update' ? 'C2-update.png' : 'C2-error.png') });
      }
      assert.deepEqual(errors, []);
      await page.close();
    }
    console.log('OK');
  } finally { await browser.close(); }
})().catch((e) => { console.error(e); process.exit(1); });
