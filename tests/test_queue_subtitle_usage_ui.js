// Browser regression for the Settings -> 字幕 -> 雲端用量 panel. PLAYWRIGHT_MODULE / PLAYWRIGHT_EXECUTABLE select tooling.
const assert = require('node:assert/strict');
const path = require('node:path');
const { pathToFileURL } = require('node:url');
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');

const NOW = 1750000000;
const HOUR0 = Math.floor(NOW / 3600) * 3600;
const snapshot = (warning) => ({
  version: 1, generated_at: NOW,
  providers: [{
    provider: 'groq', connection_id: 'c1',
    models: [
      { model: 'whisper-large-v3', kind: 'asr', requests: { limit: 2000, remaining: 1640, reset_in_s: 43 },
        tokens: { limit: null, remaining: null, reset_in_s: null },
        requests_minute: { used: 2, limit: 20 }, tokens_day: { used: 0, limit: null, reset_in_s: null },
        audio: { hour: { used_s: 1800, limit_s: 7200 }, day: { used_s: 3600, limit_s: 28800 }, limits_verified: false },
        limits_source: 'mixed', warning, updated_at: NOW },
      { model: 'llama-3.3-70b', kind: 'chat', requests: { limit: 1000, remaining: 900, reset_in_s: 100 },
        tokens: { limit: 12000, remaining: 2160, reset_in_s: 7 },
        requests_minute: { used: 5, limit: 30 }, tokens_day: { used: 25000, limit: 200000, reset_in_s: null },
        limits_source: 'mixed', warning: 'ok', updated_at: NOW },
    ],
    series: { granularity: 'hour', points: [0, 1, 2].map((i) => ({ t: HOUR0 - i * 3600, tokens: 1000 * (i + 1), audio_s: 60 * i, requests: i + 1 })) },
  }],
});

(async () => {
  const browser = await chromium.launch({ headless: true,
    ...(process.env.PLAYWRIGHT_EXECUTABLE ? { executablePath: process.env.PLAYWRIGHT_EXECUTABLE } : {}) });
  try {
    const page = await browser.newPage({ viewport: { width: 1100, height: 1000 } });
    const errors = [];
    page.on('pageerror', (e) => {
      if (e.message.startsWith("Failed to execute 'drawImage'")) return;
      errors.push(e.message);
    });
    const url = pathToFileURL(path.resolve(__dirname, '../src/oh_streamer/resources/queue_ui/index.html')).href;
    for (const skin of ['modern', 'pixel']) {
      await page.goto(`${url}?mock&skin=${skin}`);
      // pure helpers
      const h = await page.evaluate(() => {
        const u = window.__usageUi;
        return {
          pct: [u.remainingPct(1000, 180), u.remainingPct(0, 0), u.remainingPct(100, null), u.remainingPct(100, 500), u.usedPct(30, 120)],
          warn: [u.warnClass('low'), u.warnClass('critical'), u.warnClass('ok'), u.warnClass('???'), u.warnClass(undefined)],
          worst: [u.worstWarning([{ warning: 'ok' }, { warning: 'low' }]), u.worstWarning([{ warning: 'low' }, { warning: 'critical' }]), u.worstWarning([]), u.worstWarning([{ warning: 'unknown' }])],
          reset: [u.fmtResetIn(43), u.fmtResetIn(0.2), u.fmtResetIn(600), u.fmtResetIn(null)],
          count: [u.fmtCount(null), u.fmtCount(950), u.fmtCount(12000), u.fmtCount(2500000)],
          secs: [u.fmtSeconds(45), u.fmtSeconds(1800), u.fmtSeconds(7200)],
        };
      });
      assert.deepEqual(h.pct, [18, null, null, 100, 25]);
      assert.deepEqual(h.warn, ['low', 'critical', 'ok', 'unknown', 'unknown']);
      assert.deepEqual(h.worst, ['low', 'critical', 'ok', 'ok']);
      assert.deepEqual(h.reset, ['約 43 秒後重置', '即將重置', '約 10 分鐘後重置', '重置時間未知']);
      assert.deepEqual(h.count, ['—', '950', '12K', '2.5M']);
      assert.deepEqual(h.secs, ['45 秒', '30 分', '2 小時']);
      const sv = await page.evaluate((s) => window.__usageUi.seriesView(s, 24).length + ':' + window.__usageUi.seriesView({ providers: [] }, 24).length, snapshot('ok'));
      assert.equal(sv, '24:0');

      // missing bridge -> empty state, no exception
      await page.locator('#btnSettings').click();
      await page.locator('#stab-subtitles').click();
      await page.evaluate(() => { delete window.ohStreamer.subtitleUsage; });
      await page.evaluate(() => window.__usageUiRefresh());
      await page.locator('#subtitleMode').selectOption('cloud');
      await page.locator('#usageSection summary').waitFor({ state: 'visible' });
      await page.locator('#usageSection summary').click();
      await page.evaluate(() => window.__usageUiRefresh());
      assert.equal(await page.locator('#usageEmpty').isVisible(), true);
      assert.match(await page.locator('#usageEmpty').innerText(), /尚無用量資料/);

      // erroring bridge -> same empty state
      await page.evaluate(() => { window.ohStreamer.subtitleUsage = () => Promise.reject(new Error('boom')); });
      await page.evaluate(() => window.__usageUiRefresh());
      assert.equal(await page.locator('#usageEmpty').isVisible(), true);

      // data with a low warning
      await page.evaluate((s) => { window.ohStreamer.subtitleUsage = () => Promise.resolve(s); }, snapshot('low'));
      await page.evaluate(() => window.__usageUiRefresh());
      await page.locator('.usage-model').first().waitFor();
      assert.equal(await page.locator('.usage-model').count(), 2);
      assert.equal(await page.locator('#usageEmpty').isVisible(), false);
      assert.equal(await page.locator('#usageChip').isVisible(), true);
      assert.match(await page.locator('#usageChip').innerText(), /偏低/);
      assert.match(await page.locator('.usage-model').first().innerText(), /剩餘 82%.*約 43 秒後重置/);
      assert.match(await page.locator('.usage-model').nth(1).innerText(), /剩餘 18%/);
      assert.match(await page.locator('.usage-model').first().innerText(), /今日請求\(RPD\)/);
      assert.match(await page.locator('.usage-model').first().innerText(), /每分鐘請求\(RPM\)/);
      assert.match(await page.locator('.usage-model').nth(1).innerText(), /今日 token\(TPD\)/);
      assert.match(await page.locator('.usage-model').nth(1).innerText(), /每分鐘 token\(TPM\)/);
      assert.equal(await page.locator('#usageAsrNote').isVisible(), true);
      assert.equal(await page.locator('.usage-model').first().locator('.usage-fill.low').count() > 0, true);
      assert.equal(await page.locator('.usage-tokbar').count(), 24);
      assert.equal(await page.locator('.usage-audioline').count(), 1);
      await page.locator('#usageShowAudio').uncheck();
      assert.equal(await page.locator('.usage-audioline').count(), 0);
      await page.locator('#usageRange7d').click();
      assert.equal(await page.locator('.usage-tokbar').count(), 168);
      assert.equal(await page.locator('#usageRange7d').getAttribute('aria-pressed'), 'true');
      assert.match(await page.locator('#usageChart').getAttribute('aria-label'), /7 天/);
      assert.equal(await page.locator('#usageTable tr').count(), 169);

      // critical + polling stops when hidden
      await page.evaluate((s) => { window.ohStreamer.subtitleUsage = () => { window.__calls = (window.__calls || 0) + 1; return Promise.resolve(s); }; }, snapshot('critical'));
      await page.evaluate(() => window.__usageUiRefresh());
      assert.match(await page.locator('#usageChip').innerText(), /即將耗盡/);
      await page.locator('#btnCloseSettings').click();
      const before = await page.evaluate(() => window.__calls);
      await page.waitForTimeout(300);
      assert.equal(await page.evaluate(() => window.__calls), before);
    }
    assert.deepEqual(errors, []);
    console.log('usage ui ok');
  } finally {
    await browser.close();
  }
})().catch((e) => { console.error(e); process.exit(1); });
