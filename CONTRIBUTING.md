# 參與貢獻 / Contributing

## 繁體中文

歡迎回報問題與提出改進。請先看 [README.md](README.md) 與 [AGENTS.md](AGENTS.md)。

### 開發環境（目前的實際狀況）

目前的開發設定預期有並排的工作區：本儲存庫旁邊要有 `../lada`（上游 Lada 的 checkout，含 `.venv`）與 `../third_party`（RIFE 等第三方資源）。這套設定尚未整理成一鍵可重現，外部貢獻者可能無法完整重現，這是已知的缺口；可重現的 uv 開發環境在規劃中。

在那之前，任何人至少可以只用標準函式庫跑 CI 同一組離線測試（不需要 venv、網路或 GPU），內容包含安裝檔輸入與組裝、Inno 腳本約定、更新檢查、打包資源與進入點、build provenance。完整清單見 [.github/workflows/packaging-checks.yml](.github/workflows/packaging-checks.yml)。

### 執行測試

完整測試（需要上述開發環境）：

```powershell
..\lada\.venv\Scripts\python.exe run_tests.py
```

只跑單一模組，不需要 Lada 環境：

```powershell
$env:PYTHONPATH = "src"
python -m unittest tests.test_update_check tests.test_build_provenance
```

### 程式規則

- 產品名稱與技術識別碼（`oh_streamer`、`oh_*`、`OH_*`、`OH_STREAMER_*` 等）的規則請看 [AGENTS.md](AGENTS.md)：不要把產品名稱寫死在使用者看得到的字串，要走 branding API（Python 的 `oh_streamer.branding`、前端的 `BRAND_NAME` / `BRAND_LOGO`）。
- 不要重新命名既有的技術識別碼與檔案內的標籤格式，否則會破壞既有安裝與成品檔。
- 使用者看到的文字使用繁體中文（zh-TW）。
- 變更行為時請一併補測試。

### macOS 需要幫忙

macOS 路線是實驗性的，尚未在 Apple Silicon 上驗證。歡迎有 Mac 的人測試並回報，見 [docs/docs_macos_setup.md](docs/docs_macos_setup.md)。

### 隱私：貼日誌前請先遮蔽

日誌、診斷包與截圖可能含有影片標題、網址、檔名與本機路徑（含你的使用者名稱）。貼到 issue 或 PR 前，請先遮蔽路徑、網址與影片標題。

### Pull request

請描述動機與測試方式，並盡量讓一個 PR 只做一件事。

## English

Bug reports and improvements are welcome. Read [README.md](README.md) and [AGENTS.md](AGENTS.md) first.

### Dev environment (honest status)

The current dev setup expects sibling checkouts: `../lada` (the upstream Lada checkout with its `.venv`) and `../third_party` (RIFE and other third-party assets) next to this repository. It is not yet reproducible with one command, so external contributors may not be able to fully reproduce it. A reproducible uv-based dev setup is planned.

Until then, anyone can run the same offline tests as CI with only the standard library (no venv, network or GPU): installer inputs and staging, Inno script contract, update check, packaged resources and entry points, build provenance. See [.github/workflows/packaging-checks.yml](.github/workflows/packaging-checks.yml) for the exact list.

### Running tests

Full suite (needs the dev environment above):

```powershell
..\lada\.venv\Scripts\python.exe run_tests.py
```

Single modules, no Lada environment needed:

```powershell
$env:PYTHONPATH = "src"
python -m unittest tests.test_update_check tests.test_build_provenance
```

### Code rules

- Branding and technical identifier rules (`oh_streamer`, `oh_*`, `OH_*`, `OH_STREAMER_*`, ...) are in [AGENTS.md](AGENTS.md): never hard-code the product name in user-visible strings, use the branding API (`oh_streamer.branding` in Python, `BRAND_NAME` / `BRAND_LOGO` in the UI).
- Do not rename existing technical identifiers or on-disk tag formats; that would break existing installs and output files.
- User-visible text is Traditional Chinese (zh-TW).
- Add tests when you change behavior.

### macOS help wanted

The macOS route is experimental and unverified on Apple Silicon. Testers are welcome; see [docs/docs_macos_setup.md](docs/docs_macos_setup.md).

### Privacy: redact before posting logs

Logs, diagnostic bundles and screenshots can contain video titles, URLs, file names and local paths (including your user name). Redact paths, URLs and video titles before posting them in an issue or PR.

### Pull requests

Explain the motivation and how you tested it, and keep each PR focused on one thing.
