# Oh Streamer

[**繁體中文**](README.md) | [**English**](README_EN.md)

---

Oh Streamer 是建立在 [Lada](https://codeberg.org/ladaapp/lada) 去馬賽克修復模型之上的影片播放器：在 mpv 中（近）即時播放本機影片或串流網址（m3u8 / mp4，支援自訂 HTTP headers），包含音訊；也能把影片加入背景佇列慢慢完整處理。具備進度跳轉（Seeking）、畫質選擇選單、進階設定選單、進階資訊面板、常駐播放器視窗（由主程式開啟影片）、本機 AI 字幕，以及投放至 Apple TV（HLS + AirPlay）。

> **預覽版**：目前是第一個公開預覽版（0.1.0）。功能可以使用，但仍在快速演進，可能有粗糙之處或不相容的變動。已在 Windows 11 與 NVIDIA RTX 顯示卡（NVENC）上完成測試；本專案為獨立專案，非 Lada 官方組件。

**下載**：到 [GitHub Releases](https://github.com/life-hack-lab/oh-streamer/releases) 取得 Windows 安裝檔 `OhStreamer-Setup-<版本>.exe`。需要 NVIDIA 顯示卡（GTX 16 系列或 RTX 20～50 系列，驅動 570 以上）；安裝檔沒有程式碼簽章，Windows SmartScreen 可能擋下，詳見下方[安裝](#安裝windows)。

AirPlay 配對可直接在播放器完成：按投放按鈕（或 `c`），選擇「配對 裝置名稱」，使用選單中的數字輸入 Apple TV 顯示的 PIN，再選「確認 PIN」。配對後選「投放到 裝置名稱」。選單也提供刪除一位、取消配對及重新配對；等待 PIN 超過兩分鐘會逾時，可重新嘗試。不需要開啟 terminal。

## 安裝（Windows）

**需求**：Windows 10／11 64 位元（目前只在 Windows 11 測試）；NVIDIA GTX 16 系列或 RTX 20～50 系列顯示卡，驅動 570 以上（AMD、Intel 顯示卡之後才支援）；首次安裝約需 9 GB 硬碟空間與穩定的網路。

1. 從 [GitHub Releases](https://github.com/life-hack-lab/oh-streamer/releases) 下載 `OhStreamer-Setup-<版本>.exe` 並執行。不需要系統管理員權限，只為目前使用者安裝，預設位置是 `%LOCALAPPDATA%\Programs\oh_streamer`，可在安裝程式改。安裝檔沒有程式碼簽章，Windows SmartScreen 擋下時請點「其他資訊 → 仍要執行」。
2. 第一次開啟會進入設定畫面：自動偵測顯示卡並建議版本，按下安裝後下載執行環境（約 4 GB，其中 torch 約 3.2 GB）。torch 是單一大檔，下載中斷時要整個重下，其他已下載的套件會保留。
3. 完成後自動開啟主程式。要用本機 AI 字幕時，到「設定 → 字幕 → 本機」下載本機字幕模型（預設約 12.8 GB，可取消、可續傳）；只用雲端字幕不必下載。模型來源與授權見下方[字幕模型與授權](#字幕模型與授權)。

- **修復與診斷**：開始選單的「Oh Streamer 修復與診斷」可回到設定畫面，重裝、切換版本或匯出診斷包（回報問題時附上）。
- **更新**：設定裡的「檢查更新」會查詢 GitHub Releases，有新版時開啟下載頁；下載新版安裝檔直接覆蓋安裝即可，設定與佇列會保留，執行環境沒變時不會重下 torch。
- **解除安裝**：從「設定 → 應用程式 → 已安裝的應用程式」解除。會分別詢問是否刪除執行環境與下載快取（約 5 GB）、已下載的字幕模型、使用者資料（佇列與設定），預設都保留，之後重裝不必重下。
- 使用者資料放在 `%LOCALAPPDATA%\oh_streamer`，與程式資料夾分開。

## macOS（實驗性）

macOS 版目前是**實驗性、尚未驗證**：程式已加入 MPS／VideoToolbox 分支與平台介面，並通過 Windows 回歸測試，但沒有發佈 Mac 安裝檔，也還沒有在 Apple Silicon 實機上完整驗證。歡迎有 Mac 的人協助測試與回報。請參閱 [Mac 安裝說明](docs/docs_macos_setup.md)與 [macOS 打包說明](packaging/README_macos.md)。

## 使用方式

安裝版從開始選單開啟「Oh Streamer」即可。從原始碼開發時，Windows 的主入口是儲存庫根目錄的 `Oh Streamer.bat`，macOS 為 `Oh Streamer.command`。

主視窗是統一視窗：貼上網址、cURL 或本機檔案路徑後，按「串流播放」會在播放器中開啟，按「加入佇列」或 Enter 則加入背景處理；Shift+Enter 可換行。串流播放不會先下載並保存完整影片，但仍需要初始緩衝與 AI 修復啟動時間。播放器的播放設定與背景佇列的處理設定分開保存。

### 字幕（字幕圖示按鈕或 `u`，即時生效）

播放控制列的字幕圖示按鈕位於齒輪左側，以方框內兩行短線表示，可選字幕軌、顯示／隱藏字幕、調整大小（50%～200%），以及每次提早／延後 0.1 秒或重設時間偏移。選擇字幕軌會恢復顯示；沒有字幕軌時選單會顯示提示。方向鍵移動、Enter 選取、Esc 返回／關閉。這些設定作用於本次本機播放，不寫入全域設定；投放時請使用接收裝置的字幕設定。產生 AI 字幕仍在處理佇列設定。

投放前先選好字幕並開啟顯示；AirPlay／HLS 會攜帶目前選取的外掛 SRT、ASS、WebVTT 或內嵌文字字幕，並套用當時的時間偏移。雙字幕模式會投放主字幕。接收裝置可在字幕選單選擇「字幕」。投放中若要換字幕或修改時間偏移，請停止後重新投放。字幕大小、字型由接收端控制，不保留 ASS 特效；圖片字幕目前不支援，轉換失敗會顯示提示，影片仍可投放。

### 播放顯示選項（進階設定選單 `s`，即時生效）

- **RTX 超解析度**（預設開；`--vsr on|off`）：以 NVIDIA RTX Video Super Resolution 將影片放大到螢幕解析度（mpv `d3d11vpp` 濾鏡，最多 4 倍）。需 RTX 顯示卡、mpv 以 d3d11va 硬體解碼，並在 NVIDIA App 開啟「RTX 影片增強 → 超解析度」。目前狀態可在進階資訊面板（`i`）查看。
- **平滑播放**（預設開；`--smooth-playback on|off`）：依螢幕更新率重新取樣播放節奏（mpv `video-sync=display-resample` + `interpolation`、`tscale=oversample`），減少頓挫；這不是補幀。

兩者只影響本機 mpv 播放，不影響投放至 Apple TV 的串流。

投放另有獨立的 **投放 RTX 超解析** 開關（按 `c` 開啟投放選單，預設關閉）：在電腦端使用 NVIDIA VSR，可選擇「2 倍（最高 4K）」或「升到 4K」（如 720p → 4K）後再傳送至 Apple TV 或 HLS 接收端。需 Windows RTX 與選用的 NVEncC；停止投放後才能切換。安裝及限制見 [投放超解析說明](docs/docs_cast_vsr.md)。

### 修復幀率（進階設定選單 `s`）

**修復幀率**（預設原始；`--restoration-fps original|30`）：選擇「最高 30fps」會在 Lada 偵測與修復前等間隔取樣，減少運算量。60→30、59.94→29.97 fps；30fps 以下保持原樣，更高幀率以整數間隔取樣到不超過 30fps。搭配「插幀：自動」，60fps 來源可用 30fps 修復再補到 60fps；GPU 沒有餘裕時沿用重複幀退路。音訊及影片時間不變，變更時從目前位置重新開始；投放期間需先停止投放才能變更。資訊面板顯示來源與修復幀率。此設定用於即時修復播放，背景佇列處理沿用原有設定。

### 插幀（RIFE，進階設定選單 `s`，變更時從目前位置重新開始）

**插幀（RIFE）**（預設關；`--interpolation off|auto`）：設為「自動」時輸出改為修復幀率的 2 倍（例如 30→60 fps）。每兩個修復後的畫面之間，若處理領先播放夠多，就以 RIFE v4.6（`third_party/rife`，fp16 + CUDA graph；高度 > 720 時以半解析度估算光流）補一張中間畫面，否則重複前一張——GPU 跟不上時只是暫時回到原本的流暢度，不會因此卡頓。

- 判斷依據：已輸出的畫面領先 mpv 播放位置（並以 mpv 實際快取的長度為上限）的秒數，扣掉前方已偵測到、尚未修復的馬賽克片段所需的修復時間；領先 ≥ 6 秒開始補幀、< 3 秒停止。開始播放／seek 後在開始播放前一律不補幀，所以不會拖慢 seek。
- 每張補幀的 RIFE 耗時（近 15 次中位數）超過一個來源影格時間的 90% 時暫停補幀（例如 4K 來源或其他程式占用 GPU），15 秒後再試。
- 場景切換（32×32 縮圖 SSIM < 0.2）時重複前一張，不會混合兩個鏡頭。
- 重複的畫面會在編碼前被丟掉（可變幀率），所以沒在補幀時，編碼器與 mpv（解碼、RTX 超解析度）的負擔和關閉時相同；補幀時則要多處理一倍的畫面。
- 修復幀率超過 40 fps（如以原始幀率修復 60 fps 影片）或影格率不明時維持原幀率；投放到 Apple TV 不補幀，維持選定的修復幀率。
- 進階資訊面板（`i`）顯示「插幀：自動 62%（近 10 秒補幀比例）」與目前 RIFE 每張的耗時、狀態。
- RIFE 模型約占 0.3–0.7 GB VRAM，在第一次播放（每種解析度一次）時於背景載入。輸出到檔案（`--output <檔案>`）時以模擬的播放器決定何時補幀。

### 對照原始畫面（`compare`，預設關；設定選單，變更時從目前位置重新開始）

開啟後串流會多一條未修復的原始畫面（另有修復前的版本於插幀時），按鍵：`o` 按住顯示原始畫面、`Shift+o` 切換上一個階段、`[` `]` 調整目前顯示的階段、`w` 分割線對照（左：原始，右：處理後；按住滑鼠左鍵拖曳或 `Ctrl+←` `Ctrl+→` 移動分割線，`w` 或 `Esc` 離開）。分割線模式需要軟體複製解碼，進入時會暫時關閉 RTX 超解析度並顯示提示，進入約停頓 0.25 秒，離開後自動恢復；需要至少兩條畫面（開啟 `compare`）才有作用，換片或重新載入時自動離開。會多占用一個 NVENC 編碼與 VRAM。

### 背景處理佇列（`a`）

背景處理佇列在 `Oh Streamer.bat` 的統一視窗中（需 `pywebview`、`pystray`；關閉視窗會縮到系統匣繼續處理，系統匣右鍵「結束」才真正結束）。按「加入佇列」或 Enter 可加入網址 / cURL / 本機檔案；也可用「加入檔案」或拖放。支援的 HLS 隨選影片可邊下載邊處理，其他網址先下載原始檔；一次一支在背景完整修復（每約 4 秒存一段，暫停或關機後從中斷的段落接著做），完成後接成 `output.mkv` 並刪除原始檔。佇列設定：暫存上限（滿了先刪已看過、未保留、最久沒用的成品，還不夠就暫停並提示）、編碼器（預設 HEVC）、插幀（預設每格補幀，≤ 40 fps 來源輸出 2 倍幀率）、保存對照畫面（預設關）、處理中防止睡眠、全部完成後關機。每支可 🔒 保留、匯出 mp4、清除登入資訊。暫存位置：`%LOCALAPPDATA%\oh_streamer\queue`。

播放中按 `a`（或進階設定選單 `s` 的最後一列「加入處理佇列」）會把目前的影片連同目前的畫質與修復相關設定加入背景處理佇列程式（需要先開啟佇列程式，否則顯示「處理佇列未開啟」；成功顯示「已加入處理佇列」）。

- 播放器即時修復時會通知佇列暫停（每 10 秒回報一次，停止修復或關閉播放器後恢復），不會搶 GPU。
- 開啟網址或檔案時會先問佇列：已處理完成的直接播成品，顯示「播放已處理的版本」，不再即時修復。
- **已處理模式**（佇列的「播放」，或 僅供開發／內部使用的命令列： `oh_player.py --player <檔案> --processed`）：成品是普通影片檔，不經過 Lada，跳轉、快取、RTX 超解析度、平滑播放都和一般播放相同；畫質選單與需要重新修復的設定（模型、修復時間窗、插幀等）只顯示「已處理的影片」；可從投放選單投放成品。成品含 `oh:*` 對照軌時，`o` `[` `]` `w` 照常可用，否則顯示「對照未啟用」。播到 90% 或結尾時通知佇列「已看過」。
- **邊處理邊看**（佇列中還沒完成的項目按「播放」）：已處理的段落直接播（不重新編碼，畫質與成品相同），背景處理照常進行；進度條上的綠色範圍是已處理的部分。播到或跳到未處理的地方，背景處理會跳過去先做那裡，畫面顯示「處理中」等幾秒；跳過的部分之後自動補完，全部完成才合併成品。插幀設為「每格補幀」時，背景處理快被播放追上（領先不到 10 秒）會暫時改成自動插幀以跟上，領先回到 20 秒再恢復；這些段落之後會以全部插幀重做，成品仍是完整品質。還在下載的項目按「播放」會先記住播放要求，來源可讀後接上同一份背景處理成果，不另外開啟原網址。支援的 HLS 來源不需等待整部下載完，下載與處理進度會分別顯示；支援條件與恢復方式見 [下載與處理共用來源](docs/docs_queue_overlap.md)。

### 投放已處理的影片

已處理完成的影片也可從播放器投放選單投放；會使用成品的處理後軌道，無需重新進行 AI 處理。停止投放會回到接收端的播放位置。

「邊處理邊看」也可從同一個投放選單投放。投放直接讀取背景佇列已完成的處理後片段，不再跑一次 AI；只做接收端所需的 H.264／AAC 轉換。遇到尚未處理的片段時，等待並要求佇列優先處理該處；Apple TV 的播放位置與暫停狀態會回報給背景佇列。投放期間保留片段與來源音訊，即使背景處理已完成也不會提早刪除；停止投放後從接收端位置恢復本機播放。

AirPlay 投放時，可在本地播放器拖曳進度條或按方向鍵直接跳轉，使用播放按鈕或空白鍵暫停／繼續。控制透過 AirPlay 的 MRP 遙控通道送到 Apple TV，保留目前投放連線；進度和暫停狀態由接收端回報。跳到尚未處理的片段可能需要緩衝。接收端不支援遙控或指令失敗時會顯示錯誤，不會改成重新投放。純 HLS 區網串流仍需在接收端操作。

## 可選的外部解析器

針對 yt-dlp 無法處理的網頁網址，可接入自訂解析腳本。播放器優先載入平台使用者資料目錄中的 `resolvers.json`，若沒有則使用套件資源 `resolvers.default.json`；`resolvers.example.json` 是自訂設定範例。比對網頁網址中的子字串（`match`）後執行對應命令（`command`）。將命令的第一個參數設為 `"python"` 會自動使用目前的 Python 環境；相對腳本路徑以所用 resolver 設定檔所在目錄解析，絕對路徑維持指定位置。命令需將 `{"url": ..., "headers": {...}, "title": ...}` 格式的 JSON 輸出至 stdout。

預設已內建通用瀏覽器解析器（套件模組 `oh_streamer.sources.resolvers.resolve_browser`，source、wheel 與 frozen 啟動方式皆可用；`resolvers.default.json` 以 `match: "*"` 套用於所有網頁網址）。它以系統瀏覽器開啟影片頁，從播放器狀態與網路回應找出 HLS 播放清單。如需使用，只需安裝 playwright：
```bash
uv pip install playwright     # 或: pip install playwright
```
*（無需執行 `playwright install chromium` —— 程式會自動調用 Windows 系統中已安裝的 Microsoft Edge / Chrome）。*

直接把影片頁網址貼入播放器或佇列即可；既有自訂 resolver 設定仍優先，沒有符合的自訂設定時才使用內建瀏覽器解析器。瀏覽器 resolver 的行為與限制見 [瀏覽器 resolver](docs/docs_browser_resolver.md)。

## 擷取速度測試用短影片

雙擊 `Make Clip.bat`，依序輸入影片連結、開始時間、片段時長；或從儲存庫根目錄執行：

```powershell
..\lada\.venv\Scripts\python.exe make_clip.py "影片連結" --start 00:05:00 --duration 30 --quality 720p -o clips/test_720p.mp4
```

支援內建瀏覽器解析器、其他 yt-dlp 可解析的影片頁、直接影片／m3u8 與本機檔案。時間接受秒數、`MM:SS`、`HH:MM:SS`，可帶小數。省略參數會互動詢問；預設最高畫質，輸出至 `clips/`。使用 H.264 CRF 18／AAC 精確裁切，保留解析度與幀率，沒有執行 Lada 修復。需要 ffmpeg／ffprobe 位於 PATH。既有輸出不覆寫，除非加 `--overwrite`。

同名 `.json` 記錄來源、裁切時間、實際時長、尺寸／幀率與準備耗時，方便比較速度；準備耗時包含網站解析與裁切，不是 Lada 處理速度。若片段超出結尾會顯示實際時長，完全超出則失敗且不留下空白 MP4。

## 使用者資料

設定、佇列、日誌、AirPlay 配對密鑰及 `resolvers.json` 都放在平台使用者資料目錄：Windows 為 `%LOCALAPPDATA%\oh_streamer`，macOS 為 `~/Library/Application Support/oh_streamer`，Linux 為 `${XDG_DATA_HOME:-~/.local/share}/oh_streamer`。日誌位於此目錄下的 `logs/`。`--settings-file` 指定的自訂設定路徑會照指定值使用。

## 字幕模型與授權

本機 AI 字幕（日文轉繁體中文）所需的模型**由使用者在 App 內按需下載，不隨本專案散布**。請注意：

- Sakura 翻譯模型採用 CC BY-NC-SA 4.0 授權（僅限非商業使用）。
- 預設的語音辨識（ASR）模型，其模型卡沒有宣告授權條款。
- 請自行確認所下載模型的授權是否符合你的使用情境。

各元件與模型的來源及授權彙整見 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

## 開發

> 這一節以維護者為主。目前的開發環境預期有並排的工作區：本儲存庫旁邊要有 `../lada`（上游 Lada 的 checkout，含 `.venv` 與模型權重）與 `../third_party`（RIFE 等第三方資源）。這套設定尚未整理成一鍵可重現，外部貢獻者可能無法完整重現；可重現的 uv 開發環境在規劃中。在那之前，任何人都可以只用標準函式庫執行 CI 的離線測試，詳見 [CONTRIBUTING.md](CONTRIBUTING.md)。命名規則見 [AGENTS.md](AGENTS.md)。

應用程式套件位於 `src/oh_streamer/`，依播放器、佇列、處理、來源與平台服務分組。儲存庫根目錄的 `oh_player.py`、`queue_app.py` 等是相容啟動轉接，會自動加入 `src/`，因此開發及啟動不需要先安裝套件（不需 `pip install -e .`）。文件、測試與 benchmark 分別放在 `docs/`、`tests/`、`benchmarks/`。Lua、佇列 Web UI 與 resolver 預設／範例 JSON 收在 `src/oh_streamer/resources/`。

開發用的外部依賴：系統 PATH 中的 ffmpeg／ffprobe（需支援 NVENC）、mpv（放置於 `tools\mpv\mpv.exe` 或加入 PATH）、yt-dlp；投放需要 `pyatv`；進階資訊面板需要 `nvidia-ml-py`。

```powershell
# 命令列直接播放（僅供開發／診斷）：
..\lada\.venv\Scripts\python.exe oh_player.py <file-or-url> [選項]
```

`Oh Player.bat` 是僅供開發與診斷的直接播放器啟動器；一般使用者應從 Oh Streamer 開啟影片。

### 執行測試

從儲存庫根目錄執行（macOS 使用 `../lada/.venv/bin/python run_tests.py`）：

```powershell
..\lada\.venv\Scripts\python.exe run_tests.py
```

此入口會自動定位儲存庫並探索 `tests/`。CI 會用標準函式庫跑一組離線測試（清單見 `.github/workflows/`），例如：

```powershell
$env:PYTHONPATH = "src"
python -m unittest tests.test_update_check tests.test_build_provenance
```

字幕測試需要含 OpenCC 的獨立字幕環境（`opencc-python-reimplemented`，列於 `pyproject.toml` 的 `subtitles` 選用依賴）：

```powershell
$env:PYTHONPATH = "$PWD;$PWD/src"
python -m unittest discover -s tests -p "test_subtitle*.py"
```

既有 `lada/.venv` 沒有 OpenCC，因此在該環境跑全套測試會有兩個字幕 CLI 測試因 `OpenCC is required for s2twp conversion` 報錯。這是缺少測試依賴，不是字幕邏輯退步。

安裝檔用 `python packaging/build_installer.py installer --out build-installer/out` 產生（Inno Setup，約 6～8 分鐘）；發佈流程見 [發佈檢查表](docs/docs_release_checklist.md)。

## 免責聲明

- 本專案與 Apple、NVIDIA、Groq、Microsoft 沒有任何隸屬或背書關係，相關商標屬各自擁有者。
- 本專案建立在上游 [Lada](https://codeberg.org/ladaapp/lada) 之上，但並非 Lada 官方組件，與 Lada 專案沒有隸屬關係。
- 不支援也不會處理受 DRM 保護的內容。
- 使用者須自行負責所處理的內容，並遵守所在地的法律。

## 授權條款

AGPL-3.0-only，Copyright (C) 2026 life-hack-lab。詳見 [LICENSE](LICENSE)；第三方元件與模型的授權見 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。本專案建立在同樣採 AGPL-3.0 的 Lada 之上。

## 相關連結

- 回報問題與資安通報：[SECURITY.md](SECURITY.md)
- 參與貢獻：[CONTRIBUTING.md](CONTRIBUTING.md)
- 維護者的發佈流程：[docs/docs_release_checklist.md](docs/docs_release_checklist.md)
