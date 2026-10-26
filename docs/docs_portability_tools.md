# 跨平台預驗收工具

本文說明跨平台預驗收工具（診斷、驗收、benchmark、RIFE 探測）的用法。Windows 通過的結果不代表 Mac 已通過；Mac 上仍需做下面的硬體與桌面驗收。

## Windows 執行方式

在 repo 根目錄開 PowerShell，指定專案資料夾與 Lada 環境（`<專案資料夾>` 是 repo 的上一層，`lada/` 與 repo 並排）：

```powershell
$env:OH_STREAMER_PROJECT_ROOT = '<專案資料夾>'
$python = '<專案資料夾>/lada/.venv/Scripts/python.exe'
& $python portability_doctor.py --probe-encoders --output artifacts/doctor.json
& $python portability_acceptance.py --output-dir artifacts/acceptance --device cuda:0 --with-models --timeout 180
```

預設驗收包含診斷、合成素材、編碼、分段／合併及 mpv IPC／seek。`--with-models` 加入修復、佇列與 benchmark；RIFE 須明確加入 `--checks diagnostic,fixture,encode,segments,ipc,restore,queue,benchmark,rife`。每個命令有逾時限制，log 與產物保存在指定目錄。未安裝模型而要求模型測試時會失敗，不能算完整通過。

修正失敗原因後，可在同一 OS／架構重跑失敗項目：

```powershell
& $python portability_acceptance.py --output-dir artifacts/retry --retry-failed artifacts/acceptance/acceptance.json
```

重跑報告記錄 `not_retested`；不要把它當成一次全項重新驗收。Windows 報告不能拿到 Mac 當作重跑基準。

## 在 Mac 上

先照 [Mac 測試包說明](docs_borrow_mac.md) 準備固定版本的上游、套件 patch 及權重。在測試包展開後、包含 `lada/` 與 `streamer/` 的目錄執行：

```sh
lada/.venv/bin/python streamer/portability_doctor.py --device mps --probe-encoders --require-mps --require-videotoolbox --output preflight/doctor.json
lada/.venv/bin/python streamer/portability_acceptance.py --output-dir preflight/basic --device mps --require-mps --require-videotoolbox --timeout 180
lada/.venv/bin/python streamer/portability_acceptance.py --output-dir preflight/models --device mps --require-mps --require-videotoolbox --with-models --timeout 300
```

以上完整環境命令適用 Apple Silicon。Intel Mac 不能只改成 `--device cpu` 就照此安裝：本專案固定 torch 2.9.1，而官方 macOS x86_64 套件支援止於 2.2.x（[官方說明](https://docs.pytorch.org/blog/pytorch2-2/)）。先用獨立 Python 環境測不依賴模型的項目；不要直接降級既有 Lada 環境。完整模型、佇列修復與 .app 的 torch 相容性須另外研究。

Intel Mac 可先安裝 ffmpeg／mpv、Pillow，使用合成素材及 `--checks fixture,encode,segments,ipc --device cpu` 做媒體／IPC 驗收；此配置不含 doctor 的完整模型環境檢查，通過不代表可執行修復。Cocoa、剪貼簿、選單列及防睡眠可逐項獨立測試，完整 GUI 啟動仍須處理其模型依賴。VideoToolbox 依機型／codec 實測，明確要求時不得由軟體 fallback 掩蓋。

模型預驗收使用小型合成素材，只驗證管線與模型能否執行。實際畫質與速度另用 4–10 秒、含典型修復場景的短片（自行合法取得）。`--input` 會讓佇列處理整個輸入，應先裁短；不要直接用長影片做冒煙測試。

```sh
lada/.venv/bin/python streamer/benchmarks/bench_portability.py short-clip.mp4 --device mps --duration 4 --clip-lengths 30,60 --models v4-fast --encoder null --output preflight/bench-mps.json
lada/.venv/bin/python streamer/probe_rife.py --device mps --size 64x64 --fp16 off --iterations 1 --timeout 120 --json --output preflight/rife-mps.json
```

benchmark 預設不編碼；可指定實際編碼器比較端到端成本。模型載入、首幀、管線時間及總子程序時間分開記錄。MPS 記憶體是取樣值，不等同 CUDA 峰值；合成素材的 fps 不能推算修復密集影片效能。RIFE 探測成功也不會解除 Mac 播放／佇列的功能閘門，需再驗證真實工作負載與精度。

手動檢查 Finder 雙擊、Cocoa 佇列視窗、剪貼簿／拖放、選單列、seek 後音畫同步、取消工作後防睡眠解除、重啟續作、Apple TV 投放。最後才依 [打包說明](../packaging/README_macos.md) 在 Mac 建置 `.app` 並檢查乾淨機器、dylib 與簽章。

## 測試包與報告

### Intel Mac（無獨立顯卡）範例

以一台 macOS 15、無獨立顯卡的 Intel Mac 為例。此類機器可先作 macOS 媒體與系統整合測試；不要求 MPS，也不安裝完整固定版 Lada 模型環境。VideoToolbox 與 MPS 是不同功能，無獨顯不代表無法硬體編碼；H.264／HEVC 能力各自實測。

#### 從 GitHub pull 後執行

在 Mac 的儲存庫目錄更新 `main`（先保存自己的未提交修改）：

```sh
git switch main
git pull --ff-only origin main
```

以下命令直接在儲存庫根目錄執行，不需要建立 sibling `lada/`，也不需要另下載借機 zip。先準備 Python 3.12 或更新版，並確認 `ffmpeg`、`ffprobe`、`mpv` 可從 PATH 執行：

```sh
python3 -m venv .venv-intel
.venv-intel/bin/python -m pip install Pillow psutil
.venv-intel/bin/python portability_acceptance.py --output-dir artifacts/intel-media --checks fixture,encode,segments,ipc --device cpu --encoder libx264
mkdir -p artifacts/intel-media
ffmpeg -hide_banner -y -f lavfi -i testsrc2=size=320x180:rate=30 -t 2 -c:v h264_videotoolbox -allow_sw 0 artifacts/intel-media/intel-h264.mp4
ffmpeg -hide_banner -y -f lavfi -i testsrc2=size=320x180:rate=30 -t 2 -c:v hevc_videotoolbox -allow_sw 0 artifacts/intel-media/intel-hevc.mp4
```

保留 `artifacts/intel-media/acceptance.json`、logs 與硬體編碼的終端輸出，後續用來判定哪些 Mac 項目已通過。`artifacts/` 與獨立環境不提交 Git；下方則是使用 zip 的另一種目錄配置。

在解壓縮後包含 `streamer/` 的目錄，以已安裝的 Python 3.12 或更新版建立獨立環境；先確保 `ffmpeg`、`ffprobe`、`mpv` 在 PATH：

```sh
python3 -m venv .venv-intel
.venv-intel/bin/python -m pip install Pillow psutil
.venv-intel/bin/python streamer/portability_acceptance.py --output-dir preflight/intel-media --checks fixture,encode,segments,ipc --device cpu --encoder libx264
```

上述只驗證非模型流程。不要加 `--with-models`、`--require-mps` 或執行目前要求 torch 2.9.1 的 `setup_macos.command`。先用 libx264 排除硬體編碼問題，再單獨要求實際硬體編碼（`-allow_sw 0` 禁止軟體 fallback）：

```sh
ffmpeg -hide_banner -y -f lavfi -i testsrc2=size=320x180:rate=30 -t 2 -c:v h264_videotoolbox -allow_sw 0 preflight/intel-h264.mp4
ffmpeg -hide_banner -y -f lavfi -i testsrc2=size=320x180:rate=30 -t 2 -c:v hevc_videotoolbox -allow_sw 0 preflight/intel-hevc.mp4
```

保留成功／失敗 log，不能以編碼器名稱列在 `ffmpeg -encoders` 中當作可用證據。其次測 POSIX IPC／鎖／程序清理，以及獨立 Cocoa 視窗、剪貼簿、防睡眠與選單列。完整播放器與佇列視窗目前可能被模型依賴阻擋，須進一步隔離 UI 測試入口。Intel 不替代 Apple Silicon 的 MPS、arm64 dylib、模型效能及完整 .app 驗收。

在 Windows 上建立並驗證測試包：

```powershell
& $python borrow_mac_bundle.py --output artifacts/borrow-mac-preflight.zip --include-tests
& $python borrow_mac_bundle.py --verify artifacts/borrow-mac-preflight.zip
```

包內只有 allowlist 中的程式、公開文件／資源與測試，以及版本／hash 清單；不帶模型、venv、私人影片、設定、log 或 credentials。先驗證 zip 再解壓縮。報告可能包含你指定的本機路徑與工具錯誤訊息，分享前自行查看。
