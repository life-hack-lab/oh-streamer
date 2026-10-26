# AirPlay / HLS 投放 RTX 超解析

在播放器按 **c** 或點投放圖示，選擇 **投放 RTX 超解析：關**。檢查成功後變成「開」，再選擇 Apple TV 或「僅開啟區網串流」。預設關閉，選擇會保存於播放器設定檔；和本地播放的 RTX 超解析開關分開。

投放中會顯示目前設定，須停止投放後才能切換。每次新投放都有獨立的 HLS 快取，不會混用不同解析度。選單另有「超解析輸出」，點選可切換並保存：

- **2 倍（最高 4K）**（預設）：720p → 1440p、1080p → 4K。
- **升到 4K**：720p → 4K（3 倍），1080p → 4K（2 倍）；較低解析度也會以 4K 為目標，GPU 負擔較高，可自行嘗試。

兩種模式都維持比例，限制在 3840×2160 範圍內；寬螢幕或直式影片不會拉伸成 16:9。輸出選項須開啟投放 RTX 超解析才生效；投放中或檢查中無法切換。已達上限的影片會提示關閉此功能，不做假超解析或縮小後再放大。

## 安裝

需要 Windows、NVIDIA RTX GPU、相容驅動，以及完整的 NVEncC 可攜版（EXE 和 DLL 必須放在一起）。這是選用功能，AMD / macOS 和未安裝依賴的環境仍可使用一般投放。

原始碼版可執行：

```powershell
powershell -ExecutionPolicy Bypass -File tools/install_cast_vsr.ps1
```

安裝腳本從 [NVEncC 官方 9.35 release](https://github.com/rigaya/NVEnc/releases/tag/9.35) 下載並驗證 SHA256，保存在 `tools/nvencc/`。不修改系統 PATH、不修改 NVIDIA 控制面板、不需要系統管理員安裝。已有目錄時不覆寫。另可透過環境變數 `OH_STREAMER_NVENC_C` 指定 NVEncC64.exe 完整路徑，或讓它位於 PATH 中。可攜播放器也支援 EXE 旁的 `tools/nvencc/`；目前打包腳本不會自動加入這個選用依賴。

驗證環境：一台 RTX 50 系列筆電、NVEncC 9.35。9.38 在同機的短片測試曾出現結束異常或逾時，因此此安裝腳本固定使用 9.35。

## 處理流程與限制

已還原影格／已完成影片 → FFmpeg 色彩轉換 → NVEncC `ngx-vsr,vsr-quality=1` + H.264 編碼 → FFmpeg 保留時間戳封裝成 HLS → 接收端。

使用真正的 NVIDIA VSR；不是 bicubic 放大。開啟前以合成影格執行一次 VSR，確認成功輸出及 VSR 濾鏡日誌，失敗不會保存成開啟。檢查在背景進行，不凍結播放器。執行中失敗會停止投放並顯示錯誤，不靜默退回普通放大。

HLS 維持原有的 H.264 High、最高 20 Mbps、AAC 與固定片段邊界。這個開關會使用 NVIDIA 編碼，不受一般播放器的編碼器選擇影響。多增加一段 GPU 工作，尤其即時還原同時跑 4K VSR 時，速度取決於 GPU 和內容；高解析度且複雜的畫面也可能受到 20 Mbps 上限影響。接收端須支援輸出解析度與 H.264 等級；尚未以實體 Apple TV 驗證此新路徑。

GPU 整合測試（會使用實際 NVIDIA GPU，但不連線到電視）：

```powershell
$env:OH_STREAMER_TEST_VSR='1'
python -m unittest discover -s tests -p test_cast_vsr.py -v
```

測試檢查 720p → 4K、1080p → 4K（29.97 fps、無音訊）、320×180 → 640×360（含音訊）、片段時間戳、回頭跳轉及程序清理；不設定環境變數則只跑一般單元測試。另有開關保存、失敗不啟用、投放中鎖定，以及中斷排空不誤判 EOF 的測試。
