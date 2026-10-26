# 發佈檢查表（Inno + uv 路線）

架構與背景見 [docs_multi_gpu_release_plan.md](docs_multi_gpu_release_plan.md)。這份只列發佈當天的步驟。

## 1. 準備

- [ ] 版本號已更新，變更已合併，CI（packaging-checks）通過。
- [ ] 資源清單 `packaging/installer_inputs.json` 的網址、大小、SHA256 與授權說明是最新的。
- [ ] 打上版本 tag（例如 `v0.1.0`）。

## 2. 從乾淨 clone 建置

必須從該 tag 的乾淨 clone 建置：build provenance 需要 git，且 `dirty` 必須是 `false`。

```
git clone https://github.com/life-hack-lab/oh-streamer.git
cd oh-streamer
git checkout v0.1.0
python packaging/build_installer.py installer --out <輸出資料夾>
```

- 需要 Inno Setup 的 `ISCC.exe`（CI 使用 6.7.3）。
- 約 6～8 分鐘；`installer` 會先跑 stage 與 verify-stage。

## 3. 核對產物

- [ ] build manifest：`dirty=false`、revision 等於 tag 指向的 commit、版本與 `uv.lock` 雜湊正確；與前一次建置比對，只有預期的差異。
- [ ] 計算安裝檔 SHA-256（`Get-FileHash -Algorithm SHA256 <安裝檔>`），記到 release notes。
- [ ] 安裝檔檔名為 `OhStreamer-Setup-<版本>.exe`。

## 4. 測試安裝

- [ ] 在 Windows Sandbox 或乾淨的 Windows 帳號安裝，走完第一次開啟（下載執行環境）。
- [ ] 依發佈計畫「發佈前驗證」表格逐項驗證；實驗性版本至少前五項。任一項失敗就不發佈。

## 5. 發佈到 GitHub Release

- [ ] 建立正式 Release，不要勾 pre-release，附上安裝檔。GitHub 的最新版查詢（`releases/latest`）不會回傳 pre-release，app 的「檢查更新」也會略過 pre-release，勾了使用者就收不到更新。預覽版在標題註明，例如「Oh Streamer 0.1.0（預覽版 Preview）」。
- [ ] 附上所含 ffmpeg 與 mpv 對應版本的原始碼（或對應原始碼的取得連結與授權文字），符合 GPL/LGPL 義務。
- [ ] Release notes 包含：變更摘要、安裝檔 SHA-256、需求（Windows 10/11、NVIDIA 驅動 570 以上）、已知限制。
- [ ] Release notes 註明：安裝檔未簽章，SmartScreen 會警告，請點「其他資訊 → 仍要執行」；可用 SHA-256 核對下載。

## 6. 發佈後

- [ ] 在乾淨環境從 Release 下載一次，核對 SHA-256 並安裝。
- [ ] 按「檢查更新」，確認新版資訊與連結正確。
