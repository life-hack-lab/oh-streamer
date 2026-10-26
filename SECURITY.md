# 安全性政策 / Security Policy

## 繁體中文

### 如何回報

請只透過 GitHub 的私密弱點回報：到本儲存庫的 **Security** 分頁 → **Report a vulnerability**。請不要用公開 issue、討論區或其他管道回報弱點。本專案不提供電子郵件收件。

回報時請說明受影響的版本、重現步驟與影響範圍。**請不要附上個人影片的標題、檔名或網址**，也請遮蔽本機路徑；需要示範時請用假資料或自己建立的測試檔。

### 支援的版本

只支援最新的正式發佈版。舊版的問題請先升級後再確認。

### 範圍

下列元件是我們特別關心的攻擊面：

- **區網 HLS 伺服器**（投放到 Apple TV / AirPlay 時使用）：同一區網的其他裝置能否讀到不該讀的檔案、繞過存取限制。
- **127.0.0.1 轉送代理**：是否能被其他主機或網頁（例如 DNS rebinding）濫用，或洩漏自訂 HTTP headers。
- **檢查更新**：只在使用者按下按鈕時查詢 GitHub Releases；任何能讓它導向惡意網址或自動執行檔案的問題。
- **瀏覽器解析器**：使用系統 Edge / Chrome 解析串流網址時的資料外洩或任意程式碼執行。

另外，安裝檔目前沒有程式碼簽章，Windows SmartScreen 會提示警告，這是已知狀況，不屬於弱點。

## English

### How to report

Use GitHub private vulnerability reporting only: open the **Security** tab of this repository, then **Report a vulnerability**. Please do not use public issues or discussions for vulnerabilities. There is no email contact.

Include the affected version, reproduction steps and impact. **Do not include titles, file names or URLs of your personal videos**, and redact local paths. Use dummy data or a test file you made yourself when a demonstration is needed.

### Supported versions

Only the latest release is supported. Please upgrade and re-check before reporting older versions.

### Scope

- **LAN HLS server** (used for casting to Apple TV / AirPlay): other devices on the LAN reading files they should not, or bypassing access limits.
- **127.0.0.1 forwarding proxy**: abuse from other hosts or web pages (for example DNS rebinding), or leaking custom HTTP headers.
- **Update check**: only queries GitHub Releases when the user presses the button; anything that could redirect it to a malicious URL or run a file automatically.
- **Browser resolver**: data leakage or code execution when resolving stream URLs through the system Edge / Chrome.

The installer is currently unsigned and Windows SmartScreen warns about it. This is known and is not a vulnerability.
