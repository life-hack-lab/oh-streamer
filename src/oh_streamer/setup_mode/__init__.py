"""首次開啟的設定模式（Python 後端）。

``python -m oh_streamer.setup_mode`` 會開一個 pywebview 視窗，載入 ``resources/setup_ui/index.html``，
前端透過 ``window.pywebview.api`` 呼叫 ``bridge.SetupApi``：偵測顯示卡、安裝基礎環境與 GPU 版 torch、
自我檢查、匯出診斷包、啟動主程式。

- ``bridge``：``SetupApi``，不 import webview，可以直接單元測試。
- ``__main__``：建立視窗並啟動；唯一 import webview 的地方。需要環境變數 ``OH_STREAMER_INSTALL_ROOT``（launch.pyw 會設定）。
"""
