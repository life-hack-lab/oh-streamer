@echo off
REM ---------------------------------------------------------------------------------------------------------------
REM Oh Streamer: main user-facing window for playback and the background queue (queue_app.py), no console window.
REM Source code lives in src\oh_streamer; queue_app.py at this folder is a no-install CLI shim.
REM
REM   Double-click: opens the queue window. If the queue is already running (window open, or hidden in the tray),
REM     this just shows that window and exits (no second copy). Closing the window hides it to the tray; the tray
REM     menu's "Quit" really quits (and stops the running job).
REM   Options (passed through):  --hidden  start in the tray only     --instance NAME  same meaning as the player's
REM     --root DIR  queue folder (default %LOCALAPPDATA%\oh_streamer\queue)
REM   Log: %LOCALAPPDATA%\oh_streamer\logs\queue_app.log (previous run: queue_app.prev.log).
REM   Runs no GPU code itself: the Lada worker is only started while the queue has something to process.
REM
REM Autostart in the tray: press Win+R, type  shell:startup  and put a shortcut to this file there with the
REM   argument --hidden (shortcut Properties > Target: ... "Oh Streamer.bat" --hidden, Run: Minimized).
REM ---------------------------------------------------------------------------------------------------------------
setlocal
set "OH_PY=%~dp0..\lada\.venv\Scripts\pythonw.exe"
if not exist "%OH_PY%" (
    echo Python venv not found: %OH_PY%
    pause
    exit /b 1
)
start "" "%OH_PY%" "%~dp0queue_app.py" %*
