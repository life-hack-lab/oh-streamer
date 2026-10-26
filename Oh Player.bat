@echo off
REM ---------------------------------------------------------------------------------------------------------------
REM Oh Player: developer-only launcher for direct player diagnostics (oh_player.py --player), no console window. The main user-facing entry is Oh Streamer.
REM Users open videos from the main app (queue); the player window itself cannot open videos (no paste, no drag & drop).
REM Source code lives in src\oh_streamer; oh_player.py at this folder is a no-install CLI shim.
REM
REM   Double-click: opens an idle mpv window ("open videos from the main app"); the models load in the background.
REM   With an argument (file path or URL), development only: plays it. If the player is already running, the
REM     argument is handed to the running player and this exits (no second window).
REM   Log: %LOCALAPPDATA%\oh_streamer\logs\oh_player.log (previous run: oh_player.prev.log).
REM
REM Developer diagnostic shortcut: right-click this file > Send to > Desktop (create shortcut). Optionally, in the shortcut's
REM   Properties set "Run: Minimized" so the brief cmd window does not flash, and pick an icon.
REM   Alternative without any cmd window: create a shortcut whose target is
REM     "<project>\lada\.venv\Scripts\pythonw.exe" "<project>\streamer\oh_player.py" --player
REM   (replace <project> with the full path of the project folder).
REM
REM Developer convenience: press Win+R, type  shell:sendto  and put a shortcut to this file there. Selecting
REM   a video > Send to > Oh Player then opens it in the running player (or starts one).
REM   Note: cmd.exe splits command-line arguments at &, so URLs containing & need quoting.
REM ---------------------------------------------------------------------------------------------------------------
setlocal
set "OH_PY=%~dp0..\lada\.venv\Scripts\pythonw.exe"
if not exist "%OH_PY%" (
    echo Python venv not found: %OH_PY%
    pause
    exit /b 1
)
start "" "%OH_PY%" "%~dp0oh_player.py" --player %*
