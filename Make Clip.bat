@echo off
setlocal
set "CLIP_PY=%~dp0..\lada\.venv\Scripts\python.exe"
if not exist "%CLIP_PY%" (
    echo Lada Python environment not found: %CLIP_PY%
    pause
    exit /b 1
)
"%CLIP_PY%" "%~dp0make_clip.py" %*
set "CLIP_RESULT=%ERRORLEVEL%"
pause
exit /b %CLIP_RESULT%
