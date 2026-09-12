@echo off
rem launcher for windows.
rem
rem   transcribe.bat "lecture.mp4"
rem   transcribe.bat D:\records --language ru --num-speakers 2
rem
rem the interpreter is looked up in this order: TRANSCRIBE_PYTHON, a .venv or venv inside
rem the project, the system python.
setlocal
rem utf-8 console: the transcript may contain any language, cp866 would mangle it.
chcp 65001 >nul

set "PY="
if defined TRANSCRIBE_PYTHON set "PY=%TRANSCRIBE_PYTHON%"
if not defined PY if exist "%~dp0.venv\Scripts\python.exe" set "PY=%~dp0.venv\Scripts\python.exe"
if not defined PY if exist "%~dp0venv\Scripts\python.exe" set "PY=%~dp0venv\Scripts\python.exe"
if not defined PY set "PY=python"

rem a plain "no module named whisper" ten frames deep is a poor way to learn that the wrong
rem interpreter was picked, so the dependencies are checked here while it is still obvious.
"%PY%" -c "import whisper, numpy" >nul 2>nul
if errorlevel 1 (
    echo %PY% has no dependencies of this project installed.
    echo   either activate the environment they live in,
    echo   or point TRANSCRIBE_PYTHON at its interpreter:
    echo     set TRANSCRIBE_PYTHON=C:\path\to\env\Scripts\python.exe
    echo   setx writes it permanently, but only NEW terminals see it -
    echo   reopen this one, or use plain set for the current session.
    exit /b 1
)

"%PY%" -u "%~dp0transcribe.py" %*
set RC=%ERRORLEVEL%
echo.
echo [transcribe.bat] exit code: %RC%
endlocal & exit /b %RC%
