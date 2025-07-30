@echo off
echo ================================================
echo   Starting TranslateHub - Sinhala English Translator
echo ================================================

echo Activating virtual environment...
call .venv\Scripts\activate.bat

echo Checking Python environment...
where python
echo.

echo Starting Flask application...
python translator_app.py

echo.
echo Application stopped.
pause
