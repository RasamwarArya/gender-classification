@echo off
echo Starting Gender Classification Web Interface...
echo.
echo Server will be available at: http://localhost:5000
echo Press Ctrl+C to stop the server
echo.
cd /d %~dp0
python app.py

