@echo off
REM NexGen Predictive Delivery Optimizer - Windows Run Script

echo ==========================================
echo NexGen Delivery Optimizer
echo ==========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo.
    echo Please install Python 3.8 or higher from:
    echo https://www.python.org/downloads/
    echo.
    pause
    exit /b 1
)

REM Check if streamlit is installed
python -c "import streamlit" >nul 2>&1
if errorlevel 1 (
    echo Error: Streamlit is not installed
    echo.
    echo Please run setup first:
    echo   pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

REM Check data directory
if not exist "data" (
    echo Warning: data directory not found
    echo Creating data directory...
    mkdir data
    echo Created data directory
    echo.
    echo Please add your CSV files to data before proceeding
    echo.
)

REM Check for orders.csv
if not exist "data\orders.csv" (
    echo Warning: data\orders.csv not found
    echo.
    echo At minimum, you need orders.csv to run the app.
    echo The app will still launch, but you'll need to add data before training models.
    echo.
)

echo Launching NexGen Delivery Optimizer...
echo.
echo The app will open in your browser at: http://localhost:8501
echo.
echo Press Ctrl+C to stop the server
echo.
echo ==========================================
echo.

REM Run streamlit
streamlit run app.py

pause
