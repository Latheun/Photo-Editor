@echo off
echo Photo Editor Setup for Windows
echo ===========================
echo.

:: Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python is not installed. Please download and install Python 3.11 from:
    echo https://www.python.org/downloads/windows/
    echo.
    echo Important: Make sure to check "Add Python to PATH" during installation.
    echo.
    echo After installing Python, run this setup script again.
    pause
    exit /b
)

:: Create virtual environment
echo Creating Python virtual environment...
python -m venv photo_editor_env
call photo_editor_env\Scripts\activate.bat

:: Update pip
echo Updating pip...
python -m pip install --upgrade pip

:: Install required packages
echo Installing required packages (this may take a few minutes)...
pip install opencv-python numpy "ultralytics<2.6.0" mediapipe==0.10.9 tk

:: Download YOLO model
echo Downloading YOLO model...
curl -L https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt -o yolov8m.pt

:: Create input and output folders
mkdir input_photos 2>nul
mkdir output_photos 2>nul

echo.
echo Setup completed successfully!
echo.
echo To run the Photo Editor:
echo 1. Double-click on "run_photo_editor.bat"
echo 2. Place your photos in the "input_photos" folder
echo 3. Use the application to process your photos
echo.
echo Processed photos will be saved in the "output_photos" folder.
pause 