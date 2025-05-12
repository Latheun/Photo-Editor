@echo off
echo Starting Photo Editor...
echo.

:: Activate virtual environment
call photo_editor_env\Scripts\activate.bat

:: Run the application
python photo_editor_ui.py

:: Handle any errors
if %errorlevel% neq 0 (
    echo.
    echo Error running Photo Editor. 
    echo If this is the first time running the application, please run setup_windows.bat first.
    pause
)

:: Deactivate virtual environment
call photo_editor_env\Scripts\deactivate.bat 