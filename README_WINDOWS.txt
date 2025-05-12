PHOTO EDITOR FOR WINDOWS
======================

This application automatically processes photos to prepare them for official documents, ID cards, 
or passports by:
- Detecting and correcting image orientation
- Finding the person in the photo
- Adjusting the photo to standard dimensions
- Properly aligning faces to follow guidelines

INSTALLATION INSTRUCTIONS
------------------------

1. PREREQUISITES:
   - Windows 7 or newer
   - Internet connection for initial setup

2. SETUP:
   - Double-click "setup_windows.bat"
   - If Python is not installed, the script will guide you to download and install it
   - After installing Python (if needed), run "setup_windows.bat" again to complete the setup
   - The setup will create a virtual environment and install all necessary packages

3. RUNNING THE APPLICATION:
   - Double-click "run_photo_editor.bat"
   - The Photo Editor application will start

USING THE APPLICATION
--------------------

1. Place your photos in the "input_photos" folder

2. In the application:
   - You can change the input and output folders if needed
   - Adjust the target image size if necessary (default is 83x109 pixels)
   - Click "Process Photos"

3. All processed photos will be saved to the "output_photos" folder

TROUBLESHOOTING
--------------

If you encounter any issues:

1. Make sure you ran "setup_windows.bat" before trying to run the application
2. Check that Python is properly installed with the "Add Python to PATH" option selected
3. If the application crashes, try running it again
4. For persistent problems, try deleting the "photo_editor_env" folder and running setup again

SYSTEM REQUIREMENTS
-----------------

- Windows 7/8/10/11
- Python 3.9 to 3.11 (3.11 recommended)
- At least 4GB RAM
- 1GB free disk space
- Internet connection for setup 