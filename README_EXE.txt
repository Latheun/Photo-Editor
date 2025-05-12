BUILDING AN EXECUTABLE VERSION OF PHOTO EDITOR
===========================================

Creating a standalone executable (.exe) version of the Photo Editor allows distributing the 
application to Windows users without requiring them to install Python or any dependencies.

BUILDING THE EXECUTABLE
----------------------

1. Prerequisites (for the developer only):
   - Windows 10 or 11
   - Python 3.9 to 3.11 installed
   - All required packages installed (run setup_windows.bat first)

2. Run the packaging script:
   ```
   python package_to_exe.py
   ```

3. The script will:
   - Install PyInstaller if needed
   - Download the YOLO model if not present
   - Create an application icon
   - Package everything into a standalone executable
   - Create the necessary folders structure

4. When complete, you'll find the executable in:
   ```
   dist/PhotoEditor/PhotoEditor.exe
   ```

DISTRIBUTING THE APPLICATION
---------------------------

1. Zip the entire "dist/PhotoEditor" folder
   - This folder contains the executable and necessary folders

2. Share the zip file with your users

3. Instructions for users:
   - Extract the zip file to any location
   - Double-click PhotoEditor.exe to run the application
   - Place photos in the "input_photos" folder
   - Processed photos will appear in the "output_photos" folder

ADVANTAGES OF EXECUTABLE VERSION
------------------------------

- No Python installation required for end users
- No need to install any packages or dependencies
- Simple double-click to run
- All components packaged in a single executable
- Works on any Windows computer (Windows 7 or newer)

IMPORTANT NOTES
-------------

1. The executable size will be large (around 300-500 MB) because it includes:
   - Python interpreter
   - All dependencies (OpenCV, PyTorch, YOLO, MediaPipe)
   - The ML models for face detection and person detection

2. First launch may be slow as Windows validates the executable

3. Some antivirus software might flag the executable initially as suspicious
   (this is common for PyInstaller-packaged applications) 