"""
Package the photo editor application into a standalone executable
"""
import os
import shutil
import subprocess
import sys
import site
import platform

def main():
    print("Packaging Photo Editor into a standalone executable...")
    
    # Ensure we're running on Windows
    if platform.system() != "Windows":
        print("This packaging script is designed for Windows only.")
        return
    
    # Create build directory
    if not os.path.exists("build"):
        os.mkdir("build")
    
    # Install PyInstaller if not already installed
    try:
        import PyInstaller
        print("PyInstaller is already installed.")
    except ImportError:
        print("Installing PyInstaller...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])
    
    # Ensure the YOLO model exists
    if not os.path.exists("yolov8m.pt"):
        print("Downloading YOLO model...")
        import urllib.request
        urllib.request.urlretrieve(
            "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt", 
            "yolov8m.pt"
        )
    
    # Copy the YOLO model to the current directory to be included in the package
    shutil.copy("yolov8m.pt", ".")
    
    # Create the spec file for PyInstaller
    spec_content = """
# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['photo_editor_ui.py'],
    pathex=[],
    binaries=[],
    datas=[('yolov8m.pt', '.')],
    hiddenimports=['mediapipe', 'cv2', 'numpy', 'tkinter', 'PIL', 'ultralytics', 'torch'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='PhotoEditor',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='icon.ico',
)
    """
    
    # Create a simple icon for the application
    try:
        from PIL import Image, ImageDraw
        
        # Create a 256x256 image with a blue background
        icon_size = (256, 256)
        icon_img = Image.new('RGB', icon_size, color=(53, 116, 201))
        draw = ImageDraw.Draw(icon_img)
        
        # Draw a white camera icon (simplified)
        draw.rectangle([48, 88, 208, 168], fill=(255, 255, 255))
        draw.ellipse([98, 98, 158, 158], fill=(53, 116, 201))
        
        # Save as ICO file
        icon_img.save("icon.ico")
        print("Created icon.ico")
    except:
        print("Failed to create icon. The executable will use a default icon.")
        # Create an empty spec file without icon reference
        spec_content = spec_content.replace("icon='icon.ico',", "")
    
    # Write the spec file
    with open("PhotoEditor.spec", "w") as f:
        f.write(spec_content.strip())
    
    # Run PyInstaller
    print("Running PyInstaller (this may take several minutes)...")
    subprocess.check_call([
        "pyinstaller", 
        "--clean",
        "PhotoEditor.spec"
    ])
    
    # Create directories in the dist folder
    os.makedirs("dist/PhotoEditor/input_photos", exist_ok=True)
    os.makedirs("dist/PhotoEditor/output_photos", exist_ok=True)
    
    # Copy the quick start guide if it exists
    if os.path.exists("quick_guide_for_users.txt"):
        shutil.copy("quick_guide_for_users.txt", "dist/PhotoEditor/")
        print("Added quick start guide to package")
    
    # Create a simple readme file
    with open("dist/PhotoEditor/README.txt", "w") as f:
        f.write("""PHOTO EDITOR

This application automatically processes photos for official documents.

How to use:
1. Place your photos in the "input_photos" folder
2. Run PhotoEditor.exe
3. Adjust settings if needed and click "Process Photos"
4. Find your processed photos in the "output_photos" folder

For detailed instructions, see the included quick_guide_for_users.txt
        """)
    
    # Create a shortcut to the input_photos folder for easier access
    try:
        import win32com.client
        shell = win32com.client.Dispatch("WScript.Shell")
        shortcut = shell.CreateShortCut("dist/PhotoEditor/Input Folder.lnk")
        shortcut.Targetpath = os.path.join(os.path.abspath("dist/PhotoEditor"), "input_photos")
        shortcut.save()
        
        shortcut = shell.CreateShortCut("dist/PhotoEditor/Output Folder.lnk")
        shortcut.Targetpath = os.path.join(os.path.abspath("dist/PhotoEditor"), "output_photos")
        shortcut.save()
        print("Created shortcuts to input and output folders")
    except:
        print("Could not create shortcuts (pywin32 may not be installed)")
    
    print("\nPackaging complete!")
    print(f"Executable created at: {os.path.abspath('dist/PhotoEditor/PhotoEditor.exe')}")
    print("\nTo distribute the application:")
    print("1. Zip the entire 'dist/PhotoEditor' folder")
    print("2. Share the zip file with users")
    print("3. Users just need to extract the zip and run PhotoEditor.exe")

if __name__ == "__main__":
    main() 