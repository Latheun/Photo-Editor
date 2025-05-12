# Photo Editor

A desktop application that automatically processes photos for official documents, ID cards, passports, and similar uses.

## Features

- **Automatic Image Orientation** - Detects and corrects image orientation regardless of how the photo was taken
- **Face Alignment** - Ensures faces are properly aligned according to ID photo requirements
- **Person Detection** - Automatically finds and crops to the person in the image
- **Standard Sizing** - Resizes photos to standard dimensions for documents
- **Batch Processing** - Process multiple photos at once

## Screenshot

[Add a screenshot of the application here]

## Installation

### For Windows Users (Executable)

1. Download the latest release from the [Releases](https://github.com/yourusername/Photo-Editor/releases) page
2. Extract the zip file to any location
3. Double-click `PhotoEditor.exe` to run the application
4. No installation required!

### For Developers

#### Prerequisites

- Python 3.9-3.11 (recommended: 3.11)
- Git

#### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/Photo-Editor.git
cd Photo-Editor

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download YOLO model
curl -L https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt -o yolov8m.pt
```

## Usage

1. Place photos in the `input_photos` folder
2. Run the application:
   ```bash
   python photo_editor_ui.py
   ```
3. Adjust settings if needed (dimensions, folders)
4. Click "Process Photos"
5. Find processed photos in the `output_photos` folder

## Building the Executable

To build a standalone Windows executable:

1. Ensure all dependencies are installed
2. Run:
   ```bash
   python package_to_exe.py
   ```
3. The executable will be created in `dist/PhotoEditor/`

## Development

### Project Structure

```
Photo-Editor/
├── photo_editor_ui.py   # Main application UI
├── Duzenleyici.py       # Core processing logic
├── requirements.txt     # Dependencies
├── package_to_exe.py    # Script to create executable
├── input_photos/        # Input folder for photos
└── output_photos/       # Output folder for processed photos
```

### Technologies Used

- OpenCV for image processing
- YOLOv8 for person detection
- MediaPipe for face mesh and orientation detection
- Tkinter for user interface
- PyInstaller for packaging

## License

[Choose an appropriate license]

## Acknowledgements

- YOLOv8 by Ultralytics
- MediaPipe by Google
- OpenCV
