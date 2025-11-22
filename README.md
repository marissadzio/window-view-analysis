# Window View Analysis

A Python project that automatically detects and crops windows from images using Grounding DINO as a zero-shot object detection model.

## Overview

This project uses the Grounding DINO model to detect windows in images and automatically crop them. It supports multiple image formats (JPG, JPEG, PNG, HEIC) and processes all images in the input directory, saving cropped windows and annotated detection results.

## Requirements

- Python 3.x
- High-performance graphics card (GPU recommended for faster processing, but CPU is supported)

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Place all images you want to process in a folder called `input`

3. Run the script:
   ```bash
   python main.py
   ```

## Output

The script creates two output directories:
- `output_cropped/` - Contains cropped window images
- `output_detected/` - Contains original images with bounding boxes drawn around detected windows

## Supported Image Formats

- JPG/JPEG
- PNG
- HEIC

## Ratio Calculations

After cropping the window images, you can run ratio calculations on the cropped images:

1. **Prepare images**: Fix the orientation of the images to face straight manually by cycling through the images one by one on your device.

2. **Prerequisites**: Ensure you have Conda installed before continuing.

3. **Run the ratio calculations**:
   ```bash
   python ratio_calculations.py
   ```
   Or if using a Jupyter notebook:
   ```bash
   jupyter notebook ratio_calculations.ipynb
   ```

4. **Configure base directory**: The `base_dir` variable (defined near the top of the code) automatically directs to `output_cropped`. If you would like to use a different directory, change it accordingly before running the code.

5. **Select pixels**: Once running, a screen will pop up with directions on how to proceed with selecting the green and blue pixels. Every 15 images there will be a checkpoint save that allows you to run the images in batches. It is recommended to do the whole set of images in one go.

6. **View results**: Once done selecting the images, they will be saved in your chosen `base_dir`. If you would like to see all the data collectively in one Excel file, ignore the checkpoint Excel files and proceed to view the overall one.
