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
