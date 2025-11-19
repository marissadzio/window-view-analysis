import torch
import os
import glob
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from pillow_heif import register_heif_opener

# Register HEIF opener for HEIC support
register_heif_opener()

def process_image(image_path, processor, model, device, text="a window."):
    """Process a single image and return results"""
    try:
        # Open image
        image = Image.open(image_path)

        # Check image format and apply rotation accordingly
        if image_path.lower().endswith(('.png', '.heic')):
            # For PNG and HEIC images, don't rotate
            pass  # No rotation for PNG and HEIC
        else:
            # For other formats (JPG, JPEG), apply rotation
            image = image.rotate(270, expand=True)

        # Process with the model
        inputs = processor(images=image, text=text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)

        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            text_threshold=0.3,
            target_sizes=[image.size[::-1]]
        )

        return image, results[0]
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None, None

def save_results(original_image, result, image_name, cropped_dir, detected_dir):
    """Save cropped and detected images"""
    if result is None or len(result["boxes"]) == 0:
        print(f"No windows detected in {image_name}")
        return

    # Convert boxes to pixel coordinates
    boxes = []
    for box in result["boxes"]:
        x1, y1, x2, y2 = box
        # Ensure coordinates are within image bounds and in correct order
        x1 = max(0, min(int(x1), original_image.width))
        y1 = max(0, min(int(y1), original_image.height))
        x2 = max(0, min(int(x2), original_image.width))
        y2 = max(0, min(int(y2), original_image.height))

        # Ensure x1 < x2 and y1 < y2
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1

        boxes.append((x1, y1, x2, y2))

    print(f"Detected {len(boxes)} windows in {image_name}")

    # Calculate bounding box areas
    box_areas = [(x2-x1)*(y2-y1) for x1, y1, x2, y2 in boxes]
    max_area = max(box_areas)
    max_area_index = box_areas.index(max_area)

    # Strategy: If the largest box is significantly larger than others, crop to it
    # Otherwise, crop to all boxes and stitch them together
    largest_box = boxes[max_area_index]
    largest_area = max_area

    # Check if largest box is significantly larger (more than 2x) than others
    should_use_largest_only = all(area < largest_area / 2 for i, area in enumerate(box_areas) if i != max_area_index)

    if should_use_largest_only and len(boxes) > 1:
        print(f"Using largest bounding box for cropping {image_name}")
        x1, y1, x2, y2 = largest_box
        cropped_image = original_image.crop((x1, y1, x2, y2))
        cropped_image.save(os.path.join(cropped_dir, f"cropped_{image_name}"))

        # Also save annotated version
        result_image = original_image.copy()
        draw = ImageDraw.Draw(result_image)
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        result_image.save(os.path.join(detected_dir, f"detected_{image_name}"))

    else:
        print(f"Stitching multiple bounding boxes together for {image_name}")

        # Find the overall bounding box that encompasses all detections
        min_x = min(x1 for x1, y1, x2, y2 in boxes)
        min_y = min(y1 for x1, y1, x2, y2 in boxes)
        max_x = max(x2 for x1, y1, x2, y2 in boxes)
        max_y = max(y2 for x1, y1, x2, y2 in boxes)

        # Add some padding
        padding = 10
        min_x = max(0, min_x - padding)
        min_y = max(0, min_y - padding)
        max_x = min(original_image.width, max_x + padding)
        max_y = min(original_image.height, max_y + padding)

        # Crop to the overall bounding box
        cropped_image = original_image.crop((min_x, min_y, max_x, max_y))
        cropped_image.save(os.path.join(cropped_dir, f"cropped_{image_name}"))

        # Create annotated version showing all boxes
        result_image = original_image.copy()
        draw = ImageDraw.Draw(result_image)

        for i, (box, score, label) in enumerate(zip(result["boxes"], result["scores"], result["labels"])):
            x1, y1, x2, y2 = box
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Draw bounding box
            draw.rectangle([x1, y1, x2, y2], outline="red", width=2)

            # Draw label with score
            label_text = f"{label}: {score:.2f}"
            try:
                font = ImageFont.load_default()
            except:
                font = None

            # Get text size for background
            if font:
                bbox = draw.textbbox((0, 0), label_text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
            else:
                text_width = len(label_text) * 6
                text_height = 12

            # Draw background for text
            draw.rectangle([x1, y1-text_height-2, x1+text_width+4, y1], fill="red")

            # Draw text
            draw.text((x1+2, y1-text_height), label_text, fill="white", font=font)

        result_image.save(os.path.join(detected_dir, f"detected_{image_name}"))

# Main processing
model_id = "IDEA-Research/grounding-dino-base"
device = "cuda" if torch.cuda.is_available() else "cpu"

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

# Create directories
input_dir = "input"
cropped_dir = "output_cropped"
detected_dir = "output_detected"

os.makedirs(input_dir, exist_ok=True)
os.makedirs(cropped_dir, exist_ok=True)
os.makedirs(detected_dir, exist_ok=True)

# Supported image formats
supported_formats = ['*.jpg', '*.jpeg', '*.png', '*.heic', '*.JPG', '*.JPEG', '*.PNG', '*.HEIC']

# Find all images in input directory
image_files = []
for format_pattern in supported_formats:
    image_files.extend(glob.glob(os.path.join(input_dir, format_pattern)))

if not image_files:
    print(f"No images found in {input_dir} directory!")
    print(f"Supported formats: {', '.join(supported_formats)}")
    exit()

print(f"Found {len(image_files)} images to process")

# Process each image
for image_path in image_files:
    image_name = os.path.basename(image_path)
    print(f"\nProcessing: {image_name}")

    original_image, result = process_image(image_path, processor, model, device)

    if original_image is not None:
        save_results(original_image, result, image_name, cropped_dir, detected_dir)
        print(f"Completed: {image_name}")

print(f"\nAll images processed!")
print(f"Cropped images saved in: {cropped_dir}")
print(f"Detected images saved in: {detected_dir}")
