"""
Image analysis module using GPT-4.1 Vision.
"""

import os
import cv2
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from encoder import encode_jpeg
from prompt import build_analysis_prompt, build_count_prompt

# Load environment variables
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def summarize_image(image_path, style="short", model="gpt-4o-mini", max_width=512):
    """
    Analyze an image using GPT-4.1 Vision.
    
    Supports common image formats: JPEG, PNG, BMP, TIFF, WebP, PBM, PGM, PPM.
    The image is automatically converted to JPEG for processing.
    
    Args:
        image_path: Path to the image file (supports JPEG, PNG, BMP, TIFF, WebP, etc.)
        style: Analysis style - "short", "detailed", "technical", or "descriptive"
        model: Model to use (default: "gpt-4o-mini" for cost savings, options: "gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-4.1")
        max_width: Maximum frame width in pixels (default: 512, low for cost savings). Lower = cheaper.
    
    Returns:
        Text analysis of the image
    
    Raises:
        ValueError: If image cannot be opened or format is not supported
        RuntimeError: If API call fails
    """
    # Load image using OpenCV (supports multiple formats)
    frame = cv2.imread(image_path)
    if frame is None:
        raise ValueError(
            f"Could not open image: {image_path}. "
            f"Supported formats: JPEG, PNG, BMP, TIFF, WebP, PBM, PGM, PPM"
        )
    
    print(f"Analyzing image: {image_path}")
    
    # Resize image if max_width is specified and image is larger
    original_height, original_width = frame.shape[:2]
    if max_width is not None and original_width > max_width:
        scale = max_width / original_width
        new_width = max_width
        new_height = int(original_height * scale)
        frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    # Encode image as JPEG
    jpeg = encode_jpeg(frame)
    
    prompt = build_analysis_prompt(style)
    
    # Try GPT-4.1 Vision API (Responses API format) - only for gpt-4.1
    try:
        if model != "gpt-4.1":
            raise AttributeError("Responses API only for gpt-4.1")
        response = client.responses.create(
            model=model,
            input=[
                {
                    "type": "input_image",
                    "image": jpeg
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        result = response.output_text
        return f"{result}\n\n[Model used: {model}]"
    except (AttributeError, Exception) as e:
        # Fallback to standard chat.completions API if responses.create doesn't exist
        import base64
        base64_image = base64.b64encode(jpeg).decode('utf-8')
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=1000
        )
        result = response.choices[0].message.content
        return f"{result}\n\n[Model used: {model}]"


def count_items(image_path, object_name, model="gpt-4o-mini", max_width=512):
    """
    Count specific objects in an image using GPT-4.1 Vision.
    
    Supports common image formats: JPEG, PNG, BMP, TIFF, WebP, PBM, PGM, PPM.
    The image is automatically converted to JPEG for processing.
    
    Args:
        image_path: Path to the image file (supports JPEG, PNG, BMP, TIFF, WebP, etc.)
        object_name: Name of the object to count (e.g., "person", "car", "robot")
        model: Model to use (default: "gpt-4o-mini" for cost savings, options: "gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-4.1")
        max_width: Maximum frame width in pixels (default: 512, low for cost savings). Lower = cheaper.
    
    Returns:
        String containing the count and any additional information
    
    Raises:
        ValueError: If image cannot be opened or format is not supported
        RuntimeError: If API call fails
    """
    # Load image using OpenCV (supports multiple formats)
    frame = cv2.imread(image_path)
    if frame is None:
        raise ValueError(
            f"Could not open image: {image_path}. "
            f"Supported formats: JPEG, PNG, BMP, TIFF, WebP, PBM, PGM, PPM"
        )
    
    print(f"Counting {object_name} in image: {image_path}")
    
    # Resize image if max_width is specified and image is larger
    original_height, original_width = frame.shape[:2]
    if max_width is not None and original_width > max_width:
        scale = max_width / original_width
        new_width = max_width
        new_height = int(original_height * scale)
        frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    # Encode image as JPEG
    jpeg = encode_jpeg(frame)
    
    prompt = build_count_prompt(object_name)
    
    # Try GPT-4.1 Vision API (Responses API format) - only for gpt-4.1
    try:
        if model != "gpt-4.1":
            raise AttributeError("Responses API only for gpt-4.1")
        response = client.responses.create(
            model=model,
            input=[
                {
                    "type": "input_image",
                    "image": jpeg
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        result = response.output_text
        return f"{result}\n\n[Model used: {model}]"
    except (AttributeError, Exception) as e:
        # Fallback to standard chat.completions API if responses.create doesn't exist
        import base64
        base64_image = base64.b64encode(jpeg).decode('utf-8')
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=200
        )
        result = response.choices[0].message.content
        return f"{result}\n\n[Model used: {model}]"


def analyze_image_with_prompt(image_path, custom_prompt, model="gpt-4o-mini", max_width=512):
    """
    Analyze an image using GPT-4.1 Vision with a custom prompt.
    
    Supports common image formats: JPEG, PNG, BMP, TIFF, WebP, PBM, PGM, PPM.
    The image is automatically converted to JPEG for processing.
    
    Args:
        image_path: Path to the image file (supports JPEG, PNG, BMP, TIFF, WebP, etc.)
        custom_prompt: Custom prompt/question to ask about the image
        model: Model to use (default: "gpt-4o-mini" for cost savings, options: "gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-4.1")
        max_width: Maximum frame width in pixels (default: 512, low for cost savings). Lower = cheaper.
    
    Returns:
        Text response to the custom prompt
    
    Raises:
        ValueError: If image cannot be opened or format is not supported
        RuntimeError: If API call fails
    """
    # Load image using OpenCV (supports multiple formats)
    frame = cv2.imread(image_path)
    if frame is None:
        raise ValueError(
            f"Could not open image: {image_path}. "
            f"Supported formats: JPEG, PNG, BMP, TIFF, WebP, PBM, PGM, PPM"
        )
    
    print(f"Analyzing image with custom prompt: {image_path}")
    
    # Resize image if max_width is specified and image is larger
    original_height, original_width = frame.shape[:2]
    if max_width is not None and original_width > max_width:
        scale = max_width / original_width
        new_width = max_width
        new_height = int(original_height * scale)
        frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    # Encode image as JPEG
    jpeg = encode_jpeg(frame)
    
    # Try GPT-4.1 Vision API (Responses API format) - only for gpt-4.1
    try:
        if model != "gpt-4.1":
            raise AttributeError("Responses API only for gpt-4.1")
        response = client.responses.create(
            model=model,
            input=[
                {
                    "type": "input_image",
                    "image": jpeg
                },
                {
                    "role": "user",
                    "content": custom_prompt
                }
            ]
        )
        result = response.output_text
        return f"{result}\n\n[Model used: {model}]"
    except (AttributeError, Exception) as e:
        # Fallback to standard chat.completions API if responses.create doesn't exist
        import base64
        base64_image = base64.b64encode(jpeg).decode('utf-8')
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": custom_prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=1000
        )
        result = response.choices[0].message.content
        return f"{result}\n\n[Model used: {model}]"


def get_images():
    """
    List all available image files in the images directory.
    
    Returns:
        List of image file paths relative to the images directory
    """
    images_dir = Path("images")
    if not images_dir.exists():
        return []
    
    # Common image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp', '.gif', '.pbm', '.pgm', '.ppm'}
    
    image_files = []
    for file in images_dir.iterdir():
        if file.is_file() and file.suffix.lower() in image_extensions:
            image_files.append(f"images/{file.name}")
    
    return sorted(image_files)

