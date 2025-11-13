"""
Core video summarization and image analysis module using GPT-4.1 Vision.
"""

import os
import cv2
from openai import OpenAI
from dotenv import load_dotenv
from frame_extractor import extract_keyframes
from encoder import encode_jpeg
from prompt import build_summary_prompt, build_analysis_prompt, build_count_prompt

# Load environment variables
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def summarize_video(video_path, style="short"):
    """
    Summarize a video using GPT-4.1 Vision.
    
    Args:
        video_path: Path to the video file
        style: Summary style - "short", "timeline", "detailed", or "technical"
    
    Returns:
        Text summary of the video
    
    Raises:
        ValueError: If video cannot be opened
        RuntimeError: If API call fails
    """
    frames = extract_keyframes(video_path)
    print(f"Extracted {len(frames)} frames")

    image_inputs = []
    for f in frames:
        jpeg = encode_jpeg(f)
        image_inputs.append({
            "type": "input_image",
            "image": jpeg
        })

    prompt = build_summary_prompt(style)

    # Try GPT-4.1 Vision API (Responses API format)
    try:
        response = client.responses.create(
            model="gpt-4.1",
            input=[
                *image_inputs,
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        return response.output_text
    except (AttributeError, Exception) as e:
        # Fallback to standard chat.completions API if responses.create doesn't exist
        import base64
        image_content = []
        for f in frames:
            jpeg = encode_jpeg(f)
            base64_image = base64.b64encode(jpeg).decode('utf-8')
            image_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            })
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        *image_content
                    ]
                }
            ],
            max_tokens=1000
        )
        return response.choices[0].message.content


def analyze_image(image_path, style="short"):
    """
    Analyze an image using GPT-4.1 Vision.
    
    Supports common image formats: JPEG, PNG, BMP, TIFF, WebP, PBM, PGM, PPM.
    The image is automatically converted to JPEG for processing.
    
    Args:
        image_path: Path to the image file (supports JPEG, PNG, BMP, TIFF, WebP, etc.)
        style: Analysis style - "short", "detailed", "technical", or "descriptive"
    
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
    
    # Encode image as JPEG
    jpeg = encode_jpeg(frame)
    
    prompt = build_analysis_prompt(style)
    
    # Try GPT-4.1 Vision API (Responses API format)
    try:
        response = client.responses.create(
            model="gpt-4.1",
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
        return response.output_text
    except (AttributeError, Exception) as e:
        # Fallback to standard chat.completions API if responses.create doesn't exist
        import base64
        base64_image = base64.b64encode(jpeg).decode('utf-8')
        
        response = client.chat.completions.create(
            model="gpt-4o",
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
        return response.choices[0].message.content


def count_items(image_path, object_name):
    """
    Count specific objects in an image using GPT-4.1 Vision.
    
    Supports common image formats: JPEG, PNG, BMP, TIFF, WebP, PBM, PGM, PPM.
    The image is automatically converted to JPEG for processing.
    
    Args:
        image_path: Path to the image file (supports JPEG, PNG, BMP, TIFF, WebP, etc.)
        object_name: Name of the object to count (e.g., "person", "car", "robot")
    
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
    
    # Encode image as JPEG
    jpeg = encode_jpeg(frame)
    
    prompt = build_count_prompt(object_name)
    
    # Try GPT-4.1 Vision API (Responses API format)
    try:
        response = client.responses.create(
            model="gpt-4.1",
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
        return response.output_text
    except (AttributeError, Exception) as e:
        # Fallback to standard chat.completions API if responses.create doesn't exist
        import base64
        base64_image = base64.b64encode(jpeg).decode('utf-8')
        
        response = client.chat.completions.create(
            model="gpt-4o",
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
        return response.choices[0].message.content


def analyze_image_with_prompt(image_path, custom_prompt):
    """
    Analyze an image using GPT-4.1 Vision with a custom prompt.
    
    Supports common image formats: JPEG, PNG, BMP, TIFF, WebP, PBM, PGM, PPM.
    The image is automatically converted to JPEG for processing.
    
    Args:
        image_path: Path to the image file (supports JPEG, PNG, BMP, TIFF, WebP, etc.)
        custom_prompt: Custom prompt/question to ask about the image
    
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
    
    # Encode image as JPEG
    jpeg = encode_jpeg(frame)
    
    # Try GPT-4.1 Vision API (Responses API format)
    try:
        response = client.responses.create(
            model="gpt-4.1",
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
        return response.output_text
    except (AttributeError, Exception) as e:
        # Fallback to standard chat.completions API if responses.create doesn't exist
        import base64
        base64_image = base64.b64encode(jpeg).decode('utf-8')
        
        response = client.chat.completions.create(
            model="gpt-4o",
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
        return response.choices[0].message.content

