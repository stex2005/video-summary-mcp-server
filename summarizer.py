"""
Core video and image analysis module using GPT-4.1 Vision.
"""

import os
import cv2
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from frame_extractor import extract_keyframes
from encoder import encode_jpeg
from prompt import build_summary_prompt, build_analysis_prompt, build_count_prompt

# Load environment variables
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def summarize_video(video_path, style="short", start_time=None, end_time=None):
    """
    Summarize a video using GPT-4.1 Vision.
    
    Args:
        video_path: Path to the video file
        style: Summary style - "short", "timeline", "detailed", or "technical"
        start_time: Start time in seconds (None for beginning of video)
        end_time: End time in seconds (None for end of video)
    
    Returns:
        Text summary of the video
    
    Raises:
        ValueError: If video cannot be opened or time range is invalid
        RuntimeError: If API call fails
    """
    frames = extract_keyframes(video_path, start_time=start_time, end_time=end_time)
    time_range = ""
    if start_time is not None or end_time is not None:
        start_str = f"{start_time:.1f}s" if start_time is not None else "0s"
        end_str = f"{end_time:.1f}s" if end_time is not None else "end"
        time_range = f" ({start_str} - {end_str})"
    print(f"Extracted {len(frames)} frames{time_range}")

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


def analyze_video_with_prompt(video_path, custom_prompt, start_time=None, end_time=None):
    """
    Analyze a video using GPT-4.1 Vision with a custom prompt/question.
    
    Args:
        video_path: Path to the video file
        custom_prompt: Custom prompt/question to ask about the video
        start_time: Start time in seconds (None for beginning of video)
        end_time: End time in seconds (None for end of video)
    
    Returns:
        Text response to the custom prompt
    
    Raises:
        ValueError: If video cannot be opened or time range is invalid
        RuntimeError: If API call fails
    """
    frames = extract_keyframes(video_path, start_time=start_time, end_time=end_time)
    time_range = ""
    if start_time is not None or end_time is not None:
        start_str = f"{start_time:.1f}s" if start_time is not None else "0s"
        end_str = f"{end_time:.1f}s" if end_time is not None else "end"
        time_range = f" ({start_str} - {end_str})"
    print(f"Extracted {len(frames)} frames{time_range} for custom analysis")

    image_inputs = []
    for f in frames:
        jpeg = encode_jpeg(f)
        image_inputs.append({
            "type": "input_image",
            "image": jpeg
        })

    # Try GPT-4.1 Vision API (Responses API format)
    try:
        response = client.responses.create(
            model="gpt-4.1",
            input=[
                *image_inputs,
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
                            "text": custom_prompt
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


def get_videos():
    """
    List all available video files in the videos directory.
    
    Returns:
        List of video file paths relative to the videos directory
    """
    videos_dir = Path("videos")
    if not videos_dir.exists():
        return []
    
    # Common video extensions
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv', '.m4v', '.3gp'}
    
    video_files = []
    for file in videos_dir.iterdir():
        if file.is_file() and file.suffix.lower() in video_extensions:
            video_files.append(f"videos/{file.name}")
    
    return sorted(video_files)

