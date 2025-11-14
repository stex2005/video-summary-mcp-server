"""
Core video analysis module using GPT-4.1 Vision.
"""

import os
from openai import OpenAI
from dotenv import load_dotenv
from frame_extractor import extract_keyframes
from encoder import encode_jpeg
from prompt import build_summary_prompt

# Load environment variables
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def summarize_video(video_path, style="short", start_time=None, end_time=None, interval_sec=2, max_width=512):
    """
    Summarize a video using GPT-4.1 Vision.
    
    Args:
        video_path: Path to the video file
        style: Summary style - "short", "timeline", "detailed", or "technical"
        start_time: Start time in seconds (None for beginning of video)
        end_time: End time in seconds (None for end of video)
        interval_sec: Interval in seconds between extracted frames (default: 10, reasonable for cost savings)
        max_width: Maximum frame width in pixels (default: 512, low for cost savings)
    
    Returns:
        Text summary of the video
    
    Raises:
        ValueError: If video cannot be opened or time range is invalid
        RuntimeError: If API call fails
    """
    frames = extract_keyframes(video_path, start_time=start_time, end_time=end_time, interval_sec=interval_sec, max_width=max_width)
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


def analyze_video_with_prompt(video_path, custom_prompt, start_time=None, end_time=None, interval_sec=10, max_width=512):
    """
    Analyze a video using GPT-4.1 Vision with a custom prompt/question.
    
    Args:
        video_path: Path to the video file
        custom_prompt: Custom prompt/question to ask about the video
        start_time: Start time in seconds (None for beginning of video)
        end_time: End time in seconds (None for end of video)
        interval_sec: Interval in seconds between extracted frames (default: 10, reasonable for cost savings)
        max_width: Maximum frame width in pixels (default: 512, low for cost savings)
    
    Returns:
        Text response to the custom prompt
    
    Raises:
        ValueError: If video cannot be opened or time range is invalid
        RuntimeError: If API call fails
    """
    frames = extract_keyframes(video_path, start_time=start_time, end_time=end_time, interval_sec=interval_sec, max_width=max_width)
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

