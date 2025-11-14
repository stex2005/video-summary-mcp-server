"""
MCP server for video summarization and image analysis using GPT-4.1 Vision.
"""

from typing import Optional, Union
from fastmcp import FastMCP
from summarizer import (
    summarize_video as summarize_video_core,
    analyze_video_with_prompt as analyze_video_with_prompt_core,
    analyze_image as analyze_image_core,
    count_items as count_items_core,
    analyze_image_with_prompt as analyze_image_with_prompt_core,
    get_images as get_images_core,
    get_videos as get_videos_core
)

mcp = FastMCP("video-summarizer")


@mcp.tool()
def summarize_video(video_path: str, style: str = "short", start_time: Optional[Union[float, int, str]] = None, end_time: Optional[Union[float, int, str]] = None, interval_sec: Optional[Union[float, int, str]] = None, max_width: Optional[int] = None) -> str:
    """
    Summarize the content of a video using GPT-4.1 Vision.
    
    Args:
        video_path: Local path to the video file
        style: Summary style - "short", "timeline", "detailed", or "technical" (default: "short")
        start_time: Optional start time in seconds (None for beginning of video)
        end_time: Optional end time in seconds (None for end of video)
        interval_sec: Interval in seconds between extracted frames (default: 10, reasonable for cost savings). Higher = cheaper.
        max_width: Maximum frame width in pixels (default: 512, low for cost savings). Lower = cheaper.
    
    Returns:
        Text summary of the video
    """
    # Convert inputs to float if needed (for MCP compatibility)
    if start_time is not None:
        start_time = float(start_time)
    if end_time is not None:
        end_time = float(end_time)
    if interval_sec is not None:
        interval_sec = float(interval_sec)
    
    # Use defaults if not specified
    interval_sec = interval_sec if interval_sec is not None else 10
    max_width = max_width if max_width is not None else 512
    
    return summarize_video_core(video_path, style=style, start_time=start_time, end_time=end_time, interval_sec=interval_sec, max_width=max_width)


@mcp.tool()
def analyze_video_with_prompt(video_path: str, custom_prompt: str, start_time: Optional[Union[float, int, str]] = None, end_time: Optional[Union[float, int, str]] = None, interval_sec: Optional[Union[float, int, str]] = None, max_width: Optional[int] = None) -> str:
    """
    Analyze a video using GPT-4.1 Vision with a custom prompt/question.
    
    This tool allows you to ask any custom question about the video, such as:
    - Counting items in the video
    - Detecting issues or problems
    - Describing specific actions or events
    - Analyzing safety features or equipment
    
    Args:
        video_path: Local path to the video file
        custom_prompt: Custom prompt or question to ask about the video (e.g., "Count how many boxes are visible", "Are there any safety issues?")
        start_time: Optional start time in seconds (None for beginning of video). Can be number or string.
        end_time: Optional end time in seconds (None for end of video). Can be number or string.
        interval_sec: Interval in seconds between extracted frames (default: 10, reasonable for cost savings). Higher = cheaper.
        max_width: Maximum frame width in pixels (default: 512, low for cost savings). Lower = cheaper.
    
    Returns:
        Text response to the custom prompt
    """
    # Convert inputs to float if needed (for MCP compatibility)
    if start_time is not None:
        start_time = float(start_time)
    if end_time is not None:
        end_time = float(end_time)
    if interval_sec is not None:
        interval_sec = float(interval_sec)
    
    # Use defaults if not specified
    interval_sec = interval_sec if interval_sec is not None else 10
    max_width = max_width if max_width is not None else 512
    
    return analyze_video_with_prompt_core(video_path, custom_prompt, start_time=start_time, end_time=end_time, interval_sec=interval_sec, max_width=max_width)


@mcp.tool()
def analyze_image(image_path: str, style: str = "short") -> str:
    """
    Analyze the content of an image using GPT-4.1 Vision.
    
    Supports common image formats: JPEG, PNG, BMP, TIFF, WebP, PBM, PGM, PPM.
    
    Args:
        image_path: Local path to the image file (supports JPEG, PNG, BMP, TIFF, WebP, etc.)
        style: Analysis style - "short", "detailed", "technical", or "descriptive" (default: "short")
    
    Returns:
        Text analysis of the image
    """
    return analyze_image_core(image_path, style=style)


@mcp.tool()
def count_items(image_path: str, object_name: str) -> str:
    """
    Count specific objects in an image using GPT-4.1 Vision.
    
    Supports common image formats: JPEG, PNG, BMP, TIFF, WebP, PBM, PGM, PPM.
    
    Args:
        image_path: Local path to the image file (supports JPEG, PNG, BMP, TIFF, WebP, etc.)
        object_name: Name of the object to count (e.g., "person", "car", "robot", "box")
    
    Returns:
        String containing the count of the specified objects
    """
    return count_items_core(image_path, object_name)


@mcp.tool()
def analyze_image_with_prompt(image_path: str, custom_prompt: str) -> str:
    """
    Analyze an image using GPT-4.1 Vision with a custom prompt/question.
    
    Supports common image formats: JPEG, PNG, BMP, TIFF, WebP, PBM, PGM, PPM.
    This tool allows you to ask any custom question about the image.
    
    Args:
        image_path: Local path to the image file (supports JPEG, PNG, BMP, TIFF, WebP, etc.)
        custom_prompt: Custom prompt or question to ask about the image (e.g., "What color is the robot?", "Describe the safety features visible")
    
    Returns:
        Text response to the custom prompt
    """
    return analyze_image_with_prompt_core(image_path, custom_prompt)


@mcp.tool()
def get_images() -> list[str]:
    """
    List all available image files in the images directory.
    
    Returns:
        List of image file paths (e.g., ["images/photo1.jpg", "images/photo2.png"])
    """
    return get_images_core()


@mcp.tool()
def get_videos() -> list[str]:
    """
    List all available video files in the videos directory.
    
    Returns:
        List of video file paths (e.g., ["videos/video1.mp4", "videos/video2.avi"])
    """
    return get_videos_core()


if __name__ == "__main__":
    mcp.run()
