"""
MCP server for video summarization and image analysis using GPT-4.1 Vision.
"""

from fastmcp import FastMCP
from summarizer import summarize_video as summarize_video_core, analyze_image as analyze_image_core, count_items as count_items_core, analyze_image_with_prompt as analyze_image_with_prompt_core

mcp = FastMCP("video-summarizer")


@mcp.tool()
def summarize_video(video_path: str, style: str = "short") -> str:
    """
    Summarize the content of a video using GPT-4.1 Vision.
    
    Args:
        video_path: Local path to the video file
        style: Summary style - "short", "timeline", "detailed", or "technical" (default: "short")
    
    Returns:
        Text summary of the video
    """
    return summarize_video_core(video_path, style=style)


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


if __name__ == "__main__":
    mcp.run()
