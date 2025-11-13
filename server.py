"""
MCP server for video summarization using GPT-4.1 Vision.
"""

from fastmcp import FastMCP
from summarizer import summarize_video as summarize_video_core

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


if __name__ == "__main__":
    mcp.run()
