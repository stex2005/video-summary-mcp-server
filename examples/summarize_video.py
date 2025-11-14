"""
CLI runner for video summarization.
"""

import sys
from video_analysis import summarize_video


def main():
    """Main entry point for CLI usage."""
    if len(sys.argv) < 2:
        print("Usage: python main.py <video_path> [style]")
        print("Styles: short, timeline, detailed, technical")
        print("Example: python main.py videos/demo.mp4 timeline")
        sys.exit(1)
    
    video_path = sys.argv[1]
    style = sys.argv[2] if len(sys.argv) > 2 else "short"
    
    try:
        summary = summarize_video(video_path, style=style)
        print("\n" + "=" * 50)
        print("VIDEO SUMMARY")
        print("=" * 50)
        print(summary)
        print("=" * 50)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

