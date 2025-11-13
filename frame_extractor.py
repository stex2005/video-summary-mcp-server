"""
Frame extraction module for video processing.
"""

import cv2


def extract_keyframes(video_path, interval_sec=2):
    """
    Extract keyframes from a video at regular intervals.
    
    Args:
        video_path: Path to the video file
        interval_sec: Interval in seconds between extracted frames (default: 2)
    
    Returns:
        List of frames (numpy arrays)
    
    Raises:
        ValueError: If video file cannot be opened
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * interval_sec)

    frames = []
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_id % frame_interval == 0:
            frames.append(frame)
        frame_id += 1

    cap.release()
    return frames

