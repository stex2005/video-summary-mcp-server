"""
Frame extraction module for video processing.
"""

import cv2


def extract_keyframes(video_path, interval_sec=2, start_time=None, end_time=None):
    """
    Extract keyframes from a video at regular intervals.
    
    Args:
        video_path: Path to the video file
        interval_sec: Interval in seconds between extracted frames (default: 2)
        start_time: Start time in seconds (None for beginning of video)
        end_time: End time in seconds (None for end of video)
    
    Returns:
        List of frames (numpy arrays)
    
    Raises:
        ValueError: If video file cannot be opened or time range is invalid
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    # Validate and set time range
    if start_time is None:
        start_time = 0.0
    if end_time is None:
        end_time = duration
    
    if start_time < 0:
        raise ValueError(f"Start time cannot be negative: {start_time}")
    if end_time > duration:
        raise ValueError(f"End time ({end_time}s) exceeds video duration ({duration:.2f}s)")
    if start_time >= end_time:
        raise ValueError(f"Start time ({start_time}s) must be less than end time ({end_time}s)")
    
    # Convert times to frame numbers
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    frame_interval = int(fps * interval_sec)

    # Seek to start position
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frames = []
    frame_id = start_frame

    while frame_id <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_id % frame_interval == 0:
            frames.append(frame)
        frame_id += 1

    cap.release()
    return frames

