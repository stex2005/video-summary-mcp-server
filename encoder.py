"""
Frame encoding module for image compression.
"""

import cv2


def encode_jpeg(frame, quality=85):
    """
    Encode a frame as JPEG bytes.
    
    Args:
        frame: Video frame (numpy array)
        quality: JPEG quality (1-100, default: 85)
    
    Returns:
        JPEG-encoded frame as bytes
    
    Raises:
        RuntimeError: If encoding fails
    """
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    success, buffer = cv2.imencode(".jpg", frame, encode_param)
    if not success:
        raise RuntimeError("Failed to encode frame")
    return buffer.tobytes()

