"""
Core video summarization module using GPT-4.1 Vision.
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

