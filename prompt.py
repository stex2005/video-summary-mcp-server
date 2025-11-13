"""
Prompt building module for different summary and analysis styles.
"""


def build_summary_prompt(style="short"):
    """
    Build a prompt for video summarization based on style.
    
    Args:
        style: Summary style - "short", "timeline", "detailed", or "technical"
    
    Returns:
        Prompt string for the summarization task
    """
    prompts = {
        "short": (
            "Summarize what happens in this video. "
            "Focus on actions, events, and salient changes. "
            "Keep it concise."
        ),
        "timeline": (
            "Summarize the video. "
            "Then produce a timeline of key events with approximate timestamps. "
            "Format as: [Time] - Event description"
        ),
        "detailed": (
            "Provide a detailed summary of this video. "
            "Describe key actions, scene changes, objects, people, and timeline. "
            "Include as much context as possible."
        ),
        "technical": (
            "Provide a technical summary of this video. "
            "Focus on measurable actions, scene transitions, and objective observations. "
            "Use bullet points for clarity."
        )
    }
    
    return prompts.get(style, prompts["short"])


def build_analysis_prompt(style="short"):
    """
    Build a prompt for image analysis based on style.
    
    Args:
        style: Analysis style - "short", "detailed", "technical", or "descriptive"
    
    Returns:
        Prompt string for the image analysis task
    """
    prompts = {
        "short": (
            "Analyze this image. "
            "Describe what you see, including key objects, people, scenes, and any notable details. "
            "Keep it concise."
        ),
        "detailed": (
            "Provide a detailed analysis of this image. "
            "Describe all visible objects, people, scenes, colors, composition, lighting, "
            "and any other relevant details. Include context and potential interpretations."
        ),
        "technical": (
            "Provide a technical analysis of this image. "
            "Focus on objective observations: objects, colors, composition, lighting conditions, "
            "image quality, and measurable characteristics. Use bullet points for clarity."
        ),
        "descriptive": (
            "Provide a rich, descriptive analysis of this image. "
            "Describe the scene, atmosphere, mood, and visual elements in detail. "
            "Include artistic and aesthetic observations."
        )
    }
    
    return prompts.get(style, prompts["short"])


def build_count_prompt(object_name):
    """
    Build a prompt for counting specific objects in an image.
    
    Args:
        object_name: Name of the object to count
    
    Returns:
        Prompt string for the counting task
    """
    return (
        f"Count how many {object_name} are visible in this image. "
        f"Be precise and careful. Respond with only a number, or '0' if none are found. "
        f"If you cannot determine the exact count, provide your best estimate followed by a brief explanation."
    )

