"""
Prompt building module for different summary styles.
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

