# video-summary-mcp-server

## 🚀 fastMCP Example: "summarize_video" Tool

This server exposes **1 MCP tool**:

```
summarize_video(video_path: string) → summary: string
```

You can drop this file anywhere and run:

```
python server.py
```

and a Model Context Protocol–compatible client (ROS-MCP client, Claude Desktop, ChatGPT MCP) can use it immediately.

---

## 📌 Install

```bash
pip install -e .
```

Or install dependencies directly:

```bash
pip install fastmcp openai pillow opencv-python
```

---

## 🧠 What this server does

When a client calls:

```json
{
  "tool": "summarize_video",
  "arguments": { "video_path": "demo.mp4" }
}
```

fastMCP:

1. Loads the video
2. Extracts frames every 2 seconds
3. Encodes frames as JPEG
4. Sends them to GPT-4.1 Vision
5. Returns a natural language **video summary**

This works on **any computer with Python**, no GPU required.

---

## 🧪 How to run

### As MCP Server

Start the MCP server:

```bash
python server.py
```

### As Standalone CLI

Run directly from command line:

```bash
python main.py videos/demo.mp4 [style]
```

Available styles:
- `short` - Concise summary (default)
- `timeline` - Summary with timestamps
- `detailed` - Comprehensive summary
- `technical` - Technical bullet-point summary

Example:
```bash
python main.py videos/demo.mp4 timeline
```

Then use **any MCP client**, for example:

### ✔ ChatGPT MCP (Custom tools)

Add a new connection:

```
command: python /path/to/server.py
```

### ✔ ROS-MCP client

```bash
ros_mcp connect video-summarizer python server.py
```

### ✔ Claude Desktop (Tools → Add Tool)

Set executable to:

```
python3 server.py
```

---

## 📁 Project Structure

```
video-summary-mcp-server/
│
├── server.py              # MCP server (fastMCP wrapper)
├── main.py                # CLI runner for standalone use
├── summarizer.py          # Core summarization logic
├── frame_extractor.py     # Video frame extraction
├── encoder.py             # JPEG encoding
├── prompt.py              # Prompt building for different styles
├── pyproject.toml         # Project configuration
└── videos/                # Directory for video files
```

### Module Overview

- **`server.py`** - MCP server exposing `summarize_video` tool
- **`main.py`** - Standalone CLI for direct video summarization
- **`summarizer.py`** - Core module that orchestrates the summarization pipeline
- **`frame_extractor.py`** - Extracts keyframes from videos at regular intervals
- **`encoder.py`** - Compresses frames to JPEG for efficient API calls
- **`prompt.py`** - Builds prompts for different summary styles (short, timeline, detailed, technical)

---

## 🔑 Environment Setup

### Option 1: Using .env file (Recommended)

1. Create a `.env` file in the project root (any text editor is fine).

2. Add your OpenAI API key:
```
OPENAI_API_KEY=your-actual-api-key-here
```

The `.env` file is automatically loaded when you run the server.

### Option 2: Environment Variable

Set it as an environment variable:

```bash
export OPENAI_API_KEY='your-api-key-here'
```

**Get your API key from:** https://platform.openai.com/api-keys
