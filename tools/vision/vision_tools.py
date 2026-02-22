"""
Vision tools for analyzing images through the vision encoder.
"""

import os
from typing import Dict, Any
from tools.base import Tool, ToolParameter, register_tool


@register_tool
class ViewImageTool(Tool):
    """Analyze an image using the vision encoder and return a text description."""

    name = "view_image"
    description = "Analyze an image using the vision encoder. Provide an image path and an optional prompt describing what to look for. Returns a text description of the image."
    parameters = [
        ToolParameter(
            name="image_path",
            type="string",
            description="Path to the image file (e.g. /uploads/filename.png)",
            required=True
        ),
        ToolParameter(
            name="prompt",
            type="string",
            description="What to analyze or ask about the image",
            required=False,
            default="Describe this image in detail."
        )
    ]

    def execute(self, image_path: str = "", prompt: str = "Describe this image in detail.", **kwargs) -> Dict[str, Any]:
        try:
            from inference import run_vision_inference
            result_text = run_vision_inference(image_path, prompt)
            return {
                "success": True,
                "tool": self.name,
                "result": {
                    "title": f"Image analysis of {os.path.basename(image_path)}",
                    "description": result_text
                }
            }
        except FileNotFoundError as e:
            return {
                "success": False,
                "error": str(e),
                "tool": self.name
            }
        except RuntimeError as e:
            return {
                "success": False,
                "error": str(e),
                "tool": self.name
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Vision analysis failed: {str(e)}",
                "tool": self.name
            }
