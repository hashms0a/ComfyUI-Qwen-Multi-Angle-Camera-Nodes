"""
ComfyUI Custom Nodes: Qwen-Image-Edit-2511-Multiple-Angles-LoRA Camera Control
==============================================================================

A comprehensive set of nodes for controlling camera angles with the
Qwen-Image-Edit-2511-Multiple-Angles-LoRA model.

Supports all 96 camera positions:
- 8 Azimuths: front, front-right, right, back-right, back, back-left, left, front-left
- 4 Elevations: low-angle (-30°), eye-level (0°), elevated (30°), high-angle (60°)
- 3 Distances: close-up (×0.6), medium (×1.0), wide (×1.8)

Installation:
    Copy this folder to ComfyUI/custom_nodes/

Usage:
    The nodes will appear under the "Qwen Multi-Angle Camera" category.

For more information, see:
    https://huggingface.co/lovis93/Qwen-Image-Edit-2511-Multiple-Angles-LoRA
"""

from .qwen_multi_angle_camera import (
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

# Version info
__version__ = "1.0.0"
__author__ = "Custom ComfyUI Node"
