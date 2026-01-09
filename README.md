# ComfyUI Qwen Multi-Angle Camera Nodes

> **Custom ComfyUI nodes for controlling camera angles with Qwen-Image-Edit-2511-Multiple-Angles-LoRA**

<img width="1456" height="380" alt="2026-01-09_11-10" src="https://github.com/user-attachments/assets/3121a713-c4d8-4d98-bd49-230e3aa800df" />


---

## Overview

This custom node package provides comprehensive camera angle control for the [Qwen-Image-Edit-2511-Multiple-Angles-LoRA](https://huggingface.co/fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA), supporting all **96 camera positions**.

### Camera System

| Dimension | Options | Values |
|-----------|---------|--------|
| **Azimuths** | 8 | front, front-right, right, back-right, back, back-left, left, front-left |
| **Elevations** | 4 | low-angle (-30°), eye-level (0°), elevated (30°), high-angle (60°) |
| **Distances** | 3 | close-up (×0.6), medium (×1.0), wide (×1.8) |

---

## Installation

1. Navigate to your ComfyUI custom nodes directory:
   ```bash
   cd ComfyUI/custom_nodes/
   ```

2. Clone or copy this folder:
   ```bash
   git clone https://github.com/hashms0a/ComfyUI-Qwen-Multi-Angle-Camera-Nodes.git
   # OR
   # Copy the folder manually
   ```

3. Restart ComfyUI

The nodes will appear under the **"Qwen Multi-Angle Camera"** category.

---

## Available Nodes

### 📷 Camera Angle (Basic)
**`QwenMultiAngleCameraBasic`**

Simple dropdown-based camera angle selection.

| Input | Type | Description |
|-------|------|-------------|
| azimuth | Dropdown | Horizontal rotation (8 options) |
| elevation | Dropdown | Vertical angle (4 options) |
| distance | Dropdown | Camera distance (3 options) |
| prefix | String | Text before camera prompt |
| suffix | String | Text after camera prompt |

| Output | Type | Description |
|--------|------|-------------|
| prompt | STRING | Complete formatted prompt |
| azimuth_desc | STRING | Azimuth descriptor |
| elevation_desc | STRING | Elevation descriptor |
| distance_desc | STRING | Distance descriptor |

---

### 📷 Camera Angle (Advanced)
**`QwenMultiAngleCameraAdvanced`**

Numeric input with automatic snapping to nearest supported position.

| Input | Type | Range | Description |
|-------|------|-------|-------------|
| azimuth_angle | FLOAT | 0-360° | Horizontal rotation |
| elevation_angle | FLOAT | -30° to 60° | Vertical angle |
| distance_factor | FLOAT | 0.6-1.8 | Distance multiplier |

| Output | Type | Description |
|--------|------|-------------|
| prompt | STRING | Complete formatted prompt |
| actual_azimuth | FLOAT | Snapped azimuth angle |
| actual_elevation | FLOAT | Snapped elevation angle |
| actual_distance | FLOAT | Snapped distance factor |
| debug_info | STRING | Input vs. snapped values |

---

### 🔄 Camera Orbit Animation
**`QwenMultiAngleCameraOrbit`**

Generate prompts for orbital camera movement around the subject.

| Input | Type | Description |
|-------|------|-------------|
| start_azimuth | Dropdown | Starting position |
| direction | Dropdown | clockwise / counter-clockwise |
| full_rotations | FLOAT | Number of complete orbits (0.125-4.0) |
| elevation | Dropdown | Fixed elevation during orbit |
| distance | Dropdown | Fixed distance during orbit |

| Output | Type | Description |
|--------|------|-------------|
| prompts | STRING[] | List of prompts for animation |
| frame_count | INT | Total number of frames |

---

### ↕️ Camera Elevation Sweep
**`QwenMultiAngleCameraElevationSweep`**

Animate camera from one elevation to another.

| Input | Type | Description |
|-------|------|-------------|
| azimuth | Dropdown | Fixed horizontal position |
| start_elevation | Dropdown | Starting elevation |
| end_elevation | Dropdown | Ending elevation |
| distance | Dropdown | Fixed distance |
| include_reverse | BOOLEAN | Add reverse animation |

---

### ↔️ Camera Distance Transition
**`QwenMultiAngleCameraDistanceTransition`**

Animate camera distance (dolly in/out effect).

| Input | Type | Description |
|-------|------|-------------|
| azimuth | Dropdown | Fixed horizontal position |
| elevation | Dropdown | Fixed elevation |
| start_distance | Dropdown | Starting distance |
| end_distance | Dropdown | Ending distance |
| include_reverse | BOOLEAN | Add reverse animation |

---

### 🎬 Camera Path Preset
**`QwenMultiAngleCameraPath`**

Pre-defined camera animation paths.

| Path Type | Description | Frames |
|-----------|-------------|--------|
| `spiral_down` | Orbit while descending | 16 |
| `spiral_up` | Orbit while ascending | 16 |
| `hero_shot` | Low-angle dramatic orbit | 8 |
| `dramatic_reveal` | Close/high to wide/low | 6 |
| `product_showcase` | E-commerce style views | 9 |
| `full_coverage` | All 96 poses | 96 |

---

### 🔢 Camera From Index
**`QwenMultiAngleCameraFromIndex`**

Select camera position by index (0-95).

| Input | Type | Description |
|-------|------|-------------|
| pose_index | INT | Index from 0-95 |

Useful for batch processing or sequential iteration.

---

### 🎲 Random Camera Angle
**`QwenMultiAngleCameraRandom`**

Generate random camera positions with optional constraints.

| Input | Type | Description |
|-------|------|-------------|
| seed | INT | Random seed |
| allowed_azimuths | STRING | Comma-separated list or "all" |
| allowed_elevations | STRING | Comma-separated list or "all" |
| allowed_distances | STRING | Comma-separated list or "all" |

**Example constraints:**
```
allowed_azimuths: "front, front-right, front-left"
allowed_elevations: "eye-level, elevated"
allowed_distances: "all"
```

---

### 🔗 Prompt Combiner
**`QwenMultiAnglePromptCombiner`**

Combine camera prompts with additional text.

| Mode | Description |
|------|-------------|
| `camera_first` | `<sks>... + additional` |
| `camera_last` | `additional + <sks>...` |
| `replace_token` | Replace `{camera}` in template |

---

### 📑 Batch Prompt Selector
**`QwenMultiAngleBatchSelector`**

Select single prompt from a batch by index.

| Input | Type | Description |
|-------|------|-------------|
| prompts | STRING[] | Batch of prompts |
| index | INT | Selection index |
| loop | BOOLEAN | Wrap around at end |

---

### ℹ️ Camera System Info
**`QwenMultiAngleCameraInfo`**

Display camera system documentation and optionally all 96 prompts.

---

## Example Workflows

### Basic Single Shot
```
[📷 Camera Angle (Basic)] → [Qwen Image Edit Node]
        ↓
    azimuth: front-right
    elevation: eye-level
    distance: medium
```

### Product Photography Animation
```
[🎬 Camera Path Preset] → [📑 Batch Prompt Selector] → [Loop]
        ↓
    path_type: product_showcase
```

### Random Exploration with Constraints
```
[🎲 Random Camera Angle] → [Qwen Image Edit Node]
        ↓
    allowed_azimuths: "front, front-right, front-left"
    allowed_elevations: "eye-level, elevated"
```

---

## Prompt Format Reference

The LoRA expects prompts in this format:

```
<sks> [azimuth] [elevation] [distance]
```

### Examples

| Prompt | Description |
|--------|-------------|
| `<sks> front view eye-level shot medium shot` | Standard front view |
| `<sks> right side view low-angle shot close-up` | Dramatic close-up from right |
| `<sks> back-left quarter view high-angle shot wide shot` | High environmental view |

---

## Tips for Best Results

1. **Use LoRA strength 0.8-1.0** for best camera control
2. **Start with `medium shot`** - most versatile distance
3. **Low-angle shots** excel with this LoRA (trained specifically for -30°)
4. **Product shots**: Use `product_showcase` path for e-commerce style
5. **Input image quality matters** - clear subjects work best

---

## Compatibility

- **ComfyUI**: Latest version recommended
- **Python**: 3.10+
- **Base Model**: [Qwen/Qwen-Image-Edit-2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511)
- **LoRA**: [Qwen-Image-Edit-2511-Multiple-Angles-LoRA](https://huggingface.co/fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA)

---

## License

Apache 2.0

---

## Credits

- LoRA by **Lovis Odin** ([@lovis93](https://huggingface.co/lovis93))
- Trained using [fal.ai](https://fal.ai)
