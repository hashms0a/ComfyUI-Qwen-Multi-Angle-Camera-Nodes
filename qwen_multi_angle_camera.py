"""
ComfyUI Custom Nodes for Qwen-Image-Edit-2511-Multiple-Angles-LoRA
==================================================================

This module provides nodes to control camera angles for the Multi-Angle LoRA,
supporting all 96 camera positions (4 elevations × 8 azimuths × 3 distances).

Prompt format: <sks> [azimuth] [elevation] [distance]
Example: <sks> front view eye-level shot medium shot

License: Apache-2.0
"""

import math
from typing import Tuple, List, Dict, Any


# ============================================================================
# Constants and Configuration
# ============================================================================

# Azimuth definitions (horizontal rotation around subject)
AZIMUTHS = {
    "front": {"angle": 0, "descriptor": "front view"},
    "front-right": {"angle": 45, "descriptor": "front-right quarter view"},
    "right": {"angle": 90, "descriptor": "right side view"},
    "back-right": {"angle": 135, "descriptor": "back-right quarter view"},
    "back": {"angle": 180, "descriptor": "back view"},
    "back-left": {"angle": 225, "descriptor": "back-left quarter view"},
    "left": {"angle": 270, "descriptor": "left side view"},
    "front-left": {"angle": 315, "descriptor": "front-left quarter view"},
}

# Elevation definitions (vertical angle)
ELEVATIONS = {
    "low-angle": {"angle": -30, "descriptor": "low-angle shot"},
    "eye-level": {"angle": 0, "descriptor": "eye-level shot"},
    "elevated": {"angle": 30, "descriptor": "elevated shot"},
    "high-angle": {"angle": 60, "descriptor": "high-angle shot"},
}

# Distance definitions (camera distance from subject)
DISTANCES = {
    "close-up": {"factor": 0.6, "descriptor": "close-up"},
    "medium": {"factor": 1.0, "descriptor": "medium shot"},
    "wide": {"factor": 1.8, "descriptor": "wide shot"},
}

# Ordered lists for dropdowns
AZIMUTH_OPTIONS = [
    "front",
    "front-right", 
    "right",
    "back-right",
    "back",
    "back-left",
    "left",
    "front-left",
]

ELEVATION_OPTIONS = [
    "low-angle",
    "eye-level",
    "elevated",
    "high-angle",
]

DISTANCE_OPTIONS = [
    "close-up",
    "medium",
    "wide",
]


# ============================================================================
# Helper Functions
# ============================================================================

def build_prompt(azimuth: str, elevation: str, distance: str) -> str:
    """
    Build the prompt string for the Multi-Angle LoRA.
    
    Args:
        azimuth: Horizontal rotation key (e.g., "front", "back-right")
        elevation: Vertical angle key (e.g., "eye-level", "high-angle")
        distance: Distance key (e.g., "close-up", "medium", "wide")
    
    Returns:
        Formatted prompt string like "<sks> front view eye-level shot medium shot"
    """
    azimuth_desc = AZIMUTHS[azimuth]["descriptor"]
    elevation_desc = ELEVATIONS[elevation]["descriptor"]
    distance_desc = DISTANCES[distance]["descriptor"]
    
    return f"<sks> {azimuth_desc} {elevation_desc} {distance_desc}"


def angle_to_azimuth(angle: float) -> str:
    """Convert a continuous angle (0-360) to the nearest azimuth key."""
    # Normalize angle to 0-360
    angle = angle % 360
    
    # Find the closest azimuth
    min_diff = float('inf')
    closest = "front"
    
    for key, data in AZIMUTHS.items():
        diff = min(abs(angle - data["angle"]), 360 - abs(angle - data["angle"]))
        if diff < min_diff:
            min_diff = diff
            closest = key
    
    return closest


def angle_to_elevation(angle: float) -> str:
    """Convert a continuous angle (-90 to 90) to the nearest elevation key."""
    # Clamp angle to valid range
    angle = max(-90, min(90, angle))
    
    # Find the closest elevation
    min_diff = float('inf')
    closest = "eye-level"
    
    for key, data in ELEVATIONS.items():
        diff = abs(angle - data["angle"])
        if diff < min_diff:
            min_diff = diff
            closest = key
    
    return closest


def distance_to_key(distance: float) -> str:
    """Convert a continuous distance factor to the nearest distance key."""
    # Define thresholds
    if distance <= 0.8:
        return "close-up"
    elif distance <= 1.4:
        return "medium"
    else:
        return "wide"


# ============================================================================
# ComfyUI Node: Basic Camera Angle Selector
# ============================================================================

class QwenMultiAngleCameraBasic:
    """
    Basic camera angle selector with dropdown menus for azimuth, elevation, and distance.
    Outputs the formatted prompt for the Multi-Angle LoRA.
    """
    
    CATEGORY = "Qwen Multi-Angle Camera"
    FUNCTION = "generate_prompt"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("prompt", "azimuth_desc", "elevation_desc", "distance_desc")
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "azimuth": (AZIMUTH_OPTIONS, {"default": "front"}),
                "elevation": (ELEVATION_OPTIONS, {"default": "eye-level"}),
                "distance": (DISTANCE_OPTIONS, {"default": "medium"}),
            },
            "optional": {
                "prefix": ("STRING", {"default": "", "multiline": False}),
                "suffix": ("STRING", {"default": "", "multiline": False}),
            }
        }
    
    def generate_prompt(
        self,
        azimuth: str,
        elevation: str,
        distance: str,
        prefix: str = "",
        suffix: str = "",
    ) -> Tuple[str, str, str, str]:
        """Generate the camera angle prompt."""
        base_prompt = build_prompt(azimuth, elevation, distance)
        
        # Combine with optional prefix/suffix
        parts = []
        if prefix.strip():
            parts.append(prefix.strip())
        parts.append(base_prompt)
        if suffix.strip():
            parts.append(suffix.strip())
        
        full_prompt = " ".join(parts)
        
        return (
            full_prompt,
            AZIMUTHS[azimuth]["descriptor"],
            ELEVATIONS[elevation]["descriptor"],
            DISTANCES[distance]["descriptor"],
        )


# ============================================================================
# ComfyUI Node: Advanced Camera Angle Controller
# ============================================================================

class QwenMultiAngleCameraAdvanced:
    """
    Advanced camera controller with numeric inputs for precise angle control.
    Automatically snaps to the nearest supported camera position.
    """
    
    CATEGORY = "Qwen Multi-Angle Camera"
    FUNCTION = "generate_prompt"
    RETURN_TYPES = ("STRING", "FLOAT", "FLOAT", "FLOAT", "STRING")
    RETURN_NAMES = ("prompt", "actual_azimuth", "actual_elevation", "actual_distance", "debug_info")
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "azimuth_angle": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 360.0,
                    "step": 1.0,
                    "display": "slider",
                }),
                "elevation_angle": ("FLOAT", {
                    "default": 0.0,
                    "min": -30.0,
                    "max": 60.0,
                    "step": 1.0,
                    "display": "slider",
                }),
                "distance_factor": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.6,
                    "max": 1.8,
                    "step": 0.1,
                    "display": "slider",
                }),
            },
            "optional": {
                "prefix": ("STRING", {"default": "", "multiline": False}),
                "suffix": ("STRING", {"default": "", "multiline": False}),
            }
        }
    
    def generate_prompt(
        self,
        azimuth_angle: float,
        elevation_angle: float,
        distance_factor: float,
        prefix: str = "",
        suffix: str = "",
    ) -> Tuple[str, float, float, float, str]:
        """Generate prompt from numeric angles, snapping to nearest supported position."""
        # Convert to discrete positions
        azimuth_key = angle_to_azimuth(azimuth_angle)
        elevation_key = angle_to_elevation(elevation_angle)
        distance_key = distance_to_key(distance_factor)
        
        # Get actual values
        actual_azimuth = AZIMUTHS[azimuth_key]["angle"]
        actual_elevation = ELEVATIONS[elevation_key]["angle"]
        actual_distance = DISTANCES[distance_key]["factor"]
        
        # Build prompt
        base_prompt = build_prompt(azimuth_key, elevation_key, distance_key)
        
        parts = []
        if prefix.strip():
            parts.append(prefix.strip())
        parts.append(base_prompt)
        if suffix.strip():
            parts.append(suffix.strip())
        
        full_prompt = " ".join(parts)
        
        # Debug info
        debug_info = (
            f"Input: az={azimuth_angle:.1f}°, el={elevation_angle:.1f}°, dist={distance_factor:.2f}\n"
            f"Snapped: {azimuth_key} ({actual_azimuth}°), {elevation_key} ({actual_elevation}°), {distance_key} (×{actual_distance})"
        )
        
        return (
            full_prompt,
            float(actual_azimuth),
            float(actual_elevation),
            actual_distance,
            debug_info,
        )


# ============================================================================
# ComfyUI Node: Camera Orbit Animation
# ============================================================================

class QwenMultiAngleCameraOrbit:
    """
    Generate a sequence of prompts for orbital camera animation.
    Creates smooth rotation around the subject at fixed elevation and distance.
    """
    
    CATEGORY = "Qwen Multi-Angle Camera"
    FUNCTION = "generate_orbit"
    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("prompts", "frame_count")
    OUTPUT_IS_LIST = (True, False)
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "start_azimuth": (AZIMUTH_OPTIONS, {"default": "front"}),
                "direction": (["clockwise", "counter-clockwise"], {"default": "clockwise"}),
                "full_rotations": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.125,
                    "max": 4.0,
                    "step": 0.125,
                }),
                "elevation": (ELEVATION_OPTIONS, {"default": "eye-level"}),
                "distance": (DISTANCE_OPTIONS, {"default": "medium"}),
            },
            "optional": {
                "prefix": ("STRING", {"default": "", "multiline": False}),
                "suffix": ("STRING", {"default": "", "multiline": False}),
            }
        }
    
    def generate_orbit(
        self,
        start_azimuth: str,
        direction: str,
        full_rotations: float,
        elevation: str,
        distance: str,
        prefix: str = "",
        suffix: str = "",
    ) -> Tuple[List[str], int]:
        """Generate orbital animation prompts."""
        # Calculate number of frames (8 positions per rotation)
        total_positions = int(full_rotations * 8)
        
        # Get starting index
        start_idx = AZIMUTH_OPTIONS.index(start_azimuth)
        
        prompts = []
        for i in range(total_positions):
            # Calculate current azimuth index
            if direction == "clockwise":
                current_idx = (start_idx + i) % 8
            else:
                current_idx = (start_idx - i) % 8
            
            azimuth = AZIMUTH_OPTIONS[current_idx]
            base_prompt = build_prompt(azimuth, elevation, distance)
            
            parts = []
            if prefix.strip():
                parts.append(prefix.strip())
            parts.append(base_prompt)
            if suffix.strip():
                parts.append(suffix.strip())
            
            prompts.append(" ".join(parts))
        
        return (prompts, len(prompts))


# ============================================================================
# ComfyUI Node: Camera Elevation Sweep
# ============================================================================

class QwenMultiAngleCameraElevationSweep:
    """
    Generate a sequence of prompts for elevation sweep animation.
    Moves camera from low to high angle (or vice versa) at fixed azimuth.
    """
    
    CATEGORY = "Qwen Multi-Angle Camera"
    FUNCTION = "generate_sweep"
    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("prompts", "frame_count")
    OUTPUT_IS_LIST = (True, False)
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "azimuth": (AZIMUTH_OPTIONS, {"default": "front"}),
                "start_elevation": (ELEVATION_OPTIONS, {"default": "low-angle"}),
                "end_elevation": (ELEVATION_OPTIONS, {"default": "high-angle"}),
                "distance": (DISTANCE_OPTIONS, {"default": "medium"}),
                "include_reverse": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "prefix": ("STRING", {"default": "", "multiline": False}),
                "suffix": ("STRING", {"default": "", "multiline": False}),
            }
        }
    
    def generate_sweep(
        self,
        azimuth: str,
        start_elevation: str,
        end_elevation: str,
        distance: str,
        include_reverse: bool,
        prefix: str = "",
        suffix: str = "",
    ) -> Tuple[List[str], int]:
        """Generate elevation sweep prompts."""
        start_idx = ELEVATION_OPTIONS.index(start_elevation)
        end_idx = ELEVATION_OPTIONS.index(end_elevation)
        
        # Generate index sequence
        if start_idx <= end_idx:
            indices = list(range(start_idx, end_idx + 1))
        else:
            indices = list(range(start_idx, end_idx - 1, -1))
        
        # Add reverse if requested
        if include_reverse and len(indices) > 1:
            indices.extend(indices[-2::-1])  # Reverse without duplicating last
        
        prompts = []
        for idx in indices:
            elevation = ELEVATION_OPTIONS[idx]
            base_prompt = build_prompt(azimuth, elevation, distance)
            
            parts = []
            if prefix.strip():
                parts.append(prefix.strip())
            parts.append(base_prompt)
            if suffix.strip():
                parts.append(suffix.strip())
            
            prompts.append(" ".join(parts))
        
        return (prompts, len(prompts))


# ============================================================================
# ComfyUI Node: Camera Distance Transition
# ============================================================================

class QwenMultiAngleCameraDistanceTransition:
    """
    Generate a sequence of prompts for distance transition (dolly) animation.
    Moves camera closer or farther from subject at fixed angle.
    """
    
    CATEGORY = "Qwen Multi-Angle Camera"
    FUNCTION = "generate_transition"
    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("prompts", "frame_count")
    OUTPUT_IS_LIST = (True, False)
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "azimuth": (AZIMUTH_OPTIONS, {"default": "front"}),
                "elevation": (ELEVATION_OPTIONS, {"default": "eye-level"}),
                "start_distance": (DISTANCE_OPTIONS, {"default": "wide"}),
                "end_distance": (DISTANCE_OPTIONS, {"default": "close-up"}),
                "include_reverse": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "prefix": ("STRING", {"default": "", "multiline": False}),
                "suffix": ("STRING", {"default": "", "multiline": False}),
            }
        }
    
    def generate_transition(
        self,
        azimuth: str,
        elevation: str,
        start_distance: str,
        end_distance: str,
        include_reverse: bool,
        prefix: str = "",
        suffix: str = "",
    ) -> Tuple[List[str], int]:
        """Generate distance transition prompts."""
        start_idx = DISTANCE_OPTIONS.index(start_distance)
        end_idx = DISTANCE_OPTIONS.index(end_distance)
        
        # Generate index sequence
        if start_idx <= end_idx:
            indices = list(range(start_idx, end_idx + 1))
        else:
            indices = list(range(start_idx, end_idx - 1, -1))
        
        # Add reverse if requested
        if include_reverse and len(indices) > 1:
            indices.extend(indices[-2::-1])
        
        prompts = []
        for idx in indices:
            distance = DISTANCE_OPTIONS[idx]
            base_prompt = build_prompt(azimuth, elevation, distance)
            
            parts = []
            if prefix.strip():
                parts.append(prefix.strip())
            parts.append(base_prompt)
            if suffix.strip():
                parts.append(suffix.strip())
            
            prompts.append(" ".join(parts))
        
        return (prompts, len(prompts))


# ============================================================================
# ComfyUI Node: Full Camera Path
# ============================================================================

class QwenMultiAngleCameraPath:
    """
    Generate prompts for a complex camera path combining multiple movements.
    Supports orbit + elevation + distance changes in sequence.
    """
    
    CATEGORY = "Qwen Multi-Angle Camera"
    FUNCTION = "generate_path"
    RETURN_TYPES = ("STRING", "INT", "STRING")
    RETURN_NAMES = ("prompts", "frame_count", "path_description")
    OUTPUT_IS_LIST = (True, False, False)
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "path_type": ([
                    "spiral_down",      # Orbit while descending
                    "spiral_up",        # Orbit while ascending
                    "hero_shot",        # Low angle orbit
                    "dramatic_reveal",  # Start close high, pull back low
                    "product_showcase", # Multi-angle product view
                    "full_coverage",    # All 96 poses
                ], {"default": "product_showcase"}),
            },
            "optional": {
                "prefix": ("STRING", {"default": "", "multiline": False}),
                "suffix": ("STRING", {"default": "", "multiline": False}),
            }
        }
    
    def generate_path(
        self,
        path_type: str,
        prefix: str = "",
        suffix: str = "",
    ) -> Tuple[List[str], int, str]:
        """Generate complex camera path prompts."""
        prompts = []
        
        def add_prompt(az: str, el: str, dist: str):
            base = build_prompt(az, el, dist)
            parts = []
            if prefix.strip():
                parts.append(prefix.strip())
            parts.append(base)
            if suffix.strip():
                parts.append(suffix.strip())
            prompts.append(" ".join(parts))
        
        if path_type == "spiral_down":
            # Orbit twice while moving from high to low angle
            description = "360° orbit at high-angle, then 360° at eye-level"
            for elevation in ["high-angle", "eye-level"]:
                for azimuth in AZIMUTH_OPTIONS:
                    add_prompt(azimuth, elevation, "medium")
        
        elif path_type == "spiral_up":
            # Orbit twice while moving from low to high angle
            description = "360° orbit at low-angle, then 360° at elevated"
            for elevation in ["low-angle", "elevated"]:
                for azimuth in AZIMUTH_OPTIONS:
                    add_prompt(azimuth, elevation, "medium")
        
        elif path_type == "hero_shot":
            # Low angle dramatic orbit
            description = "Full 360° low-angle orbit at medium distance"
            for azimuth in AZIMUTH_OPTIONS:
                add_prompt(azimuth, "low-angle", "medium")
        
        elif path_type == "dramatic_reveal":
            # Start close and high, pull back and lower
            description = "Start close/high-angle front, end wide/low-angle front"
            # Close high angle
            add_prompt("front", "high-angle", "close-up")
            add_prompt("front", "elevated", "close-up")
            # Pull back
            add_prompt("front", "elevated", "medium")
            add_prompt("front", "eye-level", "medium")
            # Continue back and lower
            add_prompt("front", "eye-level", "wide")
            add_prompt("front", "low-angle", "wide")
        
        elif path_type == "product_showcase":
            # Standard e-commerce product views
            description = "Key angles for product photography: front, quarters, sides, back"
            for azimuth in ["front", "front-right", "right", "back", "left", "front-left"]:
                add_prompt(azimuth, "eye-level", "medium")
            # Add elevated views
            for azimuth in ["front", "front-right", "back-right"]:
                add_prompt(azimuth, "elevated", "medium")
        
        elif path_type == "full_coverage":
            # All 96 poses
            description = "Complete coverage: all 96 camera positions"
            for distance in DISTANCE_OPTIONS:
                for elevation in ELEVATION_OPTIONS:
                    for azimuth in AZIMUTH_OPTIONS:
                        add_prompt(azimuth, elevation, distance)
        
        return (prompts, len(prompts), description)


# ============================================================================
# ComfyUI Node: Prompt from Index
# ============================================================================

class QwenMultiAngleCameraFromIndex:
    """
    Select a specific camera position by index (0-95).
    Useful for batch processing or random selection.
    """
    
    CATEGORY = "Qwen Multi-Angle Camera"
    FUNCTION = "get_prompt"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "INT")
    RETURN_NAMES = ("prompt", "azimuth", "elevation", "distance", "total_poses")
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "pose_index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 95,
                    "step": 1,
                }),
            },
            "optional": {
                "prefix": ("STRING", {"default": "", "multiline": False}),
                "suffix": ("STRING", {"default": "", "multiline": False}),
            }
        }
    
    def get_prompt(
        self,
        pose_index: int,
        prefix: str = "",
        suffix: str = "",
    ) -> Tuple[str, str, str, str, int]:
        """Get prompt for a specific pose index."""
        # Index mapping: distance -> elevation -> azimuth
        # 32 poses per distance, 8 poses per elevation
        distance_idx = pose_index // 32
        remaining = pose_index % 32
        elevation_idx = remaining // 8
        azimuth_idx = remaining % 8
        
        azimuth = AZIMUTH_OPTIONS[azimuth_idx]
        elevation = ELEVATION_OPTIONS[elevation_idx]
        distance = DISTANCE_OPTIONS[distance_idx]
        
        base_prompt = build_prompt(azimuth, elevation, distance)
        
        parts = []
        if prefix.strip():
            parts.append(prefix.strip())
        parts.append(base_prompt)
        if suffix.strip():
            parts.append(suffix.strip())
        
        return (
            " ".join(parts),
            azimuth,
            elevation,
            distance,
            96,
        )


# ============================================================================
# ComfyUI Node: Random Camera Angle
# ============================================================================

class QwenMultiAngleCameraRandom:
    """
    Generate a random camera position with optional constraints.
    """
    
    CATEGORY = "Qwen Multi-Angle Camera"
    FUNCTION = "get_random"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "INT")
    RETURN_NAMES = ("prompt", "azimuth", "elevation", "distance", "seed_used")
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffff,
                }),
            },
            "optional": {
                "allowed_azimuths": ("STRING", {
                    "default": "all",
                    "multiline": False,
                }),
                "allowed_elevations": ("STRING", {
                    "default": "all",
                    "multiline": False,
                }),
                "allowed_distances": ("STRING", {
                    "default": "all",
                    "multiline": False,
                }),
                "prefix": ("STRING", {"default": "", "multiline": False}),
                "suffix": ("STRING", {"default": "", "multiline": False}),
            }
        }
    
    @classmethod
    def IS_CHANGED(cls, seed, **kwargs):
        return seed
    
    def get_random(
        self,
        seed: int,
        allowed_azimuths: str = "all",
        allowed_elevations: str = "all",
        allowed_distances: str = "all",
        prefix: str = "",
        suffix: str = "",
    ) -> Tuple[str, str, str, str, int]:
        """Generate random camera position."""
        import random
        rng = random.Random(seed)
        
        # Parse allowed values
        def parse_allowed(value: str, options: list) -> list:
            if value.lower().strip() == "all":
                return options
            parts = [p.strip() for p in value.split(",")]
            return [p for p in parts if p in options]
        
        az_options = parse_allowed(allowed_azimuths, AZIMUTH_OPTIONS)
        el_options = parse_allowed(allowed_elevations, ELEVATION_OPTIONS)
        dist_options = parse_allowed(allowed_distances, DISTANCE_OPTIONS)
        
        # Default to all if empty
        az_options = az_options or AZIMUTH_OPTIONS
        el_options = el_options or ELEVATION_OPTIONS
        dist_options = dist_options or DISTANCE_OPTIONS
        
        azimuth = rng.choice(az_options)
        elevation = rng.choice(el_options)
        distance = rng.choice(dist_options)
        
        base_prompt = build_prompt(azimuth, elevation, distance)
        
        parts = []
        if prefix.strip():
            parts.append(prefix.strip())
        parts.append(base_prompt)
        if suffix.strip():
            parts.append(suffix.strip())
        
        return (
            " ".join(parts),
            azimuth,
            elevation,
            distance,
            seed,
        )


# ============================================================================
# ComfyUI Node: Prompt Combiner
# ============================================================================

class QwenMultiAnglePromptCombiner:
    """
    Combine camera angle prompt with other prompt elements.
    Provides flexible positioning of the camera control token.
    """
    
    CATEGORY = "Qwen Multi-Angle Camera"
    FUNCTION = "combine"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("combined_prompt",)
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "camera_prompt": ("STRING", {"forceInput": True}),
                "mode": ([
                    "camera_first",     # <sks> ... + other
                    "camera_last",      # other + <sks> ...
                    "replace_token",    # Replace {camera} in template
                ], {"default": "camera_first"}),
            },
            "optional": {
                "additional_prompt": ("STRING", {"default": "", "multiline": True}),
                "template": ("STRING", {
                    "default": "A photo showing {camera}",
                    "multiline": True,
                }),
            }
        }
    
    def combine(
        self,
        camera_prompt: str,
        mode: str,
        additional_prompt: str = "",
        template: str = "",
    ) -> Tuple[str]:
        """Combine camera prompt with other elements."""
        if mode == "camera_first":
            result = camera_prompt
            if additional_prompt.strip():
                result += " " + additional_prompt.strip()
        elif mode == "camera_last":
            result = ""
            if additional_prompt.strip():
                result = additional_prompt.strip() + " "
            result += camera_prompt
        elif mode == "replace_token":
            result = template.replace("{camera}", camera_prompt)
        else:
            result = camera_prompt
        
        return (result,)


# ============================================================================
# ComfyUI Node: Batch Prompt Selector
# ============================================================================

class QwenMultiAngleBatchSelector:
    """
    Select a single prompt from a batch of prompts by index.
    Useful for iterating through animation sequences.
    """
    
    CATEGORY = "Qwen Multi-Angle Camera"
    FUNCTION = "select"
    RETURN_TYPES = ("STRING", "INT", "INT")
    RETURN_NAMES = ("prompt", "current_index", "total_count")
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "prompts": ("STRING", {"forceInput": True}),
                "index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 999,
                    "step": 1,
                }),
                "loop": ("BOOLEAN", {"default": True}),
            },
        }
    
    @classmethod
    def IS_CHANGED(cls, prompts, index, loop):
        return (prompts, index, loop)
    
    def select(
        self,
        prompts: list,
        index: int,
        loop: bool,
    ) -> Tuple[str, int, int]:
        """Select a prompt from the batch."""
        if not isinstance(prompts, list):
            prompts = [prompts]
        
        total = len(prompts)
        
        if loop:
            actual_index = index % total
        else:
            actual_index = min(index, total - 1)
        
        return (prompts[actual_index], actual_index, total)


# ============================================================================
# ComfyUI Node: Camera Info Display
# ============================================================================

class QwenMultiAngleCameraInfo:
    """
    Display information about camera angle configuration.
    Useful for debugging and visualization.
    """
    
    CATEGORY = "Qwen Multi-Angle Camera"
    FUNCTION = "get_info"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("info",)
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "show_all_prompts": ("BOOLEAN", {"default": False}),
            },
        }
    
    def get_info(self, show_all_prompts: bool) -> Tuple[str]:
        """Generate camera system information."""
        lines = [
            "Qwen-Image-Edit-2511-Multiple-Angles-LoRA",
            "=" * 45,
            "",
            "CAMERA SYSTEM OVERVIEW",
            f"Total Poses: 96 (4 elevations × 8 azimuths × 3 distances)",
            "",
            "AZIMUTHS (Horizontal Rotation):",
        ]
        
        for key, data in AZIMUTHS.items():
            lines.append(f"  {data['angle']:>3}° - {data['descriptor']}")
        
        lines.extend([
            "",
            "ELEVATIONS (Vertical Angle):",
        ])
        
        for key, data in ELEVATIONS.items():
            lines.append(f"  {data['angle']:>3}° - {data['descriptor']}")
        
        lines.extend([
            "",
            "DISTANCES:",
        ])
        
        for key, data in DISTANCES.items():
            lines.append(f"  ×{data['factor']:.1f} - {data['descriptor']}")
        
        lines.extend([
            "",
            "PROMPT FORMAT:",
            "  <sks> [azimuth] [elevation] [distance]",
            "",
            "EXAMPLE:",
            "  <sks> front view eye-level shot medium shot",
        ])
        
        if show_all_prompts:
            lines.extend([
                "",
                "ALL 96 PROMPTS:",
                "-" * 45,
            ])
            for dist in DISTANCE_OPTIONS:
                for elev in ELEVATION_OPTIONS:
                    for azim in AZIMUTH_OPTIONS:
                        lines.append(build_prompt(azim, elev, dist))
        
        return ("\n".join(lines),)


# ============================================================================
# Node Registration
# ============================================================================

NODE_CLASS_MAPPINGS = {
    "QwenMultiAngleCameraBasic": QwenMultiAngleCameraBasic,
    "QwenMultiAngleCameraAdvanced": QwenMultiAngleCameraAdvanced,
    "QwenMultiAngleCameraOrbit": QwenMultiAngleCameraOrbit,
    "QwenMultiAngleCameraElevationSweep": QwenMultiAngleCameraElevationSweep,
    "QwenMultiAngleCameraDistanceTransition": QwenMultiAngleCameraDistanceTransition,
    "QwenMultiAngleCameraPath": QwenMultiAngleCameraPath,
    "QwenMultiAngleCameraFromIndex": QwenMultiAngleCameraFromIndex,
    "QwenMultiAngleCameraRandom": QwenMultiAngleCameraRandom,
    "QwenMultiAnglePromptCombiner": QwenMultiAnglePromptCombiner,
    "QwenMultiAngleBatchSelector": QwenMultiAngleBatchSelector,
    "QwenMultiAngleCameraInfo": QwenMultiAngleCameraInfo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenMultiAngleCameraBasic": "📷 Camera Angle (Basic)",
    "QwenMultiAngleCameraAdvanced": "📷 Camera Angle (Advanced)",
    "QwenMultiAngleCameraOrbit": "🔄 Camera Orbit Animation",
    "QwenMultiAngleCameraElevationSweep": "↕️ Camera Elevation Sweep",
    "QwenMultiAngleCameraDistanceTransition": "↔️ Camera Distance Transition",
    "QwenMultiAngleCameraPath": "🎬 Camera Path Preset",
    "QwenMultiAngleCameraFromIndex": "🔢 Camera From Index",
    "QwenMultiAngleCameraRandom": "🎲 Random Camera Angle",
    "QwenMultiAnglePromptCombiner": "🔗 Prompt Combiner",
    "QwenMultiAngleBatchSelector": "📑 Batch Prompt Selector",
    "QwenMultiAngleCameraInfo": "ℹ️ Camera System Info",
}
