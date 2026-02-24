from __future__ import annotations

from typing import Dict, Any

PRESET_MAPS: dict[str, Dict[str, Any]] = {
    "open_small": {
        "map_type": "open",
        "map_width": 60,
        "map_height": 60,
        "obstacle_density": 0.03,
    },
    "rooms_default": {
        "map_type": "rooms",
        "map_width": 80,
        "map_height": 80,
        "obstacle_density": 0.10,
    },
    "maze_default": {
        "map_type": "maze",
        "map_width": 81,
        "map_height": 81,
        "obstacle_density": 0.0,
    },
}
