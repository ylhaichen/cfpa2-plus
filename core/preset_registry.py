from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PlannerPreset:
    key: str
    planner_name: str
    config_path: str
    planner_label: str


_PLANNER_PRESETS: dict[str, PlannerPreset] = {
    "cfpa2": PlannerPreset(
        key="cfpa2",
        planner_name="cfpa2",
        config_path="configs/planner_cfpa2.yaml",
        planner_label="cfpa2",
    ),
    "cfpa2_plus": PlannerPreset(
        key="cfpa2_plus",
        planner_name="cfpa2_plus",
        config_path="configs/planner_cfpa2_plus_phase1.yaml",
        planner_label="cfpa2_plus_phase1",
    ),
    "cfpa2_plus_phase1": PlannerPreset(
        key="cfpa2_plus_phase1",
        planner_name="cfpa2_plus",
        config_path="configs/planner_cfpa2_plus_phase1.yaml",
        planner_label="cfpa2_plus_phase1",
    ),
    "cfpa2_plus_phase1_calib_base": PlannerPreset(
        key="cfpa2_plus_phase1_calib_base",
        planner_name="cfpa2_plus",
        config_path="configs/planner_cfpa2_plus_phase1_calib_base.yaml",
        planner_label="cfpa2_plus_phase1_calib_base",
    ),
    "cfpa2_plus_phase1_exec0": PlannerPreset(
        key="cfpa2_plus_phase1_exec0",
        planner_name="cfpa2_plus",
        config_path="configs/planner_cfpa2_plus_phase1_exec0.yaml",
        planner_label="cfpa2_plus_phase1_exec0",
    ),
    "cfpa2_plus_phase1_no_clearance": PlannerPreset(
        key="cfpa2_plus_phase1_no_clearance",
        planner_name="cfpa2_plus",
        config_path="configs/planner_cfpa2_plus_phase1_no_clearance.yaml",
        planner_label="cfpa2_plus_phase1_no_clearance",
    ),
    "cfpa2_plus_phase1_no_turn": PlannerPreset(
        key="cfpa2_plus_phase1_no_turn",
        planner_name="cfpa2_plus",
        config_path="configs/planner_cfpa2_plus_phase1_no_turn.yaml",
        planner_label="cfpa2_plus_phase1_no_turn",
    ),
    "cfpa2_plus_phase1_no_narrowness": PlannerPreset(
        key="cfpa2_plus_phase1_no_narrowness",
        planner_name="cfpa2_plus",
        config_path="configs/planner_cfpa2_plus_phase1_no_narrowness.yaml",
        planner_label="cfpa2_plus_phase1_no_narrowness",
    ),
    "cfpa2_plus_phase1_no_team": PlannerPreset(
        key="cfpa2_plus_phase1_no_team",
        planner_name="cfpa2_plus",
        config_path="configs/planner_cfpa2_plus_phase1_no_team.yaml",
        planner_label="cfpa2_plus_phase1_no_team",
    ),
    "rh_cfpa2": PlannerPreset(
        key="rh_cfpa2",
        planner_name="rh_cfpa2",
        config_path="configs/planner_rh_cfpa2.yaml",
        planner_label="rh_cfpa2",
    ),
    "physics_rh_cfpa2": PlannerPreset(
        key="physics_rh_cfpa2",
        planner_name="physics_rh_cfpa2",
        config_path="configs/planner_physics_rh_cfpa2.yaml",
        planner_label="physics_rh_cfpa2",
    ),
}


def planner_preset_choices() -> list[str]:
    return list(_PLANNER_PRESETS.keys())


def planner_compare_choices() -> list[str]:
    return list(_PLANNER_PRESETS.keys())


def get_planner_preset(key: str) -> PlannerPreset:
    if key not in _PLANNER_PRESETS:
        raise KeyError(f"Unknown planner preset: {key}")
    return _PLANNER_PRESETS[key]
