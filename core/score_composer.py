from __future__ import annotations


def compose_cfpa2_plus_utility(
    baseline_utility: float,
    execution_penalty: float,
    cfg: dict,
) -> tuple[float, dict[str, float | bool]]:
    plus_cfg = dict(cfg.get("planning", {}).get("cfpa2_plus", {}))
    enabled_components = dict(plus_cfg.get("enabled_components", {}))
    score_mode = dict(plus_cfg.get("score_mode", {}))
    execution_cfg = dict(plus_cfg.get("execution", {}))

    execution_enabled = bool(enabled_components.get("execution_aware", True)) and bool(execution_cfg.get("enabled", True))
    baseline_weight = float(score_mode.get("baseline_weight", 1.0))
    execution_weight = float(score_mode.get("execution_weight", 1.0))
    lambda_exec = float(score_mode.get("lambda_exec", 1.0))

    execution_component = 0.0
    if execution_enabled:
        execution_component = execution_weight * lambda_exec * float(execution_penalty)

    composed = baseline_weight * float(baseline_utility) - execution_component
    return composed, {
        "baseline_weight": float(baseline_weight),
        "execution_weight": float(execution_weight),
        "lambda_exec": float(lambda_exec),
        "execution_enabled": bool(execution_enabled),
        "execution_component": float(execution_component),
        "composed_utility": float(composed),
    }

