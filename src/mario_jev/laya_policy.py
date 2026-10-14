"""Local Laya decisions over a compact state that fits its token budget."""

from time import perf_counter

from .policy import MOVEMENT_BUTTONS, compose_action

LAYA_MODELS = ("english", "multilingual", "typed-decisions")

# Laya keeps about 192 tokens per question (instructions plus options), so these are
# condensed from the Jev prompts. Movement option names must match MOVEMENT_BUTTONS.
LAYA_QUESTIONS = {
    "movement": {
        "type": "choice",
        "instructions": "Choose Mario's horizontal movement for the next frames. Progress right toward the flag. Jumping is decided separately: do not wait just because an obstacle, pit or enemy needs a jump. Brake only to avoid overshooting a landing or to back away from a very close enemy.",
        "criteria": {
            "run_right": "hold right and B: normal progress and long jumps",
            "walk_right": "hold right: slower, precise positioning",
            "brake_left": "hold left: brake or retreat",
            "wait": "release horizontal buttons",
        },
    },
    "start_jump": {
        "type": "noul",
        "instructions": "Should a grounded Mario START a jump now? True when an enemy ahead is 30-70px away at the same height, an obstacle ahead is within 48px and above his feet, he is blocked against a wall, or a pit edge is within 48px. Falling into a pit is fatal. False if airborne, A is already held, or the path ahead is clear.",
    },
    "ceiling_hop": {
        "type": "noul",
        "instructions": "Mario approaches a same-height enemy under low bricks. Is it time for a SHORT hop? True only when grounded, A released, enemy 25-45px ahead and headroom at least 16px. False when the enemy is farther than 45px (a long early jump bumps the ceiling and lands on the enemy), airborne, or A held.",
    },
    "sustain_jump": {
        "type": "noul",
        "instructions": "Mario is in the air. Should A stay held to extend the current jump? True while rising and more height or distance is needed to clear an enemy, obstacle top or pit and reach a landing surface. False when grounded, descending, after a head bump, or when clearance is already enough.",
    },
}


def _rounded(value):
    return round(value, 1) if isinstance(value, float) else value


def _pick(source, keys):
    """Copy present, non-null keys under shorter names; None when nothing is left."""
    if not source:
        return None
    picked = {
        new: _rounded(source[old])
        for old, new in keys.items()
        if source.get(old) is not None
    }
    return picked or None


def laya_state(state, transitions=2):
    """Project the full observation onto the fields the questions use.

    Laya drops tokens from the END of a state that does not fit, so the most decisive
    fields come first and null fields are omitted entirely.
    """
    mario = state["mario"]
    summary = state.get("terrain", {}).get("summary", {})
    corridor = state.get("jump_corridor", {})
    landing = state.get("landing_surfaces", {})
    compact = {
        "mario": _pick(
            mario,
            {
                "grounded": "grounded",
                "motion": "motion",
                "vx_px_per_frame": "vx",
                "vy_px_per_frame": "vy",
            },
        ),
        "a_already_held": state.get("jump_already_held"),
        "blocked_forward": state.get("blocked_forward"),
        "enemy_ahead": _pick(
            state.get("nearest_threat"),
            {
                "kind": "kind",
                "dx": "dx_px",
                "dy": "dy_px",
                "same_height": "same_height",
                "estimated_contact_in_frames": "contact_in_frames",
            },
        ),
        "obstacle_ahead": _pick(
            summary.get("nearest_obstacle"),
            {"distance_px": "distance_px", "height_above_feet_px": "height_px"},
        ),
        "pit_ahead": _pick(
            summary.get("nearest_empty_column_below_feet"),
            {"edge_distance_px": "edge_distance_px", "observed_width_px": "width_px"},
        ),
        "low_ceiling": _pick(
            corridor,
            {
                "low_ceiling_now": "now",
                "low_ceiling_before_nearest_threat": "before_enemy",
                "minimum_headroom_before_threat_px": "headroom_px",
                "possible_recent_head_bump": "head_bump",
            },
        ),
        "current_jump": _pick(
            state.get("current_jump"),
            {
                "phase": "phase",
                "elapsed_frames": "frames",
                "height_above_takeoff_px": "height_px",
                "horizontal_distance_px": "distance_px",
            },
        ),
        "landing_ahead": [
            _pick(
                surface,
                {"start_dx": "dx_px", "height_above_current_feet_px": "height_px"},
            )
            for surface in landing.get("surfaces", [])
            if surface.get("start_dx", 0) >= 0
        ][:2],
        "recent": [
            {
                "action": item["action"],
                "after": _pick(
                    item["after"],
                    {
                        "grounded": "grounded",
                        "vx_px_per_frame": "vx",
                        "vy_px_per_frame": "vy",
                    },
                ),
                **({"events": item["events"]} if item.get("events") else {}),
            }
            for item in state.get("recent_decisions", [])[-transitions:]
        ],
    }
    return {key: value for key, value in compact.items() if value not in (None, [], {})}


class LayaPolicy:
    """Answers the Jev question set with a local Laya checkpoint; no API calls."""

    def __init__(
        self, model="english", device=None, max_len=None, router=None, preload=True
    ):
        if model not in LAYA_MODELS:
            raise ValueError(f"Unknown Laya model {model!r}; choose from {LAYA_MODELS}")
        if router is None:
            try:
                from laya import Router
            except ImportError as exc:
                raise RuntimeError(
                    "Laya is not installed. Install it with: pip install -e '.[laya]'"
                ) from exc
            router = Router(device=device, default=model, max_loaded=1)
        self.router = router
        self.model = model
        self.max_len = max_len
        if preload:
            # Load the checkpoint now so download/load time is not counted as decision latency.
            self._predict({"mario": {"grounded": True}})

    def _predict(self, compact):
        return self.router.predict(
            compact, LAYA_QUESTIONS, model=self.model, max_len=self.max_len
        )

    def choose(self, state):
        compact = laya_state(state)
        started = perf_counter()
        result = self._predict(compact)
        inference_ms = (perf_counter() - started) * 1000
        answers = result["answers"]
        movement = answers["movement"]
        if movement.get("choice") not in MOVEMENT_BUTTONS:
            raise RuntimeError(f"Laya returned no usable movement: {movement!r}")
        action, decisions = compose_action(
            state,
            movement["choice"],
            answers["start_jump"]["noul"],
            answers["sustain_jump"]["noul"],
            answers["ceiling_hop"]["noul"],
        )
        usage = result.get("usage", {})
        return action, {
            "confidence": movement.get("confidence"),
            "probabilities": dict(movement.get("probabilities", {})),
            "decisions": decisions,
            "model": result.get("routing", {}).get("model", self.model),
            "usage": {
                "input_tokens": usage.get("input_tokens"),
                "state_tokens": usage.get("state_tokens"),
                "state_tokens_dropped": usage.get("state_tokens_dropped"),
                "truncated": usage.get("truncated"),
                "truncated_questions": usage.get("truncated_questions"),
                "inference_ms": round(inference_ms, 2),
            },
            "laya_state": compact,
        }

    def close(self):
        pass
