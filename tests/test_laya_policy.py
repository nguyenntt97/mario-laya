import json
from pathlib import Path

import pytest

from mario_jev.laya_policy import LAYA_QUESTIONS, LayaPolicy, laya_state
from mario_jev.policy import MOVEMENT_BUTTONS

FULL_STATE = Path(__file__).parent / "fixtures" / "full_state.json"


class FakeRouter:
    def __init__(self, movement="run_right", start=0.1, sustain=0.1, ceiling_hop=0.1):
        self.calls = []
        self.answers = {
            "movement": {
                "type": "choice",
                "choice": movement,
                "confidence": 0.7,
                "probabilities": {"run_right": 0.8, "walk_right": 0.2},
            },
            "start_jump": {"type": "noul", "noul": start},
            "ceiling_hop": {"type": "noul", "noul": ceiling_hop},
            "sustain_jump": {"type": "noul", "noul": sustain},
        }

    def predict(self, state, questions, model=None, max_len=None):
        self.calls.append({"state": state, "model": model, "max_len": max_len})
        assert questions is LAYA_QUESTIONS
        return {
            "answers": self.answers,
            "usage": {"input_tokens": 500, "state_tokens": 120, "truncated": False},
            "routing": {"model": model},
        }


def full_state(grounded=True, held=False, ceiling=False):
    return {
        "mario": {
            "grounded": grounded,
            "motion": "grounded" if grounded else "rising",
            "vx_px_per_frame": 3.0,
            "vy_px_per_frame": 0.0,
        },
        "jump_already_held": held,
        "blocked_forward": False,
        "terrain": {"summary": {"nearest_obstacle": None}},
        "nearest_threat": {"kind": "goomba", "dx": 35, "dy": 0, "same_height": True},
        "jump_corridor": {
            "low_ceiling_now": ceiling,
            "low_ceiling_before_nearest_threat": ceiling,
            "minimum_headroom_before_threat_px": 32 if ceiling else None,
        },
        "recent_decisions": [
            {"action": "right_run", "after": {"grounded": True}, "events": []}
        ]
        * 5,
    }


@pytest.mark.parametrize(
    "grounded, held, ceiling, start, sustain, ceiling_hop, expected",
    [
        (True, False, False, 0.9, 0.1, 0.1, "right_run_jump"),
        (True, True, False, 0.9, 0.9, 0.1, "right_run"),
        (False, True, False, 0.1, 0.9, 0.1, "right_run_jump"),
        (False, True, False, 0.9, 0.1, 0.1, "right_run"),
        (True, False, True, 0.9, 0.1, 0.1, "right_run"),
        (True, False, True, 0.1, 0.1, 0.9, "right_run_jump"),
    ],
)
def test_answers_compose_into_action(
    grounded, held, ceiling, start, sustain, ceiling_hop, expected
):
    router = FakeRouter(start=start, sustain=sustain, ceiling_hop=ceiling_hop)
    policy = LayaPolicy("english", max_len=1024, router=router, preload=False)
    action, diagnostics = policy.choose(full_state(grounded, held, ceiling))
    assert action == expected
    assert router.calls[0]["model"] == "english"
    assert router.calls[0]["max_len"] == 1024
    assert diagnostics["probabilities"]["run_right"] == 0.8
    assert diagnostics["usage"]["state_tokens"] == 120
    assert diagnostics["decisions"]["jump_timing_source"] == (
        "ceiling_hop" if ceiling else "start_jump"
    )


def test_preload_runs_one_prediction():
    router = FakeRouter()
    LayaPolicy("english", router=router)
    assert len(router.calls) == 1


def test_unknown_model_and_movement_are_rejected():
    with pytest.raises(ValueError):
        LayaPolicy("jev-latest", router=FakeRouter(), preload=False)
    policy = LayaPolicy("english", router=FakeRouter("dance"), preload=False)
    with pytest.raises(RuntimeError, match="movement"):
        policy.choose(full_state())


def test_movement_options_match_controller():
    assert set(LAYA_QUESTIONS["movement"]["criteria"]) == set(MOVEMENT_BUTTONS)


def test_compact_state_keeps_hazards_first_and_drops_nulls():
    compact = laya_state(full_state(ceiling=True))
    keys = list(compact)
    assert keys.index("enemy_ahead") < keys.index("recent")
    assert "obstacle_ahead" not in compact
    assert compact["low_ceiling"]["headroom_px"] == 32
    assert len(compact["recent"]) == 2
    assert "events" not in compact["recent"][0]


def test_largest_recorded_state_fits_laya_budget():
    pytest.importorskip("laya")
    from huggingface_hub import try_to_load_from_cache

    if not isinstance(
        try_to_load_from_cache("convaiinnovations/laya", "rl_agent_config.json"), str
    ):
        pytest.skip("Laya checkpoint is not cached locally")
    policy = LayaPolicy("english")
    _, diagnostics = policy.choose(json.loads(FULL_STATE.read_text()))
    assert diagnostics["usage"]["truncated"] is False
    assert diagnostics["usage"]["truncated_questions"] == []
    assert diagnostics["decisions"]["movement"] in MOVEMENT_BUTTONS
