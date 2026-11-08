import json
from pathlib import Path

import numpy as np
import pytest

from wetlab.ir import Program
from wetlab.plan import DT
from wetlab.spec import DEFAULT_LAB, load_lab
from wetlab.twin.collision import clearance
from wetlab.twin.planner import initial_state, plan_program

DEMO = Path(__file__).resolve().parent.parent / "examples" / "demo_protocol.json"


@pytest.fixture(scope="module")
def lab():
    return load_lab(DEFAULT_LAB)


def plan(lab, *steps):
    return plan_program(lab, Program(steps=list(steps)))


def codes(p):
    return [(v.step, v.code) for v in p.violations]


def test_demo_program_is_ok(lab):
    prog = Program.model_validate(json.loads(DEMO.read_text()))
    p = plan_program(lab, prog)
    assert p.ok, p.violations
    assert 10 < p.duration_s < 120 and p.duration_s == p.waypoints[-1].t
    assert p.final_state.labware_at == initial_state(lab).labware_at and p.final_state.held is None
    np.testing.assert_allclose(np.diff([w.t for w in p.waypoints]), DT, atol=1e-9)
    kinds = [e.kind for e in p.events]
    assert kinds.count("step_start") == len(prog.steps)
    assert kinds.count("attach") == kinds.count("detach") == 2 and "capture" in kinds and "instrument" in kinds
    shake = next(e for e in p.events if e.kind == "instrument")
    assert next(e for e in p.events if e.kind == "capture").t == pytest.approx(shake.t + 3.0)
    assert p.stats["max_tcp_speed_mps"] <= lab.arm.limits.tcp_speed_mps
    assert p.stats["min_clearance_m"] >= lab.safety.min_clearance_m
    assert p.stats["n_waypoints"] == len(p.waypoints) and p.lab_sha == lab.sha256() and p.ir_sha == prog.sha256()
    # the held plate sits on the shaker between the place and the next pick
    attach = [e for e in p.events if e.kind == "attach"]
    assert attach[1].data == {"labware": "plate-1", "location": "shaker-1"}


def test_unreachable_point_stops_plan(lab):
    p = plan(lab, {"op": "move_to", "target": {"xyz": [3.5, 0.5, 1.0]}}, {"op": "home"})
    assert codes(p) == [(0, "unreachable")]
    assert [e.step for e in p.events if e.kind == "step_start"] == [0]


def test_aisle_keepout(lab):
    p = plan(lab, {"op": "move_to", "target": {"xyz": [2.0, 2.0, 1.1]}})
    assert (0, "keepout") in codes(p) and not p.ok


def test_collision_with_bench(lab):
    p = plan(lab, {"op": "move_to", "target": {"xyz": [2.3, 2.5, 0.9], "z_above": 0.0}})
    assert (0, "collision") in codes(p)
    assert p.stats["min_clearance_pair"] == ["tool", "bench-1"] and p.stats["min_clearance_m"] < 0


def test_place_on_occupied_nest(lab):
    p = plan(lab, {"op": "pick", "labware": "plate-1"}, {"op": "place", "labware": "plate-1", "at": "nest-A2"})
    assert codes(p) == [(1, "state")] and "occupied by plate-2" in p.violations[0].message
    assert p.final_state.held == "plate-1"


def test_pick_while_holding(lab):
    p = plan(lab, {"op": "pick", "labware": "plate-1"}, {"op": "pick", "labware": "plate-2"})
    assert codes(p) == [(1, "state")] and "already holding plate-1" in p.violations[0].message


def test_unknown_refs(lab):
    assert codes(plan(lab, {"op": "pick", "labware": "plate-9"})) == [(0, "unknown_ref")]
    assert codes(plan(lab, {"op": "capture", "camera": "nope"})) == [(0, "unknown_ref")]


def test_clearance_helper(lab):
    st = initial_state(lab)
    d, pair = clearance(lab, st.q, st)
    assert d > lab.safety.min_clearance_m and pair[0]


def test_target_and_camera_edge_cases():
    import pydantic
    from wetlab.ir import Target
    from wetlab.spec import Camera
    for bad in ({}, {"xyz": [1, 2, 1], "labware": "plate-1"}, {"location": "nest-A1", "well": "A1"}):
        with pytest.raises(pydantic.ValidationError):
            Target(**bad)
    cam = Camera(id="c", resolution=[64, 48], intrinsics={"fx": 50, "fy": 50, "cx": 32, "cy": 24},
                 extrinsics={"position": [1, 1, 2], "look_at": [1, 1, 0]})   # straight down
    T = cam.T_room_cam()
    assert np.isfinite(T).all() and np.allclose(T[:3, 2], [0, 0, -1])
