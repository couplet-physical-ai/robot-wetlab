import time

import numpy as np
import pytest

from wetlab.ir import Program
from wetlab.media import encode_jpeg
from wetlab.spec import DEFAULT_LAB, load_lab
from wetlab.twin.planner import initial_state, plan_program
from wetlab.twin.render import COLORS, LABWARE_COLORS, LIGHT, TwinRenderer, project_points, render_plan_video


@pytest.fixture(scope="module")
def lab():
    return load_lab(DEFAULT_LAB)


def test_render_shape_and_speed(lab):
    r = TwinRenderer(lab)
    st = initial_state(lab)
    frame = r.render(st, overlay="test")
    w, h = lab.camera().resolution
    assert frame.shape == (h, w, 3) and frame.dtype == np.uint8 and frame.std() > 10
    assert encode_jpeg(frame)[:2] == b"\xff\xd8"
    ts = []
    for _ in range(5):
        t0 = time.perf_counter()
        r.render(st, door_angle_deg=45.0)
        ts.append(time.perf_counter() - t0)
    assert np.median(ts) < 0.040


def test_project_points_hand_computed(lab):
    cam = lab.camera()
    T = cam.T_room_cam()
    pc = np.array([0.2, -0.1, 2.0])                       # camera-frame point
    p_room = T[:3, :3] @ pc + T[:3, 3]
    i = cam.intrinsics
    x, y = 0.2 / 2.0, -0.1 / 2.0
    r2 = x * x + y * y                                     # 0.0125
    radial = 1 + i.dist[0] * r2 + i.dist[1] * r2**2         # p1 = p2 = k3 = 0 in the spec
    expected = [i.fx * x * radial + i.cx, i.fy * y * radial + i.cy]
    np.testing.assert_allclose(project_points(lab, cam, [p_room])[0], expected, atol=1e-9)
    # the look-at point is on the optical axis -> principal point
    np.testing.assert_allclose(project_points(lab, None, [cam.extrinsics.look_at])[0], [i.cx, i.cy], atol=1e-9)


def test_render_matches_project_points(lab):
    """Known room points land where project_points says, showing the surface they belong to."""
    frame = TwinRenderer(lab).render(initial_state(lab))
    top = 0.5 + 0.5 * LIGHT[2]                               # lambert shade of an upward face
    nest = lab.location("nest-A1")
    cases = [((1.3, 2.25, 0.9), COLORS["bench"]),               # free bench-top spot near the front-left
             ((nest.xy[0], nest.xy[1], nest.top + 0.01435), LABWARE_COLORS[0]),  # plate-1 top centre
             ((3.0, 1.5, 0.0), COLORS["floor"])]                     # floor right of the bench
    for p, color in cases:
        u, v = np.round(project_points(lab, None, [p])[0]).astype(int)
        assert 0 <= u < frame.shape[1] and 0 <= v < frame.shape[0], (p, u, v)
        np.testing.assert_allclose(frame[v, u], [int(c * top) for c in color], atol=2, err_msg=str(p))


def test_render_plan_video(lab, tmp_path):
    p = plan_program(lab, Program(steps=[{"op": "move_joints", "q_deg": [10, -100, 100, -90, -90, 0]}, {"op": "home"}]))
    assert p.ok
    out = render_plan_video(lab, p, tmp_path / "dry-run.mp4")
    data = out.read_bytes()
    assert len(data) > 1000 and data[4:8] == b"ftyp"
