"""Isaac-free logic in isaac/*.py, tested with plain python (isaacsim is only imported inside main())."""
import importlib.util
import json
import math
import py_compile
import struct
import zlib
from pathlib import Path

import numpy as np
import pytest
from pxr import Gf, Usd, UsdGeom, UsdPhysics

from wetlab.plan import Plan
from wetlab.spec import DEFAULT_LAB, load_lab, pose_matrix, rpy_matrix
from wetlab.twin.kinematics import fk_base
from wetlab.twin.usd import write_stage

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "tests/fixtures/plan_small.json"


def _load(name):
    spec = importlib.util.spec_from_file_location(f"isaac_{name}", ROOT / "isaac" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


dry = _load("dry_run")
mirror = _load("mirror")
worker = _load("worker")


@pytest.mark.parametrize("name", ["dry_run", "mirror", "worker"])
def test_compiles(name):
    py_compile.compile(str(ROOT / "isaac" / f"{name}.py"), doraise=True)


def test_fk_matches_cpu_twin():
    rng = np.random.default_rng(1)
    for q in rng.uniform(-math.pi, math.pi, (20, 6)):
        np.testing.assert_allclose(dry.fk_tool0(q), fk_base(q)[-1], atol=1e-12)


def test_quat_roundtrip_incl_180():
    for rpy in ([0, 0, 180], [180, 0, 0], [0, 180, 0], [180, 0, -90], [10, 20, 30], [0, 0, 0]):
        R = rpy_matrix(rpy)
        np.testing.assert_allclose(dry.quat_to_matrix(dry.matrix_to_quat(R)), R, atol=1e-12)


def test_pose_error():
    A = pose_matrix([0, 0, 0], [0, 0, 0])
    B = pose_matrix([0.003, 0.004, 0], [0, 0, 2])
    p, r = dry.pose_error(A, B)
    assert p == pytest.approx(0.005) and r == pytest.approx(2.0)


def test_load_plan_fixture_is_a_valid_plan():
    Plan.model_validate_json(FIXTURE.read_text())
    p = dry.load_plan(FIXTURE)
    assert p["q"].shape == (6, 6) and p["dt"] == 0.02 and len(p["lab_sha"]) == 64  # fixture is static; real plans are sha-checked against the stage
    assert [e["kind"] for e in p["events"]] == ["step_start", "attach", "detach"]


def test_load_plan_rejects_empty(tmp_path):
    f = tmp_path / "p.json"
    f.write_text(json.dumps({"waypoints": []}))
    with pytest.raises(ValueError):
        dry.load_plan(f)


def test_render_indices():
    t = np.arange(0, 1.0001, 0.02)       # 51 waypoints at 50 Hz
    idx = dry.render_indices(t, 15)
    assert idx[0] == 0 and len(idx) == 16  # 0, 1/15, ..., 15/15 s
    assert all(t[i] + 1e-9 >= k / 15 for k, i in enumerate(idx))


def test_holder_follows_tool_between_attach_and_detach():
    T_lw = pose_matrix([1.45, 2.55, 0.9])
    h = dry.Holder({"plate-1": T_lw})
    T0 = pose_matrix([1.45, 2.55, 1.1], [180, 0, 0])
    h.event({"kind": "attach", "data": {"labware": "plate-1", "location": "nest-A1"}}, T0)
    T1 = pose_matrix([2.3, 2.32, 1.3], [180, 0, 90])
    moved = h.update(T1)
    np.testing.assert_allclose(np.linalg.inv(T1) @ moved["plate-1"], np.linalg.inv(T0) @ T_lw, atol=1e-12)
    h.event({"kind": "detach", "data": {"labware": "plate-1"}}, T1)
    assert h.update(pose_matrix([0, 0, 2])) == {}
    np.testing.assert_allclose(h.poses["plate-1"], moved["plate-1"])


def test_contact_pairs_filters_self_world_and_ignored():
    r = "/World/ArmMount/arm_1/ur5e"
    raw = [(0.1, f"{r}/base_link/collisions", "/World/Room/bench_1"),   # ignored: base on its bench
           (0.2, f"{r}/forearm_link/c", f"{r}/wrist_1_link/c"),           # self
           (0.3, "/World/Room/floor", "/World/Room/bench_1"),             # world-world
           (0.4, "/World/Labware/plate_2/body", f"{r}/wrist_3_link/c"),
           (0.42, f"{r}/wrist_3_link/c", "/World/Labware/plate_2/body")]
    out = dry.contact_pairs(raw, r, {(f"{r}/base_link", "/World/Room/bench_1")})
    assert out == [{"a": f"{r}/wrist_3_link/c", "b": "/World/Labware/plate_2/body", "first_t": 0.4, "n": 2}]


def test_write_png(tmp_path):
    img = np.zeros((3, 4, 3), np.uint8)
    img[1, 2] = [255, 10, 20]
    dry.write_png(tmp_path / "a.png", img)
    b = (tmp_path / "a.png").read_bytes()
    assert b[:8] == b"\x89PNG\r\n\x1a\n" and struct.unpack(">II", b[16:24]) == (4, 3)
    raw = zlib.decompress(b[41:-12])
    assert raw[1 * 13 + 1 + 2 * 3: 1 * 13 + 1 + 3 * 3] == bytes([255, 10, 20])


def test_anchor_world_joints_follows_moved_robot():
    """The ur5e.usd root_joint (body0 = world at the origin) must be re-anchored to the mounted base."""
    s = Usd.Stage.CreateInMemory()
    m = UsdGeom.Xform.Define(s, "/World/mount")
    m.AddTranslateOp().Set(Gf.Vec3d(2.0, 2.8, 0.9))
    m.AddRotateZOp().Set(-90.0)
    r = UsdGeom.Xform.Define(s, "/World/mount/ur5e")
    r.AddRotateZOp().Set(180.0)
    UsdPhysics.RigidBodyAPI.Apply(UsdGeom.Xform.Define(s, "/World/mount/ur5e/base_link").GetPrim())
    j = UsdPhysics.FixedJoint.Define(s, "/World/mount/ur5e/root_joint")
    j.CreateBody1Rel().SetTargets(["/World/mount/ur5e/base_link"])
    j.CreateLocalPos0Attr(Gf.Vec3f(0))
    j.CreateLocalRot0Attr(Gf.Quatf(1))
    assert dry.anchor_world_joints(s, "/World/mount/ur5e") == ["/World/mount/ur5e/root_joint"]
    np.testing.assert_allclose(j.GetLocalPos0Attr().Get(), [2.0, 2.8, 0.9], atol=1e-6)
    R = np.array(Gf.Matrix3d(Gf.Rotation(Gf.Quatd(j.GetLocalRot0Attr().Get())))).T
    np.testing.assert_allclose(R, rpy_matrix([0, 0, 90]), atol=1e-6)   # -90 + 180


def test_tagged_and_stage_paths(tmp_path):
    lab = load_lab(DEFAULT_LAB)
    s = Usd.Stage.Open(str(write_stage(lab, tmp_path / "c.usda")))
    assert dry.tagged(s, "/World/Labware") == {"plate-1": "/World/Labware/plate_1", "plate-2": "/World/Labware/plate_2"}
    assert set(dry.tagged(s, "/World/Nests")) == {l.id for l in lab.locations}
    assert dry.tagged(s, "/World/Room")[lab.arm.mount] == "/World/Room/bench_1"


# -- mirror -------------------------------------------------------------------------
def test_mirror_holder_events_from_state_changes():
    assert mirror.held_events(None, "plate-1") == [{"kind": "attach", "data": {"labware": "plate-1"}}]
    assert mirror.held_events("plate-1", None) == [{"kind": "detach", "data": {"labware": "plate-1"}}]
    assert [e["kind"] for e in mirror.held_events("plate-1", "plate-2")] == ["detach", "attach"]
    assert mirror.held_events("plate-1", "plate-1") == []


def test_ws_frame_roundtrip():
    payload = json.dumps({"t": 1.0, "q": [0] * 6}).encode() * 10   # > 125 bytes: extended length
    frame = mirror.ws_frame(1, payload, mask=b"\x01\x02\x03\x04")
    import io
    op, data = mirror.ws_read(io.BytesIO(mirror.ws_frame(1, payload, mask=None)).read)
    assert (op, data) == (1, payload)
    # masked client frame: unmask by hand
    assert frame[1] & 0x80 and len(frame) == 2 + 2 + 4 + len(payload)


def test_ws_url():
    assert mirror.ws_url("http://h:8000", "cell-a") == "ws://h:8000/v1/cells/cell-a/ws"
    assert mirror.ws_url("https://h", "cell-a") == "wss://h/v1/cells/cell-a/ws"


# -- worker -------------------------------------------------------------------------
def test_worker_multipart_and_stage_sha(tmp_path):
    (tmp_path / "report.json").write_text('{"ok": true}')
    body, ctype = worker.multipart({"report": tmp_path / "report.json"})
    boundary = ctype.split("boundary=")[1]
    assert body.startswith(f"--{boundary}\r\n".encode()) and body.rstrip().endswith(f"--{boundary}--".encode())
    assert b'name="report"; filename="report.json"' in body and b'{"ok": true}' in body
    lab = load_lab(DEFAULT_LAB)
    p = write_stage(lab, tmp_path / "s.usda")
    assert worker.stage_sha(p) == lab.sha256()
    assert worker.stage_sha(tmp_path / "missing.usda") is None
