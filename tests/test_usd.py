
import numpy as np
import pytest
from pxr import Gf, Usd, UsdGeom, UsdPhysics

from wetlab.geometry import door_box, door_hinge, labware_box, location_frame, static_boxes
from wetlab.spec import DEFAULT_LAB, load_lab, pose_matrix
from wetlab.twin.usd import name, write_stage


@pytest.fixture(scope="module")
def lab():
    return load_lab(DEFAULT_LAB)


@pytest.fixture(scope="module")
def stage(lab, tmp_path_factory):
    p = write_stage(lab, tmp_path_factory.mktemp("usd") / "cell.usda")
    return Usd.Stage.Open(str(p))


def _bound(stage, path):
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    r = cache.ComputeWorldBound(stage.GetPrimAtPath(path)).ComputeAlignedRange()
    return np.array(r.GetMin()), np.array(r.GetMax())


def _world(stage, path):
    return np.array(UsdGeom.Xformable(stage.GetPrimAtPath(path)).ComputeLocalToWorldTransform(0)).T


def _assert_box(stage, path, box):
    c = box.corners()
    lo, hi = _bound(stage, path)
    np.testing.assert_allclose(lo, c.min(0), atol=1e-6)
    np.testing.assert_allclose(hi, c.max(0), atol=1e-6)
    # full pose too (aligned bounds alone would miss a 180 deg flip)
    W = _world(stage, path)
    np.testing.assert_allclose(W[:3, :3], box.T[:3, :3] * box.size, atol=1e-6)
    np.testing.assert_allclose(W[:3, 3], box.T[:3, 3], atol=1e-6)


def test_stage_header_and_prims(lab, stage):
    w = stage.GetDefaultPrim()
    assert w.GetPath() == "/World"
    assert UsdGeom.GetStageUpAxis(stage) == "Z" and UsdGeom.GetStageMetersPerUnit(stage) == 1.0
    assert w.GetCustomDataByKey("wetlab:lab_sha") == lab.sha256()
    assert w.GetCustomDataByKey("wetlab:cell") == lab.id
    assert w.GetCustomDataByKey("wetlab:survey") == lab.survey.status
    for p in ["/World/PhysicsScene", "/World/ArmProxy", "/World/Lights/dome", "/World/Lights/ceiling",
              f"/World/ArmMount/{name(lab.arm.id)}", f"/World/Cameras/{name(lab.camera().id)}"]:
        assert stage.GetPrimAtPath(p).IsValid(), p
    for loc in lab.locations:
        np.testing.assert_allclose(_world(stage, f"/World/Nests/{name(loc.id)}"), location_frame(lab, loc.id), atol=1e-9)
    m = stage.GetPrimAtPath(f"/World/ArmMount/{name(lab.arm.id)}")
    np.testing.assert_allclose(_world(stage, m.GetPath()), lab.arm.base.matrix(), atol=1e-9)
    assert m.GetCustomDataByKey("wetlab:isaac_asset") == "/Isaac/Robots/UniversalRobots/ur5e/ur5e.usd"
    assert m.GetCustomDataByKey("wetlab:base_yaw_offset_deg") == 180.0


def test_static_boxes_match_geometry(stage):
    lab = load_lab(DEFAULT_LAB)
    boxes = static_boxes(lab)
    room = stage.GetPrimAtPath("/World/Room").GetChildren()
    assert len(room) == len(boxes)
    for b in boxes:
        p = stage.GetPrimAtPath(f"/World/Room/{name(b.id)}")
        assert p.GetAttribute("wetlab:id").Get() == b.id and p.GetAttribute("wetlab:kind").Get() == b.kind
        assert p.HasAPI(UsdPhysics.CollisionAPI) == b.collide
        _assert_box(stage, p.GetPath(), b)


def test_labware_and_door_boxes(lab, stage):
    for lw in lab.labware:
        p = f"/World/Labware/{name(lw.id)}"
        assert stage.GetPrimAtPath(p).GetAttribute("physics:kinematicEnabled").Get()
        _assert_box(stage, f"{p}/body", labware_box(lab, lw.id, location_frame(lab, lw.at)))
    for d in lab.doors:
        _assert_box(stage, f"/World/Door/{name(d.id)}/panel", door_box(lab, d, 0.0))


def test_door_joint_limits_sign_and_swing(lab, stage):
    d = lab.doors[0]
    j = UsdPhysics.RevoluteJoint(stage.GetPrimAtPath(f"/World/Door/{name(d.id)}/hinge"))
    _, sign = door_hinge(lab, d)
    assert sign == -1.0   # south wall, outward, left hinge: opens toward -y = clockwise about +Z
    lo, hi = j.GetLowerLimitAttr().Get(), j.GetUpperLimitAttr().Get()
    assert (lo, hi) == ((-d.max_open_deg, 0.0) if sign < 0 else (0.0, d.max_open_deg))
    assert j.GetAxisAttr().Get() == "Z"
    drive = UsdPhysics.DriveAPI(j.GetPrim(), "angular")
    assert drive.GetTargetPositionAttr().Get() == 0.0

    def frame(pos, rot):
        T = np.eye(4)
        T[:3, :3] = np.array(Gf.Matrix3d(Gf.Rotation(Gf.Quatd(rot)))).T
        T[:3, 3] = pos
        return T

    J0 = frame(j.GetLocalPos0Attr().Get(), j.GetLocalRot0Attr().Get())
    J1 = frame(j.GetLocalPos1Attr().Get(), j.GetLocalRot1Attr().Get())
    for ang in (0.0, 30.0, d.max_open_deg):
        # PhysX revolute: body1 = joint0 * Rz(angle) * inv(joint1); open angle = sign * ang
        T = J0 @ pose_matrix([0, 0, 0], [0, 0, sign * ang]) @ np.linalg.inv(J1)
        np.testing.assert_allclose(T, door_box(lab, d, ang).T, atol=1e-6)


def test_camera_roundtrip(lab, stage):
    cam = lab.camera()
    W, H = cam.resolution
    gc = UsdGeom.Camera(stage.GetPrimAtPath(f"/World/Cameras/{name(cam.id)}")).GetCamera(Usd.TimeCode.Default())
    fr = gc.frustum
    M = fr.ComputeViewMatrix() * fr.ComputeProjectionMatrix()
    K, Tinv = cam.intrinsics.K(), np.linalg.inv(cam.T_room_cam())
    rng = np.random.default_rng(0)
    look = np.array(cam.extrinsics.look_at)
    pts = look + rng.uniform([-1.2, -0.6, -0.9], [1.2, 0.5, 0.8], size=(10, 3))
    for p in pts:
        pc = Tinv[:3, :3] @ p + Tinv[:3, 3]
        assert pc[2] > 0
        u_spec = (K @ pc)[:2] / pc[2]
        ndc = M.Transform(Gf.Vec3d(*p))
        u_usd = np.array([(ndc[0] + 1) / 2 * W, (1 - ndc[1]) / 2 * H])
        assert np.abs(u_spec - u_usd).max() < 0.5, (p, u_spec, u_usd)
    cv = stage.GetPrimAtPath(f"/World/Cameras/{name(cam.id)}").GetCustomDataByKey("wetlab:opencv")
    assert cv["fx"] == cam.intrinsics.fx and list(cv["dist"]) == cam.intrinsics.dist
    assert (cv["width"], cv["height"]) == (W, H)
    assert (gc.clippingRange.min, gc.clippingRange.max) == (pytest.approx(0.05), pytest.approx(20.0))


def test_camera_roundtrip_off_centre_principal_point(lab, tmp_path):
    """Principal-point offset and fx != fy must survive the aperture mapping (sign check)."""
    lab2 = lab.model_copy(deep=True)
    i = lab2.cameras[0].intrinsics
    i.fx, i.fy, i.cx, i.cy = 700.0, 760.0, 610.0, 385.0
    test_camera_roundtrip(lab2, Usd.Stage.Open(str(write_stage(lab2, tmp_path / "c.usda"))))


def test_cli(tmp_path):
    from wetlab.twin.usd import main
    out = tmp_path / "x.usda"
    main(["--out", str(out)])
    assert Usd.Stage.Open(str(out)).GetDefaultPrim().GetCustomDataByKey("wetlab:lab_sha") == load_lab(DEFAULT_LAB).sha256()
