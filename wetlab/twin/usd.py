"""Author the Isaac Sim twin stage (USD) from the lab spec, with usd-core only (no GPU).

Every box comes from wetlab.geometry, so this stage equals the CPU twin. Conventions:
  * stage Z-up, metersPerUnit = 1, frame = the spec's room frame.
  * prim names are Tf-valid ids ("door-1" -> "door_1"); the real id is in attribute wetlab:id.
  * camera: USD looks down -Z with +Y up; OpenCV looks down +Z with +Y down, so
    T_room_usdcam = T_room_cam (OpenCV) @ Rx(180).
    Intrinsics with pixel pitch p = HAPERTURE_MM / width (mm/px):
      focalLength = fx * p, horizontalAperture = width * p, verticalAperture = height * p * fx / fy
      horizontalApertureOffset = (width / 2 - cx) * p, verticalApertureOffset = (cy - height / 2) * p * fx / fy
    USD cannot carry distortion: customData wetlab:opencv holds {fx, fy, cx, cy, dist, width, height}
    for isaac/dry_run.py to apply with Camera.set_opencv_pinhole_properties.
  * arm: /World/ArmMount/<arm> sits at the UR controller "base" frame (our DH base). The Isaac UR5e
    asset root is ur_description's base_link; base_link -> base is a 180 deg yaw there, so the asset
    is referenced under the mount rotated by wetlab:base_yaw_offset_deg = 180 (verified on the GPU box
    by dry_run's fk_check).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, Tf, Usd, UsdGeom, UsdLux, UsdPhysics, Vt

from ..geometry import Box, door_box, door_hinge, labware_box, location_frame, static_boxes
from ..spec import DEFAULT_LAB, LabSpec, load_lab, matrix_to_quat
from .kinematics import fk_base

HAPERTURE_MM = 20.955          # Isaac/Omniverse default film back width; only the ratios matter
UR5E_ASSET = "/Isaac/Robots/UniversalRobots/ur5e/ur5e.usd"   # relative to get_assets_root_path()
BASE_YAW_OFFSET_DEG = 180.0    # ur_description: base_link -> base = Rz(pi)

COLORS = {"floor": (0.35, 0.35, 0.37), "wall": (0.85, 0.85, 0.82), "ceiling": (0.9, 0.9, 0.9),
          "bench": (0.55, 0.6, 0.65), "instrument": (0.2, 0.45, 0.8), "nest": (0.95, 0.6, 0.1),
          "door": (0.6, 0.4, 0.25), "labware": (0.9, 0.9, 0.95), "arm": (0.3, 0.6, 0.85)}
CV_TO_USD_CAM = np.diag([1.0, -1.0, -1.0, 1.0])   # Rx(180)


def name(id_: str) -> str:
    return Tf.MakeValidIdentifier(id_)


def _set_pose(prim, T: np.ndarray, scale=None) -> None:
    x = UsdGeom.Xformable(prim)
    x.ClearXformOpOrder()
    x.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*map(float, T[:3, 3])))
    w, *v = matrix_to_quat(T[:3, :3])
    x.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(w, Gf.Vec3d(*v)))
    if scale is not None:
        x.AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*map(float, scale)))


def _tag(prim, id_: str, kind: str) -> None:
    prim.CreateAttribute("wetlab:id", Sdf.ValueTypeNames.String, custom=True).Set(id_)
    prim.CreateAttribute("wetlab:kind", Sdf.ValueTypeNames.String, custom=True).Set(kind)


def _cube(stage, path: str, T: np.ndarray, size, kind: str, collide: bool):
    c = UsdGeom.Cube.Define(stage, path)
    c.CreateSizeAttr(1.0)
    c.CreateExtentAttr([(-0.5, -0.5, -0.5), (0.5, 0.5, 0.5)])
    c.CreateDisplayColorAttr([COLORS.get(kind, (0.7, 0.7, 0.7))])
    _set_pose(c.GetPrim(), T, size)
    if collide:
        UsdPhysics.CollisionAPI.Apply(c.GetPrim())
    return c


def _box_cube(stage, parent: str, b: Box):
    c = _cube(stage, f"{parent}/{name(b.id)}", b.T, b.size, b.kind, b.collide)
    _tag(c.GetPrim(), b.id, b.kind)
    return c


def _author_door(stage, lab: LabSpec, door) -> None:
    closed = door_box(lab, door, 0.0)
    hinge, sign = door_hinge(lab, door)
    path = f"/World/Door/{name(door.id)}"
    panel = UsdGeom.Xform.Define(stage, path).GetPrim()
    _set_pose(panel, closed.T)
    _tag(panel, door.id, "door")
    panel.SetCustomDataByKey("wetlab:open_sign", sign)
    UsdPhysics.RigidBodyAPI.Apply(panel)
    UsdPhysics.MassAPI.Apply(panel).CreateMassAttr(25.0)
    _cube(stage, f"{path}/panel", np.eye(4), closed.size, "door", True)
    # the panel swings through its own wall's edges: never collide with that wall
    walls = [f"/World/Room/{name(b.id)}" for b in static_boxes(lab) if b.id.startswith(f"wall-{door.wall}-")]
    UsdPhysics.FilteredPairsAPI.Apply(panel).CreateFilteredPairsRel().SetTargets([Sdf.Path(w) for w in walls])

    j = UsdPhysics.RevoluteJoint.Define(stage, f"{path}/hinge")
    j.CreateBody1Rel().SetTargets([Sdf.Path(path)])      # body0 unset = world; localPos0 is in world
    j.CreateAxisAttr("Z")
    j.CreateLocalPos0Attr(Gf.Vec3f(*map(float, hinge)))
    w, *v = matrix_to_quat(closed.T[:3, :3])
    j.CreateLocalRot0Attr(Gf.Quatf(w, Gf.Vec3f(*v)))
    j.CreateLocalPos1Attr(Gf.Vec3f(-door.width / 2, 0.0, -door.height / 2))
    j.CreateLocalRot1Attr(Gf.Quatf(1.0))
    lo, hi = sorted([0.0, sign * door.max_open_deg])
    j.CreateLowerLimitAttr(lo)
    j.CreateUpperLimitAttr(hi)
    drive = UsdPhysics.DriveAPI.Apply(j.GetPrim(), "angular")
    drive.CreateTypeAttr("force")
    drive.CreateTargetPositionAttr(0.0)    # degrees; open = open_sign * angle
    drive.CreateStiffnessAttr(1e4)
    drive.CreateDampingAttr(1e3)
    drive.CreateMaxForceAttr(1e5)


def _author_camera(stage, cam) -> None:
    W, H = cam.resolution
    i = cam.intrinsics
    p = HAPERTURE_MM / W
    c = UsdGeom.Camera.Define(stage, f"/World/Cameras/{name(cam.id)}")
    _set_pose(c.GetPrim(), cam.T_room_cam() @ CV_TO_USD_CAM)
    c.CreateProjectionAttr("perspective")
    c.CreateFocalLengthAttr(i.fx * p)
    c.CreateHorizontalApertureAttr(W * p)
    c.CreateVerticalApertureAttr(H * p * i.fx / i.fy)
    c.CreateHorizontalApertureOffsetAttr((W / 2 - i.cx) * p)
    c.CreateVerticalApertureOffsetAttr((i.cy - H / 2) * p * i.fx / i.fy)
    c.CreateClippingRangeAttr(Gf.Vec2f(0.05, 20.0))
    prim = c.GetPrim()
    _tag(prim, cam.id, "camera")
    prim.SetCustomDataByKey("wetlab:opencv", {
        "fx": i.fx, "fy": i.fy, "cx": i.cx, "cy": i.cy, "dist": Vt.DoubleArray(i.dist),
        "width": int(W), "height": int(H), "fps": float(cam.fps)})
    prim.SetCustomDataByKey("wetlab:calibration", cam.calibration.status)


def _author_arm(stage, lab: LabSpec) -> None:
    arm = lab.arm
    T_base = arm.base.matrix()
    mount = UsdGeom.Xform.Define(stage, f"/World/ArmMount/{name(arm.id)}").GetPrim()
    _set_pose(mount, T_base)
    _tag(mount, arm.id, "arm_mount")
    for k, v in {"arm_model": arm.model, "isaac_asset": UR5E_ASSET, "base_yaw_offset_deg": BASE_YAW_OFFSET_DEG,
                 "home_q_deg": Vt.DoubleArray(arm.home_q_deg), "tcp_offset_m": arm.tool.tcp_offset_m,
                 "mount": arm.mount}.items():
        mount.SetCustomDataByKey(f"wetlab:{k}", v)

    # placeholder for non-Isaac viewers: capsules between joint origins at home (dry_run deactivates it)
    proxy = UsdGeom.Xform.Define(stage, "/World/ArmProxy")
    proxy.GetPrim().SetCustomDataByKey("wetlab:note", "placeholder; Isaac uses the UR5e asset under /World/ArmMount")
    pts = [(T_base @ F)[:3, 3] for F in fk_base(np.radians(arm.home_q_deg))]
    for k, (a, b) in enumerate(zip(pts[:-1], pts[1:])):
        ln = float(np.linalg.norm(b - a))
        if ln < 1e-6:
            continue
        z = (b - a) / ln
        x = np.cross([0.0, 0.0, 1.0] if abs(z[2]) < 0.9 else [1.0, 0.0, 0.0], z)
        x /= np.linalg.norm(x)
        T = np.eye(4)
        T[:3, :3] = np.column_stack([x, np.cross(z, x), z])
        T[:3, 3] = (a + b) / 2
        cap = UsdGeom.Capsule.Define(stage, f"/World/ArmProxy/link{k}")
        cap.CreateAxisAttr("Z")
        cap.CreateHeightAttr(ln)
        cap.CreateRadiusAttr(0.045)
        cap.CreateDisplayColorAttr([COLORS["arm"]])
        _set_pose(cap.GetPrim(), T)


def write_stage(lab: LabSpec, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, "/World").GetPrim()
    stage.SetDefaultPrim(world)
    world.SetCustomDataByKey("wetlab:lab_sha", lab.sha256())
    world.SetCustomDataByKey("wetlab:cell", lab.id)
    world.SetCustomDataByKey("wetlab:survey", lab.survey.status)
    world.SetCustomDataByKey("wetlab:survey_date", lab.survey.date)

    scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    scene.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
    scene.CreateGravityMagnitudeAttr(9.81)

    UsdGeom.Scope.Define(stage, "/World/Room")
    for b in static_boxes(lab):
        _box_cube(stage, "/World/Room", b)

    for door in lab.doors:
        _author_door(stage, lab, door)

    for loc in lab.locations:
        n = UsdGeom.Xform.Define(stage, f"/World/Nests/{name(loc.id)}").GetPrim()
        _set_pose(n, location_frame(lab, loc.id))
        _tag(n, loc.id, loc.kind)

    for lw in lab.labware:
        T = location_frame(lab, lw.at)
        b = labware_box(lab, lw.id, T)
        p = f"/World/Labware/{name(lw.id)}"
        x = UsdGeom.Xform.Define(stage, p).GetPrim()
        _set_pose(x, T)    # labware frame: bottom-face centre
        _tag(x, lw.id, "labware")
        x.SetCustomDataByKey("wetlab:type", lw.type)
        x.SetCustomDataByKey("wetlab:at", lw.at)
        UsdPhysics.RigidBodyAPI.Apply(x).CreateKinematicEnabledAttr(True)
        UsdPhysics.MassAPI.Apply(x).CreateMassAttr(0.1)
        local = np.eye(4)
        local[2, 3] = b.size[2] / 2
        _cube(stage, f"{p}/body", local, b.size, "labware", True)

    for cam in lab.cameras:
        _author_camera(stage, cam)

    _author_arm(stage, lab)

    X, Y, Z = lab.room.size
    UsdLux.DomeLight.Define(stage, "/World/Lights/dome").CreateIntensityAttr(300.0)
    rect = UsdLux.RectLight.Define(stage, "/World/Lights/ceiling")   # emits along its -Z: down
    rect.CreateWidthAttr(1.2)
    rect.CreateHeightAttr(0.6)
    rect.CreateIntensityAttr(15000.0)
    _set_pose(rect.GetPrim(), np.array([[1, 0, 0, X / 2], [0, 1, 0, Y / 2], [0, 0, 1, Z - 0.02], [0, 0, 0, 1.0]]))

    stage.GetRootLayer().Save()
    return path


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Write the Isaac Sim twin stage from a lab spec.")
    ap.add_argument("--lab", default=str(DEFAULT_LAB))
    ap.add_argument("--out", default=None, help="default: out/<cell>.usda")
    a = ap.parse_args(argv)
    lab = load_lab(a.lab)
    print(write_stage(lab, a.out or f"out/{lab.id}.usda"))


if __name__ == "__main__":
    main()
