"""Isaac Sim 5.1 dry-run of a wetlab Plan on the generated twin stage.

    $ISAAC_SIM_ROOT/python.sh isaac/dry_run.py --stage out/cell-a.usda --plan plan.json --out outdir [--headless] [--fps 15]
    $ISAAC_SIM_ROOT/python.sh isaac/dry_run.py --stage out/cell-a.usda --out outdir --render-only --headless

Kinematic playback: every 50 Hz waypoint teleports the UR5e joints (positions + drive targets),
held labware follows our FK TCP rigidly between attach/detach events, the door drive stays at
its authored target (closed), and PhysX contact reports catch any arm-link contact with the
room/benches/labware. Writes video.mp4 (ffmpeg pipe; PNG frames if ffmpeg is missing), or
render.png in --render-only mode, plus report.json.

Isaac-free helpers (plan parsing, FK, attach tracking, PNG) live at module level and are unit
tested in tests/test_isaac_helpers.py; this file never imports wetlab (Isaac's python has no venv).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import struct
import subprocess
import sys
import time
import zlib
from pathlib import Path

import numpy as np

UR_JOINTS = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
# UR5e DH (same values as wetlab/twin/usd.py and the CPU twin): d, a, alpha
DH_D = [0.1625, 0.0, 0.0, 0.1333, 0.0997, 0.0996]
DH_A = [0.0, -0.425, -0.3922, 0.0, 0.0, 0.0]
DH_ALPHA = [math.pi / 2, 0.0, 0.0, math.pi / 2, -math.pi / 2, 0.0]
TOOL_LINK = "wrist_3_link"   # in the 5.1 asset wrist_3_link == ur_description tool0 == DH frame 6 (checked at q=0)


# -- Isaac-free helpers -----------------------------------------------------------
def fk_tool0(q) -> np.ndarray:
    """4x4 tool0 (flange) pose in the UR 'base' frame for joint angles q (rad)."""
    T = np.eye(4)
    for qi, d, a, al in zip(q, DH_D, DH_A, DH_ALPHA):
        ct, st, ca, sa = math.cos(qi), math.sin(qi), math.cos(al), math.sin(al)
        T = T @ np.array([[ct, -st * ca, st * sa, a * ct], [st, ct * ca, -ct * sa, a * st], [0, sa, ca, d], [0, 0, 0, 1.0]])
    return T


def quat_to_matrix(q_wxyz) -> np.ndarray:
    w, x, y, z = np.asarray(q_wxyz, float) / np.linalg.norm(q_wxyz)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def matrix_to_quat(R) -> np.ndarray:
    """3x3 rotation -> [w, x, y, z]."""
    R = np.asarray(R, float)
    t = np.trace(R)
    if t > 0:   # Shepperd: pick the largest component to stay stable near 180 deg
        s = math.sqrt(t + 1.0) * 2
        return np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    i = int(np.argmax(np.diag(R)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = math.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k]) * 2
    q = np.zeros(4)
    q[0], q[i + 1], q[j + 1], q[k + 1] = (R[k, j] - R[j, k]) / s, 0.25 * s, (R[j, i] + R[i, j]) / s, (R[k, i] + R[i, k]) / s
    return q


def pose(pos, quat_wxyz) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = quat_to_matrix(quat_wxyz), pos
    return T


def pose_error(A: np.ndarray, B: np.ndarray) -> tuple[float, float]:
    """(position error m, rotation error deg) between two 4x4 poses."""
    c = (np.trace(A[:3, :3].T @ B[:3, :3]) - 1) / 2
    return float(np.linalg.norm(A[:3, 3] - B[:3, 3])), math.degrees(math.acos(max(-1.0, min(1.0, c))))


def load_plan(path) -> dict:
    """wetlab.plan.Plan JSON -> {t (N,), q (N,6), gripper (N,), events (sorted), dt, duration_s, lab_sha, cell}."""
    p = json.loads(Path(path).read_text())
    wps = p["waypoints"]
    if not wps:
        raise ValueError("plan has no waypoints")
    q = np.array([w["q"] for w in wps], float)
    if q.shape[1] != 6:
        raise ValueError(f"waypoints need 6 joints, got {q.shape[1]}")
    return {"t": np.array([w["t"] for w in wps], float), "q": q,
            "gripper": np.array([w.get("gripper_mm", 0.0) for w in wps], float),
            "events": sorted(p.get("events", []), key=lambda e: e["t"]),
            "dt": float(p.get("dt", 0.02)), "duration_s": float(p.get("duration_s", wps[-1]["t"])),
            "lab_sha": p.get("lab_sha", ""), "cell": p.get("cell", ""), "ok": p.get("ok", True)}


def home_plan(home_q_deg) -> dict:
    return {"t": np.zeros(1), "q": np.radians([home_q_deg]), "gripper": np.zeros(1), "events": [],
            "dt": 0.02, "duration_s": 0.0, "lab_sha": "", "cell": "", "ok": True}


def event_labware(ev: dict) -> str | None:
    d = ev.get("data", {})
    return d.get("labware") or d.get("labware_id") or d.get("id")


def render_indices(t: np.ndarray, fps: float) -> list[int]:
    """Waypoint indices to render so the video samples the plan at `fps` (first waypoint at or after k/fps)."""
    out, k = [], 0
    for i, ti in enumerate(t):
        if ti + 1e-9 >= k / fps:
            out.append(i)
            while ti + 1e-9 >= k / fps:
                k += 1
    return out


class Holder:
    """Labware held rigidly by the TCP between attach and detach events."""

    def __init__(self, poses: dict[str, np.ndarray]):
        self.poses = dict(poses)   # labware id -> 4x4 world pose
        self.rel: dict[str, np.ndarray] = {}

    def event(self, ev: dict, T_tool: np.ndarray) -> None:
        lw = event_labware(ev)
        if ev["kind"] == "attach" and lw in self.poses:
            self.rel[lw] = np.linalg.inv(T_tool) @ self.poses[lw]
        elif ev["kind"] == "detach":
            self.rel.pop(lw, None)

    def update(self, T_tool: np.ndarray) -> dict[str, np.ndarray]:
        """Move held labware with the tool; returns the moved poses."""
        moved = {lw: T_tool @ R for lw, R in self.rel.items()}
        self.poses.update(moved)
        return moved


def contact_pairs(pairs, robot_prefix: str, ignore: set[tuple[str, str]]) -> list[dict]:
    """[(t, a, b)] raw collider pairs -> unique arm-vs-world pairs {a(arm), b, first_t, n}."""
    out: dict[tuple[str, str], dict] = {}
    for t, a, b in pairs:
        ina, inb = a.startswith(robot_prefix), b.startswith(robot_prefix)
        if ina == inb:
            continue   # world-world or arm self-contact
        arm, other = (a, b) if ina else (b, a)
        if any(arm.startswith(i) and other.startswith(j) for i, j in ignore):
            continue
        k = out.setdefault((arm, other), {"a": arm, "b": other, "first_t": round(t, 3), "n": 0})
        k["n"] += 1
    return list(out.values())


def write_png(path, rgb: np.ndarray) -> None:
    """Minimal stdlib PNG writer for an (H, W, 3) uint8 image."""
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].astype(np.uint8).tobytes() for y in range(h))

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    Path(path).write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                           + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def gf_to_np(M) -> np.ndarray:
    """Gf.Matrix4d (row-vector convention) -> 4x4 column-vector numpy pose."""
    return np.array(M, float).T


def anchor_world_joints(stage, root_path: str) -> list[str]:
    """Fix the asset's world-attached joints for a robot referenced somewhere other than the origin.

    NVIDIA's ur5e.usd has /ur5e/root_joint (fixed, body0 = world, localPos0 = origin): re-anchored here
    to body1's CURRENT world pose, else PhysX pulls the robot back to the origin. It also has
    joints/robot_gripper_joint (body0 = wrist_3_link, body1 = empty, for the Gripper variant): with no
    gripper loaded it would pin the wrist to the world, so it is disabled. Returns the joints changed."""
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    cache, changed = UsdGeom.XformCache(), []
    for prim in Usd.PrimRange(stage.GetPrimAtPath(root_path)):
        if not prim.IsA(UsdPhysics.Joint):
            continue
        j = UsdPhysics.Joint(prim)
        b0, b1 = j.GetBody0Rel().GetTargets(), j.GetBody1Rel().GetTargets()
        if b0 and not b1:
            j.GetJointEnabledAttr().Set(False)
            changed.append(str(prim.GetPath()))
            continue
        if b0 or not b1:
            continue
        W1 = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(b1[0])).RemoveScaleShear()
        J1 = Gf.Matrix4d().SetRotate(Gf.Quatd(j.GetLocalRot1Attr().Get() or Gf.Quatf(1)))
        J1.SetTranslateOnly(Gf.Vec3d(j.GetLocalPos1Attr().Get() or Gf.Vec3f(0)))
        Jw = J1 * W1
        j.GetLocalPos0Attr().Set(Gf.Vec3f(Jw.ExtractTranslation()))
        j.GetLocalRot0Attr().Set(Gf.Quatf(Jw.ExtractRotationQuat()))
        changed.append(str(prim.GetPath()))
    return changed


def tagged(stage, scope: str) -> dict[str, str]:
    """wetlab:id -> prim path for the children of a scope (/World/Labware, /World/Nests, ...)."""
    p = stage.GetPrimAtPath(scope)
    return {c.GetAttribute("wetlab:id").Get(): str(c.GetPath()) for c in (p.GetChildren() if p else [])
            if c.HasAttribute("wetlab:id")}


class VideoWriter:
    """ffmpeg rawvideo pipe -> H.264 MP4; PNG frames in frames/ if ffmpeg is not installed."""

    def __init__(self, out: Path, w: int, h: int, fps: float):
        self.out, self.n, self.proc = out, 0, None
        ff = shutil.which("ffmpeg")
        if ff:
            self.path = out / "video.mp4"
            self.proc = subprocess.Popen([ff, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                                          "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-c:v", "libx264",
                                          "-pix_fmt", "yuv420p", str(self.path)], stdin=subprocess.PIPE)
        else:
            self.path = out / "frames"
            self.path.mkdir(parents=True, exist_ok=True)

    def add(self, rgb: np.ndarray) -> None:
        if self.proc:
            self.proc.stdin.write(np.ascontiguousarray(rgb, np.uint8).tobytes())
        else:
            write_png(self.path / f"{self.n:05d}.png", rgb)
        self.n += 1

    def close(self) -> None:
        if self.proc:
            self.proc.stdin.close()
            self.proc.wait()


# -- Isaac ------------------------------------------------------------------------
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage", required=True)
    ap.add_argument("--plan")
    ap.add_argument("--out", required=True)
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--render-only", action="store_true", help="render the twin camera at home pose to render.png")
    a, _kit_args = ap.parse_known_args(argv)   # --/renderer/... and --portable-root go to Kit
    if not a.render_only and not a.plan:
        ap.error("--plan is required unless --render-only")
    return a


def main(argv=None) -> int:
    a = parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t_wall = time.time()
    from isaacsim.simulation_app import SimulationApp

    app = SimulationApp({"headless": a.headless})
    report: dict = {"ok": False, "mode": "render_only" if a.render_only else "dry_run"}
    try:
        run(a, out, report)
        report["ok"] = True
    except Exception as e:   # always leave a report for the worker
        import traceback
        report["error"] = f"{type(e).__name__}: {e}"
        report["traceback"] = traceback.format_exc()[-4000:]
    report["wall_s"] = round(time.time() - t_wall, 1)
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report.get(k) for k in ("ok", "frames", "fk_check", "error")}), flush=True)
    # SimulationApp.close() can hang on headless boxes; outputs are written, so exit hard.
    os._exit(0 if report["ok"] else 1)


def add_robot(stage) -> tuple[str, dict, np.ndarray]:
    """Reference NVIDIA's UR5e under /World/ArmMount/<arm>, rotated so its base_link sits where our
    DH 'base' frame says. Returns (robot prim path, mount customData wetlab dict, T_room_base)."""
    from isaacsim.core.utils.stage import add_reference_to_stage
    from isaacsim.storage.native import get_assets_root_path
    from pxr import Gf, Usd, UsdGeom

    mount = stage.GetPrimAtPath("/World/ArmMount").GetChildren()[0]
    md = mount.GetCustomDataByKey("wetlab")
    robot = f"{mount.GetPath()}/ur5e"
    add_reference_to_stage(usd_path=get_assets_root_path() + md["isaac_asset"], prim_path=robot)
    rx = UsdGeom.Xformable(stage.GetPrimAtPath(robot))
    rx.ClearXformOpOrder()
    yaw = math.radians(md["base_yaw_offset_deg"])
    rx.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)))
    stage.GetPrimAtPath("/World/ArmProxy").SetActive(False)
    anchor_world_joints(stage, robot)
    T_base = gf_to_np(UsdGeom.Xformable(mount).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
    return robot, md, T_base


def twin_camera(stage):
    """isaacsim Camera on the authored camera prim (call .initialize() + apply_opencv after world.reset())."""
    from isaacsim.sensors.camera import Camera

    prim = stage.GetPrimAtPath("/World/Cameras").GetChildren()[0]
    cv = prim.GetCustomDataByKey("wetlab:opencv")
    return Camera(prim_path=str(prim.GetPath()), resolution=(int(cv["width"]), int(cv["height"]))), cv


def start_camera(camera, cv) -> None:
    camera.initialize()
    camera.set_opencv_pinhole_properties(cx=cv["cx"], cy=cv["cy"], fx=cv["fx"], fy=cv["fy"],
                                         pinhole=(list(cv["dist"]) + [0.0] * 12)[:12])


FRAME_STATS = {"empty": 0, "mean": []}


def grab(camera, cv) -> np.ndarray:
    """Latest RGB frame; a black frame (counted in FRAME_STATS['empty']) if the annotator has none yet."""
    W, H = int(cv["width"]), int(cv["height"])
    rgba = camera.get_rgba()
    if rgba is None or np.size(rgba) != W * H * 4:
        FRAME_STATS["empty"] += 1
        return np.zeros((H, W, 3), np.uint8)
    rgb = np.asarray(rgba)[..., :3].reshape(H, W, 3).astype(np.uint8)
    FRAME_STATS["mean"].append(float(rgb.mean()))
    return rgb


def joint_setter(art):
    """set_q(q): teleport the 6 UR joints (UR order) and hold them with the drive targets."""
    from isaacsim.core.utils.types import ArticulationAction

    idx = [art.dof_names.index(n) for n in UR_JOINTS]

    def set_q(q):
        q = np.asarray(q, float)
        art.set_joint_positions(q, joint_indices=idx)
        art.set_joint_velocities(np.zeros(6), joint_indices=idx)
        art.apply_action(ArticulationAction(joint_positions=q, joint_indices=np.array(idx)))
    return set_q


def labware_setter(stage, paths: dict[str, str]):
    """set_pose(lw, T): write the kinematic labware prim's USD xformOps (authored by wetlab/twin/usd.py).
    PhysX follows USD edits of kinematic bodies and RTX renders them; the tensor-API
    SingleRigidPrim.set_world_pose did not show up in the render (Isaac 5.1, checked on the GPU box)."""
    from pxr import Gf

    ops = {lw: (stage.GetPrimAtPath(p).GetAttribute("xformOp:translate"), stage.GetPrimAtPath(p).GetAttribute("xformOp:orient"))
           for lw, p in paths.items()}

    def set_pose(lw, T):
        w, x, y, z = matrix_to_quat(T[:3, :3])
        ops[lw][0].Set(Gf.Vec3d(*map(float, T[:3, 3])))
        ops[lw][1].Set(Gf.Quatd(float(w), float(x), float(y), float(z)))
    return set_pose


def isaac_version() -> str:
    try:
        from isaacsim.core.version import get_version
        return ".".join(str(v) for v in get_version()[:3])
    except Exception:
        return Path(os.environ.get("ISAAC_SIM_ROOT", "/opt/isaacsim-5.1"), "VERSION").read_text().strip()


def run(a, out: Path, report: dict) -> None:
    from isaacsim.core.api import World
    from isaacsim.core.prims import SingleArticulation, SingleRigidPrim
    from isaacsim.core.utils.stage import get_current_stage, open_stage
    from omni.physx import get_physx_simulation_interface
    from omni.physx.bindings._physx import ContactEventType
    from pxr import PhysicsSchemaTools, PhysxSchema, Usd, UsdGeom, UsdPhysics

    report["isaac_version"] = isaac_version()
    if not open_stage(str(Path(a.stage).resolve())):
        raise RuntimeError(f"cannot open stage {a.stage}")
    stage = get_current_stage()
    lab_sha = stage.GetDefaultPrim().GetCustomDataByKey("wetlab:lab_sha")
    robot, md, T_base = add_robot(stage)
    plan = home_plan(list(md["home_q_deg"])) if a.render_only else load_plan(a.plan)
    report.update(lab_sha_stage=lab_sha, lab_sha_plan=plan["lab_sha"],
                  lab_sha_match=a.render_only or plan["lab_sha"] == lab_sha)
    for p in Usd.PrimRange(stage.GetPrimAtPath(robot)):   # contact reports on every arm link, every touch
        if p.HasAPI(UsdPhysics.RigidBodyAPI):
            PhysxSchema.PhysxContactReportAPI.Apply(p).CreateThresholdAttr().Set(0.0)

    world = World(stage_units_in_meters=1.0, physics_dt=plan["dt"], rendering_dt=plan["dt"],
                  physics_prim_path="/World/PhysicsScene")
    art = world.scene.add(SingleArticulation(prim_path=robot, name="ur5e"))
    tool = SingleRigidPrim(prim_path=f"{robot}/{TOOL_LINK}", name="tool0")
    lw_paths = tagged(stage, "/World/Labware")
    cache = UsdGeom.XformCache()
    holder = Holder({lw: gf_to_np(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(p))) for lw, p in lw_paths.items()})
    set_lw = labware_setter(stage, lw_paths)
    camera, cv = twin_camera(stage)
    W, H = int(cv["width"]), int(cv["height"])
    world.reset()
    tool.initialize()
    start_camera(camera, cv)
    set_q = joint_setter(art)
    report["dof_names"] = list(art.dof_names)

    set_q(plan["q"][0])
    for _ in range(30):   # let RTX accumulate and the annotator fill before the first frame
        world.step(render=True)

    sim = get_physx_simulation_interface()
    sim.get_contact_report()   # drop contacts from the warm-up
    renders = set(render_indices(plan["t"], a.fps))
    video = None if a.render_only else VideoWriter(out, W, H, a.fps)
    bench = tagged(stage, "/World/Room").get(md["mount"], "")
    ignore = {(f"{robot}/base_link", bench)} if bench else set()
    events, ei, raw, errs = plan["events"], 0, [], []   # errs: (pos m, rot deg, arm touching the world)
    for i, (t, q) in enumerate(zip(plan["t"], plan["q"])):
        T_tool = T_base @ fk_tool0(q)
        while ei < len(events) and events[ei]["t"] <= t + 1e-9:
            holder.event(events[ei], T_tool)
            ei += 1
        for lw, T in holder.update(T_tool).items():
            set_lw(lw, T)
        set_q(q)
        world.step(render=i in renders)
        headers, _ = sim.get_contact_report()
        step = [(float(t), str(PhysicsSchemaTools.intToSdfPath(h.collider0)), str(PhysicsSchemaTools.intToSdfPath(h.collider1)))
                for h in headers if h.type != ContactEventType.CONTACT_LOST]
        raw += step
        p, o = tool.get_world_pose()
        errs.append((*pose_error(pose(p, o), T_tool), bool(contact_pairs(step, robot, ignore))))
        if i in renders:
            if a.render_only:
                write_png(out / "render.png", grab(camera, cv))
            else:
                video.add(grab(camera, cv))
    if video:
        video.close()
        report["video"] = video.path.name

    free = [e for e in errs if not e[2]]   # PhysX pushes a colliding arm away from the plan: judge FK on free waypoints
    report.update(
        frames=1 if a.render_only else video.n, fps=a.fps, duration_s=float(plan["t"][-1]),
        waypoints=len(plan["t"]), events=len(events),
        frame_stats={"empty": FRAME_STATS["empty"], "mean_brightness": round(float(np.mean(FRAME_STATS["mean"] or [0])), 1)},
        fk_check={"link": TOOL_LINK, "waypoints": len(free),
                  "max_pos_mm": round(max([e[0] for e in free] or [0]) * 1000, 3),
                  "max_rot_deg": round(max([e[1] for e in free] or [0]), 4),
                  "max_pos_mm_incl_contact": round(max(e[0] for e in errs) * 1000, 3)},
        contacts={"method": "PhysxContactReportAPI (threshold 0) on every arm link; "
                            "get_physx_simulation_interface().get_contact_report() polled after each 50 Hz step; "
                            "base_link vs its mount bench ignored",
                  "pairs": contact_pairs(raw, robot, ignore)})


if __name__ == "__main__":
    sys.exit(main())
