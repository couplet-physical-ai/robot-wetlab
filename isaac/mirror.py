"""Isaac Sim 5.1 live mirror of a cell: the twin stage follows the cell state WebSocket.

    $ISAAC_SIM_ROOT/python.sh isaac/mirror.py --stage out/cell-a.usda --server http://HOST:8000 --cell cell-a \
        [--token $WETLAB_TOKEN] [--headless --png out/mirror.png --png-every 2]

Messages (10 Hz): {t, q:[6 rad], gripper_mm, held, labware_at:{id: loc}, door:{closed, locked, angle_deg},
interlock:{tripped, reason}, job}. Applied each frame: UR5e joints, door drive target, labware on
their nest prims (/World/Nests/*), held labware rigidly on the FK TCP. No extra dependency: the
WebSocket client below is ~40 lines of stdlib (RFC 6455 text frames, ping/pong, close; no
extensions). Reuses the stage setup from dry_run.py (same directory).
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import socket
import ssl
import struct
import sys
import threading
import time
from pathlib import Path
from urllib.parse import urlparse

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dry_run as dr  # noqa: E402  (Isaac-free at import time)


# -- tiny stdlib WebSocket client ---------------------------------------------------
def ws_url(server: str, cell: str) -> str:
    u = urlparse(server)
    return f"{'wss' if u.scheme in ('https', 'wss') else 'ws'}://{u.netloc}/v1/cells/{cell}/ws"


def ws_frame(opcode: int, payload: bytes, mask: bytes | None = None) -> bytes:
    """One FIN frame; clients must mask (pass 4 random bytes)."""
    n = len(payload)
    head = bytes([0x80 | opcode])
    mbit = 0x80 if mask else 0
    if n < 126:
        head += bytes([mbit | n])
    elif n < 1 << 16:
        head += bytes([mbit | 126]) + struct.pack(">H", n)
    else:
        head += bytes([mbit | 127]) + struct.pack(">Q", n)
    if not mask:
        return head + payload
    return head + mask + bytes(b ^ mask[i % 4] for i, b in enumerate(payload))


def ws_read(read) -> tuple[int, bytes]:
    """Read one message (joins continuation frames) with `read(n) -> bytes`. Returns (opcode, payload)."""
    opcode, data = None, b""
    while True:
        b0, b1 = _exact(read, 2)
        n = b1 & 0x7F
        if n == 126:
            n = struct.unpack(">H", _exact(read, 2))[0]
        elif n == 127:
            n = struct.unpack(">Q", _exact(read, 8))[0]
        mask = _exact(read, 4) if b1 & 0x80 else None
        chunk = _exact(read, n)
        if mask:
            chunk = bytes(c ^ mask[i % 4] for i, c in enumerate(chunk))
        op = b0 & 0x0F
        if op >= 8:            # control frames may interleave fragments
            return op, chunk
        opcode = op if op else opcode
        data += chunk
        if b0 & 0x80:
            return opcode, data


def _exact(read, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        c = read(n - len(buf))
        if not c:
            raise ConnectionError("websocket closed")
        buf += c
    return buf


def ws_connect(url: str, token: str | None = None) -> socket.socket:
    u = urlparse(url)
    port = u.port or (443 if u.scheme == "wss" else 80)
    sock = socket.create_connection((u.hostname, port), timeout=30)
    if u.scheme == "wss":
        ctx = ssl.create_default_context()
        ctx.minimum_version = ssl.TLSVersion.TLSv1_2
        sock = ctx.wrap_socket(sock, server_hostname=u.hostname)
    key = base64.b64encode(os.urandom(16)).decode()
    hdr = [f"GET {u.path or '/'}{'?' + u.query if u.query else ''} HTTP/1.1", f"Host: {u.netloc}",
           "Upgrade: websocket", "Connection: Upgrade", f"Sec-WebSocket-Key: {key}", "Sec-WebSocket-Version: 13"]
    if token:
        hdr.append(f"Authorization: Bearer {token}")
    sock.sendall(("\r\n".join(hdr) + "\r\n\r\n").encode())
    resp = b""
    while b"\r\n\r\n" not in resp:
        c = sock.recv(1)
        if not c:
            raise ConnectionError("handshake: connection closed")
        resp += c
    if b" 101 " not in resp.split(b"\r\n")[0]:
        raise ConnectionError(f"handshake failed: {resp.split(chr(13).encode())[0]!r}")
    sock.settimeout(None)
    return sock


def ws_messages(url: str, token: str | None):
    """Yield parsed JSON messages forever; reconnects with backoff."""
    while True:
        try:
            s = ws_connect(url, token)
            while True:
                op, data = ws_read(s.recv)
                if op == 1:
                    yield json.loads(data)
                elif op == 9:
                    s.sendall(ws_frame(10, data, os.urandom(4)))
                elif op == 8:
                    raise ConnectionError("server closed")
        except (OSError, ConnectionError, ValueError) as e:
            print(f"[mirror] ws: {e}; reconnecting in 2 s", flush=True)
            time.sleep(2)


# -- state -> stage ------------------------------------------------------------------
def held_events(prev: str | None, cur: str | None) -> list[dict]:
    """attach/detach events implied by a change of the `held` field."""
    if prev == cur:
        return []
    out = [{"kind": "detach", "data": {"labware": prev}}] if prev else []
    return out + ([{"kind": "attach", "data": {"labware": cur}}] if cur else [])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Isaac Sim live mirror of a wetlab cell")
    ap.add_argument("--stage", required=True)
    ap.add_argument("--server", required=True, help="http(s)://host:port of the wetlab service")
    ap.add_argument("--cell", default=None, help="default: the stage's wetlab:cell")
    ap.add_argument("--token", default=os.environ.get("WETLAB_TOKEN"))
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--png", default="out/mirror.png")
    ap.add_argument("--png-every", type=float, default=2.0, help="headless: seconds between PNG renders")
    a, _kit = ap.parse_known_args(argv)

    from isaacsim.simulation_app import SimulationApp

    app = SimulationApp({"headless": a.headless})
    from isaacsim.core.api import World
    from isaacsim.core.prims import SingleArticulation
    from isaacsim.core.utils.stage import get_current_stage, open_stage
    from pxr import UsdGeom

    open_stage(str(Path(a.stage).resolve()))
    stage = get_current_stage()
    cell = a.cell or stage.GetDefaultPrim().GetCustomDataByKey("wetlab:cell")
    robot, md, T_base = dr.add_robot(stage)
    world = World(stage_units_in_meters=1.0, physics_dt=1 / 60, rendering_dt=1 / 30,
                  physics_prim_path="/World/PhysicsScene")
    art = world.scene.add(SingleArticulation(prim_path=robot, name="ur5e"))
    lw_paths = dr.tagged(stage, "/World/Labware")
    set_lw = dr.labware_setter(stage, lw_paths)
    cache = UsdGeom.XformCache()
    nests = {loc: dr.gf_to_np(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(p)))
             for loc, p in dr.tagged(stage, "/World/Nests").items()}
    holder = dr.Holder({lw: dr.gf_to_np(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(p))) for lw, p in lw_paths.items()})
    doors = [(stage.GetPrimAtPath(f"{p}/hinge").GetAttribute("drive:angular:physics:targetPosition"),
              stage.GetPrimAtPath(p).GetCustomDataByKey("wetlab:open_sign"))
             for p in dr.tagged(stage, "/World/Door").values()]
    camera, cv = (dr.twin_camera(stage) if a.headless else (None, None))
    world.reset()
    if camera:
        dr.start_camera(camera, cv)
    set_q = dr.joint_setter(art)
    set_q(np.radians(list(md["home_q_deg"])))
    Path(a.png).parent.mkdir(parents=True, exist_ok=True)

    latest: dict = {}
    url = ws_url(a.server, cell)

    def reader():
        for m in ws_messages(url, a.token):
            latest["msg"] = m

    threading.Thread(target=reader, daemon=True).start()
    print(f"[mirror] {url} -> {a.stage}", flush=True)
    held, last_png, applied = None, 0.0, None
    while app.is_running():
        m = latest.get("msg")
        if m is not None and m is not applied:
            applied = m
            T_tool = T_base @ dr.fk_tool0(m["q"])
            for ev in held_events(held, m.get("held")):
                holder.event(ev, T_tool)
            held = m.get("held")
            for lw, loc in m.get("labware_at", {}).items():
                if lw in lw_paths and loc in nests and lw != held:
                    holder.poses[lw] = nests[loc]
                    set_lw(lw, nests[loc])
            for lw, T in holder.update(T_tool).items():
                set_lw(lw, T)
            set_q(m["q"])
            d = m.get("door") or {}
            for attr, sign in doors:   # shortcut: one door per cell in the state message; key by id when there are two
                attr.Set(float(sign * d.get("angle_deg", 0.0)))
        world.step(render=True)
        if camera and time.time() - last_png >= a.png_every:
            dr.write_png(a.png, dr.grab(camera, cv))
            last_png = time.time()
    app.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
