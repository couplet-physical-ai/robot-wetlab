"""Isaac dry-run worker: pulls sim jobs from the wetlab service, runs dry_run.py, posts the result.

    WETLAB_WORKER_TOKEN=... python3 isaac/worker.py --server https://HOST [--once] [kit args...]

Outbound HTTP only (a GPU box behind NAT works). stdlib only. Loop:
  GET  {server}/v1/internal/sim-jobs/next   (Authorization: Bearer $WETLAB_WORKER_TOKEN)
       204 -> sleep; 200 -> {job_id, cell, lab_sha, plan, result_url?, stage_url?}
  GET  {server}/v1/cells/<cell>/stage.usda  only when the cached stage's lab_sha != the plan's
  run  $ISAAC_SIM_ROOT/python.sh isaac/dry_run.py --stage ... --plan ... --out ... --headless [kit args]
  POST {server}/v1/internal/sim-jobs/<job_id>/result  multipart: report (report.json) + video (video.mp4)
Unknown CLI args (e.g. --/renderer/multiGpu/enabled=false --portable-root DIR) are forwarded to dry_run.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent


def request(method: str, url: str, token: str, body: bytes | None = None, ctype: str | None = None,
            timeout: float = 60) -> tuple[int, bytes]:
    r = urllib.request.Request(url, data=body, method=method, headers={"Authorization": f"Bearer {token}"})
    if ctype:
        r.add_header("Content-Type", ctype)
    try:
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def multipart(files: dict[str, Path]) -> tuple[bytes, str]:
    """multipart/form-data body with one file part per field."""
    b = uuid.uuid4().hex
    parts = []
    for field, path in files.items():
        ct = "application/json" if path.suffix == ".json" else "video/mp4" if path.suffix == ".mp4" else "application/octet-stream"
        parts.append(f'--{b}\r\nContent-Disposition: form-data; name="{field}"; filename="{path.name}"\r\n'
                     f"Content-Type: {ct}\r\n\r\n".encode() + path.read_bytes() + b"\r\n")
    return b"".join(parts) + f"--{b}--\r\n".encode(), f"multipart/form-data; boundary={b}"


def stage_sha(path: Path) -> str | None:
    """wetlab:lab_sha stamped on a .usda stage (wetlab/twin/usd.py), or None."""
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            m = re.search(r'string lab_sha = "([0-9a-f]+)"', f.read(8192))   # it's in the /World header
    except OSError:
        return None
    return m.group(1) if m else None


def run_job(job: dict, a, extra: list[str]) -> None:
    plan = job["plan"]
    cell, sha = job.get("cell") or plan["cell"], job.get("lab_sha") or plan["lab_sha"]
    work = Path(a.work)
    out = work / "jobs" / job["job_id"]
    out.mkdir(parents=True, exist_ok=True)
    rep = out / "report.json"
    try:
        stage = work / "stages" / f"{cell}.usda"
        if stage_sha(stage) != sha:
            st, data = request("GET", a.server + (job.get("stage_url") or f"/v1/cells/{cell}/stage.usda"), a.stage_token)
            if st != 200:
                raise RuntimeError(f"stage download {st}: {data[:200]!r}")
            stage.parent.mkdir(parents=True, exist_ok=True)
            stage.write_bytes(data)
            print(f"[worker] stage {cell} -> {stage_sha(stage)}", flush=True)
        (out / "plan.json").write_text(json.dumps(plan))
        cmd = [a.isaac_python, str(HERE / "dry_run.py"), "--stage", str(stage), "--plan", str(out / "plan.json"),
               "--out", str(out), "--headless", "--fps", str(a.fps), *extra]
        print(f"[worker] job {job['job_id']}: {' '.join(cmd)}", flush=True)
        rep.unlink(missing_ok=True)
        p = subprocess.run(cmd, env=dict(os.environ, OMNI_KIT_ACCEPT_EULA="YES"), timeout=a.timeout,
                           capture_output=True, text=True)
        tail = (p.stdout + p.stderr)[-2000:]
    except subprocess.TimeoutExpired:
        tail = f"dry_run timed out after {a.timeout} s"
    except Exception as e:   # never leave the job 'running' on the service
        tail = f"{type(e).__name__}: {e}"
    if not rep.is_file():
        rep.write_text(json.dumps({"ok": False, "error": tail}))
    files = {"report": rep}
    if (out / "video.mp4").is_file():
        files["video"] = out / "video.mp4"
    body, ctype = multipart(files)
    st, data = request("POST", a.server + (job.get("result_url") or f"/v1/internal/sim-jobs/{job['job_id']}/result"),
                       a.token, body, ctype, timeout=300)
    print(f"[worker] job {job['job_id']} result -> {st} {data[:200]!r}", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Isaac Sim dry-run worker")
    ap.add_argument("--server", required=True)
    ap.add_argument("--token", default=os.environ.get("WETLAB_WORKER_TOKEN"))
    ap.add_argument("--stage-token", default=os.environ.get("WETLAB_STAGE_TOKEN"),
                    help="token for GET stage.usda (default: the worker token)")
    ap.add_argument("--isaac-python", default=str(Path(os.environ.get("ISAAC_SIM_ROOT", "/opt/isaacsim-5.1")) / "python.sh"))
    ap.add_argument("--work", default=str(Path.home() / "wetlab" / "worker"))
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--timeout", type=float, default=1800)
    ap.add_argument("--poll", type=float, default=5.0)
    ap.add_argument("--once", action="store_true", help="handle at most one job, then exit")
    a, extra = ap.parse_known_args(argv)
    if not a.token:
        ap.error("set WETLAB_WORKER_TOKEN or --token")
    a.server, a.stage_token = a.server.rstrip("/"), a.stage_token or a.token
    while True:
        try:
            st, data = request("GET", f"{a.server}/v1/internal/sim-jobs/next", a.token)
            if st == 200:
                run_job(json.loads(data), a, extra)
            elif st != 204:
                print(f"[worker] next -> {st} {data[:200]!r}", flush=True)
        except Exception as e:   # keep polling: a bad job or a network blip must not kill the worker
            print(f"[worker] error: {type(e).__name__}: {e}", flush=True)
            st = 0
        if a.once:
            return 0
        if st != 200:
            time.sleep(a.poll)


if __name__ == "__main__":
    sys.exit(main())
