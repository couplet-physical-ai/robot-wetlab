"""Compile customer Python to IR in a sandbox. Customer code NEVER runs next to hardware.

Backend `docker` (default): throwaway container, no network, read-only root, tiny tmpfs,
256 MB RAM, half a CPU, 64 pids, all caps dropped, nobody user, SDK mounted read-only.
Backend `subprocess` (WETLAB_SANDBOX=subprocess): rlimits only, network NOT blocked. DEV ONLY.

The SDK prints the recorded IR as one MARKER line at exit; we take the last one and validate
it as ir.Program, so the worst a program can do is produce IR that the planner rejects.
"""
from __future__ import annotations

import os
import secrets
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

from pydantic import ValidationError

from .ir import Program

SDK_DIR = Path(__file__).resolve().parent.parent / "wetlab_sdk"
MARKER = "@@WETLAB_IR@@"
CAP = 1 << 20          # bytes of stdout/stderr kept
IMAGE = "python:3.12-slim"


class CompileError(Exception):
    def __init__(self, message: str, logs: str = ""):
        super().__init__(message)
        self.logs = logs


def compile_python(source: str, timeout_s: float = 20) -> tuple[Program, str]:
    """-> (validated Program, customer stdout+stderr). Raises CompileError(message, logs)."""
    backend = os.environ.get("WETLAB_SANDBOX", "docker")
    if backend == "docker":
        rc, out, err, timed_out = _docker(source, timeout_s)
    elif backend == "subprocess":
        rc, out, err, timed_out = _subprocess(source, timeout_s)
    else:
        raise CompileError(f"unknown WETLAB_SANDBOX backend {backend!r}")
    lines = out.decode(errors="replace").splitlines()
    ir_lines = [ln[len(MARKER):] for ln in lines if ln.startswith(MARKER)]
    logs = "\n".join(ln for ln in lines if not ln.startswith(MARKER))
    err_s = err.decode(errors="replace")
    logs = (logs + ("\n" if logs and err_s else "") + err_s).strip()
    if timed_out:
        raise CompileError(f"timed out after {timeout_s:g} s", logs)
    if rc != 0:
        raise CompileError(f"program exited with code {rc}", logs)
    if not ir_lines:
        raise CompileError("program did not create a protocol (from wetlab import protocol)", logs)
    try:
        return Program.model_validate_json(ir_lines[-1]), logs
    except ValidationError as e:
        raise CompileError(f"invalid protocol: {e}", logs) from None


def _docker(source: str, timeout_s: float):
    name = f"wetlab-compile-{secrets.token_hex(6)}"
    argv = ["docker", "run", "--rm", "-i", "--name", name, "--network", "none", "--read-only",
            "--tmpfs", "/tmp:size=16m,mode=1777", "--memory", "256m", "--memory-swap", "256m",
            "--cpus", "0.5", "--pids-limit", "64", "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "--user", "65534:65534", "-v", f"{SDK_DIR}:/sdk/wetlab:ro", "-e", "PYTHONPATH=/sdk",
            "-e", "PYTHONDONTWRITEBYTECODE=1", IMAGE, "python", "-"]
    kill = lambda: subprocess.run(["docker", "kill", name], capture_output=True, timeout=15)  # noqa: E731
    return _run(argv, source, timeout_s, kill)


def _subprocess(source: str, timeout_s: float):
    import resource

    tmp = tempfile.mkdtemp(prefix="wetlab-compile-")
    shutil.copytree(SDK_DIR, Path(tmp) / "wetlab")

    def limits():
        resource.setrlimit(resource.RLIMIT_CPU, (int(timeout_s) + 1,) * 2)
        # macOS maps large virtual-address regions even for tiny Python processes;
        # RLIMIT_AS below that initial mapping makes preexec_fn itself fail. This
        # backend is for trusted local examples only, not a security boundary.
        if sys.platform != "darwin":
            resource.setrlimit(resource.RLIMIT_AS, (512 << 20,) * 2)
        resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))   # no forks (counts all of this uid's processes)
        resource.setrlimit(resource.RLIMIT_FSIZE, (1 << 20,) * 2)

    boot = f"import sys; sys.path.insert(0, {tmp!r}); exec(compile(sys.stdin.read(), '<protocol>', 'exec'), {{'__name__': '__main__'}})"
    try:
        return _run([sys.executable, "-I", "-c", boot], source, timeout_s, lambda: None,
                    env={"PYTHONDONTWRITEBYTECODE": "1"}, cwd=tmp, preexec_fn=limits)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _run(argv, source: str, timeout_s: float, kill, **kw):
    p = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, **kw)
    bufs: dict[str, bytes] = {}

    def pump(key, f):  # keep the first CAP bytes, drain (discard) the rest so the child never blocks
        b = bytearray()
        while chunk := f.read(65536):
            if len(b) < CAP:
                b += chunk[: CAP - len(b)]
        bufs[key] = bytes(b)

    threads = [threading.Thread(target=pump, args=(k, f), daemon=True) for k, f in (("out", p.stdout), ("err", p.stderr))]
    for t in threads:
        t.start()
    try:
        p.stdin.write(source.encode())
        p.stdin.close()
    except BrokenPipeError:
        pass
    timed_out = False
    try:
        p.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        timed_out = True
        kill()
        p.kill()
        p.wait()
    for t in threads:
        t.join(timeout=5)
    return p.returncode, bufs.get("out", b""), bufs.get("err", b""), timed_out
