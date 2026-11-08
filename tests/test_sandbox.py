import shutil
import subprocess

import pytest

from wetlab.sandbox import CompileError, compile_python


def _docker_ok() -> bool:
    if not shutil.which("docker"):
        return False
    try:
        return subprocess.run(["docker", "image", "inspect", "python:3.12-slim"], capture_output=True, timeout=5).returncode == 0
    except Exception:
        return False


docker = pytest.mark.skipif(not _docker_ok(), reason="docker / python:3.12-slim unavailable")

BENIGN = """
from wetlab import protocol
p = protocol("ok")
p.home(); p.pick("plate-1"); p.place("plate-1", "shaker-1"); p.shake("shaker-1", rpm=800, seconds=5)
p.wait(1); p.capture("img"); p.move_to(labware="plate-2", well="A1", z_above=0.03); p.move_to(xyz=[2.0, 2.4, 1.1])
p.move_joints([0, -100, 100, -90, -90, 0]); p.gripper(90); p.mark("done")
print("hello from the sandbox")
"""


@docker
def test_benign_program_compiles():
    prog, logs = compile_python(BENIGN)
    ops = [s.op for s in prog.steps]
    assert ops == ["home", "pick", "place", "instrument", "wait", "capture", "move_to", "move_to", "move_joints", "gripper", "mark"]
    assert prog.steps[3].params == {"rpm": 800, "seconds": 5} and prog.steps[6].target.well == "A1"
    assert "hello from the sandbox" in logs and "@@WETLAB_IR@@" not in logs


HOSTILE = {
    "network": "import socket\nsocket.create_connection(('1.1.1.1', 53), timeout=3)",
    "file_write": "open('/sdk/wetlab/pwned.py', 'w').write('x')",
    "root_write": "open('/etc/pwned', 'w').write('x')",
    "fork_bomb": "import os\nwhile True:\n    os.fork()",
    "memory": "x = bytearray(1 << 30)",
}


@docker
@pytest.mark.parametrize("name", sorted(HOSTILE))
def test_hostile_programs_fail_safely(name):
    src = "from wetlab import protocol\np = protocol('evil')\np.home()\n" + HOSTILE[name]
    with pytest.raises(CompileError) as e:
        compile_python(src, timeout_s=20)
    assert "exited with code" in str(e.value) or "timed out" in str(e.value)


@docker
def test_infinite_loop_is_killed():
    with pytest.raises(CompileError, match="timed out"):
        compile_python("while True:\n    pass", timeout_s=3)
    assert "wetlab-compile" not in subprocess.run(["docker", "ps", "--format", "{{.Names}}"], capture_output=True, text=True, timeout=5).stdout


@docker
def test_bad_ir_and_missing_protocol_rejected():
    with pytest.raises(CompileError, match="did not create a protocol"):
        compile_python("print('no protocol')")
    with pytest.raises(CompileError, match="invalid protocol"):
        compile_python("from wetlab import protocol\np = protocol('x')\np.wait(-5)")
    with pytest.raises(CompileError, match="invalid protocol"):   # forged marker line still goes through validation
        compile_python("print('@@WETLAB_IR@@' + '{\"steps\": [{\"op\": \"self_destruct\"}]}')")


def test_subprocess_backend_dev_only(monkeypatch):
    monkeypatch.setenv("WETLAB_SANDBOX", "subprocess")
    prog, logs = compile_python(BENIGN)
    assert len(prog.steps) == 11 and "hello" in logs
    with pytest.raises(CompileError):
        compile_python("import os\nos.fork()")
    with pytest.raises(CompileError, match="timed out|exited"):
        compile_python("while True:\n    pass", timeout_s=2)
