"""Customer SDK (stdlib only). Mounted read-only as the `wetlab` package inside the compile sandbox.

    from wetlab import protocol
    p = protocol("demo")
    p.home(); p.pick("plate-1"); p.place("plate-1", "shaker-1"); p.shake("shaker-1", rpm=800, seconds=5)

Calls only RECORD steps. At exit the recorded IR (wetlab/ir.py) is printed as one line
prefixed with MARKER; the service validates it. Nothing here can reach the cell.
"""
import atexit
import json
import sys

MARKER = "@@WETLAB_IR@@"
_protocols = []


class Protocol:
    def __init__(self, name="untitled"):
        self.name = name
        self.steps = []

    def _add(self, op, **kw):
        self.steps.append({"op": op, **{k: v for k, v in kw.items() if v is not None}})
        return self

    def home(self, speed=1.0):
        return self._add("home", speed=speed)

    def move_joints(self, q_deg, speed=1.0):
        return self._add("move_joints", q_deg=list(q_deg), speed=speed)

    def move_to(self, labware=None, well=None, location=None, xyz=None, z_above=0.05, yaw_deg=None, linear=True, speed=1.0):
        target = {k: v for k, v in dict(labware=labware, well=well, location=location, yaw_deg=yaw_deg,
                                         xyz=list(xyz) if xyz is not None else None).items() if v is not None}
        return self._add("move_to", target={**target, "z_above": z_above}, linear=linear, speed=speed)

    def gripper(self, width_mm):
        return self._add("gripper", width_mm=width_mm)

    def pick(self, labware, approach_m=0.08):
        return self._add("pick", labware=labware, approach_m=approach_m)

    def place(self, labware, at, approach_m=0.08):
        return self._add("place", labware=labware, at=at, approach_m=approach_m)

    def wait(self, seconds):
        return self._add("wait", seconds=seconds)

    def capture(self, tag="frame", camera=None):
        return self._add("capture", tag=tag, camera=camera)

    def instrument(self, instrument, action, **params):
        return self._add("instrument", instrument=instrument, action=action, params=params)

    def shake(self, instrument, rpm, seconds):
        return self.instrument(instrument, "shake", rpm=rpm, seconds=seconds)

    def mark(self, text):
        return self._add("mark", text=str(text))

    def to_ir(self):
        return {"ir_version": "wetlab.ir/1", "name": self.name, "steps": self.steps}


def protocol(name="untitled"):
    p = Protocol(name)
    _protocols.append(p)
    return p


@atexit.register
def _emit():
    if _protocols:  # the last protocol() created is the program
        sys.stdout.write(MARKER + json.dumps(_protocols[-1].to_ir()) + "\n")
        sys.stdout.flush()
