"""Real door: guard-locking switch aux contacts read + lock solenoid request via digital I/O.

THIS IS ONLY THE SOFTWARE MIRROR. The safety function itself is hardwired: the guard-lock's
safety contacts go dual-channel into the UR controller's safeguard-stop input, and the lock
solenoid output is only energised by the controller when the arm is stopped. Software reads
the non-safety aux contacts (door closed, lock engaged) and *requests* lock/unlock.
UNTESTED ON HARDWARE.
"""
from __future__ import annotations

import socket
import struct
from typing import Protocol

from .interfaces import DoorState


class DigitalIO(Protocol):
    def read(self, channel: int) -> bool: ...
    def write(self, channel: int, value: bool) -> None: ...


class DoorRelay:
    def __init__(self, io: DigitalIO, closed_in: int = 0, locked_in: int = 1, lock_out: int = 0):
        self.io, self.closed_in, self.locked_in, self.lock_out = io, closed_in, locked_in, lock_out

    def state(self) -> DoorState:
        closed = self.io.read(self.closed_in)
        return DoorState(closed=closed, locked=closed and self.io.read(self.locked_in), angle_deg=0.0, source="relay")

    def lock(self) -> None:
        self.io.write(self.lock_out, True)

    def unlock(self) -> None:
        self.io.write(self.lock_out, False)


class URControllerIO:
    """Aux contacts on UR standard digital inputs, lock request on a standard digital output."""

    def __init__(self, host: str):
        import rtde_io
        import rtde_receive  # lazy: ur_rtde is optional
        self.r = rtde_receive.RTDEReceiveInterface(host)
        self.w = rtde_io.RTDEIOInterface(host)

    def read(self, channel: int) -> bool:
        return bool(self.r.getDigitalInState(channel))

    def write(self, channel: int, value: bool) -> None:
        self.w.setStandardDigitalOut(channel, value)


class ModbusTCPIO:
    """Minimal Modbus TCP client for a remote I/O block: FC2 read discrete input, FC5 write coil."""

    def __init__(self, host: str, port: int = 502, unit: int = 1, timeout_s: float = 0.5):
        self.addr, self.unit, self.timeout_s, self._tid = (host, port), unit, timeout_s, 0

    def _call(self, fc: int, payload: bytes) -> bytes:
        self._tid = (self._tid + 1) & 0xFFFF
        pdu = bytes([fc]) + payload
        with socket.create_connection(self.addr, timeout=self.timeout_s) as s:
            s.sendall(struct.pack(">HHHB", self._tid, 0, len(pdu) + 1, self.unit) + pdu)
            resp = s.recv(260)
        if len(resp) < 9 or resp[7] != fc:
            raise IOError(f"modbus error response: {resp.hex()}")
        return resp[8:]

    def read(self, channel: int) -> bool:
        return bool(self._call(2, struct.pack(">HH", channel, 1))[1] & 1)

    def write(self, channel: int, value: bool) -> None:
        self._call(5, struct.pack(">HH", channel, 0xFF00 if value else 0))
