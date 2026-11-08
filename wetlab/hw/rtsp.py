"""Real camera: an RTSP stream (e.g. re-published by go2rtc) decoded by an ffmpeg subprocess.

A background thread keeps only the latest frame; `heartbeat` is when it arrived. frame()
returns None once the feed is stale, which the interlock turns into `camera_lost`.
UNTESTED ON HARDWARE.
"""
from __future__ import annotations

import subprocess
import threading
import time

import numpy as np

from .interfaces import CameraInfo


class RTSPCamera:
    def __init__(self, url: str, cam_id: str, width: int, height: int, fps: float, stale_s: float = 1.0):
        self.url, self.stale_s = url, stale_s
        self.info = CameraInfo(cam_id, width, height, fps, {"source": "rtsp", "url": url})
        self.heartbeat = 0.0
        self._latest: np.ndarray | None = None
        self._stop = threading.Event()
        self._proc: subprocess.Popen | None = None
        threading.Thread(target=self._run, daemon=True, name=f"rtsp-{cam_id}").start()

    def frame(self) -> np.ndarray | None:
        return self._latest if time.time() - self.heartbeat <= self.stale_s else None

    def close(self) -> None:
        self._stop.set()
        if self._proc:
            self._proc.kill()

    def _run(self) -> None:
        w, h = self.info.width, self.info.height
        n = w * h * 3
        while not self._stop.is_set():
            self._proc = subprocess.Popen(
                ["ffmpeg", "-nostdin", "-loglevel", "error", "-rtsp_transport", "tcp", "-i", self.url,
                 "-vf", f"scale={w}:{h}", "-r", str(self.info.fps), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                stdout=subprocess.PIPE)
            while not self._stop.is_set():
                buf = self._proc.stdout.read(n)
                if len(buf) < n:
                    break
                self._latest = np.frombuffer(buf, np.uint8).reshape(h, w, 3)
                self.heartbeat = time.time()
            self._proc.kill()
            self._proc.wait()
            self._stop.wait(1.0)  # reconnect backoff
