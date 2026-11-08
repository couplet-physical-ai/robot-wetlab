"""Video/image encoding for twin renders and camera frames."""
from __future__ import annotations

import io
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image


class Mp4Writer:
    """`with Mp4Writer(path, w, h, fps) as v: v.write(rgb_frame)` -> H.264 MP4 via the ffmpeg binary."""

    def __init__(self, path: str | Path, width: int, height: int, fps: float):
        self.path, self.width, self.height, self.fps = Path(path), width, height, fps

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.proc = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{self.width}x{self.height}", "-r", str(self.fps), "-i", "-",
             "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(self.path)],
            stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        return self

    def write(self, frame: np.ndarray):
        if frame.shape != (self.height, self.width, 3) or frame.dtype != np.uint8:
            raise ValueError(f"expected uint8 frame {(self.height, self.width, 3)}, got {frame.dtype} {frame.shape}")
        self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())

    def __exit__(self, exc_type, exc, tb):
        self.proc.stdin.close()
        err = self.proc.stderr.read().decode(errors="replace")
        if self.proc.wait() != 0 and exc_type is None:
            raise RuntimeError(f"ffmpeg failed writing {self.path}: {err.strip()}")


def encode_jpeg(frame: np.ndarray, quality: int = 80) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(frame).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()
