"""输入视频与输出 AVI 的会话对象。"""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

__all__ = ["VideoSession"]

_WINDOW_NAME = "Real-time Traffic Analysis"


class VideoSession:
    """成对管理 VideoCapture / VideoWriter；预览窗口仅在 display_video 时出现。"""

    def __init__(self, video_path: str, output_path: str, display_video: bool) -> None:
        self._video_path = video_path
        self._output_path = output_path
        self._display_video = display_video
        self._cap: cv2.VideoCapture | None = None
        self._writer: cv2.VideoWriter | None = None

    @property
    def display_video(self) -> bool:
        return self._display_video

    @property
    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def open(self) -> None:
        self._cap = self._open_capture()
        self._writer = self._open_writer(self._cap)

    def read(self) -> NDArray[np.uint8] | None:
        if self._cap is None:
            raise RuntimeError("VideoSession.open() must be called first")
        ok, frame = self._cap.read()
        return frame if ok else None

    def write(self, frame: NDArray[np.uint8]) -> None:
        if self._writer is None:
            raise RuntimeError("VideoSession.open() must be called first")
        self._writer.write(frame)

    def preview(self, frame: NDArray[np.uint8]) -> bool:
        """弹出窗口。返回 False 表示使用者按了 q，应结束循环。"""
        if not self._display_video:
            return True
        cv2.imshow(_WINDOW_NAME, frame)
        return not (cv2.waitKey(1) & 0xFF == ord("q"))

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        if self._display_video:
            cv2.destroyAllWindows()

    def __enter__(self) -> VideoSession:
        self.open()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _open_capture(self) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(self._video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Unable to open input video: {self._video_path}")
        return cap

    def _open_writer(self, cap: cv2.VideoCapture) -> cv2.VideoWriter:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
        size = (int(cap.get(3)), int(cap.get(4)))
        writer = cv2.VideoWriter(self._output_path, fourcc, fps, size)
        if not writer.isOpened():
            raise RuntimeError(f"Unable to create output video: {self._output_path}")
        return writer
