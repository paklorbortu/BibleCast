
"""In-memory microphone capture for BibleCast."""

from __future__ import annotations

import threading
from queue import Empty, Full, Queue
from typing import Callable

import numpy as np
import sounddevice as sd


class MicrophoneCapture:
    """Capture overlapping mono audio segments without writing files."""

    SAMPLE_RATE = 16_000
    CHANNELS = 1
    SEGMENT_SECONDS = 4.0
    OVERLAP_SECONDS = 0.75
    BLOCK_SECONDS = 0.1
    QUEUE_BLOCKS = 50

    def __init__(
        self,
        on_segment: Callable[[np.ndarray], None],
        on_error: Callable[[str], None] | None = None,
        device: int | str | None = None,
    ) -> None:
        self.on_segment = on_segment
        self.on_error = on_error or (lambda _message: None)
        self.device = device

        self._queue: Queue[bytes] = Queue(maxsize=self.QUEUE_BLOCKS)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._lifecycle_lock = threading.Lock()
        self._error_lock = threading.Lock()
        self._reported_error = False

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        """Start a new microphone capture session."""
        with self._lifecycle_lock:
            if self.is_running:
                raise RuntimeError("Microphone capture is already running.")

            self._queue = Queue(maxsize=self.QUEUE_BLOCKS)
            self._stop_event.clear()

            with self._error_lock:
                self._reported_error = False

            self._thread = threading.Thread(
                target=self._capture_worker,
                name="BibleCastMicrophoneCapture",
                daemon=True,
            )
            self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        """Stop capture and drain audio already received."""
        self._stop_event.set()

        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=timeout)

            if thread.is_alive():
                self._report_error(
                    "Microphone capture did not stop within the time limit."
                )
                return

            self._thread = None

    def _report_error(self, message: str) -> None:
        """Report at most one capture error per session."""
        with self._error_lock:
            if self._reported_error:
                return
            self._reported_error = True

        try:
            self.on_error(message)
        except Exception:
            pass

    def _capture_worker(self) -> None:
        """Collect blocks and emit completed segments in memory."""
        block_frames = int(self.SAMPLE_RATE * self.BLOCK_SECONDS)
        segment_frames = int(self.SAMPLE_RATE * self.SEGMENT_SECONDS)
        overlap_frames = int(self.SAMPLE_RATE * self.OVERLAP_SECONDS)

        bytes_per_frame = 2 * self.CHANNELS
        segment_bytes = segment_frames * bytes_per_frame
        overlap_bytes = overlap_frames * bytes_per_frame

        audio_buffer = bytearray()
        stream = None

        def audio_callback(indata, frames, time_info, status) -> None:
            if self._stop_event.is_set():
                return

            if status:
                self._report_error(f"Microphone input warning: {status}")

            try:
                self._queue.put_nowait(bytes(indata))
            except Full:
                self._report_error(
                    "Audio processing is falling behind. "
                    "Microphone capture is stopping to avoid silent data loss."
                )
                self._stop_event.set()

        def emit_segment(raw_audio: bytes) -> None:
            # Convert signed 16-bit PCM into float32 samples in [-1, 1].
            samples = (
                np.frombuffer(raw_audio, dtype=np.int16)
                .astype(np.float32)
                / 32768.0
            )

            # Detach from the buffer and pass ownership to the consumer.
            samples = samples.copy()

            try:
                self.on_segment(samples)
            except Exception as exc:
                self._report_error(
                    f"Audio segment processing failed: "
                    f"{type(exc).__name__}: {exc}"
                )

        try:
            stream = sd.RawInputStream(
                samplerate=self.SAMPLE_RATE,
                blocksize=block_frames,
                device=self.device,
                channels=self.CHANNELS,
                dtype="int16",
                callback=audio_callback,
            )
            stream.start()

            while True:
                try:
                    block = self._queue.get(timeout=0.1)
                except Empty:
                    if self._stop_event.is_set():
                        break
                    continue

                audio_buffer.extend(block)

                while len(audio_buffer) >= segment_bytes:
                    emit_segment(bytes(audio_buffer[:segment_bytes]))
                    audio_buffer = bytearray(
                        audio_buffer[segment_bytes - overlap_bytes:]
                    )

            # Send a useful final partial segment, if there is one.
            minimum_bytes = self.SAMPLE_RATE * bytes_per_frame
            if len(audio_buffer) >= minimum_bytes:
                emit_segment(bytes(audio_buffer))

        except Exception as exc:
            self._report_error(
                f"Microphone capture failed: {type(exc).__name__}: {exc}"
            )
        finally:
            if stream is not None:
                try:
                    stream.stop()
                except Exception:
                    pass
                try:
                    stream.close()
                except Exception:
                    pass
