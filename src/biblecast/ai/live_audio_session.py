
"""Bounded live microphone transcription and Bible passage matching."""

from __future__ import annotations

from queue import Empty, Full, Queue
from threading import Event, Lock, Thread

import numpy as np
from PySide6.QtCore import QObject, Signal

from biblecast.ai.audio_reference_pipeline import (
    AudioReferencePipeline,
    AudioReferenceResult,
)
from biblecast.ai.microphone_capture import MicrophoneCapture


class LiveAudioSession(QObject):
    """Capture audio quickly and process it on a separate worker."""

    result_ready = Signal(object)
    error_occurred = Signal(str)
    listening_changed = Signal(bool)
    session_stopped = Signal()

    QUEUE_CAPACITY = 5

    def __init__(
        self,
        translation_code: str,
        pipeline: AudioReferencePipeline | None = None,
        parent: QObject | None = None,
        device: int | str | None = None,
    ) -> None:
        super().__init__(parent)
        self.translation_code = translation_code
        self.pipeline = pipeline or AudioReferencePipeline()
        self.device = device

        self._queue: Queue[np.ndarray] = Queue(
            maxsize=self.QUEUE_CAPACITY
        )
        self._stop_event = Event()
        self._state_lock = Lock()
        self._started = False
        self._accepting = False
        self._stop_controller_started = False

        self._inference_thread: Thread | None = None
        self._stop_controller: Thread | None = None
        self._capture: MicrophoneCapture | None = None

    @property
    def is_listening(self) -> bool:
        with self._state_lock:
            return self._started and self._accepting

    @property
    def is_active(self) -> bool:
        """True until capture and inference have both stopped."""
        with self._state_lock:
            return self._started

    def start(self) -> None:
        """Start inference before opening the microphone."""
        with self._state_lock:
            if self._started:
                raise RuntimeError(
                    "Live audio session is already active."
                )

            self._queue = Queue(maxsize=self.QUEUE_CAPACITY)
            reset = getattr(self.pipeline, "reset_context", None)
            if callable(reset):
                reset()
            self._stop_event.clear()
            self._started = True
            self._accepting = True
            self._stop_controller_started = False

        self._capture = MicrophoneCapture(
            on_segment=self._enqueue_segment,
            on_error=self._capture_error,
            device=self.device,
        )
        self._inference_thread = Thread(
            target=self._inference_loop,
            name="BibleCastAudioInference",
            daemon=True,
        )
        self._inference_thread.start()

        try:
            self._capture.start()
        except Exception:
            with self._state_lock:
                self._accepting = False
            self._stop_event.set()
            self._inference_thread.join()
            with self._state_lock:
                self._started = False
            self._capture = None
            self._inference_thread = None
            raise

        self.listening_changed.emit(True)

    def stop(self) -> None:
        """Request a stop without waiting for inference in the UI thread."""
        with self._state_lock:
            if not self._started or self._stop_controller_started:
                return

            self._stop_controller_started = True

        self._stop_controller = Thread(
            target=self._stop_controller_loop,
            name="BibleCastAudioShutdown",
            daemon=True,
        )
        self._stop_controller.start()

    def _request_stop(self) -> None:
        """Request shutdown safely from either worker thread."""
        self.stop()

    def _stop_controller_loop(self) -> None:
        """Stop capture, drain inference, then notify the UI."""
        capture = self._capture
        if capture is not None:
            capture.stop(timeout=5.0)

            # Never report completion while capture is still running.
            while capture.is_running:
                capture.stop(timeout=1.0)

        # Capture has ended, so no more segments can enter the queue.
        # Stop accepting segments, then drain the inference queue.
        with self._state_lock:
            self._accepting = False

        self._stop_event.set()

        worker = self._inference_thread
        if worker is not None:
            worker.join()

        with self._state_lock:
            self._started = False
            self._accepting = False

        self.listening_changed.emit(False)
        self.session_stopped.emit()

    def _enqueue_segment(self, samples: np.ndarray) -> None:
        """Called by capture; never run transcription here."""
        with self._state_lock:
            if not self._accepting:
                return

        try:
            self._queue.put_nowait(samples)
        except Full:
            with self._state_lock:
                if not self._accepting:
                    return
                self._accepting = False

            self.error_occurred.emit(
                "Audio processing fell behind. Listening is stopping "
                "to avoid silently losing audio. Wait for shutdown to "
                "finish before starting another session."
            )
            self._request_stop()

    def _capture_error(self, message: str) -> None:
        self.error_occurred.emit(message)
        self._request_stop()

    def _inference_loop(self) -> None:
        """Run transcription and matching sequentially off the UI thread."""
        while not self._stop_event.is_set() or not self._queue.empty():
            try:
                samples = self._queue.get(timeout=0.1)
            except Empty:
                continue

            try:
                result: AudioReferenceResult = self.pipeline.process_audio(
                    samples,
                    translation_code=self.translation_code,
                    sample_rate=16_000,
                    limit=5,
                )
                if result.success and result.candidates:
                    self.result_ready.emit(result)
                elif result.error not in {
                    "",
                    "No speech was recognized.",
                    "No matching Bible passages were found.",
                }:
                    self.error_occurred.emit(result.error)
            except Exception as exc:
                self.error_occurred.emit(
                    f"Audio analysis failed: {type(exc).__name__}: {exc}"
                )
            finally:
                self._queue.task_done()
