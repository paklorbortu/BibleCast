
"""Bounded live microphone transcription and Bible passage matching."""

from __future__ import annotations

from queue import Empty, Full, Queue
from threading import Event, Thread

import numpy as np
from PySide6.QtCore import QObject, Signal

from biblecast.ai.audio_reference_pipeline import (
    AudioReferencePipeline,
    AudioReferenceResult,
)
from biblecast.ai.microphone_capture import MicrophoneCapture


class LiveAudioSession(QObject):
    """Capture audio quickly and process segments on a separate worker."""

    result_ready = Signal(object)
    error_occurred = Signal(str)
    listening_changed = Signal(bool)

    QUEUE_CAPACITY = 3

    def __init__(
        self,
        translation_code: str,
        pipeline: AudioReferencePipeline | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.translation_code = translation_code
        self.pipeline = pipeline or AudioReferencePipeline()

        self._queue: Queue[np.ndarray] = Queue(
            maxsize=self.QUEUE_CAPACITY
        )
        self._stop_event = Event()
        self._worker: Thread | None = None
        self._capture: MicrophoneCapture | None = None
        self._state_lock = __import__("threading").Lock()
        self._started = False
        self._overloaded = False

    @property
    def is_listening(self) -> bool:
        return self._started and not self._stop_event.is_set()

    def start(self) -> None:
        """Start inference processing, then begin microphone capture."""
        with self._state_lock:
            if self._started:
                raise RuntimeError("Live audio session is already running.")

            self._queue = Queue(maxsize=self.QUEUE_CAPACITY)
            self._stop_event.clear()
            self._overloaded = False
            self._started = True

        self._worker = Thread(
            target=self._inference_loop,
            name="BibleCastAudioInference",
            daemon=True,
        )
        self._worker.start()

        self._capture = MicrophoneCapture(
            on_segment=self._enqueue_segment,
            on_error=self._capture_error,
        )

        try:
            self._capture.start()
        except Exception:
            self._stop_event.set()
            self._worker.join(timeout=2.0)
            with self._state_lock:
                self._started = False
            raise

        self.listening_changed.emit(True)

    def stop(self) -> None:
        """Stop capture and let the worker drain already queued segments."""
        with self._state_lock:
            if not self._started:
                return
            self._started = False

        # This signals the capture worker to stop and waits for it to finish.
        # The segment callback only enqueues audio, so it should return quickly.
        capture = self._capture
        if capture is not None:
            capture.stop(timeout=5.0)

        self._stop_event.set()
        self.listening_changed.emit(False)

    def wait_until_stopped(self, timeout: float = 10.0) -> bool:
        """Wait for inference to finish after stop() has been requested."""
        worker = self._worker
        if worker is None:
            return True

        worker.join(timeout=timeout)
        if worker.is_alive():
            return False

        self._worker = None
        self._capture = None
        return True

    def _enqueue_segment(self, samples: np.ndarray) -> None:
        """Called by the capture worker. Never run inference here."""
        if self._stop_event.is_set():
            return

        try:
            self._queue.put_nowait(samples)
        except Full:
            with self._state_lock:
                if self._overloaded:
                    return
                self._overloaded = True

            self.error_occurred.emit(
                "Audio processing fell behind. Listening has stopped "
                "to avoid silently losing audio. Please start listening "
                "again when the system is ready."
            )

            # stop() only signals the capture worker when called from that
            # worker, avoiding a self-join.
            capture = self._capture
            if capture is not None:
                capture.stop()

            self._stop_event.set()

    def _capture_error(self, message: str) -> None:
        self.error_occurred.emit(message)
        self._stop_event.set()

    def _inference_loop(self) -> None:
        """Run the model sequentially, outside the microphone callback."""
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
                elif result.error:
                    # Avoid spamming the UI for every ordinary segment with
                    # no recognizable reference.
                    if result.error not in {
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
