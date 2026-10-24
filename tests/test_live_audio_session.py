
from _pytest import pytester_assertions
import threading
import time

import numpy as np

from biblecast.ai.audio_reference_pipeline import AudioReferenceResult
from biblecast.ai.live_audio_session import LiveAudioSession


class FakeCapture:
    def __init__(self, on_segment, on_error=None, device=None):
        self.on_segment = on_segment
        self.on_error = on_error
        self.running = False

    @property
    def is_running(self):
        return self.running

    def start(self):
        self.running = True

    def stop(self, timeout=5.0):
        self.running = False


class FakePipeline:
    def __init__(self, block_first=False):
        self.block_first = block_first
        self.entered = threading.Event()
        self.release = threading.Event()
        self.calls = 0
        self.thread_ids = []
        self.lock = threading.Lock()

    def process_audio(
        self,
        audio,
        translation_code,
        sample_rate=16000,
        limit=5,
    ):
        with self.lock:
            self.calls += 1
            call_number = self.calls
            self.thread_ids.append(threading.get_ident())

        self.entered.set()
        if self.block_first and call_number == 1:
            self.release.wait(timeout=5.0)

        return AudioReferenceResult(
            success=False,
            error="No speech was recognized.",
        )


def wait_for_event(event, timeout=3.0):
    assert event.wait(timeout), "Timed out waiting for the worker."


def wait_for_calls(pipeline, expected, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with pipeline.lock:
            if pipeline.calls >= expected:
                return
        time.sleep(0.01)
    raise AssertionError(f"Expected at least {expected} pipeline calls.")


def make_session(monkeypatch, pipeline):
    monkeypatch.setattr(
        "biblecast.ai.live_audio_session.MicrophoneCapture",
        FakeCapture,
    )
    return LiveAudioSession("WEB", pipeline=pipeline)


def test_inference_runs_off_the_calling_thread(monkeypatch):
    pipeline = FakePipeline()
    session = make_session(monkeypatch, pipeline)
    caller_thread_id = threading.get_ident()

    session.start()
    try:
        session._enqueue_segment(np.ones(16000, dtype=np.float32))
        wait_for_event(pipeline.entered)

        assert pipeline.thread_ids
        assert pipeline.thread_ids[0] != caller_thread_id
    finally:
        session.stop()
        wait_for_calls(pipeline, 1)
        pipeline.release.set()
        assert session._stop_controller is not None
        session._stop_controller.join(timeout=3.0)


def test_queue_overflow_requests_shutdown(monkeypatch):
    pipeline = FakePipeline(block_first=True)
    session = make_session(monkeypatch, pipeline)
    stopped = threading.Event()
    errors = []
    session.session_stopped.connect(stopped.set)
    session.error_occurred.connect(errors.append)

    session.start()
    session._enqueue_segment(np.ones(16000, dtype=np.float32))
    wait_for_event(pipeline.entered)

    # One segment is being processed. Fill all three waiting slots.
    for _ in range(session.QUEUE_CAPACITY):
        session._enqueue_segment(np.ones(16000, dtype=np.float32))

    # The next segment must trigger controlled shutdown.
    session._enqueue_segment(np.ones(16000, dtype=np.float32))

    assert session._queue.qsize() == session.QUEUE_CAPACITY
    assert not session.is_listening
    assert any("fell behind" in message for message in errors)

    pipeline.release.set()
    wait_for_shutdown(session)


def test_stop_drains_already_queued_segments(monkeypatch):
    pipeline = FakePipeline(block_first=True)
    session = make_session(monkeypatch, pipeline)
    stopped = threading.Event()
    session.session_stopped.connect(stopped.set)

    session.start()
    session._enqueue_segment(np.ones(16000, dtype=np.float32))
    wait_for_event(pipeline.entered)

    session._enqueue_segment(np.ones(16000, dtype=np.float32))
    session._enqueue_segment(np.ones(16000, dtype=np.float32))
    session.stop()

    # Release inference after stop was requested.
    pipeline.release.set()
    wait_for_shutdown(session)

    assert pipeline.calls == 3


def test_stop_is_idempotent(monkeypatch):
    pipeline = FakePipeline()
    session = make_session(monkeypatch, pipeline)
    stopped = threading.Event()
    session.session_stopped.connect(stopped.set)

    session.start()
    session.stop()
    session.stop()

    wait_for_shutdown(session)

def wait_for_shutdown(session, timeout=3.0):
    controller = session._stop_controller
    assert controller is not None, "Shutdown controller was not started."

    controller.join(timeout=timeout)

    assert not controller.is_alive(), "Shutdown controller timed out."
    assert not session.is_active, "Session is still active after shutdown."