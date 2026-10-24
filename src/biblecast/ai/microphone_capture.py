
"""Offline microphone capture for BibleCast."""

from __future__ import annotations

import shutil
import tempfile
import threading
import wave
from pathlib import Path
from queue import Empty, Full, Queue
from typing import Callable

import sounddevice as sd


class MicrophoneCapture:
    """Capture continuous microphone audio into overlapping WAV segments.

    Audio is captured on a background thread. Each segment is four seconds
    long, with a half-second overlap between consecutive segments.

    stop() stops capture but preserves segment files.
    cleanup() removes the session files and must only be called after
    downstream transcription has finished using them.
    """

    SAMPLE_RATE = 16_000
    CHANNELS = 1
    SAMPLE_WIDTH_BYTES = 2
    SEGMENT_SECONDS = 4.0
    OVERLAP_SECONDS = 0.5
    BLOCK_SECONDS = 0.1
    QUEUE_BLOCKS = 30

    def __init__(
        self,
        on_segment: Callable[[Path], None],
        on_error: Callable[[str], None] | None = None,
        output_dir: str | Path | None = None,
        device: int | str | None = None,
    ) -> None:
        self.on_segment = on_segment
        self.on_error = on_error or (lambda _message: None)
        self.output_dir = Path(output_dir) if output_dir else None
        self.device = device

        self._queue: Queue[bytes] = Queue(maxsize=self.QUEUE_BLOCKS)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._stream = None
        self._session_dir: Path | None = None
        self._error_lock = threading.Lock()
        self._reported_error = False
        self._lifecycle_lock = threading.Lock()

    @property
    def is_running(self) -> bool:
        """Return whether the capture worker is active."""
        return self._thread is not None and self._thread.is_alive()

    @property
    def session_dir(self) -> Path | None:
        """Directory containing this session's WAV files."""
        return self._session_dir

    def start(self) -> None:
        """Start a new capture session."""
        with self._lifecycle_lock:
            if self.is_running:
                raise RuntimeError("Microphone capture is already running.")

            if self._session_dir is not None:
                raise RuntimeError(
                    "Call cleanup() before starting another session."
                )

            if self.output_dir is None:
                parent = None
            else:
                self.output_dir.mkdir(parents=True, exist_ok=True)
                parent = str(self.output_dir)

            self._session_dir = Path(
                tempfile.mkdtemp(prefix="biblecast_audio_", dir=parent)
            )
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
        """Stop capture, drain queued audio, and preserve WAV files."""
        self._stop_event.set()

        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=timeout)
            if thread.is_alive():
                self._report_error(
                    "Microphone capture did not stop in time. "
                    "Audio files have been preserved."
                )
                return

        if thread is not threading.current_thread():
            self._thread = None

    def cleanup(self) -> None:
        """Remove session files after all consumers have finished."""
        if self.is_running:
            raise RuntimeError(
                "Stop microphone capture before cleaning up."
            )

        if self._thread is not None:
            self._thread.join(timeout=0)

        if self._session_dir is not None:
            shutil.rmtree(self._session_dir, ignore_errors=True)
            self._session_dir = None

    def _report_error(self, message: str) -> None:
        """Report only the first error in a session."""
        with self._error_lock:
            if self._reported_error:
                return
            self._reported_error = True

        try:
            self.on_error(message)
        except Exception:
            pass

    def _capture_worker(self) -> None:
        """Capture, drain queued audio, and emit completed WAV segments."""
        if self._session_dir is None:
            self._report_error("Capture session directory is unavailable.")
            return

        block_frames = int(self.SAMPLE_RATE * self.BLOCK_SECONDS)
        segment_frames = int(self.SAMPLE_RATE * self.SEGMENT_SECONDS)
        overlap_frames = int(self.SAMPLE_RATE * self.OVERLAP_SECONDS)

        bytes_per_frame = self.SAMPLE_WIDTH_BYTES * self.CHANNELS
        segment_bytes = segment_frames * bytes_per_frame
        overlap_bytes = overlap_frames * bytes_per_frame

        audio_buffer = bytearray()
        segment_number = 1
        stream = None

        def emit_segment(audio_data: bytes) -> None:
            nonlocal segment_number

            path = self._session_dir / f"segment_{segment_number:06d}.wav"
            self._write_wav(path, audio_data)

            try:
                self.on_segment(path)
            except Exception as exc:
                self._report_error(
                    f"Audio segment callback failed: "
                    f"{type(exc).__name__}: {exc}"
                )

            segment_number += 1

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
                    "Capture has been asked to stop."
                )
                self._stop_event.set()

        try:
            stream = sd.RawInputStream(
                samplerate=self.SAMPLE_RATE,
                blocksize=block_frames,
                device=self.device,
                channels=self.CHANNELS,
                dtype="int16",
                callback=audio_callback,
            )
            self._stream = stream
            stream.start()

            while True:
                if self._stop_event.is_set():
                    # The worker owns stream shutdown. This avoids closing
                    # the PortAudio stream concurrently from another thread.
                    try:
                        stream.stop()
                    except Exception:
                        pass
                    try:
                        stream.close()
                    except Exception:
                        pass
                    stream = None
                    self._stream = None

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

            # Preserve a useful final partial segment on shutdown.
            minimum_final_bytes = self.SAMPLE_RATE * bytes_per_frame
            if len(audio_buffer) >= minimum_final_bytes:
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

            self._stream = None

    def _write_wav(self, path: Path, audio_data: bytes) -> None:
        """Write mono 16-bit PCM audio as a WAV file."""
        with wave.open(str(path), "wb") as wav_file:
            wav_file.setnchannels(self.CHANNELS)
            wav_file.setsampwidth(self.SAMPLE_WIDTH_BYTES)
            wav_file.setframerate(self.SAMPLE_RATE)
            wav_file.writeframes(audio_data)
