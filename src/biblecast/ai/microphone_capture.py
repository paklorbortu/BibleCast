
"""Offline microphone capture for BibleCast."""

from __future__ import annotations

import tempfile
import threading
import time
import wave
from pathlib import Path
from queue import Empty, Full, Queue
from typing import Callable

import sounddevice as sd


class MicrophoneCapture:
    """Capture microphone audio and emit short WAV segments.

    Audio capture runs in a background thread. The callback only copies
    audio into a bounded queue; WAV files are written by the worker thread.

    Each segment is four seconds long. Consecutive segments overlap by
    half a second to reduce the chance of cutting off spoken references.
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
        self.output_dir = (
            Path(output_dir) if output_dir is not None else None
        )
        self.device = device

        self._queue: Queue[bytes | None] = Queue(
            maxsize=self.QUEUE_BLOCKS
        )
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._stream = None
        self._temp_dir: tempfile.TemporaryDirectory | None = None
        self._error_lock = threading.Lock()
        self._reported_error = False

    @property
    def is_running(self) -> bool:
        """Whether the capture worker is active."""
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        """Start microphone capture, raising if already running."""
        if self.is_running:
            raise RuntimeError("Microphone capture is already running.")

        self._stop_event.clear()
        self._queue = Queue(maxsize=self.QUEUE_BLOCKS)
        self._reported_error = False

        if self.output_dir is None:
            self._temp_dir = tempfile.TemporaryDirectory(
                prefix="biblecast_audio_"
            )
            segment_dir = Path(self._temp_dir.name)
        else:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            segment_dir = self.output_dir

        self._thread = threading.Thread(
            target=self._capture_worker,
            args=(segment_dir,),
            name="BibleCastMicrophoneCapture",
            daemon=True,
        )
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        """Stop capture and wait for the worker to finish."""
        self._stop_event.set()

        stream = self._stream
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

        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=timeout)

            if thread.is_alive():
                self._report_error(
                    "Microphone capture did not stop in time."
                )
                return

        self._thread = None

        if self._temp_dir is not None:
            self._temp_dir.cleanup()
            self._temp_dir = None

    def _report_error(self, message: str) -> None:
        """Report only the first capture error."""
        with self._error_lock:
            if self._reported_error:
                return
            self._reported_error = True

        try:
            self.on_error(message)
        except Exception:
            # UI callback errors must not crash the capture worker.
            pass

    def _capture_worker(self, segment_dir: Path) -> None:
        """Read queued microphone blocks and write rolling WAV segments."""
        block_frames = int(self.SAMPLE_RATE * self.BLOCK_SECONDS)
        segment_frames = int(self.SAMPLE_RATE * self.SEGMENT_SECONDS)
        overlap_frames = int(self.SAMPLE_RATE * self.OVERLAP_SECONDS)

        bytes_per_frame = self.SAMPLE_WIDTH_BYTES * self.CHANNELS
        segment_bytes = segment_frames * bytes_per_frame
        overlap_bytes = overlap_frames * bytes_per_frame

        audio_buffer = bytearray()
        next_segment_number = 1

        def audio_callback(indata, frames, time_info, status) -> None:
            if self._stop_event.is_set():
                return

            if status:
                # Overflow can mean that audio was lost. Report it without
                # stopping capture, since short glitches may be recoverable.
                self._report_error(
                    f"Microphone input warning: {status}"
                )

            try:
                self._queue.put_nowait(bytes(indata))
            except Full:
                self._report_error(
                    "Audio processing is falling behind. "
                    "Stop and restart listening."
                )
                self._stop_event.set()

        try:
            self._stream = sd.RawInputStream(
                samplerate=self.SAMPLE_RATE,
                blocksize=block_frames,
                device=self.device,
                channels=self.CHANNELS,
                dtype="int16",
                callback=audio_callback,
            )
            self._stream.start()

            while not self._stop_event.is_set():
                try:
                    block = self._queue.get(timeout=0.1)
                except Empty:
                    continue

                if block is None:
                    break

                audio_buffer.extend(block)

                while len(audio_buffer) >= segment_bytes:
                    segment_data = bytes(audio_buffer[:segment_bytes])
                    segment_path = (
                        segment_dir
                        / f"segment_{next_segment_number:06d}.wav"
                    )
                    self._write_wav(segment_path, segment_data)

                    try:
                        self.on_segment(segment_path)
                    except Exception as exc:
                        self._report_error(
                            f"Audio segment callback failed: "
                            f"{type(exc).__name__}: {exc}"
                        )

                    next_segment_number += 1
                    audio_buffer = bytearray(
                        audio_buffer[segment_bytes - overlap_bytes:]
                    )

            # Flush one final partial segment on a normal stop if it
            # contains enough audio to be useful for transcription.
            minimum_final_bytes = int(
                self.SAMPLE_RATE
                * 1.0
                * bytes_per_frame
            )
            if len(audio_buffer) >= minimum_final_bytes:
                segment_path = (
                    segment_dir
                    / f"segment_{next_segment_number:06d}.wav"
                )
                self._write_wav(segment_path, bytes(audio_buffer))

                try:
                    self.on_segment(segment_path)
                except Exception as exc:
                    self._report_error(
                        f"Final audio segment callback failed: "
                        f"{type(exc).__name__}: {exc}"
                    )

        except Exception as exc:
            self._report_error(
                f"Microphone capture failed: {type(exc).__name__}: {exc}"
            )
        finally:
            stream = self._stream
            self._stream = None

            if stream is not None:
                try:
                    stream.stop()
                except Exception:
                    pass
                try:
                    stream.close()
                except Exception:
                    pass

    def _write_wav(self, path: Path, audio_data: bytes) -> None:
        """Write mono 16-bit PCM audio as a WAV file."""
        with wave.open(str(path), "wb") as wav_file:
            wav_file.setnchannels(self.CHANNELS)
            wav_file.setsampwidth(self.SAMPLE_WIDTH_BYTES)
            wav_file.setframerate(self.SAMPLE_RATE)
            wav_file.writeframes(audio_data)
