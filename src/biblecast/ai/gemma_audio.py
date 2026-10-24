
"""Native audio inference adapter for the local Gemma 4 model."""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


MODEL_REPO = "ggml-org/gemma-4-E2B-it-GGUF"

DEFAULT_MODEL_PATH = (
    Path.home()
    / ".cache"
    / "huggingface"
    / "hub"
    / "models--ggml-org--gemma-4-E2B-it-GGUF"
    / "snapshots"
    / "6938c58e698502037401a5f19f6905f2fb2f7dee"
    / "gemma-4-E2B-it-Q8_0.gguf"
)

DEFAULT_PROJECTOR_PATH = (
    DEFAULT_MODEL_PATH.parent
    / "mmproj-gemma-4-E2B-it-Q8_0.gguf"
)


@dataclass(frozen=True)
class AudioInferenceResult:
    """Raw result from Gemma audio inference."""

    success: bool
    output: str = ""
    error: str = ""


class GemmaAudioAdapter:
    """Run local Gemma 4 audio inference through llama-mtmd-cli.

    This adapter does not control the projector or make live-casting
    decisions. Its output must be validated by later pipeline stages.
    """

    def __init__(
        self,
        executable: str = "llama-mtmd-cli",
        model_path: Path = DEFAULT_MODEL_PATH,
        projector_path: Path = DEFAULT_PROJECTOR_PATH,
        timeout_seconds: int = 90,
    ) -> None:
        self.executable = executable
        self.model_path = Path(model_path)
        self.projector_path = Path(projector_path)
        self.timeout_seconds = timeout_seconds

    def infer(self, audio_path: str | Path) -> AudioInferenceResult:
        """Analyze an audio file using Gemma's native audio input."""

        audio = Path(audio_path).expanduser()

        if not audio.is_file():
            return AudioInferenceResult(
                success=False,
                error=f"Audio file not found: {audio}",
            )

        if not self.model_path.is_file():
            return AudioInferenceResult(
                success=False,
                error=f"Gemma model file not found: {self.model_path}",
            )

        if not self.projector_path.is_file():
            return AudioInferenceResult(
                success=False,
                error=f"Multimodal projector not found: {self.projector_path}",
            )

        prompt = (
            "You are BibleCast, an assistant helping a church operator identify "
            "Bible references spoken aloud. Listen to the supplied audio. "
            "Identify an explicitly spoken Bible book, chapter, and verse, "
            "or a clearly identifiable Bible passage being quoted. Do not "
            "invent a reference. If the audio is unclear or no reference can "
            "be identified, say UNCERTAIN. "
            "Return a concise result with these fields: "
            "BOOK: <book or UNKNOWN> | CHAPTER: <number or UNKNOWN> | "
            "VERSE: <number, range, or UNKNOWN> | "
            "CONFIDENCE: <low, medium, or high> | "
            "HEARD: <short description of what was heard>. "
            "Treat confidence as an estimate, not proof."
        )

        command = [
            self.executable,
            "-m",
            str(self.model_path),
            "--mmproj",
            str(self.projector_path),
            "--audio",
            str(audio),
            "-p",
            prompt,
            "--jinja",
            "--no-warmup",
        ]

        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.timeout_seconds,
                check=False,
                creationflags=(
                    subprocess.CREATE_NO_WINDOW
                    if os.name == "nt"
                    else 0
                ),
            )
        except subprocess.TimeoutExpired:
            return AudioInferenceResult(
                success=False,
                error=f"Gemma audio inference timed out after "
                f"{self.timeout_seconds} seconds.",
            )
        except OSError as exc:
            return AudioInferenceResult(
                success=False,
                error=f"Could not start llama-mtmd-cli: {exc}",
            )

        output = completed.stdout.strip()
        stderr = completed.stderr.strip()

        if completed.returncode != 0:
            details = stderr or output or "No diagnostic output was returned."
            return AudioInferenceResult(
                success=False,
                output=output,
                error=f"llama-mtmd-cli exited with code "
                f"{completed.returncode}: {details[-2000:]}",
            )

        if not output:
            return AudioInferenceResult(
                success=False,
                error="Gemma returned no standard output.",
            )

        return AudioInferenceResult(success=True, output=output)
