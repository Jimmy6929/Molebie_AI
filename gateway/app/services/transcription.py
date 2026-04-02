"""
Local speech-to-text using faster-whisper (CTranslate2).

Model is lazy-loaded on first request and cached in memory.
The 'tiny' model (~75 MB) auto-downloads to ~/.cache/huggingface/hub/.
"""

import asyncio
import os
import tempfile

_model = None


def _get_model():
    global _model
    if _model is not None:
        return _model

    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise RuntimeError(
            "faster-whisper is not installed. "
            "Voice transcription requires PyTorch and faster-whisper — "
            "install with: pip install -r requirements-ml.txt"
        )

    print("[transcription] Loading Whisper tiny model (first call may download ~75 MB)...")
    _model = WhisperModel("tiny", device="cpu", compute_type="int8")
    print("[transcription] Whisper model loaded.")
    return _model


def _transcribe_sync(audio_bytes: bytes, suffix: str = ".webm") -> str:
    model = _get_model()
    fd, temp_path = tempfile.mkstemp(suffix=suffix)
    try:
        os.write(fd, audio_bytes)
        os.close(fd)
        segments, _ = model.transcribe(temp_path, beam_size=1, language="en")
        return " ".join(seg.text for seg in segments).strip()
    finally:
        os.unlink(temp_path)


async def transcribe_audio(audio_bytes: bytes, filename: str = "audio.webm") -> str:
    suffix = os.path.splitext(filename)[1] or ".webm"
    return await asyncio.to_thread(_transcribe_sync, audio_bytes, suffix)
