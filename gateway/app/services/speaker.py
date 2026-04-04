"""
Speaker verification via MFCC voice embeddings.

Uses only numpy + ffmpeg (both already available). No PyTorch needed.
Voice profiles are stored as JSON files in ~/.local-ai/voice-profiles/.
"""

import asyncio
import json
import os
import subprocess
import tempfile
import wave

import numpy as np

PROFILES_DIR = os.path.expanduser("~/.local-ai/voice-profiles")
SIMILARITY_THRESHOLD = 0.82
REQUIRED_SAMPLES = 3


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _profile_path(user_id: str) -> str:
    _ensure_dir(PROFILES_DIR)
    return os.path.join(PROFILES_DIR, f"{user_id}.json")


def _audio_to_pcm(audio_bytes: bytes, suffix: str = ".webm") -> tuple:
    """Convert audio bytes to 16 kHz mono float32 PCM via ffmpeg."""
    fd_in, in_path = tempfile.mkstemp(suffix=suffix)
    fd_out, out_path = tempfile.mkstemp(suffix=".wav")
    try:
        os.write(fd_in, audio_bytes)
        os.close(fd_in)
        os.close(fd_out)
        subprocess.run(
            [
                "ffmpeg", "-i", in_path,
                "-ar", "16000", "-ac", "1", "-f", "wav", "-y", out_path,
            ],
            capture_output=True,
            check=True,
        )
        with wave.open(out_path, "r") as wf:
            raw = wf.readframes(wf.getnframes())
            rate = wf.getframerate()
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        return rate, data
    finally:
        for p in (in_path, out_path):
            try:
                os.unlink(p)
            except OSError:
                pass


def _extract_mfcc(signal: np.ndarray, rate: int, n_mfcc: int = 20) -> np.ndarray:
    """Extract mean MFCC vector from a float32 mono signal."""
    if len(signal) < int(rate * 0.5):
        return np.zeros(n_mfcc)

    emph = np.append(signal[0], signal[1:] - 0.97 * signal[:-1])

    frame_len = int(0.025 * rate)
    frame_step = int(0.01 * rate)
    n_frames = max(1, (len(emph) - frame_len) // frame_step + 1)

    indices = np.arange(frame_len)[None, :] + (np.arange(n_frames) * frame_step)[:, None]
    indices = np.clip(indices, 0, len(emph) - 1)
    frames = emph[indices] * np.hamming(frame_len)

    NFFT = 512
    power = np.abs(np.fft.rfft(frames, NFFT)) ** 2 / NFFT

    n_filt = 40
    low_mel = 0.0
    high_mel = 2595.0 * np.log10(1.0 + rate / 2.0 / 700.0)
    mel_pts = np.linspace(low_mel, high_mel, n_filt + 2)
    hz_pts = 700.0 * (10.0 ** (mel_pts / 2595.0) - 1.0)
    bins = np.floor((NFFT + 1) * hz_pts / rate).astype(int)

    fbank = np.zeros((n_filt, NFFT // 2 + 1))
    for i in range(1, n_filt + 1):
        left, center, right = int(bins[i - 1]), int(bins[i]), int(bins[i + 1])
        if center > left:
            fbank[i - 1, left:center] = (np.arange(left, center) - left) / (center - left)
        if right > center:
            fbank[i - 1, center:right] = (right - np.arange(center, right)) / (right - center)

    fb = np.dot(power, fbank.T)
    fb = np.where(fb == 0, np.finfo(float).eps, fb)
    fb = 20.0 * np.log10(fb)

    N = fb.shape[1]
    k = np.arange(n_mfcc)[:, None]
    n_idx = np.arange(N)[None, :]
    dct_m = np.cos(np.pi * k * (2 * n_idx + 1) / (2 * N)) * np.sqrt(2.0 / N)
    dct_m[0] *= 1.0 / np.sqrt(2.0)

    mfcc = fb @ dct_m.T
    return np.mean(mfcc, axis=0)


def _extract_embedding(audio_bytes: bytes, suffix: str = ".webm") -> np.ndarray:
    rate, signal = _audio_to_pcm(audio_bytes, suffix)
    return _extract_mfcc(signal, rate)


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _load_profile(user_id: str):
    path = _profile_path(user_id)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    return {
        "embedding": np.array(data["embedding"]),
        "n_samples": data["n_samples"],
    }


def _save_profile(user_id: str, embedding: np.ndarray, n_samples: int):
    path = _profile_path(user_id)
    with open(path, "w") as f:
        json.dump({"embedding": embedding.tolist(), "n_samples": n_samples}, f)


# -- Public async API --------------------------------------------------------

def _enroll_sync(audio_bytes: bytes, user_id: str, suffix: str) -> dict:
    new_emb = _extract_embedding(audio_bytes, suffix)
    profile = _load_profile(user_id)

    if profile is None:
        _save_profile(user_id, new_emb, 1)
        return {"enrolled": True, "n_samples": 1, "complete": False, "required": REQUIRED_SAMPLES}

    n = profile["n_samples"]
    avg = (profile["embedding"] * n + new_emb) / (n + 1)
    _save_profile(user_id, avg, n + 1)
    return {
        "enrolled": True,
        "n_samples": n + 1,
        "complete": n + 1 >= REQUIRED_SAMPLES,
        "required": REQUIRED_SAMPLES,
    }


def _verify_sync(audio_bytes: bytes, user_id: str, suffix: str) -> tuple:
    profile = _load_profile(user_id)
    if profile is None or profile["n_samples"] < REQUIRED_SAMPLES:
        return True, 1.0  # no profile or incomplete → allow

    emb = _extract_embedding(audio_bytes, suffix)
    confidence = _cosine_sim(emb, profile["embedding"])
    return confidence >= SIMILARITY_THRESHOLD, confidence


def _delete_sync(user_id: str) -> bool:
    path = _profile_path(user_id)
    if os.path.exists(path):
        os.unlink(path)
        return True
    return False


def _status_sync(user_id: str) -> dict:
    profile = _load_profile(user_id)
    if profile is None:
        return {"enrolled": False, "n_samples": 0, "complete": False, "required": REQUIRED_SAMPLES}
    return {
        "enrolled": True,
        "n_samples": profile["n_samples"],
        "complete": profile["n_samples"] >= REQUIRED_SAMPLES,
        "required": REQUIRED_SAMPLES,
    }


async def enroll_voice_sample(audio_bytes: bytes, user_id: str, filename: str = "audio.webm") -> dict:
    suffix = os.path.splitext(filename)[1] or ".webm"
    return await asyncio.to_thread(_enroll_sync, audio_bytes, user_id, suffix)


async def verify_speaker(audio_bytes: bytes, user_id: str, filename: str = "audio.webm") -> tuple:
    suffix = os.path.splitext(filename)[1] or ".webm"
    return await asyncio.to_thread(_verify_sync, audio_bytes, user_id, suffix)


async def delete_voice_profile(user_id: str) -> bool:
    return await asyncio.to_thread(_delete_sync, user_id)


async def get_voice_profile_status(user_id: str) -> dict:
    return await asyncio.to_thread(_status_sync, user_id)
