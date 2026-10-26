"""Rebuildable, public synthetic transport/audio fixture; no model benchmark footage."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading


def generate_fixture(output: Path, duration=4, width=320, height=180, fps=30) -> dict:
    """Encode moving colour bars, visible frame/timecode and a 440 Hz AAC tone.

    Pillow's bundled bitmap font avoids FFmpeg drawtext/system font dependencies.
    This exercises transport, segmentation, seek and audio, not mosaic detection.
    Output is replaced only after successful encoding. FFmpeg runs with a timeout.
    """
    from PIL import Image, ImageDraw, ImageFont

    output = Path(output).resolve()
    if (not math.isfinite(float(duration)) or not 0 < duration <= 600
            or not isinstance(fps, int) or not 1 <= fps <= 120
            or not isinstance(width, int) or not isinstance(height, int)
            or width < 160 or height < 100 or width > 1920 or height > 1080
            or width % 2 or height % 2):
        raise ValueError("Use duration (0,600], integer fps [1,120], even dimensions [160x100,1920x1080]")
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("FFmpeg is required to generate the synthetic fixture")
    frames = max(1, round(float(duration) * fps))
    actual_duration = frames / fps
    output.parent.mkdir(parents=True, exist_ok=True)
    errors = []
    with tempfile.TemporaryDirectory(prefix="fixture-", dir=output.parent) as tmp:
        encoded = Path(tmp) / "fixture.mp4"
        command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
                   "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", f"{width}x{height}",
                   "-framerate", str(fps), "-i", "pipe:0", "-f", "lavfi", "-i",
                   "sine=frequency=440:sample_rate=48000", "-map", "0:v:0", "-map", "1:a:0",
                   "-t", str(actual_duration), "-c:v", "libx264", "-preset", "veryfast",
                   "-crf", "18", "-threads", "1", "-pix_fmt", "yuv420p", "-c:a", "aac",
                   "-b:a", "96k", "-movflags", "+faststart", "-map_metadata", "-1", str(encoded)]
        font = ImageFont.load_default()
        colours = [(225, 220, 40), (35, 210, 210), (30, 200, 50), (210, 35, 190), (220, 45, 35), (35, 50, 220)]
        with tempfile.TemporaryFile() as stderr:
            proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=stderr)

            def write_frames():
                try:
                    for frame in range(frames):
                        picture = Image.new("RGB", (width, height), (20, 20, 20))
                        draw = ImageDraw.Draw(picture)
                        for index, colour in enumerate(colours):
                            draw.rectangle((index * width // 6, 0, (index + 1) * width // 6, height), fill=colour)
                        x = round(frame * (width - 24) / max(1, frames - 1))
                        draw.rectangle((x, height // 2, x + 23, height // 2 + 23), fill="white", outline="black")
                        seconds, fraction = divmod(frame, fps)
                        hours, remainder = divmod(seconds, 3600)
                        minutes, seconds = divmod(remainder, 60)
                        draw.rectangle((0, 0, width, 42), fill="black")
                        draw.text((5, 4), f"SYNTHETIC  frame {frame:06d}", font=font, fill="white")
                        draw.text((5, 22), f"{hours:02d}:{minutes:02d}:{seconds:02d}:{fraction:02d}  {fps} fps / 440 Hz", font=font, fill="white")
                        proc.stdin.write(picture.tobytes())
                except (OSError, ValueError) as exc:
                    errors.append(str(exc))
                finally:
                    proc.stdin.close()

            writer = threading.Thread(target=write_frames, daemon=True)
            writer.start()
            try:
                proc.wait(timeout=max(30, actual_duration * 5))
            except subprocess.TimeoutExpired as exc:
                proc.kill()
                proc.wait(timeout=10)
                raise RuntimeError("Synthetic fixture FFmpeg timed out") from exc
            finally:
                writer.join(timeout=10)
            if proc.returncode or errors or writer.is_alive():
                stderr.seek(0)
                raise RuntimeError("Synthetic fixture encoding failed: " + stderr.read().decode("utf-8", "replace")[-4000:] + "; ".join(errors))
        encoded.replace(output)
    return {"path": str(output), "duration": actual_duration, "width": width, "height": height,
            "fps": fps, "frame_count": frames, "audio_hz": 440, "audio_sample_rate": 48000,
            "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "purpose": "synthetic transport/audio test; not a real mosaic detection benchmark"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=4)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=180)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()
    print(json.dumps(generate_fixture(args.output, args.duration, args.width, args.height, args.fps), indent=2))
