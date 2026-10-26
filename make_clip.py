"""Create an accurately trimmed MP4 benchmark input from a page, stream or file."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from oh_streamer.sources import source_resolve, qualities


def timestamp(value: str) -> float:
    """Seconds, MM:SS, or HH:MM:SS; fractions are allowed on seconds."""
    try:
        parts = value.strip().split(":")
        if not 1 <= len(parts) <= 3:
            raise ValueError
        seconds = float(parts[-1])
        if len(parts) > 1:
            if not 0 <= seconds < 60 or any(not p.isdigit() for p in parts[:-1]):
                raise ValueError
            if len(parts) == 3 and int(parts[1]) >= 60:
                raise ValueError
            seconds += int(parts[-2]) * 60
            if len(parts) == 3:
                seconds += int(parts[0]) * 3600
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError
        return seconds
    except ValueError:
        raise argparse.ArgumentTypeError("請輸入非負秒數、MM:SS 或 HH:MM:SS，例如 90、01:30、00:01:30.5") from None


def resolve_input(target: str, headers: dict, quality: str) -> dict:
    if not source_resolve.is_url(target):
        path = Path(target).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"找不到影片：{path}")
        return {"url": str(path), "http_headers": {}, "title": path.stem, "label": "original"}
    title = target
    if resolver := source_resolve.find_resolver(target):
        target, headers, title = source_resolve.resolve_external(target, headers, resolver)
    if source_resolve.looks_like_media_url(target):
        return {"url": target, "http_headers": headers, "title": title, "label": "original"}
    formats, resolved_title, _ = source_resolve.resolve_with_ytdlp(target, headers)
    selected = dict(qualities.pick_quality(formats, qualities.parse_quality_spec(quality)))
    selected["http_headers"] = {**selected.get("http_headers", {}), **headers}
    selected["title"] = title if title != target else resolved_title
    return selected


def ffmpeg_command(ffmpeg: str, source: dict, start: float, duration: float,
                   output: Path, crf: int, hls_options: list[str]) -> list[str]:
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "warning", "-stats", "-nostdin", "-y"]
    inputs = [source["url"]]
    if source.get("audio_url"):
        inputs.append(source["audio_url"])
    for url in inputs:
        if source_resolve.is_url(url):
            cmd += source_resolve.FFmpegHttpOptions(source.get("http_headers", {})).cli_args()
            cmd += ["-rw_timeout", "20000000"]
        if ".m3u8" in url.split("?", 1)[0].lower():
            cmd += hls_options
        cmd += ["-ss", str(start), "-i", url]
    cmd += ["-t", str(duration), "-map", "0:v:0", "-map", "1:a:0" if len(inputs) == 2 else "0:a:0?",
            "-c:v", "libx264", "-preset", "fast", "-crf", str(crf), "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(output)]
    return cmd


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="擷取短影片供 Lada 速度測試（保持原解析度與幀率，精確裁切）")
    parser.add_argument("url", nargs="?", help="影片頁、直接串流網址或本機檔案；省略時互動輸入")
    parser.add_argument("--start", type=timestamp, help="開始時間：秒數／MM:SS／HH:MM:SS")
    parser.add_argument("--duration", type=timestamp, help="片段時長，格式同 --start")
    parser.add_argument("-o", "--output", type=Path, help="輸出 MP4 路徑")
    parser.add_argument("--quality", default="best", help="best／worst／720p 等；預設最高畫質")
    parser.add_argument("--header", action="append", default=[], help="額外 HTTP 標頭，格式 Name: value")
    parser.add_argument("--crf", type=int, default=18, help="H.264 品質，0–51；預設 18")
    parser.add_argument("--overwrite", action="store_true", help="允許覆寫既有輸出")
    args = parser.parse_args(argv)
    try:
        for stream in (sys.stdout, sys.stderr):
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8", errors="replace")
        target = args.url or input("影片連結或檔案路徑：").strip().strip('"')
        start = args.start if args.start is not None else timestamp(input("開始時間（例如 00:05:00）："))
        duration = args.duration if args.duration is not None else timestamp(input("片段時長（例如 30）："))
        if not target or duration <= 0 or not 0 <= args.crf <= 51:
            raise ValueError("連結不能為空、時長必須大於 0，CRF 必須介於 0–51")
        qualities.parse_quality_spec(args.quality)
        output = (args.output or ROOT / "clips" / f"clip_{int(time.time())}_{start:g}s_{duration:g}s.mp4").resolve()
        if output.suffix.lower() != ".mp4":
            raise ValueError("輸出檔案請使用 .mp4 副檔名")
        if output.exists() and not args.overwrite:
            raise ValueError(f"檔案已存在：{output}；請指定其他路徑或 --overwrite")
        if not source_resolve.is_url(target) and Path(target).expanduser().resolve() == output:
            raise ValueError("輸出路徑不能與輸入檔案相同")
        ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
        if not ffmpeg or not ffprobe:
            raise RuntimeError("找不到 ffmpeg／ffprobe，請將它們加入 PATH")
        print("解析來源中…", flush=True)
        began = time.monotonic()
        source = resolve_input(target, source_resolve.parse_headers(args.header), args.quality)
        print(f"畫質：{source['label']}；開始 {start:g} 秒；擷取 {duration:g} 秒", flush=True)
        # Some CDNs use .jpeg names for valid MPEG-TS segments. Newer FFmpeg also
        # checks segment extensions separately; only add its option when supported.
        help_text = subprocess.run([ffmpeg, "-hide_banner", "-h", "demuxer=hls"],
                                   capture_output=True, text=True, check=True).stdout
        hls_options = ["-allowed_extensions", "ALL"]
        if "allowed_segment_extensions" in help_text:
            hls_options += ["-allowed_segment_extensions", "ALL"]
        if "extension_picky" in help_text:
            hls_options += ["-extension_picky", "0"]
        output.parent.mkdir(parents=True, exist_ok=True)
        # Publish only a successfully probed output; failed runs leave no partial MP4.
        with tempfile.TemporaryDirectory(prefix=".make-clip-", dir=output.parent) as temporary:
            partial = Path(temporary) / "clip.mp4"
            subprocess.run(ffmpeg_command(ffmpeg, source, start, duration, partial, args.crf, hls_options), check=True)
            probe = subprocess.run([ffprobe, "-v", "error", "-show_streams", "-show_format", "-of", "json", str(partial)],
                                   capture_output=True, text=True, encoding="utf-8", check=True)
            info = json.loads(probe.stdout)
            if not any(s.get("codec_type") == "video" for s in info.get("streams", [])):
                raise RuntimeError("擷取結果沒有影片影格；開始時間可能超出影片長度")
            actual = float(info["format"].get("duration", 0))
            if actual <= 0:
                raise RuntimeError("擷取結果為空；請確認開始時間")
            if actual < duration - 0.5:
                print(f"來源可能已到結尾：實際時長 {actual:.3f} 秒", file=sys.stderr)
            if output.exists() and not args.overwrite:
                raise ValueError(f"輸出檔案已被另一個程序建立：{output}")
            partial.replace(output)
        metadata = {"input": target, "start_seconds": start, "requested_duration_seconds": duration,
                    "actual_duration_seconds": actual, "quality": source["label"], "crf": args.crf,
                    "title": source["title"], "preparation_seconds": round(time.monotonic() - began, 3),
                    "streams": [{k: s[k] for k in ("codec_type", "codec_name", "width", "height", "avg_frame_rate") if k in s}
                                for s in info["streams"]]}
        output.with_suffix(".json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"完成：{output}\n實際時長：{actual:.3f} 秒；測試資料：{output.with_suffix('.json')}")
        return 0
    except KeyboardInterrupt:
        print("已取消", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"擷取失敗：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
