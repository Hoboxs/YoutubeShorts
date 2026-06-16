from pathlib import Path
from datetime import datetime, timedelta
from CreatePostImage import createPostImage
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess
import shutil
import json
from dataclasses import dataclass


@dataclass
class ClipResult:
    src: str
    out: str | None
    success: bool
    reason: str = ""


def format_date(dt: datetime) -> str:
    return f"{dt.strftime('%B')} {dt.day}, {dt.year}"


def has_audio_stream(filepath: Path) -> bool:
    """Return True if the file contains at least one audio stream."""
    probe = subprocess.run(
        [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_streams",
            filepath.as_posix(),
        ],
        capture_output=True,
        text=True,
    )

    try:
        streams = json.loads(probe.stdout).get("streams", [])
        return any(s.get("codec_type") == "audio" for s in streams)
    except json.JSONDecodeError as e:
        print(f"  ⚠️  Could not parse ffprobe output: {e}")
        return False


def validate_sources(source_files: list[Path]) -> None:
    """Abort if any source file is missing an audio stream."""
    missing_audio = [f for f in source_files if not has_audio_stream(f)]

    if missing_audio:
        print("\n❌ The following files have no audio stream:")
        for f in missing_audio:
            print(f"   - {f}")
        raise RuntimeError(
            f"{len(missing_audio)} file(s) are missing audio. "
            "Fix the source files before running again."
        )

    print(f"✅ All {len(source_files)} files have audio streams.")


def normalize_clip(src: Path, out: Path, idx: int, total: int) -> ClipResult:
    """Normalize a single clip to 1080p/60fps with stereo audio."""
    print(f"\nNormalizing ({idx}/{total}): {src.name}")

    cmd = [
        "ffmpeg", "-y",
        "-err_detect", "ignore_err",
        "-i", src.as_posix(),
        "-map", "0:v:0",
        "-map", "0:a:0",
        "-vf", "scale=-2:1080,fps=60",
        "-af", "aresample=48000",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-profile:v", "high",
        "-level", "4.2",
        "-preset", "veryfast",
        "-c:a", "aac",
        "-ac", "2",
        "-ar", "48000",
        "-b:a", "192k",
        out.as_posix(),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        return ClipResult(
            src=str(src),
            out=None,
            success=False,
            reason=result.stderr[-800:],
        )

    return ClipResult(src=str(src), out=str(out), success=True)


def normalize_all(source_files: list[Path], temp_dir: Path) -> list[ClipResult]:
    """Normalize all clips in parallel."""
    total = len(source_files)
    jobs = {
        idx: (src, temp_dir / f"clip_{idx:04d}.mp4")
        for idx, src in enumerate(source_files, start=1)
    }

    results: dict[int, ClipResult] = {}

    with ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(normalize_clip, src, out, idx, total): idx
            for idx, (src, out) in jobs.items()
        }

        for future in as_completed(futures):
            idx = futures[future]
            results[idx] = future.result()

    # Return in original order
    return [results[i] for i in sorted(results)]


def main():
    # --------------------------------------------------
    # SETTINGS — dates derived automatically from today
    # --------------------------------------------------
    YEAR, MONTH, DAY = 2026, 6, 14  # Set this manually each run

    today      = datetime(YEAR, MONTH, DAY)
    end_date   = today
    start_date = end_date - timedelta(days=6)

    BASE_DIR   = Path.cwd() / "Results"
    OUTPUT_DIR = Path.cwd() / "Compilation"
    TEMP_DIR   = OUTPUT_DIR / "temp"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        TEMP_DIR.mkdir(parents=True, exist_ok=True)

        FINAL_FILE     = OUTPUT_DIR / f"compilation_60fps_{timestamp}.mp4"
        thumbnail_path = OUTPUT_DIR / f"ThumbnailIntroImage_{timestamp}.png"

        # Dynamically generated thumbnail title
        title = (
            f"Weekly Shorts Recap: Best Clips from "
            f"{format_date(start_date)} to {format_date(end_date)}!"
        )
        createPostImage(title, thumbnail_path.as_posix(), True)

        # --------------------------------------------------
        # COLLECT result*.mp4 FILES (RECURSIVE, 7 DAYS)
        # --------------------------------------------------
        source_files: list[Path] = []
        current_date = start_date

        while current_date <= end_date:
            folder_path = BASE_DIR / current_date.strftime("%Y_%m_%d")
            print(f"Scanning: {folder_path}")

            if folder_path.is_dir():
                source_files.extend(sorted(folder_path.rglob("result*.mp4")))
            else:
                print(f"  ⚠️  Folder not found: {folder_path}")

            current_date += timedelta(days=1)

        if not source_files:
            raise RuntimeError("No result*.mp4 files found.")

        print(f"\nFound {len(source_files)} file(s) across 7 folders.")

        # --------------------------------------------------
        # STEP 1: VALIDATE ALL CLIPS HAVE AUDIO
        # --------------------------------------------------
        validate_sources(source_files)

        # --------------------------------------------------
        # STEP 2: NORMALIZE EACH CLIP (PARALLEL)
        # --------------------------------------------------
        clip_results = normalize_all(source_files, TEMP_DIR)

        failed = [r for r in clip_results if not r.success]
        if failed:
            print(f"\n❌ {len(failed)} clip(s) failed FFmpeg normalization:")
            for r in failed:
                print(f"   - {r.src}")
                print(f"     Reason: {r.reason}")
            raise RuntimeError("Fix the above files and run again.")

        normalized_files = [Path(r.out) for r in clip_results if r.out]

        # --------------------------------------------------
        # STEP 3: CONCATENATE (NO RE-ENCODE)
        # --------------------------------------------------
        concat_file = TEMP_DIR / "concat.txt"
        concat_file.write_text(
            "\n".join(f"file '{p.as_posix()}'" for p in normalized_files),
            encoding="utf-8",
        )

        subprocess.run(
            [
                "ffmpeg", "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", concat_file.as_posix(),
                "-c", "copy",
                FINAL_FILE.as_posix(),
            ],
            check=True,
        )

        print(f"\n✅ SUCCESS — {FINAL_FILE}")

    finally:
        # --------------------------------------------------
        # STEP 4: CLEAN UP (always runs)
        # --------------------------------------------------
        if TEMP_DIR.exists():
            print("\nCleaning up temp files...")
            try:
                shutil.rmtree(TEMP_DIR)
                print("🧹 Temp files deleted.")
            except Exception as e:
                print(f"⚠️  Could not delete temp files: {e}")


if __name__ == "__main__":
    main()