from moviepy import VideoFileClip, AudioFileClip, CompositeVideoClip, ImageClip
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from CreatePostImage import createPostImage
from CreateTTS import generate_tts
import subprocess
import random
import re
import os
import shutil
import time


# ---------------------------------------------------------------------------
# Config dataclass — tweak here, never touch the rest of the code
# ---------------------------------------------------------------------------

@dataclass
class Config:
    test_mode:           bool  = True

    # Audio limits (seconds)
    max_audio_length:    int   = 180
    max_audio_cutoff:    int   = 176

    # Background video
    bg_speed_up:         float = 1.5
    bg_fps:              int   = 60
    bg_duration_buffer:  float = 5.0   # extra seconds so bg is always longer than audio

    # Output resolution
    final_width:         int   = 1080
    final_height:        int   = 1920

    # ffmpeg retry
    ffmpeg_retries:      int   = 3
    ffmpeg_retry_delay:  int   = 3     # seconds between retries

    # Job-level retry (retries the full pipeline, not just ffmpeg)
    job_retries:         int   = 1

    # Folders
    texts_dir:           str   = "Texts"
    processing_dir:      str   = "Processing"
    bg_clips_dir:        str   = "BackgroundClips"
    results_dir:         str   = "Results"
    tests_dir:           str   = "Tests"
    failed_dir:          str   = "Failed"

    sentence_end:        frozenset = field(default_factory=lambda: frozenset({".", "?", "!"}))


# ---------------------------------------------------------------------------
# Per-file paths — one fresh instance per job, nothing leaks between runs
# ---------------------------------------------------------------------------

@dataclass
class JobPaths:
    timestamp:        str
    text_path:        Path
    audio_path:       Path
    subtitle_path:    Path
    intro_image_path: Path
    used_videos_path: Path
    bg_video_path:    Path
    final_path:       Path

    @classmethod
    def from_timestamp(cls, ts: str, cfg: Config) -> "JobPaths":
        p = Path(cfg.processing_dir)
        return cls(
            timestamp        = ts,
            text_path        = p / f"text_{ts}.txt",
            audio_path       = p / f"audio_{ts}.wav",
            subtitle_path    = p / f"subtitles_{ts}.ass",
            intro_image_path = p / f"IntroImage_{ts}.png",
            used_videos_path = p / f"VideosUsed_{ts}.txt",
            bg_video_path    = p / f"background_{ts}.mp4",
            final_path       = p / f"result_{ts}.mp4",
        )


# ---------------------------------------------------------------------------
# ffmpeg helper — retries with full stderr printed on every failure
# ---------------------------------------------------------------------------

def run_ffmpeg(args: list[str], cfg: Config, label: str = "ffmpeg") -> None:
    for attempt in range(1, cfg.ffmpeg_retries + 1):
        result = subprocess.run(args, capture_output=True, text=True)
        if result.returncode == 0:
            return

        print(f"\n  [{label}] attempt {attempt}/{cfg.ffmpeg_retries} failed (exit {result.returncode})")
        print("  --- ffmpeg stderr (last 3000 chars) ---")
        print(result.stderr[-3000:])
        print("  --- end stderr ---\n")

        if attempt < cfg.ffmpeg_retries:
            time.sleep(cfg.ffmpeg_retry_delay)
        else:
            raise RuntimeError(f"ffmpeg step '{label}' failed after {cfg.ffmpeg_retries} attempts")


# ---------------------------------------------------------------------------
# Clip pre-screener — runs once at startup, filters out bad files
# ---------------------------------------------------------------------------

def probe_clip(path: Path) -> tuple:
    """Returns (path, duration_or_None, is_healthy)."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error",
             "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1",
             str(path)],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return path, None, False
        return path, float(result.stdout.strip()), True
    except Exception:
        return path, None, False


def prescreen_clips(cfg: Config) -> list:
    """
    Probes every .mp4 in BackgroundClips/ in parallel using ffprobe.
    Returns a list of (path, duration) for healthy clips only.
    Logs any bad files so you know to replace them.
    """
    print("\n---------------------------------")
    print("Pre-screening Background Clips")
    print("---------------------------------")

    clips_dir = Path(cfg.bg_clips_dir).resolve()
    all_clips = list(clips_dir.glob("*.mp4"))
    if not all_clips:
        raise ValueError(f"No .mp4 files found in {clips_dir}")

    healthy, skipped = [], []

    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(probe_clip, p): p for p in all_clips}
        for fut in as_completed(futures):
            path, duration, ok = fut.result()
            if ok and duration and duration > 0:
                healthy.append((path, duration))
            else:
                skipped.append(path)

    if skipped:
        print(f"  ⚠️  Skipped {len(skipped)} corrupt/unreadable clip(s):")
        for p in sorted(skipped):
            print(f"     {p.name}")

    print(f"  ✅ {len(healthy)} healthy clip(s) ready")

    if not healthy:
        raise RuntimeError("No healthy background clips found — check BackgroundClips/")

    return healthy


# ---------------------------------------------------------------------------
# Step 1 — pick & concat background clips
# ---------------------------------------------------------------------------

def pick_random_file(target_length: float, paths: JobPaths, cfg: Config,
                     healthy_clips: list) -> Path:
    print("\n---------------------------------")
    print("Picking Background Clips")
    print("---------------------------------")

    buffered_target = target_length + cfg.bg_duration_buffer
    clips = random.sample(healthy_clips, k=len(healthy_clips))

    selected = []
    total_duration = 0.0

    for path, duration in clips:
        if total_duration >= buffered_target:
            break
        usable = duration / cfg.bg_speed_up
        if usable <= 0:
            continue
        selected.append(path)
        total_duration += usable

    if not selected:
        raise RuntimeError("Could not select enough background clips for target duration.")

    print(f"  Selected {len(selected)} clip(s), usable: {total_duration:.2f}s "
          f"(need {buffered_target:.2f}s)")

    with paths.used_videos_path.open("w", encoding="utf-8") as f:
        for v in selected:
            f.write(f"file '{v.resolve()}'\n")

    run_ffmpeg([
        "ffmpeg", "-y",
        "-loglevel", "error",
        "-err_detect", "ignore_err",
        "-fflags", "+genpts+igndts",
        "-avoid_negative_ts", "make_zero",
        "-f", "concat", "-safe", "0",
        "-i", str(paths.used_videos_path),
        # Force CFR output at target FPS so no frames are held/frozen at clip boundaries.
        # fps filter runs BEFORE setpts so each clip is resampled to a consistent
        # timeline first, then the speedup is applied cleanly on top.
        "-filter_complex",
        f"[0:v]fps={cfg.bg_fps},setpts={1/cfg.bg_speed_up}*PTS[vout]",
        "-map", "[vout]",
        "-an",
        "-vsync", "cfr",
        "-r", str(cfg.bg_fps),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-profile:v", "high",
        "-level", "4.2",
        str(paths.bg_video_path),
    ], cfg=cfg, label="concat_background")

    print(f"  Background video ready: {paths.bg_video_path.name}")
    return paths.bg_video_path


# ---------------------------------------------------------------------------
# Step 2 — generate audio + subtitles + intro image (with clear error context)
# ---------------------------------------------------------------------------

def generate_audio(paths: JobPaths) -> tuple:
    if not paths.text_path.exists():
        raise FileNotFoundError(f"TTS text file not found: {paths.text_path}")

    file_text = paths.text_path.read_text(encoding="utf-8").strip()
    if not file_text:
        raise ValueError(f"TTS text file is empty: {paths.text_path}")

    first_line = file_text.splitlines()[0].upper()

    try:
        createPostImage(first_line, str(paths.intro_image_path))
    except Exception as e:
        raise RuntimeError(f"createPostImage failed for '{first_line[:40]}': {e}") from e

    # intro_length, total_length = generate_tts(
    #         audio_path=str(paths.audio_path),
    #         ass_path=str(paths.subtitle_path),
    #         text=file_text,
    #     )
    
    # try:
    #     intro_length, total_length = generate_tts(
    #         audio_path=str(paths.audio_path),
    #         ass_path=str(paths.subtitle_path),
    #         text=file_text,
    #     )
    # except Exception as e:
    #     raise RuntimeError(f"generate_tts failed: {e}") from e

    try:
        intro_length, total_length = generate_tts(
            audio_path=str(paths.audio_path),
            ass_path=str(paths.subtitle_path),
            text=file_text,
        )
    except Exception as e:
        import traceback
        traceback.print_exc()  # prints full traceback to stderr
        raise RuntimeError(f"generate_tts failed: {e}") from e

    return first_line, intro_length, total_length


# ---------------------------------------------------------------------------
# Step 3 — resize + crop to 9:16
# ---------------------------------------------------------------------------

def resize_and_crop(clip, cfg: Config):
    resized = clip.resized(height=cfg.final_height)
    if resized.w < cfg.final_width:
        resized = clip.resized(width=cfg.final_width)

    w, h = resized.size
    x1 = max(0, (w - cfg.final_width)  // 2)
    y1 = max(0, (h - cfg.final_height) // 2)

    return resized.cropped(
        x1=x1, y1=y1,
        x2=x1 + cfg.final_width,
        y2=y1 + cfg.final_height,
    )


# ---------------------------------------------------------------------------
# Step 4 — compose & export (two-pass: moviepy render, then ffmpeg subtitle burn)
# ---------------------------------------------------------------------------

def compose_and_export(paths: JobPaths, intro_length: float, cfg: Config) -> None:
    print("\n---------------------------------")
    print("Composing and Exporting")
    print("---------------------------------")

    audio_clip = AudioFileClip(str(paths.audio_path))
    video_clip = VideoFileClip(str(paths.bg_video_path))
    temp_path  = paths.final_path.with_suffix(".tmp.mp4")

    try:
        trimmed   = video_clip.subclipped(0, audio_clip.duration).without_audio()
        trimmed   = resize_and_crop(trimmed, cfg)
        intro_img = (ImageClip(str(paths.intro_image_path))
                     .with_position("center")
                     .with_duration(intro_length))
        composite = CompositeVideoClip([trimmed, intro_img])
        final     = composite.with_audio(audio_clip)

        # Pass 1: moviepy renders clean ProRes — no filter interference
        final.write_videofile(
            str(temp_path),
            fps=cfg.bg_fps,
            codec="libx264",
            audio_codec="aac",
            ffmpeg_params=["-profile:v", "high", "-level", "4.2", "-crf", "18", "-movflags", "+faststart"],
            threads=4,
        )

    finally:
        video_clip.close()
        audio_clip.close()
        try:
            final.close()
        except Exception:
            pass

    # Pass 2: ffmpeg burns subtitles onto the clean ProRes file
    # Windows-safe path: forward slashes, drive colon escaped as \:
    sub_str = str(paths.subtitle_path.resolve()).replace("\\", "/").replace(":", "\\:")
    vf_arg  = f"ass=filename='{sub_str}',format=yuv420p"

    run_ffmpeg([
        "ffmpeg", "-y",
        "-loglevel", "error", "-stats",
        "-i", str(temp_path),
        "-vf", vf_arg,
        "-c:v", "libx264",
        "-profile:v", "high",
        "-level", "4.2",
        "-crf", "18",
        "-movflags", "+faststart",
        "-pix_fmt", "yuv420p",
        "-c:a", "copy",
        str(paths.final_path),
    ], cfg=cfg, label="burn_subtitles")

    temp_path.unlink(missing_ok=True)
    print(f"  Final file: {paths.final_path.name}")


# ---------------------------------------------------------------------------
# Step 5 — move results to dated folder
# ---------------------------------------------------------------------------

def move_results(paths: JobPaths, first_sentence: str, cfg: Config) -> None:
    date_str    = (datetime.now() + timedelta(days=1)).strftime("%Y_%m_%d")
    clean_label = re.sub(r"[^a-z0-9]+", "_", first_sentence.lower()).strip("_")

    base = cfg.tests_dir if cfg.test_mode else cfg.results_dir
    dest = Path(base) / date_str / clean_label
    dest.mkdir(parents=True, exist_ok=True)

    for item in Path(cfg.processing_dir).iterdir():
        if item.is_file():
            shutil.move(str(item), dest)
            print(f"  Moved {item.name} → {dest}")


# ---------------------------------------------------------------------------
# Cleanup on failure — restore text first, then move remains to Failed/
# ---------------------------------------------------------------------------

def cleanup_failed_job(paths: JobPaths, source_txt: Path, cfg: Config) -> None:
    print("\n  🧹 Cleaning up failed job...")

    # 1. Restore text BEFORE touching any other files
    if not cfg.test_mode:
        restored = False
        if paths.text_path.exists():
            try:
                source_txt.write_text(
                    paths.text_path.read_text(encoding="utf-8"), encoding="utf-8"
                )
                print(f"  Restored text → {source_txt}")
                restored = True
            except Exception as e:
                print(f"  Could not restore text: {e}")
        if not restored:
            print(f"  ⚠️  Could not restore {source_txt.name} — check manually.")
    else:
        print("  (test mode — original file untouched)")

    # 2. Move remaining job files to Failed/<timestamp>/ for inspection
    failed_dir = Path(cfg.failed_dir) / paths.timestamp
    failed_dir.mkdir(parents=True, exist_ok=True)

    for item in Path(cfg.processing_dir).iterdir():
        if item.is_file() and paths.timestamp in item.name:
            try:
                shutil.move(str(item), failed_dir)
                print(f"  → Failed/{paths.timestamp}/{item.name}")
            except Exception as e:
                print(f"  Could not move {item.name}: {e}")


# ---------------------------------------------------------------------------
# Per-file orchestrator — parallel audio+video generation, job-level retry
# ---------------------------------------------------------------------------

def run_pipeline(paths: JobPaths, cfg: Config, healthy_clips: list) -> tuple:
    """
    Generates audio first (needed for duration), then runs background
    concat concurrently with any future parallel steps.
    Returns (first_sentence, intro_length).
    """
    print("\n  [1/3] Generating audio + intro image...")
    first_sentence, intro_length, total_length = generate_audio(paths)

    print("\n  [2/3] Building background video...")
    pick_random_file(total_length, paths, cfg, healthy_clips)

    print("\n  [3/3] Composing and exporting...")
    compose_and_export(paths, intro_length, cfg)

    return first_sentence, intro_length


def process_file(source_txt: Path, cfg: Config, healthy_clips: list) -> bool:
    """
    Runs the full pipeline for one text file.
    Retries cfg.job_retries times on failure before giving up.
    Returns True on success, False on total failure.
    """
    total_attempts = cfg.job_retries + 1

    for attempt in range(1, total_attempts + 1):
        ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
        paths = JobPaths.from_timestamp(ts, cfg)


        # shutil.copy2(source_txt, paths.text_path)
        # print(f"  Copied {source_txt.name} → {paths.text_path.name}")

        # if not cfg.test_mode:
        #     source_txt.write_text("", encoding="utf-8")

        # first_sentence, _ = run_pipeline(paths, cfg, healthy_clips)
        # move_results(paths, first_sentence, cfg)
        # return True
        
        try:
            shutil.copy2(source_txt, paths.text_path)
            print(f"  Copied {source_txt.name} → {paths.text_path.name}")

            if not cfg.test_mode:
                source_txt.write_text("", encoding="utf-8")

            first_sentence, _ = run_pipeline(paths, cfg, healthy_clips)
            move_results(paths, first_sentence, cfg)
            return True

        except Exception as e:
            print(f"\n  ❌ Attempt {attempt}/{total_attempts} failed: {e}")
            cleanup_failed_job(paths, source_txt, cfg)

            if attempt < total_attempts:
                print(f"  🔁 Retrying job in 2s...")
                time.sleep(2)
            else:
                print(f"  ❌ All {total_attempts} attempt(s) exhausted for {source_txt.name}")
                return False


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    cfg = Config(
        test_mode          = False,   # flip to False for production
        max_audio_length   = 180,
        max_audio_cutoff   = 176,
        bg_duration_buffer = 5.0,
        job_retries        = 1,      # retry the full job once before giving up
    )

    Path(cfg.processing_dir).mkdir(exist_ok=True)
    Path(cfg.failed_dir).mkdir(exist_ok=True)

    # Pre-screen all background clips once — skips bad files for every job this run
    healthy_clips = prescreen_clips(cfg)

    texts_dir = Path(cfg.texts_dir)
    txt_files = sorted(
        f for f in texts_dir.iterdir()
        if f.is_file() and f.suffix.lower() == ".txt" and f.stat().st_size > 0
    )

    if not txt_files:
        print("No non-empty .txt files found in Texts/")
        return

    successes, failures = 0, 0

    for txt_file in txt_files:
        print(f"\n\n{'='*50}")
        print(f"Processing: {txt_file.name}")
        print(f"{'='*50}")

        ok = process_file(txt_file, cfg=cfg, healthy_clips=healthy_clips)
        if ok:
            successes += 1
        else:
            failures += 1

    print(f"\n\nAll done!  ✅ {successes} succeeded  ❌ {failures} failed")


if __name__ == "__main__":
    main()