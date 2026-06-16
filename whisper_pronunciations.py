import json
import re
import hashlib
import numpy as np
import soundfile as sf
from pathlib import Path
from TTS.api import TTS
import whisper
import torch
from difflib import SequenceMatcher

# ==========================
# CONFIG
# ==========================
MODEL_NAME = "tts_models/en/vctk/vits"
SPEAKER = "p250"
USE_GPU = False

BASE_SPEED = 0.9
NOISE_SCALE = 0.4
NOISE_SCALE_W = 0.6
LENGTH_SCALE = 1.15

CONFIDENCE_THRESHOLD = 0.85
MAX_VARIANTS = 25

OUTPUT_JSON = Path("AdditionalFiles/whisper_corrections_p250.json")
CACHE_DIR = Path(".pronunciation_cache")
WORDS_JSON = Path("AdditionalFiles/10k_common_words.json")  # Input JSON

OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

# ==========================
# MODELS
# ==========================
tts = TTS(model_name=MODEL_NAME, progress_bar=False, gpu=USE_GPU)
SAMPLE_RATE = tts.synthesizer.output_sample_rate

whisper_model = whisper.load_model("base")
device = "cuda" if torch.cuda.is_available() else "cpu"
whisper_model.to(device)

# ==========================
# TEXT UTILITIES
# ==========================
def normalize(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.lower())

def tokenize(text: str):
    return re.findall(r"[A-Za-z']+", text)

def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()

def hash_key(word: str, variant: str) -> str:
    return hashlib.md5(f"{word}|{variant}".encode()).hexdigest()

# ==========================
# SAFE AUDIO WRITE
# ==========================
def safe_write_wav(path: Path, wav, sample_rate: int):
    if wav is None:
        raise RuntimeError("TTS returned None")

    wav = np.asarray(wav)

    if wav.ndim == 2:
        wav = wav.squeeze()

    if wav.ndim != 1 or len(wav) == 0:
        raise RuntimeError("Invalid audio buffer")

    if not np.isfinite(wav).all():
        raise RuntimeError("NaNs detected in audio")

    sf.write(str(path), wav.astype(np.float32), sample_rate)

# ==========================
# VARIANT GENERATION
# ==========================
def generate_variants(text: str):
    variants = [
        text,
        text.lower(),
        text.upper(),
        re.sub(r"([aeiou])", r"\1\1", text, flags=re.I),
        re.sub(r"([aeiou])", r"\1 ", text, flags=re.I),
        " ".join(text),
        "-".join(text),
    ]

    for v in "aeiou":
        variants.append(text.replace(v, v * 3))

    seen = set()
    final = []
    for v in variants:
        if v not in seen:
            seen.add(v)
            final.append(v)
        if len(final) >= MAX_VARIANTS:
            break

    return final

# ==========================
# AUDIO + WHISPER
# ==========================
def synthesize_cached(key_text: str, variant: str):
    key = CACHE_DIR / f"{hash_key(key_text, variant)}.wav"
    if key.exists():
        return key

    try:
        wav = tts.tts(
            text=variant,
            speaker=SPEAKER,
            speed=BASE_SPEED,
            noise_scale=NOISE_SCALE,
            noise_scale_w=NOISE_SCALE_W,
            length_scale=LENGTH_SCALE
        )
        safe_write_wav(key, wav, SAMPLE_RATE)
        return key
    except Exception as e:
        print(f"⚠️ TTS failed for '{variant}': {e}")
        return None

def whisper_transcribe(wav_path: Path) -> str:
    result = whisper_model.transcribe(
        str(wav_path),
        language="en",
        condition_on_previous_text=False,
        temperature=0.0
    )
    return result.get("text", "").lower()

# ==========================
# WORD LEARNING
# ==========================
def learn_word(word: str):
    target = normalize(word)
    best = (0.0, None)

    for variant in generate_variants(word):
        wav = synthesize_cached(word, variant)
        if not wav:
            continue

        heard = whisper_transcribe(wav)
        heard_norm = normalize(heard)
        score = similarity(heard_norm, target)

        if score > best[0]:
            best = (score, heard_norm)

        if score >= CONFIDENCE_THRESHOLD:
            return heard_norm, word

    if best[0] >= 0.6:
        return best[1], word

    return None, None

# ==========================
# SENTENCE LEARNING
# ==========================
def learn_sentence(sentence: str):
    words = tokenize(sentence)
    learned = {}

    # First: learn all words individually
    for w in words:
        heard, intended = learn_word(w)
        if heard:
            learned[heard] = intended

    # Second: test full sentence context
    wav = synthesize_cached(sentence, sentence)
    if not wav:
        return learned

    transcript = whisper_transcribe(wav)
    transcript_words = tokenize(transcript)

    for intended in words:
        if intended.lower() not in [w.lower() for w in transcript_words]:
            # Context failure → retry word with emphasis
            emphasized = f"{intended}, {intended}"
            wav2 = synthesize_cached(intended, emphasized)
            if not wav2:
                continue

            heard2 = whisper_transcribe(wav2)
            heard2_norm = normalize(heard2)

            if similarity(heard2_norm, normalize(intended)) >= 0.7:
                learned[heard2_norm] = intended

    return learned

# ==========================
# MAIN PIPELINE
# ==========================
def build_corrections(inputs):
    corrections = {}

    for item in inputs:
        print(f"\n🔎 Processing: {item}")

        if len(item.split()) == 1:
            heard, intended = learn_word(item)
            if heard:
                corrections[heard] = intended
                print(f"✅ Learned word: {heard} → {intended}")
            else:
                print("❌ Failed word")

        else:
            learned = learn_sentence(item)
            corrections.update(learned)
            print(f"✅ Learned {len(learned)} words from sentence")

    return corrections

# ==========================
# ENTRY POINT
# ==========================
if __name__ == "__main__":
    # Load the JSON list of words/sentences
    if not WORDS_JSON.exists():
        raise FileNotFoundError(f"Input file not found: {WORDS_JSON}")

    with open(WORDS_JSON, "r", encoding="utf-8") as f:
        INPUTS = json.load(f)

    if not isinstance(INPUTS, list):
        raise ValueError("10k_common_words.json must contain a JSON array of words/sentences.")

    # Load existing corrections if available
    existing = {}
    if OUTPUT_JSON.exists():
        existing = json.loads(OUTPUT_JSON.read_text(encoding="utf-8"))

    learned = build_corrections(INPUTS)
    existing.update(learned)

    OUTPUT_JSON.write_text(
        json.dumps(existing, ensure_ascii=False, indent=4),
        encoding="utf-8"
    )

    print("\n📘 Whisper correction JSON updated successfully.")
