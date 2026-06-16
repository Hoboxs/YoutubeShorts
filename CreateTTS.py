import re
import difflib
import unicodedata
import datetime
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Union, Tuple, Optional
from scipy.signal import butter, sosfiltfilt

# --------------------------
# TTS CONFIG
# --------------------------
MODEL_NAME = "tts_models/en/vctk/vits"
USE_GPU = False
SPEAKER = "p250"

BASE_SPEED = 0.82
LONG_SENTENCE_SPEED = 0.9
NOISE_SCALE = 0.4
NOISE_SCALE_W = 0.6
LENGTH_SCALE = 1.15

# Audio config
AUDIO_PAD_SEC = 0.5          # silence padding added to start and end of audio
AUDIO_FILTER_CUTOFF = 8000   # low-pass filter cutoff in Hz
SENTENCE_PAUSE_SEC = 0.1     # silence between sentences 

# Sentence pause scaling: short sentences get PAUSE_SHORT, long sentences get
# PAUSE_LONG, with a linear ramp between SHORT_WORDS and LONG_WORDS thresholds.
PAUSE_SHORT_SEC = 0.18       # pause after a short sentence (≤ SHORT_WORDS words)
PAUSE_LONG_SEC  = 0.42       # pause after a long sentence  (≥ LONG_WORDS words)
PAUSE_SHORT_WORDS = 8        # word count considered "short"
PAUSE_LONG_WORDS  = 20       # word count considered "long"

# Subtitle config
CHUNK_FLUSH_DUR = 0.6        # seconds of speech before starting a new subtitle chunk
MIN_CHUNK_DUR = 0.01         # minimum duration (sec) to assign a subtitle entry
SUBTITLE_FONT_SIZE = 52
SUBTITLE_MARGIN_LR = 80
SUBTITLE_MARGIN_V = 140
SUBTITLE_POS_X = 960
SUBTITLE_POS_Y = 420

# Maps currency symbols to their spoken suffix (placed after the number)
CURRENCY_SUFFIXES: dict[str, str] = {
    "$": "dollars",
    "£": "pounds",
    "€": "euros",
    "¥": "yen",
}

# Contraction suffixes Whisper may emit as a separate token.
# Only bare forms (no apostrophe) are listed here — apostrophe-prefixed forms
# like "'s" and "'t" are caught by the startswith("'") check.
_BARE_CONTRACTION_SUFFIXES: frozenset[str] = frozenset([
    "s", "t", "re", "ve", "ll", "d", "m",
])

# --------------------------
# Lazy model loading
# --------------------------
_TTS_MODEL = None
_WHISPER_MODEL = None
_SAMPLE_RATE: Optional[int] = None


def get_tts_model():
    """Load and cache the TTS model on first use."""
    global _TTS_MODEL, _SAMPLE_RATE
    if _TTS_MODEL is None:
        from TTS.api import TTS
        _TTS_MODEL = TTS(model_name=MODEL_NAME, progress_bar=True, gpu=USE_GPU)
        _SAMPLE_RATE = _TTS_MODEL.synthesizer.output_sample_rate
    return _TTS_MODEL


def get_sample_rate() -> int:
    """Return the TTS model sample rate, loading the model if needed."""
    if _SAMPLE_RATE is None:
        get_tts_model()
    return _SAMPLE_RATE


def get_whisper_model():
    """Load and cache the Whisper model on first use."""
    global _WHISPER_MODEL
    if _WHISPER_MODEL is None:
        import whisper
        import torch
        _WHISPER_MODEL = whisper.load_model("base")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _WHISPER_MODEL.to(device)
    return _WHISPER_MODEL


# --------------------------
# Text helpers
# --------------------------
def number_to_words(n: int) -> str:
    """Convert a non-negative integer to its English word form.

    Tens and ones are joined with a hyphen (e.g. thirty-four), which is both
    grammatically correct and keeps hyphenated compounds like '34-year-old'
    as a single token after expansion ('thirty-four-year-old').
    """
    if n < 0:
        return "minus " + number_to_words(-n)
    ones = ["", "one", "two", "three", "four", "five", "six", "seven",
            "eight", "nine", "ten", "eleven", "twelve", "thirteen",
            "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
    tens = ["", "", "twenty", "thirty", "forty", "fifty",
            "sixty", "seventy", "eighty", "ninety"]
    if n == 0:
        return "zero"
    if n < 20:
        return ones[n]
    if n < 100:
        return tens[n // 10] + ("-" + ones[n % 10] if n % 10 else "")
    if n < 1000:
        rest = number_to_words(n % 100) if n % 100 else ""
        return ones[n // 100] + " hundred" + (" " + rest if rest else "")
    if n < 1_000_000:
        rest = number_to_words(n % 1000) if n % 1000 else ""
        return number_to_words(n // 1000) + " thousand" + (" " + rest if rest else "")
    rest = number_to_words(n % 1_000_000) if n % 1_000_000 else ""
    return number_to_words(n // 1_000_000) + " million" + (" " + rest if rest else "")


def year_to_words(year: int) -> str:
    """Convert a year integer to the spoken form used by both TTS and Whisper.

    Using the same spoken form on both sides ensures the alignment group lookup
    succeeds. 

    Examples:
        1900 -> 'nineteen hundred'
        1997 -> 'nineteen ninety-seven'
        2000 -> 'two thousand'
        2003 -> 'two thousand three'
        2010 -> 'twenty ten'
        2020 -> 'twenty twenty'
    """
    if year < 1000 or year > 2999:
        return number_to_words(year)
    if 2000 <= year <= 2009:
        remainder = year - 2000
        return "two thousand" + (f" {number_to_words(remainder)}" if remainder else "")
    if 2010 <= year <= 2099:
        return f"twenty {number_to_words(year % 100)}"
    # 1000–1999
    century, remainder = year // 100, year % 100
    century_words = number_to_words(century)
    if remainder == 0:
        return f"{century_words} hundred"
    if remainder < 10:
        return f"{century_words} oh {number_to_words(remainder)}"
    return f"{century_words} {number_to_words(remainder)}"


def ordinal_to_words(n: int) -> tuple[str, str]:
    """Convert an integer to its ordinal spoken form and display suffix.

    Returns (spoken_form, display_suffix) e.g.:
        1  -> ("first",        "st")
        2  -> ("second",       "nd")
        3  -> ("third",        "rd")
        12 -> ("twelfth",      "th")
        21 -> ("twenty-first", "st")
    """
    ones_ordinals = [
        "", "first", "second", "third", "fourth", "fifth", "sixth",
        "seventh", "eighth", "ninth", "tenth", "eleventh", "twelfth",
        "thirteenth", "fourteenth", "fifteenth", "sixteenth", "seventeenth",
        "eighteenth", "nineteenth",
    ]
    tens = ["", "", "twenty", "thirty", "forty", "fifty",
            "sixty", "seventy", "eighty", "ninety"]
    tens_ordinals = ["", "", "twentieth", "thirtieth", "fortieth", "fiftieth",
                     "sixtieth", "seventieth", "eightieth", "ninetieth"]

    suffix_map = {1: "st", 2: "nd", 3: "rd"}
    last_two = n % 100
    last_one = n % 10
    if 11 <= last_two <= 13:
        suffix = "th"
    else:
        suffix = suffix_map.get(last_one, "th")

    if n <= 0 or n > 31:
        return number_to_words(n), suffix
    if n < 20:
        return ones_ordinals[n], suffix
    if n % 10 == 0:
        return tens_ordinals[n // 10], suffix
    return f"{tens[n // 10]}-{ones_ordinals[n % 10]}", suffix


def _expand_times(text: str) -> tuple[str, list]:
    """Replace time expressions with their spoken word form.

    Handles 12-hour and 24-hour times, with optional a.m./p.m. or am/pm suffix.
    The original time string is preserved as the display form.

    e.g. '7:45'     -> ('seven forty-five', [(["seven", "forty-five"], "7:45")])
         '7:45 a.m.' -> ('seven forty-five a.m.', [..., "7:45 a.m."])
         '12:00'    -> ('twelve o\' clock', [...])
    """
    replacements: list = []
    # Match H:MM or HH:MM with optional am/pm or a.m./p.m.
    pattern = r"\b(\d{1,2}):(\d{2})\b"

    def repl(m: re.Match) -> str:
        hours, mins = int(m.group(1)), int(m.group(2))
        display = m.group(0)
        hour_word = number_to_words(hours)
        if mins == 0:
            min_word = "o\' clock"
        elif mins < 10:
            min_word = f"oh {number_to_words(mins)}"
        else:
            min_word = number_to_words(mins)
        spoken = f"{hour_word} {min_word}"
        words = spoken.split()
        replacements.append((words, display))
        return spoken

    text = re.sub(pattern, repl, text)
    return text, replacements


def _expand_ampm(text: str) -> tuple[str, list]:
    """Replace a.m./p.m. (with or without dots) with spoken form, keeping dots in display.

    TTS speaks 'am' and 'pm' without dots. The display form preserves dots.
    Uses a non-word-boundary pattern since trailing dots break \b matching.

    e.g. 'a.m.' -> spoken 'am', display 'a.m.'
         'p.m.' -> spoken 'pm', display 'p.m.'
         'AM'   -> spoken 'am', display 'AM'
    """
    replacements: list = []
    # Two alternates: dotted form (a.m./p.m.) and plain form (am/pm).
    # Dotted form uses lookaround instead of \b since dots break word boundaries.
    pattern = r"(?<!\w)(a\.m\.|p\.m\.|A\.M\.|P\.M\.)(?!\w)|\b(am|pm|AM|PM)\b"

    def repl(m: re.Match) -> str:
        original = m.group(1) or m.group(2)
        spoken = original.replace(".", "").lower()
        replacements.append(([spoken], original))
        return spoken

    text = re.sub(pattern, repl, text)
    return text, replacements


def _expand_currency(text: str) -> tuple[str, list]:
    """Replace currency expressions with their spoken word form.

    Includes cents when present. Returns (word_list, display_string) pairs.

    e.g. '$19'     -> ('nineteen dollars', [(["nineteen", "dollars"], "$19")])
         '$19.99'  -> ('nineteen dollars and ninety-nine cents',
                       [(["nineteen", "dollars", "and", "ninety-nine", "cents"], "$19.99")])
         '£1,200'  -> ('one thousand two hundred pounds', [...])
    """
    replacements: list = []
    pattern = r"([$£€¥])(\d[\d,]*(?:\.\d{1,2})?)"

    def repl(m: re.Match) -> str:
        symbol, amount = m.group(1), m.group(2)
        suffix = CURRENCY_SUFFIXES.get(symbol, "")
        parts = amount.replace(",", "").split(".")
        whole = int(parts[0])
        cents = int(parts[1].ljust(2, "0")) if len(parts) > 1 and parts[1] else 0

        whole_str = f"{whole:,}"
        words = number_to_words(whole).split() + ([suffix] if suffix else [])
        if cents:
            cent_word = number_to_words(cents) or ""
            words += ["and"] + cent_word.split() + ["cents"]
            display = f"{symbol}{whole_str}.{parts[1].ljust(2, '0')}"
        else:
            display = f"{symbol}{whole_str}"

        replacements.append((words, display))
        return " ".join(words)

    text = re.sub(pattern, repl, text)
    return text, replacements


# Precompiled year-context pattern used by _expand_plain_numbers.
# Matches years preceded by a context word or month name (with optional day).
# Variable-width lookbehind is not supported in Python's re, so the context
# word is captured as group 1 and the year as group 2 — only group 2 is used.
_YEAR_CONTEXT_PATTERN = re.compile(
    r"\b(?:"
    r"(?:in|since|by|until|before|after|from|of|circa|around|about|year)[\s,]+"
    r"|(?:january|february|march|april|may|june|july|august|september|october|"
    r"november|december|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)"
    # Optional day: digit (e.g. "14"), ordinal suffix (e.g. "14th"),
    # or ordinal word (e.g. "fourteenth", "twenty-first") — all valid separators
    r"(?:[\s,]+(?:\d{1,2}(?:st|nd|rd|th)?|[a-z]+(?:-[a-z]+)*))?[\s,]+"
    r")(\d{4})\b",
    re.IGNORECASE,
)

# Sentinel character used to tag comma-formatted numbers through expansion.
# U+E000 (Unicode private-use area) — safe since it never appears in natural
# language text and cannot be produced by NFKC normalisation.
_SENTINEL = "\uE000"


def _expand_plain_numbers(text: str) -> tuple[str, list]:
    """Replace standalone integers with their English word form.

    Year detection: a bare 4-digit number (1000–2999) is treated as a year only
    when preceded by a year-context word or month name. Comma-formatted numbers
    (e.g. 1,204) are always expanded regardless.

    Hyphenated compounds containing digits (e.g. 34-year-old, mid-36) are
    expanded first so the whole compound becomes a single token.

    Returns:
        (expanded_text, replacements) where replacements is a list of
        ([word, ...], display_string) pairs for word_groups.
    """
    replacements: list = []

    # --- Step 1: hyphenated compounds containing digits ---
    # Must run before standalone expansion so "34-year-old" is handled as a
    # whole token rather than "34" being consumed first.
    def repl_compound(m: re.Match) -> str:
        full = m.group(0)
        if not re.search(r"\d", full):
            return full
        expanded = re.sub(r"(\d+)", lambda dm: number_to_words(int(dm.group(1))), full)
        if expanded != full:
            replacements.append(([expanded], full))
        return expanded

    text = re.sub(
        r"\b[\w\u00C0-\u024F\d]+(?:-[\w\u00C0-\u024F\d]+)+\b",
        repl_compound,
        text,
    )

    # --- Step 2: comma-formatted standalone numbers (e.g. 9,876) ---
    # Tag them with a sentinel so the year guard ignores them.
    text = re.sub(
        r"(\d{1,3}(?:,\d{3})+)",
        lambda m: f"{_SENTINEL}{m.group().replace(',', '')}{_SENTINEL}",
        text,
    )

    # --- Step 3: calculate year spans (after sentinel tagging) ---
    year_spans: set[tuple[int, int]] = set()
    for m in _YEAR_CONTEXT_PATTERN.finditer(text):
        n = int(m.group(1))
        if 1000 <= n <= 2999:
            year_spans.add((m.start(1), m.end(1)))

    # --- Step 4: expand remaining standalone integers ---
    def repl(m: re.Match) -> str:
        raw = m.group()
        was_comma_formatted = raw.startswith(_SENTINEL) and raw.endswith(_SENTINEL)
        n = int(raw.strip(_SENTINEL))
        if not was_comma_formatted and (m.start(), m.end()) in year_spans:
            return str(n)  # year in context — leave for TTS to speak naturally
        if not was_comma_formatted and 1000 <= n <= 2999:
            words = year_to_words(n).split()
        else:
            words = number_to_words(n).split()
        display = f"{n:,}"
        replacements.append((words, display))
        return " ".join(words)

    sentinel_or_number = rf"{re.escape(_SENTINEL)}\d+{re.escape(_SENTINEL)}|\b\d+\b"
    text = re.sub(sentinel_or_number, repl, text)
    return text, replacements


def tts_clarity_fixes(text: str) -> str:
    """Add a space after sentence-ending punctuation to improve TTS pacing."""
    return re.sub(r'([.,!?])', r'\1 ', text)


def preprocess_text(text: str) -> tuple[str, list]:
    """Normalise and expand text for TTS synthesis and Whisper alignment.

    Returns:
        tts_text:   Fully expanded text (numbers as words) for TTS and alignment.
        word_groups: List of ([word, ...], display_string) pairs. Each entry maps
                     the expanded word form(s) back to the original digit display
                     (e.g. ['nineteen', 'dollars'] -> '$19'). Used by the aligner
                     to show digits in the ASS file instead of spoken words.

    Processing order:
        1. NFKC Unicode normalisation
        2. Currency expansion  ($19 -> 'nineteen dollars', display '$19')
        3. Date expansion      (12/03/2024 -> '12 March 2024')
        4. Plain number expansion (hyphenated compounds first, then standalone)
        5. TTS clarity fixes   (space after punctuation)
    """
    if not text or not text.strip():
        return text, []

    text = unicodedata.normalize("NFKC", text)
    all_replacements: list = []

    # Step 2: times (before currency so "7:45" is not confused with decimals)
    text, time_reps = _expand_times(text)
    all_replacements.extend(time_reps)

    # Step 2b: a.m./p.m. (before currency/numbers so dots aren't consumed)
    text, ampm_reps = _expand_ampm(text)
    all_replacements.extend(ampm_reps)

    # Step 3: currency (now handles cents)
    text, currency_reps = _expand_currency(text)
    all_replacements.extend(currency_reps)

    # Step 4: dates (dd/mm/yyyy or mm-dd-yyyy)
    # Day numbers are converted to ordinal spoken form (e.g. 12 -> "twelfth")
    # and logged as word_groups so they display with ordinal suffix (e.g. "12th").
    date_pattern = r"(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})"
    def date_repl(m: re.Match) -> str:
        d, mth, y = m.groups()
        try:
            month_name = datetime.date(1900, int(mth), 1).strftime("%B")
        except ValueError:
            return m.group()
        day_n = int(d)
        spoken, suffix = ordinal_to_words(day_n)
        display = f"{day_n}{suffix}"
        all_replacements.append(([spoken], display))
        return f"{spoken} {month_name} {y}"
    text = re.sub(date_pattern, date_repl, text)

    # Also handle bare "Month DD" or "Month DDth" patterns already in text
    # e.g. "March 14," -> "March fourteenth,"
    # Match month name followed by a day number with optional ordinal suffix
    bare_date_pattern = (
        r"\b(january|february|march|april|may|june|july|august|september|"
        r"october|november|december)\s+(\d{1,2})(st|nd|rd|th)?\b"
    )
    def bare_date_repl(m: re.Match) -> str:
        month, day_str, existing_suffix = m.group(1), m.group(2), m.group(3)
        day_n = int(day_str)
        spoken, suffix = ordinal_to_words(day_n)
        display = f"{day_n}{existing_suffix or suffix}"
        all_replacements.append(([spoken], display))
        return f"{month} {spoken}"
    text = re.sub(bare_date_pattern, bare_date_repl, text, flags=re.IGNORECASE)

    # Step 5: plain numbers
    text, number_reps = _expand_plain_numbers(text)
    all_replacements.extend(number_reps)

    # Step 6: clarity fixes applied here so word_groups indices stay consistent
    # with what the aligner sees when it tokenizes the same tts_text.
    text = tts_clarity_fixes(text)

    return text, all_replacements


# --------------------------
# Audio helpers
# --------------------------
def format_ass_time(sec: float) -> str:
    """Format a timestamp in seconds as an ASS subtitle time string (H:MM:SS.cc)."""
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    cs = int((sec - int(sec)) * 100)
    return f"{h}:{m:02}:{s:02}.{cs:02}"


def apply_fade(wav: np.ndarray, fade_ms: int = 10) -> np.ndarray:
    """Apply a short linear fade-in and fade-out to avoid audio clicks."""
    sample_rate = get_sample_rate()
    n = int(sample_rate * fade_ms / 1000)
    if len(wav) < n * 2:
        return wav
    wav[:n] *= np.linspace(0, 1, n)
    wav[-n:] *= np.linspace(1, 0, n)
    return wav


def smooth_audio(wav: np.ndarray, cutoff: int = AUDIO_FILTER_CUTOFF) -> np.ndarray:
    """Apply a zero-phase low-pass Butterworth filter to reduce TTS harshness.

    Uses sosfiltfilt (second-order sections, forward-backward) instead of
    lfilter to avoid phase delay, producing cleaner audio without timing shift.
    """
    sample_rate = get_sample_rate()
    sos = butter(2, cutoff / (sample_rate / 2), btype="low", output="sos")
    return sosfiltfilt(sos, wav).astype(np.float32)


# --------------------------
# Transcription
# --------------------------
def transcribe_whisper(audio_path: Path) -> list[tuple[str, float, float]]:
    """Transcribe audio using Whisper and return word-level timestamps.

    Results are cached to a JSON sidecar file (<audio_path>.whisper.json)
    keyed on the audio file's modification time. Re-transcription is skipped
    if the audio file has not changed since the last run, saving significant
    time during subtitle iteration.

    Apostrophes in Whisper output are normalised to straight ' at source
    so all downstream code sees them consistently.

    Returns:
        List of (word, start_sec, end_sec) tuples.
    """
    import json

    cache_path = audio_path.with_suffix(".whisper.json")
    audio_mtime = audio_path.stat().st_mtime

    # Return cached result if audio file has not changed
    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if cached.get("mtime") == audio_mtime:
                return [tuple(w) for w in cached["words"]]
        except (json.JSONDecodeError, KeyError):
            pass  # cache corrupt — fall through to re-transcribe

    model = get_whisper_model()
    try:
        result = model.transcribe(str(audio_path), word_timestamps=True)
    except Exception as exc:
        print(f"[Whisper] Warning: transcription failed for {audio_path}: {exc}")
        return []
    out = []
    for seg in result["segments"]:
        for w in seg["words"]:
            word = w["word"].strip()
            if word:
                word = re.sub(r"[\u2018\u2019\u02bc\u02b9\u0060\u00b4]", "'", word)
                out.append((word, w["start"], w["end"]))

    # Write cache
    try:
        cache_path.write_text(
            json.dumps({"mtime": audio_mtime, "words": out}),
            encoding="utf-8",
        )
    except OSError:
        pass  # cache write failure is non-fatal

    return out


# --------------------------
# Alignment
# --------------------------
def tokenize_tts(text: str) -> list[str]:
    """Tokenize text into words, preserving contractions, hyphenated words,
    accented characters, and surrounding quotation marks.

    All apostrophe variants (curly, unicode, backtick, etc.) are normalised to
    a straight apostrophe before tokenizing so contractions are always one token.

    Examples:
        "didn't"          -> ["didn't"]
        "mother-in-law"   -> ["mother-in-law"]
        '"Hello,"'        -> ['"Hello,"']
        "café"            -> ["café"]
        "34-year-old"     -> ["34-year-old"]
    """
    # Use explicit unicode escapes — literal curly quote chars in a character
    # class are misinterpreted by Python as a matching pair, not two codepoints.
    text = re.sub(r"[\u2018\u2019\u02bc\u02b9\u0060\u00b4\u0027]", "'", text)
    return re.findall(
        r'(?:[\u201c\u201d"])*[\w\u00C0-\u024F]+'
        r'(?:[-\'"][\w\u00C0-\u024F]+)*'
        r'(?:[.,!?](?:[\u201c\u201d"]*)|(?:[\u201c\u201d"])+)?',
        text,
    )


def normalize_token(token: str) -> str:
    """Lowercase and strip all non-word characters for fuzzy alignment.

    Apostrophes and hyphens are stripped so "didn't" and "didnt", or
    "thirty-four" and "thirtyfour", compare as equal.

    Note: this means two different surface forms can collide (e.g.
    "thirty-four-year-old" and "thirtyfouryearold" both normalize to the same
    string). This is intentional for matching but means group_lookup keys must
    be unique normalized forms.
    """
    return re.sub(r"[^\w]", "", token.lower())


def merge_whisper_contractions(whisper_words: list) -> list:
    """Rejoin Whisper tokens that are split contractions into a single token.

    Handles both apostrophe-prefixed suffixes ("'s", "'t") and bare suffixes
    ("s", "t") that Whisper emits when the apostrophe is dropped. Only merges
    when the preceding token is purely alphabetic (not a number or punctuation)
    to reduce false positives on common standalone words like "s" or "t".

    Note: bare suffixes like "s", "t", "d" are also common standalone words.
    The alphabetic-only guard reduces but does not eliminate false merges
    (e.g. "he d something" may still merge to "he'd something").

    Examples:
        [("sister", ...), ("s", ...)]  -> [("sister's", ...)]
        [("didn", ...), ("'t", ...)]   -> [("didn't", ...)]
        [("the", ...), ("house", ...)] -> [("the", ...), ("house", ...)]
    """
    whisper_words = [
        (re.sub(r"[\u2018\u2019\u02bc\u02b9\u0060\u00b4]", "'", w), s, e)
        for w, s, e in whisper_words
    ]
    merged = []
    i = 0
    while i < len(whisper_words):
        word, start, end = whisper_words[i]
        if i + 1 < len(whisper_words):
            next_word, _, next_end = whisper_words[i + 1]
            next_lower = next_word.lower()
            prev_is_alpha = word.replace("-", "").isalpha()
            is_contraction_suffix = (
                next_word.startswith("'") or next_lower in _BARE_CONTRACTION_SUFFIXES
            )
            if prev_is_alpha and is_contraction_suffix:
                suffix = next_word if next_word.startswith("'") else f"'{next_word}"
                merged.append((word + suffix, start, next_end))
                i += 2
                continue
        merged.append((word, start, end))
        i += 1
    return merged


def align_whisper_to_tts(
    tts_text: str,
    whisper_words: list,
    word_groups: list,
    sentences: list[str],
) -> tuple[list[tuple[str, float, float]], list[int]]:
    """Align TTS tokens to Whisper word timestamps using fuzzy sequence matching.

    Args:
        tts_text:      Expanded TTS text (e.g. 'nineteen dollars twenty four').
        whisper_words: List of (word, start_sec, end_sec) from Whisper.
        word_groups:   List of ([word, ...], display_string) from preprocess_text.
        sentences:     List of TTS sentences — used to compute per-sentence
                       aligned word counts so sentence partitioning in
                       build_subtitle_chunks remains accurate after token merging.

    Returns:
        (aligned, aligned_sentence_counts) where:
        - aligned is a list of (display_word, start_sec, end_sec) tuples.
        - aligned_sentence_counts[i] is the number of aligned entries for sentence i.

    Notes:
        - When number groups are merged (multiple TTS tokens → 1 display token),
          the aligned list is shorter than the raw TTS token count. Returning
          per-sentence counts ensures build_subtitle_chunks slices correctly.
        - group_lookup is sorted longest-first so longer groups are matched first.
    """
    whisper_words = merge_whisper_contractions(whisper_words)
    tts_tokens = tokenize_tts(tts_text)
    tts_norm = [normalize_token(t) for t in tts_tokens]
    whisper_norm = [normalize_token(w) for w, _, _ in whisper_words]

    # Build group lookup sorted longest-first so longer groups match before
    # shorter subsets (e.g. "nineteen dollars and ninety-nine cents" before "nineteen").
    group_lookup: list[tuple[tuple, str]] = sorted(
        [(tuple(normalize_token(w) for w in words), display)
         for words, display in word_groups],
        key=lambda x: len(x[0]),
        reverse=True,
    )

    # Build a mapping from TTS token index -> sentence index so we can tag
    # each raw_aligned entry with its sentence during the opcode loop.
    # This is more reliable than cumulative token counts because "insert" opcodes
    # skip TTS tokens without adding to raw_aligned, making counts diverge.
    sentence_token_counts = [len(tokenize_tts(s)) for s in sentences]
    tts_token_to_sentence: list[int] = []
    for sent_i, tc in enumerate(sentence_token_counts):
        tts_token_to_sentence.extend([sent_i] * tc)

    matcher = difflib.SequenceMatcher(None, whisper_norm, tts_norm)
    # raw_aligned: (token, start, end, sentence_index)
    raw_aligned: list[tuple[str, float, float, int]] = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                _, start, end = whisper_words[i1 + k]
                tts_idx = j1 + k
                sent_i = tts_token_to_sentence[tts_idx] if tts_idx < len(tts_token_to_sentence) else 0
                raw_aligned.append((tts_tokens[tts_idx], start, end, sent_i))

        elif tag == "replace":
            t_count = j2 - j1
            _, span_start, _ = whisper_words[i1]
            _, _, span_end = whisper_words[i2 - 1]
            duration = (span_end - span_start) / t_count
            for k in range(t_count):
                tts_idx = j1 + k
                sent_i = tts_token_to_sentence[tts_idx] if tts_idx < len(tts_token_to_sentence) else 0
                raw_aligned.append((
                    tts_tokens[tts_idx],
                    span_start + k * duration,
                    span_start + (k + 1) * duration,
                    sent_i,
                ))

        # "delete": Whisper token has no TTS counterpart — skip
        # "insert": TTS token has no Whisper counterpart — skip

    # Post-pass 1: scan for group matches across the full raw_aligned list.
    raw_norm = [normalize_token(tok) for tok, _, _, _ in raw_aligned]
    group_disp: dict[int, str] = {}
    i = 0
    while i < len(raw_norm):
        matched = False
        for norm_tuple, display in group_lookup:
            glen = len(norm_tuple)
            if tuple(raw_norm[i:i + glen]) == norm_tuple:
                group_disp[i] = display
                for k in range(1, glen):
                    group_disp[i + k] = ""
                i += glen
                matched = True
                break
        if not matched:
            i += 1

    # Post-pass 2: apply overrides, merge suppressed tokens, and count aligned
    # entries per sentence using the sentence_index tag on each raw entry.
    aligned: list[tuple[str, float, float]] = []
    aligned_sentence_counts: list[int] = [0] * len(sentences)
    for raw_idx, (tok, start, end, sent_i) in enumerate(raw_aligned):
        disp = group_disp.get(raw_idx, tok)
        if disp == "":
            if aligned:
                prev_disp, prev_start, _ = aligned[-1]
                aligned[-1] = (prev_disp, prev_start, end)
            # suppressed — don't increment count, duration absorbed into prev
        else:
            aligned.append((disp, start, end))
            if sent_i < len(aligned_sentence_counts):
                aligned_sentence_counts[sent_i] += 1

    return aligned, aligned_sentence_counts


# --------------------------
# Subtitle helpers
# --------------------------
def build_karaoke_from_aligned(aligned_words: list) -> str:
    # ORIGINAL FUNCTION PRESERVED AS REQUESTED
    parts = []
    for word, s, e in aligned_words:
        dur_cs = max(1, int((e - s) * 100))
        parts.append(f"{{\\k{dur_cs}}}{word.upper()}")
    return " ".join(parts)


def chunk_words_by_duration(
    words: list,
    flush_dur: float = CHUNK_FLUSH_DUR,
) -> list:
    """Group words into subtitle chunks.

    A new chunk starts whenever the accumulated *spoken* duration reaches
    flush_dur seconds. Spoken duration is measured as the sum of individual
    word durations (end - start) rather than the wall-clock span from the
    first word's start to the current word's start. This prevents merged
    number entries (which have a large end time) from causing premature flushes
    on subsequent words.

    The word that trips the threshold starts the NEXT chunk so no word appears
    at both the end of one chunk and the start of the next.

    Args:
        words:     List of (word, start_sec, end_sec) tuples.
        flush_dur: Accumulated spoken duration threshold that triggers a new chunk.
                   The final chunk is always emitted regardless of its length.
    """
    chunks = []
    buf: list = []
    buf_spoken_dur = 0.0
    for word in words:
        if buf and buf_spoken_dur >= flush_dur:
            chunks.append(buf)
            buf = []
            buf_spoken_dur = 0.0
        buf.append(word)
        buf_spoken_dur += word[2] - word[1]
    if buf:
        chunks.append(buf)
    return chunks


def write_ass_file(chunks_by_sentence: list, ass_path: Path) -> None:
    """Write subtitle chunks to an ASS file.

    Each chunk becomes one Dialogue entry. End times are clamped to the next
    entry's start time only when genuine overlap exists, preserving the full
    duration of merged number entries.

    Args:
        chunks_by_sentence: Nested list — outer list is sentences, inner list
                            is (start_sec, end_sec, ass_text) tuples.
        ass_path:           Output path for the .ass file.
    """
    sample_rate = get_sample_rate()  # ensures model is loaded for SAMPLE_RATE usage
    ass = [
        "[Script Info]",
        "ScriptType: v4.00+",
        "PlayResX: 1920",
        "PlayResY: 900",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding",
        f"Style: Default,Montserrat,{SUBTITLE_FONT_SIZE},"
        f"&H00FFFFFF,&H00808080,&H00000000,&H64000000,"
        f"-1,0,0,0,100,100,0,0,1,3,4,5,"
        f"{SUBTITLE_MARGIN_LR},{SUBTITLE_MARGIN_LR},{SUBTITLE_MARGIN_V},1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]

    pos_tag = "{" + f"\\an8\\pos({SUBTITLE_POS_X},{SUBTITLE_POS_Y})" + "}"

    for sent_idx, sentence_chunks in enumerate(chunks_by_sentence):
        is_last_sentence = sent_idx == len(chunks_by_sentence) - 1
        next_sentence_start = None
        if not is_last_sentence:
            next_sent = chunks_by_sentence[sent_idx + 1]
            if next_sent and next_sent[0]:
                next_sentence_start = next_sent[0][0]
        for chunk_idx, (s, e, txt) in enumerate(sentence_chunks):
            is_last_in_sentence = chunk_idx == len(sentence_chunks) - 1
            if not is_last_in_sentence:
                # Within a sentence: extend to next chunk's start for continuity.
                e = sentence_chunks[chunk_idx + 1][0]
            else:
                # Last chunk of sentence: always extend to the next sentence's
                # first word start. This eliminates the on-screen gap entirely
                # while keeping sentences visually separated by token partitioning.
                if next_sentence_start is not None:
                    e = next_sentence_start
            ass.append(
                f"Dialogue: 0,{format_ass_time(s)},{format_ass_time(e)},"
                f"Default,,0,0,0,,{pos_tag}{txt}"
            )

    ass_path.write_text("\n".join(ass), encoding="utf-8")


# --------------------------
# Pipeline steps
# --------------------------
def sentence_pause_for_length(word_count: int) -> float:
    """Return a pause duration scaled to sentence length.

    Short sentences (≤ PAUSE_SHORT_WORDS words) get PAUSE_SHORT_SEC.
    Long sentences (≥ PAUSE_LONG_WORDS words) get PAUSE_LONG_SEC.
    Sentences in between get a linearly interpolated value.
    """
    if word_count <= PAUSE_SHORT_WORDS:
        return PAUSE_SHORT_SEC
    if word_count >= PAUSE_LONG_WORDS:
        return PAUSE_LONG_SEC
    t = (word_count - PAUSE_SHORT_WORDS) / (PAUSE_LONG_WORDS - PAUSE_SHORT_WORDS)
    return PAUSE_SHORT_SEC + t * (PAUSE_LONG_SEC - PAUSE_SHORT_SEC)


def trim_trailing_silence(
    wav: np.ndarray,
    sample_rate: int,
    threshold: float = 0.01,
    min_keep_sec: float = 0.05,
) -> np.ndarray:
    """Remove trailing silence from a TTS waveform.

    The TTS model typically appends 100-300ms of silence after each sentence.
    Trimming it allows sentences to be concatenated without audible gaps.

    Args:
        wav:          Audio waveform as float32 array.
        threshold:    Amplitude below which a sample is considered silent.
        min_keep_sec: Minimum audio to keep even if it ends in silence,
                      to avoid cutting off natural sentence endings.

    Returns:
        Trimmed waveform.
    """
    min_keep = int(sample_rate * min_keep_sec)
    # Find last sample above threshold
    above = np.where(np.abs(wav) > threshold)[0]
    if len(above) == 0:
        return wav
    last = above[-1]
    # Keep at least min_keep samples
    cut = max(last + 1, min_keep)
    return wav[:cut]


def synthesise_audio(
    sentences: list[str],
    audio_path: Path,
    pause_sec: float = SENTENCE_PAUSE_SEC,
    original_sentences: list[str] = None,
) -> tuple[list[float], list[float], float]:
    """Synthesise speech for each sentence and write the combined audio file.

    Adds leading/trailing silence pads and inter-sentence pauses. Returns
    sentence boundary times already shifted by the pad offset so they align
    with Whisper timestamps (which are relative to the padded file).

    Args:
        sentences:          List of expanded TTS sentence strings.
        audio_path:         Output path for the .wav file.
        pause_sec:          Duration of silence inserted after each sentence.
        original_sentences: Optional list of pre-expansion sentences used to
                            determine speaking speed. If not provided, the
                            expanded sentences are used, which can cause number-
                            heavy sentences to be spoken faster than intended.

    Returns:
        (sentence_start_times, sentence_end_times, total_duration_sec)
    """
    tts = get_tts_model()
    sample_rate = get_sample_rate()
    audio_parts: list[np.ndarray] = []
    sentence_start_times: list[float] = []
    sentence_end_times: list[float] = []
    elapsed = 0.0

    for sent_idx, sent in enumerate(sentences):
        # Use original (pre-expansion) sentence for word count so number
        # expansion doesn't push a short sentence into the faster speed tier.
        try:
            orig_sent = original_sentences[sent_idx] if original_sentences else sent
        except IndexError:
            orig_sent = sent
        speed = BASE_SPEED if len(orig_sent.split()) <= 12 else LONG_SENTENCE_SPEED
        try:
            wav = tts.tts(
                text=sent,
                speaker=SPEAKER,
                speed=speed,
                noise_scale=NOISE_SCALE,
                noise_scale_w=NOISE_SCALE_W,
                length_scale=LENGTH_SCALE,
            )
        except (RuntimeError, ValueError) as exc:
            # TTS failed for this sentence (e.g. unusual characters, model error).
            # Insert silence to keep timing consistent rather than crashing.
            print(f"[TTS] Warning: synthesis failed for sentence {sent!r}: {exc}")
            wav = None

        if not wav:
            # TTS returned empty or None — insert silence to preserve timing.
            wav = [0.0] * int(sample_rate * 0.5)

        wav = apply_fade(np.asarray(wav, dtype=np.float32))
        # Trim trailing silence the TTS model produces at the end of each sentence.
        wav = trim_trailing_silence(wav, sample_rate)
        sentence_start_times.append(elapsed)
        audio_parts.append(wav)
        elapsed += len(wav) / sample_rate

        # Scale pause duration by sentence length: short sentences get a brief
        # pause, long sentences get a longer breath before the next begins.
        try:
            orig_sent = original_sentences[sent_idx] if original_sentences else sent
        except (ValueError, IndexError):
            orig_sent = sent
        word_count = len(orig_sent.split())
        scaled_pause = sentence_pause_for_length(word_count)
        silence = np.zeros(int(sample_rate * scaled_pause), dtype=np.float32)
        audio_parts.append(silence)
        elapsed += scaled_pause
        sentence_end_times.append(elapsed)

    # smooth_audio runs before padding — the filter should only process speech,
    # not the leading/trailing silence pads.
    audio = smooth_audio(np.concatenate(audio_parts))

    pad = int(AUDIO_PAD_SEC * sample_rate)
    pad_offset = pad / sample_rate
    audio = np.concatenate([
        np.zeros(pad, dtype=np.float32),
        audio,
        np.zeros(pad, dtype=np.float32),
    ])

    sentence_start_times = [t + pad_offset for t in sentence_start_times]
    sentence_end_times = [t + pad_offset for t in sentence_end_times]

    audio_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(audio_path, audio, sample_rate)

    return sentence_start_times, sentence_end_times, len(audio) / sample_rate


def build_subtitle_chunks(
    words: list[tuple[str, float, float]],
    sentences: list[str],
    aligned_sentence_counts: list[int],
) -> list[list[tuple[float, float, str]]]:
    """Partition aligned words into per-sentence subtitle chunks.

    Sentence 0 is intentionally skipped (it is the lead-in sentence used by
    the caller). Words are partitioned using aligned_sentence_counts — the exact
    number of aligned entries per sentence as returned by align_whisper_to_tts.
    This correctly accounts for token merging (where multiple TTS tokens become
    one display token, making the aligned list shorter than the TTS token list).

    Args:
        words:                   Aligned (display_word, start_sec, end_sec) tuples.
        sentences:               List of TTS sentences (sentence 0 is skipped).
        aligned_sentence_counts: Per-sentence aligned word counts from aligner.

    Returns:
        Nested list: outer list is sentences, inner list is
        (start_sec, end_sec, ass_karaoke_text) tuples.
    """
    chunks_by_sentence: list = []

    next_word_idx = min(aligned_sentence_counts[0] if aligned_sentence_counts else 0, len(words))

    for i in range(1, len(sentences)):
        count = aligned_sentence_counts[i] if i < len(aligned_sentence_counts) else 0
        end_idx = min(next_word_idx + count, len(words))
        sentence_words = words[next_word_idx:end_idx]

        if not sentence_words:
            continue

        next_word_idx = end_idx
        word_chunks = chunk_words_by_duration(sentence_words)

        sentence_chunks: list = []
        for ch in word_chunks:
            s = ch[0][1]
            e = max(ch[-1][2], s + MIN_CHUNK_DUR)
            txt = build_karaoke_from_aligned(ch)
            sentence_chunks.append((s, e, txt))

        chunks_by_sentence.append(sentence_chunks)

    return chunks_by_sentence


# --------------------------
# Main entry point
# --------------------------
def generate_tts(
    audio_path: Union[str, Path],
    ass_path: Union[str, Path],
    text: str,
    pause_sec: float = SENTENCE_PAUSE_SEC,
) -> Tuple[float, float]:
    """Synthesise speech from text and write a word-timed ASS subtitle file.

    Orchestrates four independent pipeline steps:
        1. preprocess_text   — normalise and expand numbers/currency
        2. synthesise_audio  — TTS synthesis + write .wav
        3. transcribe_whisper + align_whisper_to_tts — word timestamps
        4. build_subtitle_chunks + write_ass_file — ASS output

    The first sentence of text is always skipped in the subtitle output
    (by design — it is used as a lead-in by the caller).

    Args:
        audio_path: Output path for the synthesised .wav file.
        ass_path:   Output path for the .ass subtitle file.
        text:       Input text. May contain numbers, dates, currency,
                    contractions, and accented characters.
        pause_sec:  Duration of silence inserted between sentences (seconds).

    Returns:
        (first_chunk_start_sec, total_audio_duration_sec).
        first_chunk_start_sec is the Whisper timestamp of the first subtitle
        entry — use this as the handoff point for any preceding content.
    """
    if not text or not text.strip():
        return 0.0, 0.0

    audio_path = Path(audio_path)
    ass_path = Path(ass_path)

    import traceback

    # Step 1: preprocess
    tts_text, word_groups = preprocess_text(text)
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', tts_text) if s.strip()]
    if not sentences:
        return 0.0, 0.0

    # Step 2: synthesise
    original_sentences = [
        s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()
    ]
    sentence_start_times, sentence_end_times, total_dur = synthesise_audio(
        sentences, audio_path, pause_sec, original_sentences
    )

    # Step 3: transcribe and align
    words = transcribe_whisper(audio_path)
    if not words:
        write_ass_file([], ass_path)
        return 0.0, total_dur

    try:
        words, aligned_sentence_counts = align_whisper_to_tts(tts_text, words, word_groups, sentences)
    except Exception:
        print("[generate_tts] ERROR in align_whisper_to_tts:")
        traceback.print_exc()
        print(f"  tts_text length: {len(tts_text)}")
        print(f"  words count: {len(words)}")
        print(f"  sentences count: {len(sentences)}")
        raise

    # Step 4: build subtitles
    try:
        chunks_by_sentence = build_subtitle_chunks(words, sentences, aligned_sentence_counts)
    except Exception:
        print("[generate_tts] ERROR in build_subtitle_chunks:")
        traceback.print_exc()
        print(f"  words count: {len(words)}")
        print(f"  sentences count: {len(sentences)}")
        print(f"  aligned_sentence_counts: {aligned_sentence_counts}")
        raise

    try:
        write_ass_file(chunks_by_sentence, Path(ass_path))
    except Exception:
        print("[generate_tts] ERROR in write_ass_file:")
        traceback.print_exc()
        raise

    try:
        first_chunk_start = chunks_by_sentence[0][0][0]
    except (IndexError, TypeError):
        first_chunk_start = 0.0
    return first_chunk_start, total_dur