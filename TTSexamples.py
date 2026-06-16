from TTS.api import TTS
import os

tts = TTS(model_name="tts_models/en/vctk/vits", progress_bar=True)

female_speakers = [
    "p226", "p227", "p229",
    "p236", "p237",
    "p247", "p248", "p249",
    "p252", "p254",
    "p281", "p303"
]

male_speakers = [
    "p228", "p230", "p232",
    "p234", "p240",
    "p250", "p255", "p260",
    "p268", "p275",
    "p310", "p312"
]

text = "Hey! This is a quick, energetic voice test. Let's make things exciting!"

out_dir = "voice_samples"
os.makedirs(out_dir, exist_ok=True)

def generate_samples(speakers, label):
    for s in speakers:
        out_path = os.path.join(out_dir, f"{label}_{s}.wav")
        print(f"Generating {label} {s} …")
        tts.tts_to_file(text=text, speaker=s, file_path=out_path)

print("Generating female voices…")
generate_samples(female_speakers, "female")

print("Generating male voices…")
generate_samples(male_speakers, "male")

print("Done! Check the `voice_samples` folder.")
