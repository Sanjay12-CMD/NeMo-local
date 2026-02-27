import os
from nemo.collections.tts.models import FastPitchModel, HifiGanModel
from pydub import AudioSegment

print("Loading FastPitch model...")
fastpitch = FastPitchModel.from_pretrained("tts_en_fastpitch_align")
print("FastPitch loaded successfully!")

print("Loading HiFi-GAN model...")
hifigan = HifiGanModel.from_pretrained("tts_en_hifigan")  # ← THIS LINE IS CRITICAL
print("HiFi-GAN loaded successfully!")

text = "Hello! This is a test of NVIDIA NeMo text to speech."
mel = fastpitch.generate_spectrogram(text)
audio = hifigan.convert_spectrogram_to_audio(mel)

output_file = "output.wav"
audio_path = os.path.join(os.getcwd(), output_file)
AudioSegment(
    audio.tobytes(),
    frame_rate=fastpitch.sample_rate,
    sample_width=audio.dtype.itemsize,
    channels=1
).export(audio_path, format="wav")

print(f"✅ Audio saved to {audio_path}")
