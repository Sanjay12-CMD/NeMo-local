import torch
import soundfile as sf
import nemo.collections.tts as nemo_tts

# Load FastPitch (text → spectrogram)
spectrogram_generator = nemo_tts.models.FastPitchModel.from_pretrained("tts_en_fastpitch")

# Load HiFi-GAN (spectrogram → audio)
vocoder = nemo_tts.models.HifiGanModel.from_pretrained("tts_en_hifigan")

# Move to GPU
spectrogram_generator = spectrogram_generator.cuda()
vocoder = vocoder.cuda()

text = "Hello, this is a test of NVIDIA NeMo text to speech running on GPU."

# Generate spectrogram
spectrogram = spectrogram_generator.generate_spectrogram(tokens=spectrogram_generator.parse(text))

# Generate audio
audio = vocoder.convert_spectrogram_to_audio(spec=spectrogram)

# Save file
sf.write("output.wav", audio[0].cpu().numpy(), 22050)

print("Audio generated successfully!")