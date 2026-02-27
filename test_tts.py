import nemo.collections.tts as nemo_tts
import soundfile as sf
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# Load models directly on GPU
spectrogram_generator = nemo_tts.models.FastPitchModel.from_pretrained(
    "tts_en_fastpitch"
).to(device)

vocoder = nemo_tts.models.HifiGanModel.from_pretrained(
    "tts_en_hifigan"
).to(device)

spectrogram_generator.eval()
vocoder.eval()

text = "X-TOWN 250i ABS - KYMCO MéxicoThe Kymco X-Town is a series of maxi-scooters available in 125cc and 300cc (276cc) displacements, designed for urban commuting and long-haul comfort. Known for its sporty styling, shaped taillight, and features like under-seat storage for two helmets, it is aimed at delivering a balanced, comfortable riding experience. The 250i ABS model offers a powerful 276cc engine, advanced safety with ABS, and a modern design, making it a popular choice for riders seeking a blend of performance and practicality in the maxi-scooter segment."

with torch.no_grad():
    parsed = spectrogram_generator.parse(text)
    spectrogram = spectrogram_generator.generate_spectrogram(tokens=parsed)
    audio = vocoder.convert_spectrogram_to_audio(spec=spectrogram)

sf.write("output_gpu.wav", audio.to("cpu").numpy()[0], 22050)

print("Saved output_gpu.wav successfully!")