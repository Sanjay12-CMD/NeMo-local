import nemo.collections.tts as nemo_tts
import soundfile as sf
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# Load multispeaker FastPitch
spectrogram_generator = nemo_tts.models.FastPitchModel.from_pretrained(
    "tts_en_fastpitch_multispeaker"
).to(device)

# Load vocoder
vocoder = nemo_tts.models.HifiGanModel.from_pretrained(
    "tts_en_hifigan"
).to(device)

spectrogram_generator.eval()
vocoder.eval()

# Your reference voice file
reference_audio = "audio 2 (online-audio-converter.com).wav"

# New text you want spoken
text = "Hello Sanjay, this is your cloned voice speaking."

with torch.no_grad():
    # Extract speaker embedding from reference audio
    speaker_embedding = spectrogram_generator.extract_speaker_embedding(
        reference_audio
    ).to(device)

    parsed = spectrogram_generator.parse(text)

    spectrogram = spectrogram_generator.generate_spectrogram(
        tokens=parsed,
        speaker_embeddings=speaker_embedding
    )

    audio = vocoder.convert_spectrogram_to_audio(spec=spectrogram)

# Save result
sf.write("output_clone.wav", audio.to("cpu").numpy()[0], 22050)

print("Saved output_clone.wav successfully!")