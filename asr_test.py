import nemo.collections.asr as nemo_asr
import torch
import torchaudio

# -------------------------------
# GPU / CPU setup
# -------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# -------------------------------
# Load the pretrained ASR model
# -------------------------------
model_name = "stt_en_quartznet15x5"
asr_model = nemo_asr.models.EncDecCTCModel.from_pretrained(model_name=model_name)
asr_model = asr_model.to(device)

# -------------------------------
# Example audio file
# -------------------------------
audio_file = r"/home/vishnu/Xtown-Report/NeMo/audio 2 (online-audio-converter.com).wav"

# -------------------------------
# Load and convert to mono
# -------------------------------
waveform, sample_rate = torchaudio.load(audio_file)
if waveform.shape[0] > 1:
    # Convert stereo to mono by averaging channels
    waveform = torch.mean(waveform, dim=0, keepdim=True)

# Save temporary mono file
mono_file = "temp_mono.wav"
torchaudio.save(mono_file, waveform, sample_rate)

# -------------------------------
# Run transcription
# -------------------------------
try:
    transcription = asr_model.transcribe([mono_file])
    print("\n--- Transcription ---")
    print(transcription[0])
except FileNotFoundError:
    print(f"Audio file '{audio_file}' not found. Please add a WAV file to test.")