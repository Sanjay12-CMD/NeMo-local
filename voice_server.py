from flask import Flask, request, send_file, jsonify
import torch
import soundfile as sf
import io
import nemo.collections.tts as nemo_tts

app = Flask(__name__)

# ==========================================================
# 🔊 Load NeMo TTS Models ONCE
# ==========================================================

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

print("Loading NeMo TTS models...")

spectrogram_generator = nemo_tts.models.FastPitchModel.from_pretrained(
    "tts_en_fastpitch"
).to(device)

vocoder = nemo_tts.models.HifiGanModel.from_pretrained(
    "tts_en_hifigan"
).to(device)

spectrogram_generator.eval()
vocoder.eval()

print("NeMo models loaded successfully!")

# ==========================================================
# ✂️ TEXT SPLITTING
# ==========================================================

MAX_CHARS_PER_CHUNK = 400

def split_text(text, max_chars=MAX_CHARS_PER_CHUNK):
    chunks = []
    while len(text) > max_chars:
        split_at = text.rfind(" ", 0, max_chars)
        if split_at == -1:
            split_at = max_chars
        chunks.append(text[:split_at])
        text = text[split_at:].strip()
    chunks.append(text)
    return chunks


# ==========================================================
# 🎤 TTS API
# ==========================================================

@app.route("/tts", methods=["POST"])
def tts():
    data = request.json
    text = data.get("text")

    if not text:
        return jsonify({"error": "Text is required"}), 400

    chunks = split_text(text)
    all_audio = []

    with torch.no_grad():
        for chunk in chunks:
            parsed = spectrogram_generator.parse(chunk)
            spectrogram = spectrogram_generator.generate_spectrogram(tokens=parsed)
            audio = vocoder.convert_spectrogram_to_audio(spec=spectrogram)
            all_audio.append(audio)

    final_audio = torch.cat(all_audio, dim=1)

    # Save to memory buffer
    buffer = io.BytesIO()
    sf.write(buffer, final_audio.cpu().numpy()[0], 22050, format="WAV")
    buffer.seek(0)

    return send_file(
        buffer,
        mimetype="audio/wav",
        as_attachment=False
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)