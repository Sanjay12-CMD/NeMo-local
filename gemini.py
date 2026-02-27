import os
import torch
import soundfile as sf
import wave
import nemo.collections.tts as nemo_tts
from google import genai

# ==========================================================
# 🔐 CONFIGURATION
# ==========================================================

GEMINI_API_KEY = "AIzaSyAzUoD51tcpTZegRlGgpSjebIJXXYVTIt4"
MAX_CHARS_PER_CHUNK = 300   # Safe limit for NeMo


# ==========================================================
# 🤖 GEMINI 2.5 FLASH SETUP
# ==========================================================

client = genai.Client(api_key=GEMINI_API_KEY)

def get_gemini_answer(query: str) -> str:
    """
    Forces Gemini to return a long detailed answer.
    """

    prompt = f"""
    Provide a detailed explanation of at least 600 words
    about the following topic.

    Topic: {query}

    Explain geography, history, importance, facts, and interesting details.
    """

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config={
            "max_output_tokens": 1200,
            "temperature": 0.7
        }
    )

    if not response.text:
        return "Sorry, I could not generate a response."

    return response.text.strip()


# ==========================================================
# 🔊 NeMo TTS SETUP (LOAD ONCE)
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
# ✂️ SAFE TEXT SPLITTING
# ==========================================================

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
# 🎤 TEXT → SPEECH FUNCTION
# ==========================================================

def text_to_speech(text, filename="output.wav"):

    chunks = split_text(text)

    print("\n================ DEBUG INFO ================")
    print(f"Total text length: {len(text)} characters")
    print(f"Number of chunks: {len(chunks)}")
    print("============================================\n")

    all_audio = []

    with torch.no_grad():
        for i, chunk in enumerate(chunks):
            print(f"Generating chunk {i+1}/{len(chunks)}")

            parsed = spectrogram_generator.parse(chunk)
            spectrogram = spectrogram_generator.generate_spectrogram(tokens=parsed)
            audio = vocoder.convert_spectrogram_to_audio(spec=spectrogram)

            all_audio.append(audio)

    final_audio = torch.cat(all_audio, dim=1)

    sf.write(filename, final_audio.to("cpu").numpy()[0], 22050)

    # Print actual duration
    with wave.open(filename, "r") as f:
        duration = f.getnframes() / f.getframerate()
        print(f"\nActual audio duration: {round(duration, 2)} seconds")

    print(f"Audio saved as {filename}")
    return filename


# ==========================================================
# 🚀 MAIN LOOP
# ==========================================================

if __name__ == "__main__":

    while True:
        user_question = input("\nAsk something (type 'exit' to quit): ")

        if user_question.lower() == "exit":
            break

        # Step 1: Get Gemini answer
        answer = get_gemini_answer(user_question)

        print("\n========== GEMINI ANSWER ==========\n")
        print(answer)
        print("\n===================================\n")

        # Step 2: Convert to speech
        audio_file = text_to_speech(answer)

        print("\nDone! Play the file using:")
        print("aplay output.wav\n")