import asyncHandler from "../../shared/asyncHandler.js";
import { askRag } from "./rag.service.js";
import {
  chunkText,
  textToSpeech,
} from "../../shared/services/voice.service.js";
import Class from "../classes/classes.model.js";

const normalizeClassLevel = (value) => {
  if (!value) return value;
  const str = String(value).trim().toLowerCase();
  const digitMatch = str.match(/\d+/);
  if (digitMatch) return digitMatch[0];
  return str.replace(/^class\s*/, "");
};

export const askQuestion = asyncHandler(async (req, res) => {
  const { question, classLevel } = req.body;
  const voiceEnabled = req.query.voice === "true";

  if (!question) {
    return res.status(400).json({ message: "Question is required" });
  }

  let effectiveClassLevel = normalizeClassLevel(classLevel);

  if (req.user?.role === "student" && req.user?.class_id) {
    const cls = await Class.findOne({
      where: { id: req.user.class_id, school_id: req.user.school_id },
      attributes: ["class_name"],
    });
    if (cls?.class_name) {
      effectiveClassLevel = normalizeClassLevel(cls.class_name);
    }
  }

  let result;
  try {
    result = await askRag({
      question,
      classLevel: effectiveClassLevel,
      userId: req.user.id,
    });
  } catch (err) {
    console.error("RAG ask failed:", err?.message || err);
    result = {
      answer:
        "I could not generate an answer right now. Please try again in a moment.",
      sources: [],
      source_type: "fallback",
    };
  }

  // 🔹 TEXT-ONLY (default)
  if (!voiceEnabled) {
    return res.json({
      question,
      answer: result.answer,
      sources: result.sources,
      ...(result.billing_warning ? { billing_warning: result.billing_warning } : {}),
    });
  }

  // 🔹 VOICE MODE
  res.setHeader("Content-Type", "audio/wav");
  res.setHeader("Transfer-Encoding", "chunked");

  const sentences = chunkText(result.answer);

  for (const sentence of sentences) {
    try {
      const wavBuffer = await textToSpeech(sentence);
      res.write(wavBuffer);
    } catch (err) {
      console.error("TTS failed:", err.message);
      break; // stop voice, keep text correctness
    }
  }

  res.end();
});

export const speakText = asyncHandler(async (req, res) => {
  const { text } = req.body;

  if (!text || !String(text).trim()) {
    return res.status(400).json({ message: "Text is required" });
  }

  res.setHeader("Content-Type", "audio/wav");
  res.setHeader("Transfer-Encoding", "chunked");

  const sentences = chunkText(String(text));

  for (const sentence of sentences) {
    try {
      const wavBuffer = await textToSpeech(sentence);
      res.write(wavBuffer);
    } catch (err) {
      console.error("TTS failed:", err.message);
      break;
    }
  }

  res.end();
});
