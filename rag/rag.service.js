import { ChromaClient } from "chromadb";
import { GoogleGenAI } from "@google/genai";
import AiChatLog from "../ai-chat-logs/ai-chat-log.model.js";
import { deductTokens } from "../tokens/token.service.js";
import AppError from "../../shared/appError.js";

const CHROMA_URL = process.env.CHROMA_URL || "http://localhost:8000";
const COLLECTION_NAME = "cbse_books";

// Gemini setup
const GEMINI_MODEL = (process.env.GEMINI_MODEL || "gemini-2.5-flash-lite").replace(/^models\//, "");
const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });

// Chroma setup
const chromaUrl = new URL(
  CHROMA_URL.startsWith("http") ? CHROMA_URL : `http://${CHROMA_URL}`
);
const chroma = new ChromaClient({
  host: chromaUrl.hostname,
  port: chromaUrl.port
    ? Number(chromaUrl.port)
    : chromaUrl.protocol === "https:"
    ? 443
    : 80,
  ssl: chromaUrl.protocol === "https:",
});

const normalizeClassLevel = (value) => {
  if (!value) return null;
  const str = String(value).trim().toLowerCase();
  const digitMatch = str.match(/\d+/);
  if (digitMatch) return digitMatch[0];
  return str.replace(/^class\s*/, "");
};

const STOPWORDS = new Set([
  "the",
  "and",
  "or",
  "of",
  "to",
  "a",
  "an",
  "in",
  "on",
  "for",
  "with",
  "about",
  "tell",
  "what",
  "is",
  "are",
  "was",
  "were",
  "do",
  "does",
  "did",
  "i",
  "you",
  "we",
  "they",
  "he",
  "she",
  "it",
]);

const extractKeywords = (text) => {
  const words = String(text || "")
    .toLowerCase()
    .match(/[a-z0-9]+/g);
  if (!words) return [];
  return words.filter((w) => w.length >= 4 && !STOPWORDS.has(w));
};

const keywordSearch = async ({ collection, query, limit = 5 }) => {
  const keywords = extractKeywords(query);
  if (!keywords.length) {
    return { chunks: [], metadatas: [] };
  }

  // Fetch all docs (small dataset) and score by keyword hits
  const all = await collection.get({
    limit: 10000,
    include: ["documents", "metadatas"],
  });

  const scored = [];
  for (let i = 0; i < (all.documents || []).length; i++) {
    const doc = all.documents[i] || "";
    const lower = doc.toLowerCase();
    let score = 0;
    for (const k of keywords) {
      if (lower.includes(k)) score += 1;
    }
    if (score > 0) {
      scored.push({ doc, meta: all.metadatas?.[i] || null, score });
    }
  }

  scored.sort((a, b) => b.score - a.score);
  const top = scored.slice(0, limit);

  return {
    chunks: top.map((t) => t.doc),
    metadatas: top.map((t) => t.meta),
  };
};

export const formatRagSources = (metadatas) => {
  if (!Array.isArray(metadatas)) return [];
  return [
    ...new Set(
      metadatas.map((m) => {
        const title = m.chapter || m.book || "Source";
        return `Class ${m.class} - ${title}`;
      })
    ),
  ];
};

const extractGeminiText = (result) =>
  result?.text ||
  result?.candidates?.[0]?.content?.parts?.map((p) => p.text).join("") ||
  "";

const runGemini = async (prompt) => {
  if (!process.env.GEMINI_API_KEY) {
    throw new AppError("GEMINI_API_KEY is missing", 500);
  }

  const result = await ai.models.generateContent({
    model: GEMINI_MODEL,
    contents: prompt,
  });

  return {
    text: extractGeminiText(result),
    tokensUsed: result?.usageMetadata?.totalTokenCount || 0,
  };
};

const buildDirectPrompt = ({ question, classLevel }) => `
You are a friendly school tutor.
Explain clearly for a Class ${classLevel || "school"} student.

Question:
${question}

Instructions:
- Give a correct answer in simple words.
- Choose the best visual format for this question: diagram, table, or pie.
- If process/explanation question (like photosynthesis): prefer diagram.
- If compare/classify question: prefer table.
- If parts/percent composition question: prefer pie.
- If this is a science question (e.g., photosynthesis), include key terms and real-life example.
- Keep it concise and easy to remember.

Output format (mandatory):
Best Format: diagram|table|pie

Visual Explanation:
- (for diagram: arrow steps)
- (for table: markdown table rows)
- (for pie: item with percentages)

Text Explanation:
- (simple explanation points)
`;

export async function retrieveRagContext({
  query,
  classLevel,
  allowGlobal = true,
}) {
  try {
    const collection = await chroma.getCollection({
      name: COLLECTION_NAME,
    });

    const results = await collection.query({
      queryTexts: [query],
      nResults: 5,
    });

    return {
      chunks: (results.documents || []).flat(),
      metadatas: (results.metadatas || []).flat(),
      filter: "global",
      classLevel: normalizeClassLevel(classLevel),
      chromaAvailable: true,
    };
  } catch (err) {
    // Chroma unavailable: continue with Gemini direct answering path.
    return {
      chunks: [],
      metadatas: [],
      filter: "gemini_direct",
      classLevel: normalizeClassLevel(classLevel),
      chromaAvailable: false,
    };
  }
}

export async function askRag({ question, classLevel, userId }) {
  const context = await retrieveRagContext({
    query: question,
    classLevel,
    allowGlobal: true,
  });

  const chunks = context.chunks;

  let answer;
  let tokensUsed = 0;
  let usedFilter = context.filter;
  let finalChunks = chunks;
  let finalMetadatas = context.metadatas;
  const normalizedClassLevel = normalizeClassLevel(classLevel);
  let billingWarning = null;

  if (!chunks.length) {
    if (context.chromaAvailable) {
      try {
        const collection = await chroma.getCollection({ name: COLLECTION_NAME });
        const keyword = await keywordSearch({
          collection,
          query: question,
          limit: 5,
        });
        if (keyword.chunks.length) {
          finalChunks = keyword.chunks;
          finalMetadatas = keyword.metadatas;
          usedFilter = "keyword";
        }
      } catch (err) {
        // ignore and use Gemini direct mode below
      }
    }

    if (!finalChunks.length) {
      const direct = await runGemini(
        buildDirectPrompt({ question, classLevel: normalizedClassLevel })
      );
      answer = direct.text || "I could not generate an answer right now.";
      tokensUsed = direct.tokensUsed;
      usedFilter = "gemini_direct";
    }
  }

  if (finalChunks.length && !answer) {
    const context = finalChunks.join("\n\n");

    const prompt = `
You are a school tutor.
Answer ONLY using the textbook content below.
If the answer is not present, say "I don't know".

Textbook content:
${context}

Question:
${question}

Output format (mandatory):
Best Format: diagram|table|pie

Visual Explanation:
- (for diagram: arrow steps)
- (for table: markdown table rows)
- (for pie: item with percentages)

Text Explanation:
- (simple explanation points)
`;

    const ragResult = await runGemini(prompt);
    tokensUsed = ragResult.tokensUsed || 0;
    answer = ragResult.text || "";
  }

  // If model still says "I don't know", try keyword context (if not already)
  if (
    answer &&
    answer.trim().toLowerCase() === "i don't know" &&
    usedFilter !== "keyword" &&
    context.chromaAvailable
  ) {
    try {
      const collection = await chroma.getCollection({ name: COLLECTION_NAME });
      const keyword = await keywordSearch({
        collection,
        query: question,
        limit: 5,
      });
      if (keyword.chunks.length) {
        finalChunks = keyword.chunks;
        finalMetadatas = keyword.metadatas;
        usedFilter = "keyword";

        const retryContext = finalChunks.join("\n\n");
        const retryPrompt = `
You are a school tutor.
Answer ONLY using the textbook content below.
If the answer is not present, say "I don't know".

Textbook content:
${retryContext}

Question:
${question}

Output format (mandatory):
Best Format: diagram|table|pie

Visual Explanation:
- (for diagram: arrow steps)
- (for table: markdown table rows)
- (for pie: item with percentages)

Text Explanation:
- (simple explanation points)
`;

        const retryResult = await runGemini(retryPrompt);
        tokensUsed = retryResult.tokensUsed || tokensUsed;
        answer = retryResult.text || answer;
      }
    } catch (err) {
      // Keep existing answer if keyword fallback fails.
    }
  }

  // 🔹 Log AI usage
  const log = await AiChatLog.create({
    user_id: userId,
    user_query: question,
    ai_response: answer,
    tokens_used: tokensUsed,
    model_used: GEMINI_MODEL,
    ai_type: "rag",
    class_level: classLevel ?? null,
  });

  // 🔹 Deduct tokens (only if tokens used)
  if (tokensUsed > 0) {
    try {
      await deductTokens({
        userId,
        amount: tokensUsed,
        reason: "rag",
        refId: log.id,
      });
    } catch (err) {
      // Keep answer successful even when billing/subscription blocks deduction.
      billingWarning = err?.message || "Token deduction failed";
    }
  }

  return {
    answer,
    sources: formatRagSources(finalMetadatas),
    source_type: finalChunks.length ? "rag" : "gemini_direct",
    filters_used: usedFilter,
    ...(billingWarning ? { billing_warning: billingWarning } : {}),
  };
}


