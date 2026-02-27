import fs from "fs";
import path from "path";
import { pathToFileURL } from "url";
import * as pdfjsLib from "pdfjs-dist/legacy/build/pdf.mjs";
import { ChromaClient } from "chromadb";
import dotenv from "dotenv";

dotenv.config();



// -------- CONFIG --------
const BOOKS_DIR = path.resolve(process.cwd(), process.argv[2]);
if (!BOOKS_DIR) {
  console.error("❌ Please provide books folder path");
  process.exit(1);
}

const CHROMA_URL = process.env.CHROMA_URL || "http://localhost:8000";
const COLLECTION_NAME = "cbse_books";

// -------- CHROMA SETUP --------
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

// -------- HELPERS --------
function getAllPdfFiles(dir) {
  let results = [];
  const list = fs.readdirSync(dir);

  list.forEach((file) => {
    const filePath = path.join(dir, file);
    const stat = fs.statSync(filePath);

    if (stat && stat.isDirectory()) {
      results = results.concat(getAllPdfFiles(filePath));
    } else if (file.endsWith(".pdf")) {
      results.push(filePath);
    }
  });

  return results;
}

// -------- MAIN INGEST FUNCTION --------
async function ingest() {
  console.log("📚 Starting ingestion...");
  console.log("Books folder:", BOOKS_DIR);

  const collection = await chroma.getOrCreateCollection({
    name: COLLECTION_NAME,
  });

  const pdfFiles = getAllPdfFiles(BOOKS_DIR);
  console.log(`Found ${pdfFiles.length} PDF files`);

  const chunkSize = 500;
  const chunkOverlap = 50;

  const splitText = (text) => {
    const cleaned = text.replace(/\s+/g, " ").trim();
    if (!cleaned) return [];

    const chunks = [];
    let start = 0;

    while (start < cleaned.length) {
      const end = Math.min(start + chunkSize, cleaned.length);
      chunks.push(cleaned.slice(start, end));
      if (end === cleaned.length) break;
      start = Math.max(0, end - chunkOverlap);
    }

    return chunks;
  };

  const standardFontDataUrl = `${pathToFileURL(
    path.resolve(process.cwd(), "node_modules/pdfjs-dist/standard_fonts")
  ).toString()}/`;

  for (const filePath of pdfFiles) {
    console.log(`\n📄 Processing: ${filePath}`);

    // ---- Extract metadata from path ----
    // Example: books/class7/science.pdf
    // Example: books/class6/english/chapter-1.pdf
    const parts = filePath.split(path.sep);
    const classIndex = parts.findIndex((p) => p.toLowerCase().startsWith("class"));
    const className = classIndex >= 0 ? parts[classIndex] : "unknown";
    const subjectName =
      classIndex >= 0 && classIndex + 1 < parts.length - 1
        ? parts[classIndex + 1]
        : "unknown";
    const fileName = path.basename(filePath);
    const chapterName = fileName.replace(/\.pdf$/i, "");
    const relativePath = path.relative(BOOKS_DIR, filePath);
    const safeIdPrefix = relativePath.replace(/[^\w.-]/g, "_");

    // ---- Read PDF ----
    let data;
    try {
      data = new Uint8Array(fs.readFileSync(filePath));
    } catch (err) {
      if (err && err.code === "ENOENT") {
        console.warn("⚠️ File missing, skipping:", filePath);
        continue;
      }
      throw err;
    }
    const pdf = await pdfjsLib.getDocument({
      data,
      standardFontDataUrl,
    }).promise;

    let fullText = "";

    for (let pageNum = 1; pageNum <= pdf.numPages; pageNum++) {
      const page = await pdf.getPage(pageNum);
      const content = await page.getTextContent();
      const pageText = content.items
        .map((item) => item.str)
        .join(" ")
        .replace(/\s+/g, " ")
        .trim();
      if (pageText) {
        fullText += `${pageText}\n`;
      }
    }

    if (!fullText.trim()) {
      console.warn("⚠️ Empty text, skipping:", filePath);
      continue;
    }

    // ---- Chunk text ----
    const chunks = splitText(fullText);
    console.log(`✂️ Created ${chunks.length} chunks`);

    // ---- Store chunks ----
    for (let i = 0; i < chunks.length; i++) {
      const chunkText = chunks[i];

      await collection.upsert({
        ids: [`${safeIdPrefix}-${i}`],
        documents: [chunkText],
        metadatas: [
          {
            syllabus: "CBSE",
            class: className.toLowerCase().replace("class", ""),
            subject: subjectName.toLowerCase(),
            book: fileName,
            chapter: chapterName,
            source_path: relativePath,
          },
        ],
      });
    }

    console.log(`✅ Finished ${fileName}`);
  }

  console.log("\n🎉 Ingestion completed successfully!");
}

ingest().catch((err) => {
  console.error("❌ Ingestion failed:", err);
  process.exit(1);
});
