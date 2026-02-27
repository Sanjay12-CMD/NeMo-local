import express from "express";
import { askQuestion, speakText } from "./rag.controller.js";
import { protect } from "../../shared/middlewares/auth.js";
import { ragRateLimit } from "../../shared/middlewares/rateLimit.js";

const router = express.Router();

// student / teacher / parent can all use this
router.post("/ask", ragRateLimit, protect, askQuestion);
router.post("/speak", ragRateLimit, protect, speakText);

export default router;
