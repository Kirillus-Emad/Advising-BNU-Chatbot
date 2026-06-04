"""
rag.py
RAG pipeline: retrieve → build context → generate answer.
Uses Groq API (llama-3.3-70b-versatile) — no local model download.
Retrieval: bge-m3 semantic search (cosine similarity), top-5 chunks.
"""

import os
import re
import json
import hashlib
from typing import Optional, Dict, List
from pathlib import Path

from groq import Groq
from dotenv import load_dotenv

from vectorstore import get_vector_store
from utils import detect_language, normalize_egyptian_arabic, format_sources

load_dotenv()

# 'llama-3.3-70b-versatile' 2048 
#"qwen/qwen3-32b"  4096

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_MODEL_NAME = 'qwen/qwen3-32b'  
LLM_CACHE_FILE = "./llm_cache.json"
MAX_TOKENS = 3000
TOP_K_RETRIEVAL = 10


SYSTEM_PROMPT_AR = """أنت مساعد ذكي متخصص حصرياً في الإجابة على أسئلة جامعة بنها الأهلية (BNU).

تعليمات صارمة:
1. أجب فقط بناءً على المعلومات الموجودة في السياق المعطى. لا تستخدم أي معلومة من خارج السياق مهما كان السؤال.
2. إذا لم يكن السؤال متعلقاً مباشرةً بجامعة بنها الأهلية أو إذا لم تجد إجابته في السياق، رد فقط بـ: "أنا متخصص فقط في أسئلة جامعة بنها الأهلية. هل عندك سؤال عن الكليات أو المصاريف أو القبول؟"
3. لا تخترع أو تفترض أو تستنتج أي معلومة غير موجودة حرفياً في السياق.
4. اللغة: إذا كتب المستخدم بالعامية المصرية، رد بالعامية المصرية. إذا كتب بالفصحى، رد بالفصحى. إذا كتب بالإنجليزية، رد بالإنجليزية.
5. إذا كان السؤال عن رسوم أو مصاريف، اذكر الأرقام بدقة كما وردت في السياق.
6. نسّق إجابتك بشكل واضح: استخدم عناوين ونقاط مرتبة عند الحاجة."""

SYSTEM_PROMPT_EN = """You are a smart assistant specialized exclusively in answering questions about Benha National University (BNU).

Strict instructions:
1. Answer ONLY based on the information provided in the context. Never use knowledge from outside the context, no matter what the question is.
2. If the question is not directly about BNU, or if the answer is not found in the context, reply ONLY with: "I'm specialized only in Benha National University questions. Do you have a question about faculties, fees, or admission?"
3. Do NOT fabricate, assume, or infer any information not explicitly stated in the context.
4. Language: if the user writes in Egyptian Arabic dialect, reply in Egyptian Arabic dialect. If they write in English, reply in English. If in Modern Standard Arabic, reply in Modern Standard Arabic.
5. When discussing fees, state exact figures as they appear in the context.
6. Format your answer clearly: use headers and bullet points where appropriate."""


class ResponseCache:
    """Simple JSON-based cache for LLM responses."""

    def __init__(self, cache_file: str = LLM_CACHE_FILE):
        self.cache_file = cache_file
        self._cache: Dict[str, str] = {}
        self._load()

    def _load(self):
        if Path(self.cache_file).exists():
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    self._cache = json.load(f)
            except Exception:
                self._cache = {}

    def _save(self):
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self._cache, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def get(self, key: str) -> Optional[str]:
        return self._cache.get(key)

    def set(self, key: str, value: str):
        self._cache[key] = value
        self._save()

    @staticmethod
    def make_key(query: str, context: str) -> str:
        content = f"{query}||{context[:200]}"
        return hashlib.md5(content.encode()).hexdigest()


class GroqLLM:
    """Wrapper around Groq API (llama-3.3-70b-versatile)."""

    def __init__(self):
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not found in .env file")
        self._client = Groq(api_key=GROQ_API_KEY)
        print(f"⚙️  LLM: {LLM_MODEL_NAME} via Groq API")

    def chat(self, messages: List[Dict]) -> str:
        """Send a chat request to Groq and return the response text."""
        try:
            response = self._client.chat.completions.create(
                model=LLM_MODEL_NAME,
                messages=messages,
                max_tokens=MAX_TOKENS,
                temperature=0.1,
            )
            text = response.choices[0].message.content or ""
            # Qwen3 models emit <think>...</think> reasoning blocks — strip them
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
            return text
        except Exception as e:
            return f"[Groq API Error: {e}]"


class RAGPipeline:
    """
    Full RAG pipeline:
    1. Retrieve relevant chunks from vector store
    2. Build structured context
    3. Generate answer with Groq LLM
    """

    def __init__(self):
        self._vs = get_vector_store()
        self._llm = GroqLLM()
        self._cache = ResponseCache()

    def _build_context(self, chunks: List[Dict]) -> str:
        if not chunks:
            return "لا توجد معلومات ذات صلة متاحة. / No relevant information available."

        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            source = chunk["metadata"].get("source", "Unknown")
            section = chunk["metadata"].get("section", "")
            header = f"[{i}] من: {source}"
            if section and section != "general":
                header += f" | قسم: {section}"
            context_parts.append(f"{header}\n{chunk['text']}")

        return "\n\n---\n\n".join(context_parts)

    def _extract_sources(self, chunks: List[Dict]) -> List[Dict]:
        return [chunk["metadata"] for chunk in chunks]

    def answer(
        self,
        question: str,
        top_k: int = TOP_K_RETRIEVAL,
        conversation_history: Optional[List[Dict]] = None,
    ) -> Dict:
        """Full RAG answer pipeline with optional conversation history."""
        # 1. Detect language & normalize Arabic
        language = detect_language(question)
        normalized_q = normalize_egyptian_arabic(question) if language == 'ar' else question

        # 2. Hybrid retrieval + reranking: dense + BM25 candidates → cross-encoder rerank
        chunks = self._vs.hybrid_rerank_search(normalized_q, top_k=top_k)
        if not chunks and normalized_q != question:
            chunks = self._vs.hybrid_rerank_search(question, top_k=top_k)

        # 3. Check cache (only for fresh questions without history context)
        if chunks and not conversation_history:
            cache_key = ResponseCache.make_key(question, chunks[0]["text"][:200])
            cached = self._cache.get(cache_key)
            if cached:
                return {
                    "answer": cached,
                    "sources": self._extract_sources(chunks),
                    "language": language,
                    "chunks_found": len(chunks),
                    "cached": True,
                }

        # 4. Build context
        context = self._build_context(chunks)

        # 5. Build messages with conversation history
        system_prompt = SYSTEM_PROMPT_AR if language == 'ar' else SYSTEM_PROMPT_EN
        messages = [{"role": "system", "content": system_prompt}]

        # Include last 3 turns so the LLM knows what was discussed
        if conversation_history:
            for turn in conversation_history[-3:]:
                messages.append({"role": "user", "content": turn["user"]})
                messages.append({"role": "assistant", "content": turn["assistant"]})

        # Current question with retrieved context
        user_message = f"السياق:\n{context}\n\nالسؤال: {question}" if language == 'ar' \
            else f"Context:\n{context}\n\nQuestion: {question}"
        messages.append({"role": "user", "content": user_message})

        # 6. Generate answer
        answer_text = self._llm.chat(messages)

        # 7. Fallback if LLM failed or returned empty
        if not answer_text or len(answer_text.strip()) < 10 or answer_text.startswith("[Groq"):
            answer_text = self._fallback_answer(chunks, language)

        # 8. Cache result (only when no history to keep cache simple)
        if chunks and not conversation_history:
            cache_key = ResponseCache.make_key(question, chunks[0]["text"][:200])
            self._cache.set(cache_key, answer_text)

        return {
            "answer": answer_text,
            "sources": self._extract_sources(chunks),
            "language": language,
            "chunks_found": len(chunks),
            "cached": False,
        }

    def translate_last_answer(self, last_answer: str, target_language: str) -> str:
        """Translate the last answer to the requested language without new retrieval."""
        if target_language == 'en':
            system = "You are a helpful assistant. Translate the given Arabic text to clear, professional English."
            user_msg = f"Please translate this answer to English:\n\n{last_answer}"
        else:
            system = "أنت مساعد متخصص. ترجم النص التالي إلى اللغة العربية الفصحى بدقة واحترافية."
            user_msg = f"من فضلك ترجم هذه الإجابة إلى العربية:\n\n{last_answer}"

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg},
        ]
        return self._llm.chat(messages)

    def _fallback_answer(self, chunks: List[Dict], language: str) -> str:
        if not chunks:
            if language == 'ar':
                return (
                    "عذراً، لم أجد معلومات كافية للإجابة على هذا السؤال. 🤔\n"
                    "يرجى زيارة الموقع الرسمي: www.bnu.edu.eg\n"
                    "أو التواصل عبر الفيسبوك: https://www.facebook.com/BenhaNationalUniversity"
                )
            else:
                return (
                    "Sorry, I couldn't find enough information to answer this question. 🤔\n"
                    "Please visit the official website: www.bnu.edu.eg\n"
                    "Or contact via Facebook: https://www.facebook.com/BenhaNationalUniversity"
                )
        top_chunk = chunks[0]["text"]
        source = chunks[0]["metadata"].get("source", "")
        if language == 'ar':
            return f"بناءً على المعلومات المتاحة:\n\n{top_chunk}\n\n📄 المصدر: {source}"
        else:
            return f"Based on available information:\n\n{top_chunk}\n\n📄 Source: {source}"
