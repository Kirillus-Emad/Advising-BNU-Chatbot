"""
main.py
BNU University Chatbot — CLI Interface
Supports Egyptian Arabic, Modern Standard Arabic, and English.

Usage:
    python main.py                  # Normal chat mode
    python main.py --rebuild        # Force rebuild vector store
    python main.py --no-llm         # Use retrieval-only mode (faster, no LLM)
"""

import sys
import os
import argparse
import time
from pathlib import Path

# ─── Color output ────────────────────────
try:
    from colorama import init, Fore, Style
    init(autoreset=True)
    COLORS = True
except ImportError:
    COLORS = False

def c(text, color):
    if not COLORS:
        return text
    colors = {
        'cyan': Fore.CYAN, 'green': Fore.GREEN, 'yellow': Fore.YELLOW,
        'red': Fore.RED, 'blue': Fore.BLUE, 'magenta': Fore.MAGENTA,
        'white': Fore.WHITE, 'bold': Style.BRIGHT,
    }
    return colors.get(color, '') + str(text) + Style.RESET_ALL


def print_banner():
    banner = """
╔══════════════════════════════════════════════════════════════╗
║         🎓  جامعة بنها الأهلية  |  BNU Chatbot  🎓          ║
║         Benha National University — Smart Assistant          ║
║                                                              ║
║   Languages: العربية (عامية + فصحى)  |  English             ║
║   Type 'help' for commands  |  اكتب 'مساعدة' للأوامر        ║
╚══════════════════════════════════════════════════════════════╝
"""
    print(c(banner, 'cyan'))


def print_help():
    help_text = """
📋 Available Commands / الأوامر المتاحة:
  help / مساعدة     — Show this help
  clear / مسح       — Clear conversation history
  sources / مصادر   — Toggle source display
  quit / خروج / exit — Exit the chatbot
  
💡 Example Questions / أمثلة:
  • ما هي مصاريف كلية الهندسة؟
  • كيف أتقدم للجامعة؟
  • What are the admission requirements?
  • هل في منح دراسية؟
  • ايه الكليات الموجودة في الجامعة؟
  • What is the minimum score for medicine?
"""
    print(c(help_text, 'yellow'))


def initialize_system(rebuild: bool = False):
    """Initialize vector store with documents."""
    print(c("\n🚀 Initializing BNU Chatbot System...", 'cyan'))

    from vectorstore import get_vector_store
    from loader import load_pdfs, BNU_KNOWLEDGE_BASE

    vs = get_vector_store()

    # Check if rebuild needed
    if rebuild and vs.is_populated():
        print(c("🔄 Rebuilding vector store...", 'yellow'))
        vs.reset()

    if not vs.is_populated():
        print(c("📚 Building knowledge base...", 'yellow'))

        all_docs = []

        # Try loading from PDFs first
        data_dir = Path("data")
        if data_dir.exists() and (list(data_dir.glob("*.pdf")) or list(data_dir.glob("*.docx"))):
            try:
                pdf_docs = load_pdfs("data")
                all_docs.extend(pdf_docs)
                print(c(f"✅ Loaded {len(pdf_docs)} chunks from PDFs", 'green'))
            except Exception as e:
                print(c(f"⚠️  PDF loading failed: {e}", 'yellow'))
                print(c("   Using built-in knowledge base instead.", 'yellow'))

        # Always add built-in knowledge (ensures completeness)
        existing_ids = {doc["id"] for doc in all_docs}
        for doc in BNU_KNOWLEDGE_BASE:
            if doc["id"] not in existing_ids:
                all_docs.append(doc)

        print(c(f"📊 Total knowledge chunks: {len(all_docs)}", 'cyan'))

        # Index into ChromaDB
        vs.add_documents(all_docs)

        # Build BM25 sparse index for hybrid retrieval
        vs.build_bm25(all_docs)
    else:
        print(c(f"✅ Vector store ready: {vs.get_count()} chunks loaded", 'green'))

    return vs


def initialize_rag(no_llm: bool = False):
    """Initialize RAG pipeline."""
    if no_llm:
        print(c("⚡ Running in retrieval-only mode (no LLM)", 'yellow'))
        return None

    from rag import RAGPipeline
    return RAGPipeline()


def retrieval_only_answer(question: str, vs) -> dict:
    """
    Simple retrieval without LLM — returns top chunks directly.
    Useful for fast responses or when LLM is not available.
    """
    from utils import detect_language, normalize_egyptian_arabic, format_sources

    language = detect_language(question)
    normalized = normalize_egyptian_arabic(question) if language == 'ar' else question

    chunks = vs.hybrid_search(normalized, top_k=3)
    if not chunks:
        chunks = vs.hybrid_search(question, top_k=3)

    if not chunks:
        if language == 'ar':
            answer = "عذراً، لم أجد معلومات ذات صلة. تفضل بزيارة www.bnu.edu.eg"
        else:
            answer = "Sorry, no relevant information found. Visit www.bnu.edu.eg"
    else:
        if language == 'ar':
            answer = "بناءً على المعلومات المتاحة:\n\n"
        else:
            answer = "Based on available information:\n\n"
        for i, chunk in enumerate(chunks[:2], 1):
            answer += f"{chunk['text']}\n\n"

    return {
        "answer": answer.strip(),
        "sources": [c["metadata"] for c in chunks],
        "language": language,
        "chunks_found": len(chunks),
    }


def format_answer(result: dict, show_sources: bool = True) -> str:
    """Format the chatbot response for display."""
    from utils import format_sources

    output = []

    # Main answer
    output.append(c("\n🤖 BNU Assistant:", 'green'))
    output.append(result["answer"])

    # Sources
    if show_sources and result.get("sources"):
        sources_text = format_sources(result["sources"])
        if sources_text:
            output.append(c(f"\n{sources_text}", 'blue'))

    # Debug info (optional)
    chunks = result.get("chunks_found", 0)
    cached = "⚡ (cached)" if result.get("cached") else ""
    if chunks > 0:
        output.append(c(f"\n[Retrieved {chunks} relevant sections {cached}]", 'magenta'))

    return "\n".join(output)


def chat_loop(rag_pipeline, vs, show_sources: bool = True, no_llm: bool = False):
    """Main chat loop."""
    from router import (classify_intent, detect_language_switch,
                       get_greeting_response, get_acknowledgment_response,
                       get_out_of_scope_response, get_offensive_response,
                       get_crisis_response, get_last_real_answer)
    from utils import detect_language

    conversation_history = []
    print(c("\n✅ System ready! Ask your question below.\n", 'green'))
    print(c("─" * 60, 'cyan'))

    while True:
        try:
            # Get user input
            user_input = input(c("\n👤 You: ", 'yellow')).strip()

            if not user_input:
                continue

            # ── Commands ──────────────────────────────
            cmd = user_input.lower()
            if cmd in ['quit', 'exit', 'q', 'خروج', 'وداعاً']:
                print(c("\n👋 شكراً لاستخدامك مساعد جامعة بنها الأهلية! / Goodbye!", 'cyan'))
                break

            if cmd in ['help', 'مساعدة', '?']:
                print_help()
                continue

            if cmd in ['clear', 'مسح']:
                os.system('cls' if os.name == 'nt' else 'clear')
                print_banner()
                conversation_history = []
                print(c("✅ Conversation cleared.", 'green'))
                continue

            if cmd in ['sources', 'مصادر']:
                show_sources = not show_sources
                state = "ON ✅" if show_sources else "OFF ❌"
                print(c(f"Sources display: {state}", 'yellow'))
                continue

            # ── Route intent ──────────────────────────
            intent  = classify_intent(user_input)
            language = detect_language(user_input)

            print(c("⏳ Thinking...", 'magenta'), end='\r')
            start_time = time.time()

            if intent == 'language_switch':
                target_lang = detect_language_switch(user_input)
                last_answer = get_last_real_answer(conversation_history)
                if last_answer and rag_pipeline:
                    answer_text = rag_pipeline.translate_last_answer(last_answer, target_lang)
                else:
                    answer_text = get_acknowledgment_response(language, bool(conversation_history))
                result = {"answer": answer_text, "sources": [], "language": target_lang or language, "chunks_found": 0}

            elif intent == 'greeting':
                answer_text = get_greeting_response(language)
                result = {"answer": answer_text, "sources": [], "language": language, "chunks_found": 0}

            elif intent == 'acknowledgment':
                answer_text = get_acknowledgment_response(language, bool(conversation_history))
                result = {"answer": answer_text, "sources": [], "language": language, "chunks_found": 0}

            elif intent == 'offensive':
                answer_text = get_offensive_response(language)
                result = {"answer": answer_text, "sources": [], "language": language, "chunks_found": 0}

            elif intent == 'crisis':
                answer_text = get_crisis_response(language)
                result = {"answer": answer_text, "sources": [], "language": language, "chunks_found": 0}

            elif intent == 'out_of_scope':
                answer_text = get_out_of_scope_response(language)
                result = {"answer": answer_text, "sources": [], "language": language, "chunks_found": 0}

            else:
                # University question → RAG
                if no_llm or rag_pipeline is None:
                    result = retrieval_only_answer(user_input, vs)
                else:
                    result = rag_pipeline.answer(user_input, conversation_history=conversation_history)

            elapsed = time.time() - start_time
            print(" " * 30, end='\r')  # clear "thinking" line

            # ── Display response ──────────────────────
            print(format_answer(result, show_sources=show_sources))
            print(c(f"[{elapsed:.1f}s]", 'magenta'), end='')

            # Store in conversation history
            conversation_history.append({
                "user": user_input,
                "assistant": result["answer"],
                "intent": intent,
            })

        except KeyboardInterrupt:
            print(c("\n\n👋 Goodbye! / وداعاً!", 'cyan'))
            break
        except Exception as e:
            print(c(f"\n❌ Error: {e}", 'red'))
            print(c("Please try again.", 'yellow'))


def main():
    parser = argparse.ArgumentParser(
        description="BNU University Chatbot — مساعد جامعة بنها الأهلية"
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force rebuild the vector store from PDFs",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Run in retrieval-only mode (no LLM, much faster)",
    )
    parser.add_argument(
        "--no-sources",
        action="store_true",
        help="Hide source citations in responses",
    )
    args = parser.parse_args()

    print_banner()

    # Initialize vector store
    vs = initialize_system(rebuild=args.rebuild)

    # Initialize RAG pipeline
    rag = initialize_rag(no_llm=args.no_llm)

    # Start chat
    chat_loop(
        rag_pipeline=rag,
        vs=vs,
        show_sources=not args.no_sources,
        no_llm=args.no_llm,
    )


if __name__ == "__main__":
    main()
