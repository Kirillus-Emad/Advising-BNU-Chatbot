"""
gui.py
Tkinter GUI for BNU Chatbot.
- Correct Arabic display (no terminal encoding issues)
- RTL alignment for Arabic messages, LTR for English
- Long sentences wrap automatically
- Conversation memory + smart intent routing
"""

import tkinter as tk
from tkinter import scrolledtext, messagebox
import threading
import webbrowser
import re
import sys
import os

URL_RE = re.compile(r'(https?://[^\s؀-ۿ،؛»«\)\]\}]+|www\.[^\s؀-ۿ،؛»«\)\]\}]+)')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class BNUChatGUI:
    # ── Theme ────────────────────────────────────────────────────────
    BG          = "#f5f6fa"
    HEADER_BG   = "#1a3a5c"
    INPUT_BG    = "#ffffff"
    CHAT_BG     = "#ffffff"
    SEND_BG     = "#1a3a5c"
    USER_FG     = "#1a3a5c"
    BOT_FG      = "#1e8449"
    SOURCE_FG   = "#7f8c8d"
    SYSTEM_FG   = "#c0392b"
    MSG_FG      = "#2c3e50"
    DIVIDER_FG  = "#e0e0e0"
    FONT_MAIN   = ("Arial", 12)
    FONT_BOLD   = ("Arial", 12, "bold")
    FONT_SMALL  = ("Arial", 10)
    FONT_ITALIC = ("Arial", 11, "italic")

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("BNU Chatbot | مساعد جامعة بنها الأهلية")
        self.root.geometry("900x660")
        self.root.minsize(640, 480)
        self.root.configure(bg=self.BG)

        self.rag_pipeline  = None
        self.vs            = None
        self.conv_history  = []
        self.is_ready      = False
        self._link_tags    = set()

        self._build_ui()
        threading.Thread(target=self._initialize, daemon=True).start()

    # ── UI ───────────────────────────────────────────────────────────

    def _build_ui(self):
        # Header
        hdr = tk.Frame(self.root, bg=self.HEADER_BG, height=55)
        hdr.pack(fill=tk.X)
        hdr.pack_propagate(False)
        tk.Label(
            hdr,
            text="BNU Chatbot  |  مساعد جامعة بنها الأهلية",
            bg=self.HEADER_BG, fg="white",
            font=("Arial", 14, "bold"),
        ).pack(side=tk.LEFT, expand=True, padx=(10, 0))
        tk.Button(
            hdr,
            text="⟳ Rebuild KB",
            command=self._rebuild,
            bg="#2e5c8a", fg="white",
            font=("Arial", 9, "bold"),
            relief=tk.FLAT, padx=10, pady=5,
            cursor="hand2",
        ).pack(side=tk.RIGHT, padx=10, pady=8)

        # Status bar — pack BEFORE chat so it's not squeezed out
        self.status_var = tk.StringVar(value="Initializing...")
        tk.Label(
            self.root,
            textvariable=self.status_var,
            bg="#dcdde1", fg="#555",
            font=("Arial", 9),
            anchor=tk.W, padx=8, pady=3,
        ).pack(fill=tk.X, side=tk.BOTTOM)

        # Input row — pack BEFORE chat so it's not squeezed out
        inp_row = tk.Frame(self.root, bg=self.BG)
        inp_row.pack(fill=tk.X, side=tk.BOTTOM, padx=12, pady=(0, 10))

        self.entry = tk.Entry(
            inp_row,
            font=self.FONT_MAIN,
            relief=tk.SOLID, bd=1,
            bg=self.INPUT_BG, fg=self.MSG_FG,
            insertbackground=self.MSG_FG,
        )
        self.entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=9, padx=(0, 8))
        self.entry.bind("<Return>", self._on_send)

        self.send_btn = tk.Button(
            inp_row,
            text="Send | إرسال",
            command=self._on_send,
            bg=self.SEND_BG, fg="white",
            font=("Arial", 11, "bold"),
            relief=tk.FLAT, padx=18, pady=8,
            cursor="hand2",
            activebackground="#2e5c8a", activeforeground="white",
        )
        self.send_btn.pack(side=tk.RIGHT)

        # Chat area — pack LAST so it fills remaining space
        chat_outer = tk.Frame(self.root, bg=self.BG)
        chat_outer.pack(fill=tk.BOTH, expand=True, padx=12, pady=(10, 6))

        self.chat = scrolledtext.ScrolledText(
            chat_outer,
            wrap=tk.WORD,
            font=self.FONT_MAIN,
            bg=self.CHAT_BG,
            fg=self.MSG_FG,
            state=tk.DISABLED,
            relief=tk.FLAT,
            bd=0,
            highlightthickness=1,
            highlightbackground="#dcdde1",
            padx=14,
            pady=10,
            spacing1=2,
            spacing3=4,
        )
        self.chat.pack(fill=tk.BOTH, expand=True)

        # Tags
        self.chat.tag_config("user_lbl",  foreground=self.USER_FG,   font=self.FONT_BOLD,   justify="right")
        self.chat.tag_config("bot_lbl",   foreground=self.BOT_FG,    font=self.FONT_BOLD,   justify="left")
        self.chat.tag_config("msg_ar",    foreground=self.MSG_FG,    font=self.FONT_MAIN,   justify="right")
        self.chat.tag_config("msg_en",    foreground=self.MSG_FG,    font=self.FONT_MAIN,   justify="left")
        self.chat.tag_config("source",    foreground=self.SOURCE_FG, font=self.FONT_SMALL,  justify="left")
        self.chat.tag_config("system",    foreground=self.SYSTEM_FG, font=self.FONT_ITALIC, justify="left")
        self.chat.tag_config("divider",    foreground=self.DIVIDER_FG)
        self.chat.tag_config("link_style", foreground="#2980b9", underline=True)

    # ── Initialization ───────────────────────────────────────────────

    def _initialize(self):
        self._sys("🚀 Initializing BNU Chatbot...")
        self._status("Loading vector store...")

        from vectorstore import get_vector_store
        from loader import load_pdfs, BNU_KNOWLEDGE_BASE
        from pathlib import Path

        self.vs = get_vector_store()
        if not self.vs.is_populated():
            self._sys("📚 Building knowledge base from PDFs...")
            docs = []
            data_dir = Path("data")
            if data_dir.exists() and list(data_dir.glob("*.pdf")):
                try:
                    docs = load_pdfs("data")
                except Exception:
                    pass
            existing = {d["id"] for d in docs}
            for doc in BNU_KNOWLEDGE_BASE:
                if doc["id"] not in existing:
                    docs.append(doc)
            self.vs.add_documents(docs)
        else:
            self._sys(f"✅ Vector store ready: {self.vs.get_count()} chunks loaded")

        self._status("Loading RAG pipeline...")
        self._sys("⚙️  Connecting to Groq LLM + loading bge-m3 embeddings...")

        from rag import RAGPipeline
        self.rag_pipeline = RAGPipeline()

        from embed import get_embedding_model
        get_embedding_model().load()   # pre-warm so first query is fast

        self.is_ready = True
        self._sys("✅ System ready! Type your question below.\n")
        self._status("Ready")
        self.root.after(0, lambda: self.entry.focus())

    # ── Message handling ─────────────────────────────────────────────

    def _on_send(self, event=None):
        if not self.is_ready:
            self._sys("⏳ Still initializing, please wait...")
            return
        q = self.entry.get().strip()
        if not q:
            return
        self.entry.delete(0, tk.END)
        self._append_msg("user_lbl", "You", q, lang=self._lang(q))
        self.send_btn.configure(state=tk.DISABLED)
        self._status("Thinking...")
        threading.Thread(target=self._process, args=(q,), daemon=True).start()

    def _process(self, question: str):
        from router import (classify_intent, detect_language_switch,
                            get_greeting_response, get_acknowledgment_response,
                            get_out_of_scope_response, get_offensive_response,
                            get_crisis_response, get_last_real_answer)
        from utils import detect_language

        intent   = classify_intent(question)
        language = detect_language(question)

        if intent == 'language_switch':
            target      = detect_language_switch(question)
            last_answer = get_last_real_answer(self.conv_history)
            if last_answer and self.rag_pipeline:
                answer = self.rag_pipeline.translate_last_answer(last_answer, target)
            else:
                answer = get_acknowledgment_response(language, bool(self.conv_history))
            result = {"answer": answer, "sources": [], "chunks_found": 0, "lang": target or language}

        elif intent == 'greeting':
            result = {"answer": get_greeting_response(language), "sources": [], "chunks_found": 0, "lang": language}

        elif intent == 'acknowledgment':
            result = {"answer": get_acknowledgment_response(language, bool(self.conv_history)),
                      "sources": [], "chunks_found": 0, "lang": language}

        elif intent == 'offensive':
            result = {"answer": get_offensive_response(language), "sources": [], "chunks_found": 0, "lang": language}

        elif intent == 'crisis':
            result = {"answer": get_crisis_response(language), "sources": [], "chunks_found": 0, "lang": language}

        elif intent == 'out_of_scope':
            result = {"answer": get_out_of_scope_response(language), "sources": [], "chunks_found": 0, "lang": language}

        else:
            result = self.rag_pipeline.answer(question, conversation_history=self.conv_history)
            result["lang"] = result.get("language", language)

        self.conv_history.append({"user": question, "assistant": result["answer"], "intent": intent})
        self.root.after(0, self._show, result)

    def _show(self, result: dict):
        lang = result.get("lang", "en")
        self._append_msg("bot_lbl", "BNU Assistant", result["answer"],
                         lang=lang, sources=result.get("sources", []))
        chunks = result.get("chunks_found", 0)
        cached = " (cached)" if result.get("cached") else ""
        self._status(f"Ready  —  retrieved {chunks} sections{cached}")
        self.send_btn.configure(state=tk.NORMAL)
        self.entry.focus()

    # ── Display helpers ───────────────────────────────────────────────

    def _lang(self, text: str) -> str:
        """Quick language detection for display alignment."""
        ar = sum(1 for c in text if '؀' <= c <= 'ۿ')
        return 'ar' if ar > len(text) * 0.2 else 'en'

    def _insert_line_with_links(self, line: str, base_tag: str):
        """Insert a single line of text, making any URLs clickable."""
        parts = URL_RE.split(line)
        for part in parts:
            if URL_RE.match(part):
                url = part if part.startswith('http') else 'https://' + part
                tag_id = f"url_{abs(hash(url)) % 9999999}"
                self.chat.insert(tk.END, part, (base_tag, "link_style", tag_id))
                if tag_id not in self._link_tags:
                    self._link_tags.add(tag_id)
                    self.chat.tag_bind(tag_id, "<Button-1>",
                                       lambda e, u=url: webbrowser.open(u))
                    self.chat.tag_bind(tag_id, "<Enter>",
                                       lambda e: self.chat.configure(cursor="hand2"))
                    self.chat.tag_bind(tag_id, "<Leave>",
                                       lambda e: self.chat.configure(cursor=""))
            else:
                self.chat.insert(tk.END, part, base_tag)

    def _append_msg(self, label_tag: str, sender: str, message: str,
                    lang: str = 'en', sources: list = None):
        msg_tag = "msg_ar" if lang == 'ar' else "msg_en"
        self.chat.configure(state=tk.NORMAL)
        self.chat.insert(tk.END, f"{sender}:\n", label_tag)
        for line in message.split('\n'):
            self._insert_line_with_links(line, msg_tag)
            self.chat.insert(tk.END, '\n', msg_tag)
        if sources:
            seen = set()
            for src in sources:
                lbl = src.get("source", "")
                sec = src.get("section", "")
                key = f"{lbl}|{sec}"
                if key not in seen and lbl:
                    seen.add(key)
                    self.chat.insert(tk.END, f"  📄 {lbl}  —  {sec}\n", "source")
        self.chat.insert(tk.END, "─" * 70 + "\n", "divider")
        self.chat.configure(state=tk.DISABLED)
        self.chat.see(tk.END)

    def _rebuild(self):
        if not self.is_ready:
            return
        if not messagebox.askyesno("Rebuild Knowledge Base",
                                   "This will re-index all PDFs from the data/ folder.\nContinue?"):
            return
        self.is_ready = False
        self.send_btn.configure(state=tk.DISABLED)
        self._link_tags.clear()
        threading.Thread(target=self._do_rebuild, daemon=True).start()

    def _do_rebuild(self):
        self._sys("🔄 Rebuilding knowledge base from new PDF data...")
        self._status("Rebuilding...")
        if self.vs:
            self.vs.reset()
        self._initialize()

    def _sys(self, text: str):
        def _do():
            self.chat.configure(state=tk.NORMAL)
            self.chat.insert(tk.END, text + "\n", "system")
            self.chat.configure(state=tk.DISABLED)
            self.chat.see(tk.END)
        self.root.after(0, _do)

    def _status(self, text: str):
        self.root.after(0, lambda: self.status_var.set(text))


def main():
    root = tk.Tk()
    BNUChatGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
