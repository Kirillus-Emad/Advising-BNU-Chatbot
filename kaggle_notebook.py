"""
kaggle_notebook.py
══════════════════════════════════════════════
BNU Chatbot — Kaggle / Colab Ready Version
══════════════════════════════════════════════

Run this in a Kaggle notebook with GPU enabled.
Just paste each section in a separate cell.
"""

# ════════════════════════════════════════
# CELL 1: Install dependencies
# ════════════════════════════════════════
"""
!pip install -q \
    chromadb==0.5.3 \
    sentence-transformers==3.0.1 \
    transformers==4.44.2 \
    accelerate==0.33.0 \
    bitsandbytes \
    pymupdf \
    pdfplumber \
    langdetect \
    colorama \
    pyarabic
"""

# ════════════════════════════════════════
# CELL 2: Upload your PDFs and run setup
# ════════════════════════════════════════
"""
import os
os.makedirs("data", exist_ok=True)

# If running on Kaggle with a dataset:
# !cp /kaggle/input/your-dataset/*.pdf data/

# OR upload manually to data/ folder
"""

# ════════════════════════════════════════
# CELL 3: Copy all Python files
# (Copy each .py file from the project)
# Then run:
# ════════════════════════════════════════
"""
# Initialize everything
from loader import load_pdfs, BNU_KNOWLEDGE_BASE
from vectorstore import get_vector_store

vs = get_vector_store()
if not vs.is_populated():
    docs = []
    try:
        docs = load_pdfs("data")
    except Exception as e:
        print(f"PDF loading: {e}")
    for doc in BNU_KNOWLEDGE_BASE:
        if not any(d["id"] == doc["id"] for d in docs):
            docs.append(doc)
    vs.add_documents(docs)
    print(f"✅ Indexed {vs.get_count()} chunks")
"""

# ════════════════════════════════════════
# CELL 4: Interactive chat widget for Kaggle
# ════════════════════════════════════════

KAGGLE_CHAT_CODE = '''
from IPython.display import display, HTML, clear_output
import ipywidgets as widgets
from vectorstore import get_vector_store
from router import classify_intent, get_greeting_response, get_out_of_scope_response
from utils import detect_language, normalize_egyptian_arabic, format_sources

vs = get_vector_store()

# Try to load RAG pipeline (requires LLM)
try:
    from rag import RAGPipeline
    rag = RAGPipeline()
    USE_LLM = True
    print("✅ LLM loaded — full RAG mode")
except Exception as e:
    rag = None
    USE_LLM = False
    print(f"⚡ Retrieval-only mode (LLM not loaded: {e})")


def get_answer(question):
    """Get answer for a question."""
    intent = classify_intent(question)
    language = detect_language(question)
    
    if intent == "greeting":
        return get_greeting_response(language), []
    
    if intent == "out_of_scope":
        return get_out_of_scope_response(language), []
    
    if USE_LLM and rag:
        result = rag.answer(question)
        return result["answer"], result["sources"]
    else:
        # Retrieval only
        normalized = normalize_egyptian_arabic(question) if language == "ar" else question
        chunks = vs.search(normalized, top_k=3)
        if not chunks:
            if language == "ar":
                return "عذراً، لم أجد معلومات كافية. تفضل بزيارة www.bnu.edu.eg", []
            return "Sorry, no relevant information found. Visit www.bnu.edu.eg", []
        
        answer = "\\n\\n".join([c["text"] for c in chunks[:2]])
        sources = [c["metadata"] for c in chunks]
        return answer, sources


# ── Build Kaggle UI ──────────────────────────────────
chat_history = []

title = widgets.HTML("""
<div style="background: linear-gradient(135deg, #1a237e, #0d47a1); 
            padding: 20px; border-radius: 12px; margin-bottom: 15px; text-align: center;">
    <h2 style="color: white; margin: 0; font-family: Arial;">
        🎓 جامعة بنها الأهلية | BNU Chatbot
    </h2>
    <p style="color: #90caf9; margin: 5px 0 0 0; font-size: 13px;">
        Benha National University — Smart Assistant
    </p>
</div>
""")

chat_output = widgets.Output(
    layout=widgets.Layout(
        height="450px",
        overflow_y="auto",
        border="1px solid #e0e0e0",
        border_radius="8px",
        padding="10px",
        background_color="#fafafa",
    )
)

question_box = widgets.Text(
    placeholder="اكتب سؤالك هنا / Type your question here...",
    layout=widgets.Layout(width="75%", height="40px"),
    style={"font_size": "15px"},
)

send_btn = widgets.Button(
    description="إرسال / Send",
    button_style="primary",
    layout=widgets.Layout(width="20%", height="40px"),
    style={"font_weight": "bold"},
)

clear_btn = widgets.Button(
    description="مسح / Clear",
    button_style="warning",
    layout=widgets.Layout(width="10%", height="40px"),
)

input_row = widgets.HBox(
    [question_box, send_btn, clear_btn],
    layout=widgets.Layout(gap="5px", margin="10px 0"),
)

def display_message(role, text, sources=None):
    """Render a chat message in the output widget."""
    if role == "user":
        bubble_style = """
            background: #1565c0; color: white; 
            border-radius: 12px 12px 4px 12px;
            padding: 10px 14px; margin: 5px 0 5px 30%;
            font-size: 14px; line-height: 1.5;
        """
        prefix = "👤"
    else:
        bubble_style = """
            background: white; color: #1a1a1a;
            border: 1px solid #e0e0e0;
            border-radius: 12px 12px 12px 4px;
            padding: 10px 14px; margin: 5px 30% 5px 0;
            font-size: 14px; line-height: 1.6;
        """
        prefix = "🤖"
    
    text_html = text.replace("\\n", "<br>").replace("  ", "&nbsp;&nbsp;")
    
    sources_html = ""
    if sources:
        src_text = format_sources(sources)
        if src_text:
            sources_html = f"""
            <div style="font-size: 11px; color: #1565c0; margin-top: 6px; 
                        border-top: 1px solid #e3f2fd; padding-top: 4px;">
                {src_text.replace(chr(10), "<br>")}
            </div>
            """
    
    html = f"""
    <div style="{bubble_style}">
        <strong>{prefix}</strong><br>
        {text_html}
        {sources_html}
    </div>
    """
    display(HTML(html))


def on_send(b):
    """Handle send button click."""
    question = question_box.value.strip()
    if not question:
        return
    
    question_box.value = ""
    
    with chat_output:
        display_message("user", question)
    
    # Get answer
    answer, sources = get_answer(question)
    
    with chat_output:
        display_message("bot", answer, sources)
    
    chat_history.append({"user": question, "bot": answer})


def on_clear(b):
    """Clear chat history."""
    global chat_history
    chat_history = []
    with chat_output:
        clear_output()


send_btn.on_click(on_send)
clear_btn.on_click(on_clear)

# Handle Enter key
def handle_submit(widget):
    on_send(None)

question_box.on_submit(handle_submit)

# Display initial greeting
with chat_output:
    display_message("bot", 
        "أهلاً وسهلاً! 🎓\\nأنا المساعد الذكي لجامعة بنها الأهلية.\\n"
        "يسعدني الإجابة على أسئلتك عن الكليات، المصروفات، شروط القبول، والمزيد!\\n\\n"
        "Welcome! Ask me anything about BNU. 😊"
    )

# Show the UI
display(widgets.VBox([title, chat_output, input_row]))
'''

if __name__ == "__main__":
    print("This file contains Kaggle notebook code.")
    print("Copy the code from KAGGLE_CHAT_CODE into a Kaggle cell.")
    print("Or run: python main.py --no-llm")
