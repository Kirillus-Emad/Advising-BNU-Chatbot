# 🎓 BNU Chatbot — مساعد جامعة بنها الأهلية

A fully local RAG-based chatbot for Benha National University (BNU).
Supports **Egyptian Arabic dialect**, **Modern Standard Arabic**, and **English**.

---

## 📁 Project Structure

```
bnu_chatbot/
│── main.py           # CLI chatbot entry point
│── rag.py            # RAG pipeline (retrieve + generate)
│── embed.py          # Local embedding model (multilingual-e5-base)
│── loader.py         # PDF loader + text cleaning + chunking
│── vectorstore.py    # ChromaDB vector store management
│── router.py         # Intent router (greeting / RAG / out-of-scope)
│── utils.py          # Language detection, Arabic normalization, helpers
│── requirements.txt  # Python dependencies
│── README.md         # This file
│── data/             # Place your PDF files here
│   ├── fees.pdf
│   ├── apply.pdf
│   └── links.pdf
│── chroma_db/        # Auto-created — ChromaDB persistent store
│── model_cache/      # Auto-created — HuggingFace model cache
│── llm_cache.json    # Auto-created — LLM response cache
```

---

## ⚙️ Installation

### Step 1 — Clone / Download the project
```bash
# If using git:
git clone <your-repo-url>
cd bnu_chatbot

# Or just copy all files to a folder called bnu_chatbot/
```

### Step 2 — Create a virtual environment (recommended)
```bash
python -m venv venv

# Activate on Linux/Mac:
source venv/bin/activate

# Activate on Windows:
venv\Scripts\activate
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

> ⚠️ If you're on **CPU only** (no GPU), also run:
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cpu
> ```

> ⚠️ If you're on **Kaggle / Colab with GPU**, install bitsandbytes for 4-bit quantization:
> ```bash
> pip install bitsandbytes
> ```

### Step 4 — Add your PDF files
```bash
mkdir data
# Copy your 3 PDF files into the data/ folder:
#   data/fees.pdf
#   data/apply.pdf
#   data/links.pdf
```

> 💡 **Note:** Even without PDFs, the chatbot works using the built-in BNU knowledge base extracted from your documents.

---

## 🚀 How to Run

### Normal mode (with full LLM — recommended on GPU):
```bash
python main.py
```

### Fast mode — No LLM (retrieval only, runs on any CPU):
```bash
python main.py --no-llm
```

### Rebuild vector store (if you updated PDFs):
```bash
python main.py --rebuild
```

### Hide source citations:
```bash
python main.py --no-sources
```

### Combine options:
```bash
python main.py --no-llm --rebuild
```

---

## 🧠 Tech Stack

| Component | Technology |
|-----------|-----------|
| Embeddings | `intfloat/multilingual-e5-base` (HuggingFace local) |
| Vector DB | ChromaDB (local persistence) |
| LLM | `Qwen/Qwen2.5-3B-Instruct` (local, 4-bit on GPU) |
| PDF Parsing | PyMuPDF + pdfplumber |
| Arabic NLP | Custom normalization + langdetect |

---

## 💬 Example Chat Session

```
╔══════════════════════════════════════════════════════════════╗
║         🎓  جامعة بنها الأهلية  |  BNU Chatbot  🎓          ║
╚══════════════════════════════════════════════════════════════╝

✅ System ready! Ask your question below.

👤 You: السلام عليكم

🤖 BNU Assistant:
أهلاً وسهلاً! 🎓
أنا المساعد الذكي لجامعة بنها الأهلية.
يسعدني الإجابة على استفساراتك عن:
• الكليات والبرامج الدراسية
• المصروفات والمنح الدراسية
• شروط وخطوات القبول
• الروابط والمعلومات العامة

اسألني أي سؤال! 😊

────────────────────────────────────────────────────────────

👤 You: ايه مصاريف كلية الطب؟

🤖 BNU Assistant:
مصاريف كلية الطب البشري لعام 2025/2026 هي:
• للطلاب المصريين: 155,000 جنيه مصري سنوياً
• للطلاب الوافدين: 6,500 دولار أمريكي سنوياً

مدة الدراسة: 5 سنوات + سنتين امتياز

ملاحظة مهمة: لا توجد زيادة سنوية في المصروفات —
يتم تثبيت الرسوم لكل دفعة طوال مدة الدراسة.

📄 Faculties & Fees Info — fees
[Retrieved 4 relevant sections]

────────────────────────────────────────────────────────────

👤 You: How do I apply to BNU?

🤖 BNU Assistant:
To apply to Benha National University (BNU), follow these steps:

1. Register at: https://admission.bnu.edu.eg/login
2. Activate your account via the email you receive
3. Fill in your personal and academic information
4. Upload required documents:
   - Birth certificate
   - National ID (student + guardian)
   - Academic certificate copy
5. Pay the application fee (1,500 EGP) online
6. Wait for the coordination result notification
7. Upon acceptance email, visit the university within 48 hours
8. Pay tuition fees and submit all original documents

📄 FAQ - General Questions — admission
[Retrieved 5 relevant sections]

────────────────────────────────────────────────────────────

👤 You: هل في منح للطلاب المتفوقين؟

🤖 BNU Assistant:
نعم، توفر جامعة بنها الأهلية منح تفوق للطلاب على النحو التالي:

🏆 منح التفوق الدراسي (للطلاب أثناء الدراسة):
• الطالب الأول في كليته: خصم 25%
• الطالب الثاني: خصم 20%
• الطالب الثالث: خصم 15%
• الطالب الرابع والخامس: خصم 10%
(شرط: ألا يقل التقدير العام عن 90%)

👨‍👩‍👧 تخفيضات إضافية:
• أبناء أعضاء هيئة التدريس: خصم 20%
• الأخ الملتحق حديثاً (وأخوه لا يزال بالجامعة): خصم 10%
• أبناء شهداء القوات المسلحة والشرطة: خصم 100%
• ذوو الهمم: خصم يعادل نسبة الإعاقة

⚠️ قواعد المنح:
- تطبق على الطلاب المصريين فقط
- لا يمكن الجمع بين أكثر من منحة (تُطبق الأعلى)
- لا تشمل الرسوم الإدارية

📄 Faculties & Fees Info — scholarships
[Retrieved 3 relevant sections]

────────────────────────────────────────────────────────────

👤 You: quit
👋 شكراً لاستخدامك مساعد جامعة بنها الأهلية! / Goodbye!
```

---

## 🖥️ Running on Kaggle

```python
# Kaggle Notebook — paste this in a cell:

!pip install -q chromadb sentence-transformers transformers \
             accelerate bitsandbytes pymupdf pdfplumber \
             langdetect colorama pyarabic

# Upload your files via Kaggle dataset, then:
import subprocess
subprocess.run(["python", "main.py", "--no-llm"])  # Fast mode
# OR for full LLM (needs GPU):
subprocess.run(["python", "main.py"])
```

---

## ❓ Troubleshooting

| Problem | Solution |
|---------|----------|
| `No PDF files found` | Add PDF files to `data/` folder, or run as-is (built-in KB works) |
| Out of memory (RAM) | Use `--no-llm` flag for retrieval-only mode |
| Slow on CPU | Use `--no-llm` or switch to a smaller model |
| Wrong language detection | The bot auto-detects; mix of Arabic/English is handled |
| ChromaDB errors | Delete `chroma_db/` folder and run with `--rebuild` |
| Model download fails | Set `HF_HOME=/your/cache/path` or check internet connection |

---

## 📞 Official BNU Links

- 🌐 Website: https://www.bnu.edu.eg
- 📝 Apply: https://admission.bnu.edu.eg/login
- 💰 Fees: https://bnu.edu.eg/ar/page/tuition-fees
- 🏫 Faculties: https://bnu.edu.eg/ar/page/faculties
- 📋 Admission Policies: https://bnu.edu.eg/ar/page/admissions_policies
- 📘 Facebook: https://www.facebook.com/BenhaNationalUniversity
- 🌍 International Students: https://admission.study-in-egypt.gov.eg
