#!/bin/bash
# ════════════════════════════════════════════════════
# BNU Chatbot — One-Click Setup Script
# Run: bash setup.sh
# ════════════════════════════════════════════════════

set -e  # exit on error

echo "╔══════════════════════════════════════════╗"
echo "║  🎓 BNU Chatbot Setup                    ║"
echo "╚══════════════════════════════════════════╝"

# ── Step 1: Python version check ─────────────────
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "✅ Python: $PYTHON_VERSION"

# ── Step 2: Create virtual env ───────────────────
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

echo "🔄 Activating virtual environment..."
source venv/bin/activate

# ── Step 3: Upgrade pip ──────────────────────────
pip install --upgrade pip -q

# ── Step 4: Install PyTorch ──────────────────────
echo "📥 Installing PyTorch..."
if command -v nvidia-smi &> /dev/null; then
    echo "   GPU detected — installing CUDA version"
    pip install torch --index-url https://download.pytorch.org/whl/cu118 -q
    pip install bitsandbytes -q
else
    echo "   No GPU — installing CPU version"
    pip install torch --index-url https://download.pytorch.org/whl/cpu -q
fi

# ── Step 5: Install other dependencies ───────────
echo "📥 Installing dependencies..."
pip install -r requirements.txt -q

# ── Step 6: Create data directory ────────────────
mkdir -p data
mkdir -p model_cache
mkdir -p chroma_db

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║  ✅ Setup Complete!                       ║"
echo "╚══════════════════════════════════════════╝"
echo ""
echo "📂 Next steps:"
echo "   1. Copy your PDF files to the data/ folder:"
echo "      cp /path/to/your/pdfs/*.pdf data/"
echo ""
echo "   2. Run the chatbot:"
echo "      source venv/bin/activate"
echo "      python main.py --no-llm    # Fast mode (CPU)"
echo "      python main.py             # Full mode (GPU recommended)"
echo ""
echo "   3. For Kaggle, copy all .py files to your notebook"
echo "      and run: python main.py --no-llm"
