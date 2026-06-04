"""
utils.py
Utilities: language detection, Arabic normalization, text helpers.
"""

import re
from typing import List
from langdetect import detect, DetectorFactory

# Fix seed for consistent language detection
DetectorFactory.seed = 42


# ─────────────────────────────────────────
# Egyptian Arabic slang normalization map
# ─────────────────────────────────────────
EGYPTIAN_SLANG_MAP = {
    # Greetings
    "ازيك": "كيف حالك",
    "ازيكم": "كيف حالكم",
    "عامل ايه": "كيف حالك",
    "عاملة ايه": "كيف حالك",
    "ايه الاخبار": "ما الأخبار",
    "ايه احوالك": "كيف حالك",
    "هاي": "مرحباً",
    "هاى": "مرحباً",
    "هلو": "مرحباً",

    # Common Egyptian words
    "مصاريف": "رسوم دراسية",
    "فلوس": "أموال",
    "كام": "كم",
    "بكام": "بكم",
    "امتى": "متى",
    "فين": "أين",
    "ازاي": "كيف",
    "ليه": "لماذا",
    "ايه": "ما",
    "إيه": "ما",
    "مش": "لا / ليس",
    "مش عارف": "لا أعرف",
    "مش فاهم": "لا أفهم",
    "عايز": "أريد",
    "عايزة": "أريد",
    "عوزين": "نريد",
    "بيقولوا": "يقولون",
    "بتقولوا": "تقولون",
    "ممكن": "هل يمكن",
    "محتاج": "أحتاج",
    "محتاجة": "أحتاج",
    "لو سمحت": "من فضلك",
    "لو سمحتي": "من فضلك",
    "شكراً": "شكراً",
    "شكرا": "شكراً",
    "متشكر": "شكراً",
    "تمام": "موافق",
    "ماشي": "موافق",
    "اوكي": "موافق",
    "اوكى": "موافق",
    "يعني": "أي أن",
    "بس": "لكن / فقط",
    "زي": "مثل",
    "زى": "مثل",
    "كتير": "كثير",
    "أوي": "جداً",
    "قوي": "جداً",
    "برضو": "أيضاً",
    "برضه": "أيضاً",
    "دلوقتي": "الآن",
    "دلوقت": "الآن",
    "هنا": "هنا",
    "هناك": "هناك",
    "هناه": "هناك",
    "كلية الطب بشري": "كلية الطب البشري",
    "دكتوراة": "دكتوراه",
    "ادخال": "قبول",
    "انتساب": "التحاق",
    "مدخل": "مقبول",
    "اتقبلت": "قُبلت",
}


def normalize_egyptian_arabic(text: str) -> str:
    """Normalize Egyptian dialect slang to Modern Standard Arabic."""
    result = text
    for slang, msa in EGYPTIAN_SLANG_MAP.items():
        result = re.sub(rf'\b{re.escape(slang)}\b', msa, result, flags=re.IGNORECASE)
    return result


def detect_language(text: str) -> str:
    """
    Detect language of input text.
    Returns: 'ar' for Arabic (including Egyptian), 'en' for English, 'mixed' otherwise.
    """
    # Quick heuristic: if text contains Arabic chars, it's Arabic
    arabic_pattern = re.compile(r'[\u0600-\u06FF\u0750-\u077F]')
    english_pattern = re.compile(r'[a-zA-Z]')

    arabic_chars = len(arabic_pattern.findall(text))
    english_chars = len(english_pattern.findall(text))

    if arabic_chars > 0 and arabic_chars >= english_chars:
        return 'ar'
    elif english_chars > arabic_chars:
        try:
            detected = detect(text)
            return detected if detected in ['en', 'ar'] else 'en'
        except Exception:
            return 'en'
    else:
        return 'ar'


def clean_text(text: str) -> str:
    """Clean extracted PDF text."""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove page numbers patterns
    text = re.sub(r'\n\d+\n', '\n', text)
    # Remove repeated dashes/underscores
    text = re.sub(r'[-_]{3,}', '', text)
    # Fix common OCR artifacts
    text = text.replace('ﺔ', 'ة').replace('ﻟ', 'ل').replace('ﺎ', 'ا')
    # Normalize Arabic whitespace
    text = re.sub(r'[\u200b\u200c\u200d\ufeff]', '', text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Split text into overlapping chunks by word count.
    Arabic-aware splitting.
    """
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = ' '.join(words[start:end])
        if len(chunk.strip()) > 30:  # skip tiny chunks
            chunks.append(chunk.strip())
        start += chunk_size - overlap

    return chunks


def format_sources(sources: list[dict]) -> str:
    """Format retrieved sources for display."""
    if not sources:
        return ""
    seen = set()
    formatted = []
    for src in sources:
        label = src.get('source', 'Unknown')
        section = src.get('section', '')
        key = f"{label}_{section}"
        if key not in seen:
            seen.add(key)
            if section:
                formatted.append(f"📄 {label} — {section}")
            else:
                formatted.append(f"📄 {label}")
    return "\n".join(formatted)


def is_greeting(text: str) -> bool:
    """
    Check if the user message is just a greeting.
    Uses word-boundary matching for short words to avoid substring false-positives
    e.g. 'hi' must not match inside 'scholarship' or 'this'.
    """
    greetings = [
        'hi', 'hello', 'hey', 'good morning', 'good evening', 'good afternoon',
        'مرحبا', 'مرحباً', 'السلام عليكم', 'اهلا', 'أهلاً', 'أهلا', 'هلا',
        'صباح الخير', 'مساء الخير', 'صباح النور', 'مساء النور',
        'هاي', 'هلو', 'ازيك', 'ازيكم', 'عامل ايه', 'كيف حالك',
    ]
    text_lower = text.lower().strip()
    if len(text_lower.split()) > 4:
        return False
    for g in greetings:
        if len(g) <= 4:
            # Short words: must match as a whole word, not a substring
            if re.search(r'\b' + re.escape(g) + r'\b', text_lower):
                return True
        else:
            if g in text_lower:
                return True
    return False
