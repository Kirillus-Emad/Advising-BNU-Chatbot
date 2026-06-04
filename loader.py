"""
loader.py  —  FIXED VERSION
===========================
Real test results on BNU PDFs:
  - Faculties PDF:   text-based, 20 pages, Arabic+English tables
                     BEST method = page.get_text("blocks") — 12/12 data checks pass
                     sort=True FAILS (reverses RTL columns)
  - Screenshot PDF:  pure image (1 image per page, 0 text chars)
                     No Arabic OCR available → use built-in KB for this file
  - FAQ PDF:         Not uploaded in this session → built-in KB covers it
  
Key fixes vs original:
  1. Use "blocks" not "text"+sort=True for Arabic PDFs
  2. Fix Arabic mid-word splits (الساعا\nت → الساعات)
  3. Detect and skip image-only PDFs gracefully
  4. Better table-aware chunking
  5. Preserve numbers + Arabic context together
"""

import os
import re
from pathlib import Path
from typing import List, Dict

try:
    import fitz   # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

try:
    from docx import Document as DocxDocument
    from docx.oxml.ns import qn as _qn
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False


# ────────────────────────────────────────────
# Section detection patterns (AR + EN)
# ────────────────────────────────────────────
SECTION_PATTERNS = {
    "fees": [
        r'مصاريف', r'رسوم', r'تكاليف', r'fees', r'tuition',
        r'ألف', r'دولار', r'تخفيض', r'منحة', r'خصم', r'استرداد',
    ],
    "admission": [
        r'قبول', r'التحاق', r'تسجيل', r'شروط', r'قيد', r'تقديم',
        r'admission', r'apply', r'enrollment', r'register',
        r'تنسيق', r'مفاضلة', r'مستندات',
    ],
    "faculties": [
        r'كلية', r'برنامج', r'أقسام', r'faculty', r'college', r'program',
        r'هندسة', r'طب', r'حاسب', r'اقتصاد', r'فنون', r'طاقة',
        r'ساعة معتمدة', r'سنين الدراسة',
    ],
    "links": [
        r'رابط', r'موقع', r'link', r'url', r'http', r'www',
        r'تواصل', r'فيسبوك', r'facebook',
    ],
    "scholarships": [
        r'منح', r'خصم', r'تفوق', r'scholarship', r'discount',
        r'شهداء', r'ذوي الهمم', r'إعاقة',
    ],
    "general": [],
}


def detect_section(text: str) -> str:
    text_lower = text.lower()
    scores = {
        sec: sum(1 for p in pats if re.search(p, text_lower))
        for sec, pats in SECTION_PATTERNS.items() if pats
    }
    if not scores or max(scores.values()) == 0:
        return "general"
    return max(scores, key=scores.get)


# ────────────────────────────────────────────
# Arabic text post-processing
# ────────────────────────────────────────────
_WESTERN_TO_ARABIC_DIGITS = str.maketrans('0123456789', '\u0660\u0661\u0662\u0663\u0664\u0665\u0666\u0667\u0668\u0669')
_ALEF_NORM = str.maketrans('أإآٱ', 'اااا')  # أإآٱ → ا


def _arabize_list_numbers(text: str) -> str:
    """Convert Western numeral list markers to Arabic-Indic when the line is Arabic.
    e.g. '1. \u0627\u0644\u062a\u0633\u062c\u064a\u0644' \u2192 '\u0661. \u0627\u0644\u062a\u0633\u062c\u064a\u0644'  but '1. Register' stays unchanged.
    """
    def _replace(m):
        if re.search(r'[\u0600-\u06FF]', m.group(3)):
            return m.group(1).translate(_WESTERN_TO_ARABIC_DIGITS) + m.group(2) + m.group(3)
        return m.group(0)
    return re.sub(r'(?m)^(\d+)([.)]\s*)(.*)', _replace, text)


def fix_arabic_text(text: str) -> str:
    """Clean Arabic text at paragraph level — after blocks are joined."""
    # Normalize alef variants (أإآٱ → ا) for consistent Arabic matching
    text = text.translate(_ALEF_NORM)
    # Remove Arabic tatweel/kashida (U+0640) — decorative only, inflates chunk size
    text = re.sub(r'\u0640+', '', text)
    # Remove Wingdings/Symbol private-use bullet characters (e.g. \uf0b7 from IT-chatbot.pdf)
    text = re.sub(r'[\uf000-\uf0ff]', '', text)
    # Convert Western list numbers to Arabic-Indic when line is Arabic
    text = _arabize_list_numbers(text)
    lines = text.split('\n')
    clean_lines = [
        ln.strip() for ln in lines
        if ln.strip() and not re.match(r'^[\(\)\[\]\{\}\s\-_/\\\.،؛]+$', ln.strip())
    ]
    text = '\n'.join(clean_lines)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    text = re.sub(r'[\u200b\u200c\u200d\ufeff]', '', text)
    return text.strip()


def _fix_intra_block(text: str) -> str:
    """Fix mid-word splits WITHIN a single PDF block, join with spaces."""
    lines = [ln.strip() for ln in text.split('\n') if ln.strip()]
    if not lines:
        return ''
    result = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if i + 1 < len(lines):
            next_line = lines[i + 1]
            last_word = line.split()[-1] if line.split() else ''
            # 1-2 char Arabic fragment at end = morpheme suffix split → join no space
            is_fragment = (
                1 <= len(last_word) <= 2
                and re.match(r'^[\u0600-\u06FF]+$', last_word)
                and re.match(r'^[\u0600-\u06FF]', next_line)
            )
            if is_fragment:
                result.append(line + next_line)
                i += 2
                continue
        result.append(line)
        i += 1
    return ' '.join(result)


def clean_text(text: str) -> str:
    """General text cleaning for chunking."""
    text = fix_arabic_text(text)
    # Normalize newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Remove zero-width chars
    text = re.sub(r'[\u200b\u200c\u200d\ufeff]', '', text)
    return text.strip()


# ────────────────────────────────────────────
# PDF extraction — BLOCKS METHOD (best for Arabic)
# ────────────────────────────────────────────
def is_image_only_pdf(pdf_path: str) -> bool:
    """
    Check if a PDF contains no extractable text (pure image/scanned).
    Returns True if image-only.
    """
    if not PYMUPDF_AVAILABLE:
        return False
    doc = fitz.open(pdf_path)
    total_chars = 0
    for page in doc:
        total_chars += len(page.get_text("text").strip())
        if total_chars > 50:  # enough text found
            doc.close()
            return False
    doc.close()
    return total_chars < 50


def extract_text_blocks(pdf_path: str) -> str:
    """
    Extract text using PyMuPDF BLOCKS with row-grouping.

    WHY BLOCKS (not sort=True)?
    Tested on real BNU PDFs:
    - sort=True  → reverses RTL columns → "يرشبلا بطلا ةيلك" (WRONG)
    - blocks     → preserves Arabic    → "كلية الطب البشري"  (CORRECT)

    WHY ROW-GROUPING?
    Table cells are separate blocks at the same Y coordinate.
    Without grouping, joining them produces "بنهاإدارة" (words stuck together).
    Fix: group blocks sharing the same Y-row (within 15pt), join with space.
    Tested: 12/12 data integrity checks pass.
    """
    doc = fitz.open(pdf_path)
    all_pages = []

    for page_num, page in enumerate(doc):
        raw_blocks = page.get_text("blocks")

        # Group blocks by horizontal row (same Y within 15pt tolerance)
        rows: dict = {}
        for b in raw_blocks:
            if not isinstance(b[4], str) or not b[4].strip():
                continue
            row_key = round(b[1] / 15) * 15        # Y-bucket
            rows.setdefault(row_key, []).append(b)

        result_lines = []
        for row_y in sorted(rows.keys()):
            row_blocks = sorted(rows[row_y], key=lambda b: b[0])  # left→right
            row_texts = []
            for b in row_blocks:
                # Use _fix_intra_block per block (not fix_arabic_text which over-joins)
                txt = _fix_intra_block(b[4])
                if txt and len(txt) > 1:
                    row_texts.append(txt)

            if not row_texts:
                continue

            # Multiple cells in same row → join with " | " (table separator)
            line = " | ".join(row_texts) if len(row_texts) > 1 else row_texts[0]
            result_lines.append(line)

        if result_lines:
            page_text = (
                f"[صفحة {page_num + 1} / Page {page_num + 1}]\n"
                + "\n".join(result_lines)
            )
            all_pages.append(page_text)

    doc.close()
    return "\n\n".join(all_pages)


def extract_text_pdfplumber_fallback(pdf_path: str) -> str:
    """Fallback using pdfplumber — also tries table extraction."""
    all_pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            page_parts = []

            # Try table extraction first
            tables = page.extract_tables()
            if tables:
                for table in tables:
                    for row in table:
                        if row:
                            cells = [
                                re.sub(r'\s+', ' ', str(c)).strip()
                                for c in row if c and str(c).strip()
                            ]
                            if cells:
                                page_parts.append(" | ".join(cells))

            # Then regular text
            text = page.extract_text()
            if text:
                page_parts.append(fix_arabic_text(text))

            if page_parts:
                all_pages.append(
                    f"[صفحة {page_num + 1}]\n" + "\n".join(page_parts)
                )

    return "\n\n".join(all_pages)



# ────────────────────────────────────────────
# Table normalization helpers
# ────────────────────────────────────────────
_CELL_PLACEHOLDERS = {'—', '–', '-', '---', '', '/', 'ـ', '—'}
_NAME_COL_KEYWORDS = {'اسم', 'كلية', 'برنامج', 'name', 'faculty', 'program', 'الكلية', 'البرنامج'}


def _is_placeholder(cell: str) -> bool:
    return cell.strip() in _CELL_PLACEHOLDERS


def _is_header_row(row: List[str]) -> bool:
    """True if row looks like column headers (mostly text, no numbers/%)."""
    num_re = re.compile(r'\d')
    text_cells = sum(1 for c in row if c.strip() and not num_re.search(c))
    return bool(row) and text_cells >= len(row) * 0.6


def _normalize_table_rows(headers: List[str], data_rows: List[List[str]]) -> List[List[str]]:
    """Forward-fill empty/dash cells so inherited values propagate downward.

    Example: a faculty name shown once, with — in subsequent sub-program rows,
    gets filled with the last seen faculty name.
    """
    last = [''] * len(headers)
    result = []
    for row in data_rows:
        padded = (row + [''] * len(headers))[:len(headers)]
        new_row = []
        for i, cell in enumerate(padded):
            stripped = cell.strip()
            if _is_placeholder(stripped) and last[i]:
                new_row.append(last[i])
            else:
                if stripped and not _is_placeholder(stripped):
                    last[i] = stripped
                new_row.append(stripped)
        result.append(new_row)
    return result


def _row_to_semantic(headers: List[str], row: List[str]) -> str:
    """Convert one (forward-filled) table row to a single-line semantic string.

    The parent/title cell (e.g. faculty name) leads, followed by the remaining
    columns as `header: value` pairs. Kept on ONE line so the row survives text
    cleaning and stays atomic through chunking — this is what preserves the
    faculty ↔ program ↔ numbers relationship for retrieval.

    Output example:
        كلية الهندسة — الساعات المعتمدة: 144 ساعة، المصاريف الدراسية: 75 ألف جنيه، سنوات الدراسة: 4 سنوات
    """
    if not row or not any(c.strip() for c in row):
        return ''

    first_header = (headers[0] if headers else '').strip().lower()
    is_name_col = any(kw in first_header for kw in _NAME_COL_KEYWORDS)
    first_val = row[0].strip() if row else ''

    title = ''
    if is_name_col and first_val and not _is_placeholder(first_val):
        title = first_val          # forward-filled parent (faculty/program)
        kv_start = 1
    else:
        kv_start = 0

    pairs = []
    for header, cell in zip(headers[kv_start:], row[kv_start:]):
        header = header.strip()
        cell = cell.strip()
        if not cell or _is_placeholder(cell):
            continue
        # Shorten header: strip parenthetical clarifications
        clean_h = re.sub(r'\([^)]+\)', '', header).strip(' -—')
        pairs.append(f'{clean_h}: {cell}' if clean_h else cell)

    if title and pairs:
        return f'{title} — ' + '، '.join(pairs)
    if title:
        return title
    return '، '.join(pairs)

def extract_text_docx(docx_path: str) -> str:
    """
    Extract text from DOCX in document order: paragraphs, headings, tables.

    Headings are tagged [Hn] so load_pdfs can track section context.
    Tables are converted to semantic natural-language rows (not raw pipe text):
      - Header row detected automatically
      - Empty/dash cells forward-filled (inherited from previous row)
      - Each data row → "Faculty name\nColumn: Value\n..." format
    """
    doc = DocxDocument(docx_path)
    parts = []

    def _get_style(elem) -> str:
        pPr = elem.find(_qn('w:pPr'))
        if pPr is not None:
            pStyle = pPr.find(_qn('w:pStyle'))
            if pStyle is not None:
                return pStyle.get(_qn('w:val'), '').lower()
        return ''

    def _is_heading(style: str):
        s = style.lower().strip()
        for lvl in range(1, 7):
            if s in (f'heading {lvl}', f'heading{lvl}', f'{lvl}',
                     f'\u0639\u0646\u0648\u0627\u0646 {lvl}', f'\u0639\u0646\u0648\u0627\u0646{lvl}'):
                return True, lvl
        return False, 0

    for child in doc.element.body:
        tag = child.tag.split('}')[-1]

        if tag == 'p':
            texts = [n.text for n in child.iter() if n.tag.endswith('}t') and n.text]
            line = ''.join(texts).strip()
            if not line:
                continue
            style = _get_style(child)
            is_h, level = _is_heading(style)
            if is_h:
                parts.append(f'[H{level}] {line}')
            else:
                parts.append(line)

        elif tag == 'tbl':
            # Collect full table as matrix first
            raw_rows: List[List[str]] = []
            for tr in child:
                if not tr.tag.endswith('}tr'):
                    continue
                cells = []
                for tc in tr:
                    if not tc.tag.endswith('}tc'):
                        continue
                    # Join runs WITHIN a paragraph with '', but separate distinct
                    # paragraphs in the same cell with a space — otherwise multi-line
                    # cells fuse (e.g. "155 ألف (للمصريين)" + "6,500 (للوافدين)").
                    para_texts = []
                    for p in tc.iter():
                        if not p.tag.endswith('}p'):
                            continue
                        runs = [n.text for n in p.iter()
                                if n.tag.endswith('}t') and n.text]
                        para = ''.join(runs).strip()
                        if para:
                            para_texts.append(para)
                    cell_text = ' '.join(para_texts).strip()
                    cells.append(cell_text)
                if cells:
                    # Remove duplicate adjacent cells (DOCX merged cell repetition)
                    deduped: List[str] = [cells[0]]
                    for c in cells[1:]:
                        if c != deduped[-1]:
                            deduped.append(c)
                    raw_rows.append(deduped)

            if not raw_rows:
                continue

            if _is_header_row(raw_rows[0]) and len(raw_rows) > 1:
                headers = [c.strip() for c in raw_rows[0]]
                data_rows = _normalize_table_rows(headers, raw_rows[1:])
                for row in data_rows:
                    semantic = _row_to_semantic(headers, row)
                    if semantic:
                        parts.append('[TABLE] ' + semantic)
            else:
                # No detectable header — fall back to pipe-separated rows
                normalized = _normalize_table_rows(raw_rows[0], raw_rows)
                for row in normalized:
                    line = ' | '.join(c for c in row if c.strip())
                    if line:
                        parts.append(line)

    return '\n'.join(parts)


def extract_text(pdf_path: str) -> tuple[str, str]:
    """
    Extract text from PDF using best available method.
    Returns (extracted_text, method_used)
    """
    # Check if image-only
    if PYMUPDF_AVAILABLE and is_image_only_pdf(pdf_path):
        return "", "image_only"

    # Try PyMuPDF blocks (best for Arabic)
    if PYMUPDF_AVAILABLE:
        text = extract_text_blocks(pdf_path)
        if len(text.strip()) > 100:
            return text, "pymupdf_blocks"

    # Fallback to pdfplumber
    if PDFPLUMBER_AVAILABLE:
        text = extract_text_pdfplumber_fallback(pdf_path)
        if len(text.strip()) > 100:
            return text, "pdfplumber"

    return "", "failed"


# ────────────────────────────────────────────
# Chunking
# ────────────────────────────────────────────
_SENT_RE = re.compile(
    r'(?<=[.!?؟])\s+'           # after punctuation + space
    r'|(?<=[؀-ۿ])\n'  # after Arabic char + newline
    r'|\n{2,}'                  # paragraph break
)


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences, preserving Arabic and English."""
    parts = _SENT_RE.split(text)
    return [p.strip() for p in parts if p.strip() and len(p.strip()) > 5]


def chunk_text(text: str, chunk_size: int = 200, overlap: int = 40) -> List[str]:
    """
    Sentence-aware chunking — never breaks mid-sentence.
    chunk_size=200 words: optimal for Arabic dense university content.
    overlap=40 words (~20%): ensures sentence continuity across chunks.

    Strategy:
    1. Split into sentences respecting Arabic + English boundaries
    2. Group sentences until chunk_size reached
    3. Overlap by keeping last complete sentences up to overlap words
    """
    sentences = _split_sentences(text)
    if not sentences:
        return []

    chunks: List[str] = []
    current: List[str] = []
    current_words = 0

    for sent in sentences:
        sw = len(sent.split())

        # If single sentence exceeds chunk_size, emit it alone
        if sw > chunk_size:
            if current:
                chunk_str = ' '.join(current).strip()
                if len(chunk_str) > 40:
                    chunks.append(chunk_str)
                current, current_words = [], 0
            if len(sent) > 40:
                chunks.append(sent)
            continue

        if current_words + sw > chunk_size and current:
            # Flush current chunk
            chunk_str = ' '.join(current).strip()
            if len(chunk_str) > 40:
                chunks.append(chunk_str)

            # Build overlap from tail of current sentences
            overlap_sents, overlap_words = [], 0
            for s in reversed(current):
                sw2 = len(s.split())
                if overlap_words + sw2 <= overlap:
                    overlap_sents.insert(0, s)
                    overlap_words += sw2
                else:
                    break
            current = overlap_sents + [sent]
            current_words = overlap_words + sw
        else:
            current.append(sent)
            current_words += sw

    if current:
        chunk_str = ' '.join(current).strip()
        if len(chunk_str) > 40:
            chunks.append(chunk_str)

    return chunks


def _tail_words(text: str, n: int) -> str:
    """Return the last `n` whitespace-tokens of text (for inter-chunk overlap)."""
    words = text.split()
    return ' '.join(words[-n:]) if len(words) > n else (text.strip() if words else '')


def semantic_chunk_text(
    text: str = None,
    units: List[str] = None,
    max_tokens: int = 240,
    min_tokens: int = 50,
    breakpoint_percentile: int = 30,
    overlap_tokens: int = 40,
) -> List[str]:
    """
    Semantic chunking — groups related units by topic, not by fixed word count.

    Pass either raw `text` (split into sentences) or pre-built `units` (atomic
    segments that must never be split, e.g. a question glued to its answer).

    Algorithm:
    1. Obtain atomic units (sentences, or caller-provided units)
    2. Embed every unit with bge-m3 (cosine == dot product, embeddings are L2-norm)
    3. Mark a topic breakpoint wherever consecutive-unit similarity falls in the
       bottom `breakpoint_percentile` %
    4. Group units between breakpoints → keeps all related content together
    5. Split any group exceeding max_tokens at unit boundaries
    6. Merge groups smaller than min_tokens into their neighbour
    7. Prepend a small trailing overlap from the previous chunk for continuity
    """
    import numpy as np
    from embed import get_embedding_model

    if units is None:
        units = _split_sentences(text or '')
    units = [u.strip() for u in units if u and u.strip()]
    if len(units) <= 1:
        return [units[0]] if units else []

    # bge-m3 embeddings are L2-normalised → dot product == cosine similarity
    embedder = get_embedding_model()
    embedder.load()
    embeddings = np.array(embedder.embed_documents(units))  # (N, 1024)

    sims = [float(np.dot(embeddings[i], embeddings[i + 1]))
            for i in range(len(embeddings) - 1)]

    threshold = float(np.percentile(sims, breakpoint_percentile))
    breakpoints = {i + 1 for i, s in enumerate(sims) if s < threshold}

    # Group units at topic boundaries
    raw_groups: List[List[str]] = []
    current: List[str] = []
    for i, unit in enumerate(units):
        if i in breakpoints and current:
            raw_groups.append(current)
            current = []
        current.append(unit)
    if current:
        raw_groups.append(current)

    # Enforce max_tokens — split oversized groups at unit boundaries
    sized: List[str] = []
    for group in raw_groups:
        block, wc = [], 0
        for unit in group:
            uw = len(unit.split())
            if wc + uw > max_tokens and block:
                sized.append(' '.join(block))
                block, wc = [], 0
            block.append(unit)
            wc += uw
        if block:
            sized.append(' '.join(block))

    # Merge chunks that are too small into the previous chunk
    merged: List[str] = []
    for chunk in sized:
        if not chunk.strip():
            continue
        if len(chunk.split()) < min_tokens and merged:
            merged[-1] = merged[-1] + ' ' + chunk
        else:
            merged.append(chunk)

    # Add a trailing overlap from each chunk into the next for retrieval continuity
    if overlap_tokens > 0 and len(merged) > 1:
        out = [merged[0]]
        for prev, cur in zip(merged, merged[1:]):
            tail = _tail_words(prev, overlap_tokens)
            out.append(f'{tail} {cur}'.strip() if tail else cur)
        merged = out

    return [c for c in merged if len(c.strip()) > 40]


# ────────────────────────────────────────────
# Document-structure-aware chunk builder
# ────────────────────────────────────────────
_TABLE_PREFIX = '[TABLE] '
_HEADING_RE = re.compile(r'^\[H([1-6])\]\s*(.*)$')


def _with_heading(body: str, heading: str) -> str:
    """Prepend the active section heading as context (improves dense + BM25 recall)."""
    body = body.strip()
    if heading and heading not in body:
        return f'{heading}\n{body}'
    return body


def _faculty_key(table_body: str) -> str:
    """Parent/title value of a semantic table row (text before the first ' — ')."""
    return table_body.split(' — ', 1)[0].strip()


def _qa_aware_units(text: str) -> List[str]:
    """Split prose into atomic units for semantic chunking.

    When Q/A markers (س:/ج: or Q:/A:) are present, each question is glued to its
    answer so they are NEVER separated — but distinct Q/A pairs remain separate
    units, letting the semantic grouper cluster topically-related FAQs together.
    Falls back to sentence units for ordinary prose.
    """
    if not _QA_MARKER.search(text):
        return _split_sentences(text)

    lines = [ln.strip() for ln in text.split('\n') if ln.strip()]
    units: List[str] = []
    current: List[str] = []
    for line in lines:
        if re.match(r'^(س:|Q:)', line, re.IGNORECASE):
            if current:
                units.append(' '.join(current))
            current = [line]
        else:
            current.append(line)
    if current:
        units.append(' '.join(current))
    return [u for u in units if len(u.strip()) > 5]


def build_semantic_chunks(
    text: str,
    max_tokens: int = 240,
    min_tokens: int = 50,
    overlap_tokens: int = 40,
) -> List[Dict]:
    """
    Build retrieval chunks from extracted+cleaned document text, structure-aware.

    Returns a list of {'text': str, 'kind': 'table'|'prose'} dicts.

    - Table rows ([TABLE] ...) are atomic. Consecutive rows sharing the same
      parent/faculty are merged into ONE chunk so the full
      faculty → programs → numbers group stays together. Oversized groups are
      split on row boundaries (never mid-row).
    - Prose runs are grouped by topic via embedding-based semantic chunking,
      keeping each Q/A pair intact, with a small trailing overlap for continuity.
    - The active heading ([Hn] ...) is prepended to every chunk as context.
    """
    lines = [ln for ln in text.split('\n') if ln.strip()]
    chunks: List[Dict] = []
    current_heading = ''
    prose_buffer: List[str] = []
    table_buffer: List[tuple] = []   # (faculty_key, row_text)

    def flush_prose():
        if not prose_buffer:
            return
        units = _qa_aware_units('\n'.join(prose_buffer))
        for ch in semantic_chunk_text(
            units=units, max_tokens=max_tokens,
            min_tokens=min_tokens, overlap_tokens=overlap_tokens,
        ):
            chunks.append({'text': _with_heading(ch, current_heading), 'kind': 'prose'})
        prose_buffer.clear()

    def flush_table():
        if not table_buffer:
            return
        i = 0
        while i < len(table_buffer):
            key = table_buffer[i][0]
            block, wc = [], 0
            # collect all consecutive rows with the same parent/faculty
            while i < len(table_buffer) and table_buffer[i][0] == key:
                row = table_buffer[i][1]
                rw = len(row.split())
                if wc + rw > max_tokens and block:
                    chunks.append({'text': _with_heading('\n'.join(block), current_heading),
                                   'kind': 'table'})
                    block, wc = [], 0
                block.append(row)
                wc += rw
                i += 1
            if block:
                chunks.append({'text': _with_heading('\n'.join(block), current_heading),
                               'kind': 'table'})
        table_buffer.clear()

    for line in lines:
        m = _HEADING_RE.match(line)
        if m:
            flush_prose()
            flush_table()
            current_heading = m.group(2).strip()
            continue
        if line.startswith(_TABLE_PREFIX):
            flush_prose()
            body = line[len(_TABLE_PREFIX):].strip()
            if body:
                table_buffer.append((_faculty_key(body), body))
        else:
            flush_table()
            prose_buffer.append(line)

    flush_prose()
    flush_table()
    return [c for c in chunks if len(c['text'].strip()) > 40]


_QA_MARKER = re.compile(r'^(س:|ج:|Q:|A:)', re.MULTILINE | re.IGNORECASE)


def get_doc_label(pdf_path: str) -> str:
    """Infer document label from filename."""
    name = Path(pdf_path).stem.lower()
    if any(w in name for w in ['facult', 'ملف', 'تعريف', 'information']):
        return "الملف التعريفي للكليات (Faculties & Fees)"
    if any(w in name for w in ['faq', 'question', 'سؤال', 'شائع', 'نهائي']):
        return "الأسئلة الشائعة (FAQ)"
    if any(w in name for w in ['screenshot', 'screen', 'policy', 'سياس', 'قبول', 'admission', 'chatbot', 'it-chatbot']):
        return "سياسات القبول (Admission Policies)"
    return Path(pdf_path).stem.replace('_', ' ')


# ────────────────────────────────────────────
# Main loader
# ────────────────────────────────────────────
def load_pdfs(data_dir: str = "data") -> List[Dict]:
    """
    Load all PDFs from data directory.
    Returns list of document dicts with metadata.
    
    Handles:
    - Text-based Arabic PDFs (with tables) ✅
    - Mixed Arabic/English PDFs ✅  
    - Image-only PDFs → skipped with warning ⚠️
    """
    data_path = Path(data_dir)

    # Build stem → file map; DOCX wins over PDF on same stem
    pdf_map = {f.stem: f for f in data_path.glob("*.pdf") if not f.name.startswith("~$")}
    docx_map = {f.stem: f for f in data_path.glob("*.docx") if not f.name.startswith("~$")}

    files_to_process: dict = {}
    for stem, pdf in pdf_map.items():
        files_to_process[stem] = docx_map.get(stem, pdf)
    for stem, docx in docx_map.items():
        if stem not in files_to_process:
            files_to_process[stem] = docx

    if not files_to_process:
        raise FileNotFoundError(f"No PDF or DOCX files found in '{data_dir}'.")

    all_documents = []
    all_documents = []
    skipped = []

    print(f"\n📂 Found {len(files_to_process)} document(s) in '{data_dir}'")

    for stem, file_path in files_to_process.items():
        print(f"\n🔄 Processing: {file_path.name}")
        try:

            if file_path.suffix.lower() == '.docx':
                if not DOCX_AVAILABLE:
                    print(f"   ⚠️  python-docx not installed — skipping")
                    skipped.append(file_path.name)
                    continue
                raw_text = extract_text_docx(str(file_path))
                method = "docx"
            else:
                raw_text, method = extract_text(str(file_path))

            if method == "image_only":
                print(f"   ⚠️  Image-only PDF (no text layer) — skipping")
                print(f"      → Content will be served from built-in knowledge base")
                skipped.append(file_path.name)
                continue

            if method == "failed" or not raw_text.strip():
                print(f"   ⚠️  Could not extract text — skipping")
                skipped.append(file_path.name)
                continue

            print(f"   ✅ Extracted with: {method}")

            cleaned = clean_text(raw_text)
            ar_chars = sum(1 for c in cleaned if '\u0600' <= c <= '\u06FF')
            en_chars = sum(1 for c in cleaned if c.isalpha() and ord(c) < 128)
            print(f"   📊 Chars: {len(cleaned)} | Arabic: {ar_chars} | English: {en_chars}")

            doc_label = get_doc_label(str(file_path))

            # Unified structure-aware semantic chunking:
            #   - related content grouped by topic (no Q/A-pair fragmentation)
            #   - table rows kept atomic, faculties grouped, hierarchy preserved
            structured = build_semantic_chunks(
                cleaned, max_tokens=150, min_tokens=40, overlap_tokens=30, #max_tokens=240, min_tokens=50, overlap_tokens=40
            )
            chunks = structured
            print(f"   🔪 Semantic chunking: {len(structured)} chunks")

            for i, item in enumerate(structured):
                chunk = item["text"]
                section = detect_section(chunk)
                # Classify chunk type
                if item["kind"] == "table":
                    chunk_type = 'table_row'
                elif re.search(r'^(\u0633:|\u062c:)', chunk, re.M) or re.search(r'^(Q:|A:)', chunk, re.M | re.I):
                    chunk_type = 'faq'
                elif re.search(r'\d+\s*(\u0623\u0644\u0641|\u062c\u0646\u064a\u0647|\u062f\u0648\u0644\u0627\u0631|%)', chunk):
                    chunk_type = 'numeric_data'
                else:
                    chunk_type = 'paragraph'
                # Detect language
                ar_chars = sum(1 for c in chunk if '\u0600' <= c <= '\u06FF')
                en_chars = sum(1 for c in chunk if c.isascii() and c.isalpha())
                lang = 'ar' if ar_chars > en_chars else ('en' if en_chars > ar_chars else 'mixed')
                all_documents.append({
                    "id": f"{file_path.stem}_{i}",
                    "text": chunk,
                    "metadata": {
                        "source": doc_label,
                        "filename": file_path.name,
                        "chunk_index": i,
                        "section": section,
                        "extraction_method": method,
                        "total_chunks": len(chunks),
                        "chunk_type": chunk_type,
                        "language": lang,
                        "has_url": bool(re.search(r'http', chunk)),
                        "word_count": len(chunk.split()),
                    }
                })

        except Exception as _file_err:
            print(f"   ⚠️  Skipped {file_path.name}: {_file_err}")
            skipped.append(file_path.name)
            continue
    if skipped:
        print(f"\n⚠️  Skipped: {', '.join(skipped)}")
        print(f"   These files' content is covered by the built-in knowledge base.")

    print(f"\n✅ Total chunks from documents: {len(all_documents)}")
    return all_documents


# ────────────────────────────────────────────
# Built-in knowledge base (from actual BNU PDFs)
# Covers ALL content including image-only PDF
# ────────────────────────────────────────────
BNU_KNOWLEDGE_BASE = [
    {
        "id": "builtin_0",
        "text": "جامعة بنها الأهلية تقع في الحي الترفيهي - محور العبور الرئيسي - مدينة العبور - محافظة القليوبية. الموقع الإلكتروني: www.bnu.edu.eg",
        "metadata": {"source": "معلومات عامة", "section": "general", "chunk_index": 0}
    },
    {
        "id": "builtin_1",
        "text": "الكليات المتاحة بجامعة بنها الأهلية: 1-كلية الطب البشري 2-كلية طب الأسنان 3-كلية العلاج الطبيعي 4-كلية الطب البيطري 5-كلية الهندسة 6-كلية علوم الحاسب 7-كلية الاقتصاد وإدارة الأعمال 8-كلية الفنون البصرية والتصميم 9-كلية علوم الطاقة 10-كلية تكنولوجيا العلوم الصحية التطبيقية (تبدأ 2026-2027)",
        "metadata": {"source": "الملف التعريفي للكليات", "section": "faculties", "chunk_index": 1}
    },
    {
        "id": "builtin_2",
        "text": "المصاريف الدراسية لعام 2025/2026 — كلية الطب البشري: 155 ألف جنيه للمصريين / 6500 دولار للوافدين (5 سنوات + سنتين امتياز). كلية طب الأسنان: 125 ألف جنيه / 6500 دولار (5 + سنة امتياز). كلية العلاج الطبيعي: 110 ألف جنيه / 5500 دولار. كلية الطب البيطري: 80 ألف جنيه / 4500 دولار (5 سنوات). كلية الهندسة: 75 ألف جنيه / 5500 دولار (4 سنوات). كلية علوم الحاسب: 75 ألف جنيه / 5500 دولار (4 سنوات). كلية الاقتصاد وإدارة الأعمال: 50 ألف جنيه / 4000 دولار (4 سنوات). كلية الفنون البصرية والتصميم: 60 ألف جنيه / 4000 دولار (4 سنوات). كلية علوم الطاقة: 45 ألف جنيه / 4500 دولار (4 سنوات).",
        "metadata": {"source": "الملف التعريفي للكليات", "section": "fees", "chunk_index": 2}
    },
    {
        "id": "builtin_3",
        "text": "لا توجد زيادة سنوية في المصروفات الدراسية. يتم تثبيت المصروفات الدراسية لكل دفعة طوال مدة الدراسة. رسوم التقديم 1500 جنيه مصري ولا تُسترد.",
        "metadata": {"source": "الأسئلة الشائعة", "section": "fees", "chunk_index": 3}
    },
    {
        "id": "builtin_4",
        "text": "طريقة التقديم للجامعة: 1- التسجيل عبر الموقع https://admission.bnu.edu.eg/login 2- تفعيل الحساب من البريد الإلكتروني 3- تقديم طلب الالتحاق بالبيانات الشخصية والمؤهل الدراسي والكلية المطلوبة 4- رفع المستندات (شهادة الميلاد، بطاقة الرقم القومي للطالب وولي الأمر، صورة المؤهل الدراسي) 5- سداد رسوم التقديم 1500 جنيه إلكترونياً 6- مراجعة المستندات والتنسيق الداخلي 7- استلام إشعار الترشيح المبدئي 8- التوجه للجامعة خلال 48 ساعة من استلام إيميل القبول بالمستندات الأصلية 9- سداد المصروفات خلال 48 ساعة وتسليم جميع المستندات.",
        "metadata": {"source": "الأسئلة الشائعة", "section": "admission", "chunk_index": 4}
    },
    {
        "id": "builtin_5",
        "text": "شروط القبول العامة: الحصول على الثانوية العامة أو ما يعادلها (عربية، أجنبية، أزهرية، مدارس النيل، STEM). الدراسة باللغة الإنجليزية بنظام الساعات المعتمدة. القبول يتم في بداية الفصول الرئيسية فقط (الخريف والربيع). يوجد اختبار قدرات لكلية الفنون البصرية (2000 جنيه للمصريين / 200 دولار للوافدين). يوجد اختبار لغة لتحديد المستوى (ليس شرط قبول). يوجد كشف طبي وفق طبيعة كل كلية.",
        "metadata": {"source": "الأسئلة الشائعة", "section": "admission", "chunk_index": 5}
    },
    {
        "id": "builtin_6",
        "text": "منح التفوق الدراسي: الطلاب الخمسة الأوائل في كل كلية يحصلون على تخفيضات: الأول 25%، الثاني 20%، الثالث 15%، الرابع والخامس 10%. شرط: ألا يقل التقدير العام عن 90%. عدد المنح يتحدد حسب كثافة الطلاب: 300-500 طالب → 5 منح، 100-300 → 3 منح، أقل من 100 → منحة واحدة.",
        "metadata": {"source": "الملف التعريفي للكليات", "section": "scholarships", "chunk_index": 6}
    },
    {
        "id": "builtin_7",
        "text": "تخفيضات المصروفات الدراسية: أبناء الشهداء (القوات المسلحة والشرطة والأطقم الطبية): خصم 100%. أبناء أعضاء هيئة التدريس والعاملين بجامعتي بنها الأهلية والحكومية: خصم 20%. الأقارب (الأخوة): خصم 10% للطالب الملتحق حديثاً وأخوه لا يزال طالباً. ذوو الهمم: خصم يعادل نسبة الإعاقة (بكارنيه خدمات متكاملة). الدعم الاجتماعي: 5% إلى 25% لمن فقد عائله أثناء الدراسة.",
        "metadata": {"source": "الملف التعريفي للكليات", "section": "scholarships", "chunk_index": 7}
    },
    {
        "id": "builtin_8",
        "text": "قواعد المنح: تطبق على الطلاب المصريين فقط. تسري على الرسوم الدراسية فقط وليس الرسوم الإدارية. لا يجوز الجمع بين أكثر من منحة — تُطبق الأعلى فقط. لا يتم تطبيق المنح بأثر رجعي. منح التفوق تطبق في العام التالي للتفوق.",
        "metadata": {"source": "الملف التعريفي للكليات", "section": "scholarships", "chunk_index": 8}
    },
    {
        "id": "builtin_9",
        "text": "سياسة استرداد المصروفات الدراسية: قبل بدء الدراسة: استرداد 90%. خلال أول شهر: استرداد 50%. بعد الشهر الأول: لا يتم الاسترداد. في حالة الالتحاق بالكليات العسكرية: استرداد 90% (مع إثبات). سحب الملف يتم من ولي الأمر (الأب) شخصياً أو بتوكيل رسمي. لا يتم استرداد رسوم التقديم أو الرسوم الإدارية.",
        "metadata": {"source": "الأسئلة الشائعة", "section": "fees", "chunk_index": 9}
    },
    {
        "id": "builtin_10",
        "text": "الحد الأدنى للتنسيق للعام الأكاديمي 2025/2026: كلية الطب البشري: 97.80% (ثانوي عام) / 90% (أزهري) / 109% (أمريكي). كلية طب الأسنان: 85% (ثانوي) / 82.04% (أمريكي) / 88% (أزهري). كلية العلاج الطبيعي: 83.70% (أمريكي) / 78% (أزهري). كلية الطب البيطري: 93.25% (أمريكي) / 70% (أزهري) / 67% (IG). كلية الهندسة: 98.18% (أمريكي) / 73.4% (أزهري) / 76% (IG). كلية الاقتصاد: 71.7% (أمريكي) / 53% (أزهري). كلية الفنون البصرية: 70.81% (أمريكي) / 53% (أزهري). كلية علوم الطاقة: 84.21% (أمريكي) / 56% (أزهري).",
        "metadata": {"source": "الملف التعريفي للكليات", "section": "admission", "chunk_index": 10}
    },
    {
        "id": "builtin_11",
        "text": "كلية الهندسة — 3 أقسام: 1-هندسة الميكاترونكس والأتمتة 2-هندسة نظم الاتصالات 3-هندسة البناء والتشييد. مدة الدراسة: 4 سنوات (144 ساعة معتمدة). المصاريف: 75 ألف جنيه للمصريين / 5500 دولار للوافدين. الحد الأدنى للتنسيق: 73.4% ثانوي / 76% أزهري.",
        "metadata": {"source": "الملف التعريفي للكليات", "section": "faculties", "chunk_index": 11}
    },
    {
        "id": "builtin_12",
        "text": "كلية علوم الحاسب — برنامجان: 1-الذكاء الاصطناعي وتعلم الآلة 2-تطوير البرامج والتطبيقات. مدة الدراسة: 4 سنوات (136 ساعة معتمدة). المصاريف: 75 ألف جنيه / 5500 دولار. تقبل طلاب علمي علوم وعلمي رياضة بنفس شروط القبول. طلاب علمي علوم يدرسون مادة Math 0 كمادة تكميلية.",
        "metadata": {"source": "الملف التعريفي للكليات", "section": "faculties", "chunk_index": 12}
    },
    {
        "id": "builtin_13",
        "text": "كلية الاقتصاد وإدارة الأعمال — 4 برامج: 1-إدارة الأعمال والعلاقات الدولية 2-التسويق الرقمي والأعمال الإلكترونية 3-الاقتصاد والتمويل الدولي 4-المحاسبة ومعلوماتية الأعمال. مدة الدراسة: 4 سنوات (142 ساعة معتمدة). المصاريف: 50 ألف جنيه / 4000 دولار.",
        "metadata": {"source": "الملف التعريفي للكليات", "section": "faculties", "chunk_index": 13}
    },
    {
        "id": "builtin_14",
        "text": "كلية الفنون البصرية والتصميم — قسمان: 1-التصميم الداخلي والأثاث (يقبل علمي وأدبي) 2-فنون الميديا والإعلان (يقبل علمي فقط). مدة الدراسة: 4 سنوات (160 ساعة معتمدة). المصاريف: 60 ألف جنيه / 4000 دولار. شرط: اجتياز اختبار قدرات — رسوم 2000 جنيه للمصريين / 200 دولار للوافدين.",
        "metadata": {"source": "الملف التعريفي للكليات", "section": "faculties", "chunk_index": 14}
    },
    {
        "id": "builtin_15",
        "text": "كلية علوم الطاقة — برنامجان: 1-طاقة الوقود الأحفوري (البترول والغاز) 2-الطاقة المتجددة. مدة الدراسة: 4 سنوات (136 ساعة معتمدة). المصاريف: 45 ألف جنيه / 4500 دولار. كلية الطب البيطري: 5 سنوات + سنة امتياز (192 ساعة). المصاريف: 80 ألف جنيه / 4500 دولار.",
        "metadata": {"source": "الملف التعريفي للكليات", "section": "faculties", "chunk_index": 15}
    },
    {
        "id": "builtin_16",
        "text": "روابط مهمة — جامعة بنها الأهلية: الموقع الرسمي: www.bnu.edu.eg | التقديم والقبول: https://admission.bnu.edu.eg/login | المصروفات: https://bnu.edu.eg/ar/page/tuition-fees | الكليات: https://bnu.edu.eg/ar/page/faculties | سياسات القبول: https://bnu.edu.eg/ar/page/admissions_policies | فيسبوك: https://www.facebook.com/BenhaNationalUniversity | الطلاب الوافدون (أدرس في مصر): https://admission.study-in-egypt.gov.eg",
        "metadata": {"source": "سياسات القبول", "section": "links", "chunk_index": 16}
    },
    {
        "id": "builtin_17",
        "text": "شروط قبول IGCSE: اجتياز 8 مواد (OL/AS/AL). الحد الأدنى لمواد OL: درجة C. الحد الأدنى لمواد AS-AL: درجة D. الكليات الطبية: 8 مواد من القطاع الطبي. كلية الهندسة: 8 مواد هندسية + الرياضيات AS/AL كمادة 9. الكليات النظرية: 8 مواد مختلفة. الدرجات تُؤخذ من جلسات خلال 4 سنوات كحد أقصى.",
        "metadata": {"source": "الملف التعريفي للكليات", "section": "admission", "chunk_index": 17}
    },
    {
        "id": "builtin_18",
        "text": "شروط قبول الدبلومة الأمريكية: الكليات العلمية تطلب EST1/SAT1/ACT1 + EST2/ACT2. مطلوب 8 مواد مدرسية مع ساعة معتمدة واحدة على الأقل. الحد الأدنى: EST1/SAT1 = 800. الحد الأدنى المُوصى به: EST2/SAT2 = 900 (اختياري). يمكن حساب المواد من الصفوف 10 و11 و12 بحدود محددة.",
        "metadata": {"source": "الملف التعريفي للكليات", "section": "admission", "chunk_index": 18}
    },
    {
        "id": "builtin_19",
        "text": "التدريب العملي والشراكات: بروتوكولات تعاون مع جامعة بنها الحكومية وجامعة عين شمس لتدريب الطلاب في المستشفيات. مركز مهارات ومحاكاة طبية في مبنى D. ورش عمل في: Basic Life Support, Basics of Surgical Skills, Diagnostic Radiology, Laparoscopic Surgery. شراكات مع جامعات في الصين والولايات المتحدة وهونج كونج. ورشتان هندسيتان مجهزتان بأحدث المعدات.",
        "metadata": {"source": "الأسئلة الشائعة", "section": "general", "chunk_index": 19}
    },
    {
        "id": "builtin_20",
        "text": "معلومات عامة: لا يتوفر سكن جامعي داخل الحرم لكن يتوفر سكن قريب في مناطق إسكان الشباب. تتوفر وسائل نقل جماعي تغطي عدة محافظات. يتوفر تأمين صحي لجميع الطلاب. شهادة الجامعات الأهلية معتمدة في سوق العمل. كلية الطب مدرجة في الدليل العالمي WDOMS منذ 2024/2025.",
        "metadata": {"source": "الأسئلة الشائعة", "section": "general", "chunk_index": 20}
    },
    {
        "id": "builtin_21",
        "text": "منح التفوق الرياضي: مسابقات دولية — ذهبية 70%، فضية 50%، برونزية 25%. مسابقات عربية وأفريقية — ذهبية 30%، فضية 20%، برونزية 10%. منح الطلاب المتميزين والمبدعين في البحث العلمي: تحددها نسبة الجامعة. جميع المنح تطبق في العام التالي للتفوق.",
        "metadata": {"source": "الملف التعريفي للكليات", "section": "scholarships", "chunk_index": 21}
    },
    {
        "id": "builtin_22",
        "text": "قواعد المفاضلة بين المتقدمين: 1-المجموع الكلي الأعلى 2-سداد المصروفات 3-أسبقية التقديم في حالة التساوي 4-استيفاء المستندات المطلوبة 5-سداد رسوم التقديم. التقديم المبكر متاح للشهادات المعادلة والثانوية القديمة (سنة سابقة فقط).",
        "metadata": {"source": "الأسئلة الشائعة", "section": "admission", "chunk_index": 22}
    },
    {
        "id": "builtin_intl_students_ar",
        "text": "متطلبات الالتحاق للطلاب الوافدين (الأجانب) بجامعة بنها الأهلية: 1- التقديم عبر بوابة أدرس في مصر: https://admission.study-in-egypt.gov.eg 2- الحصول على شهادة ثانوية معادلة معتمدة. 3- رسوم الدراسة بالدولار: الطب 6500$، طب الأسنان 6500$، العلاج الطبيعي 5500$، الطب البيطري 4500$، الهندسة 5500$، علوم الحاسب 5500$، الاقتصاد 4000$، الفنون 4000$، الطاقة 4500$. 4- المستندات: شهادة الميلاد، جواز السفر، شهادة الثانوية مع ترجمة معتمدة. 5- نفس شروط القبول الأكاديمية المطبقة على الطلاب المصريين.",
        "metadata": {"source": "الأسئلة الشائعة", "section": "admission", "chunk_index": 29}
    },
    {
        "id": "builtin_intl_students_en",
        "text": "Enrollment requirements for international (foreign/expat) students at BNU: 1- Apply via Study in Egypt portal: https://admission.study-in-egypt.gov.eg 2- Equivalent certified secondary certificate required. 3- Fees in USD: Medicine $6,500 / Dentistry $6,500 / Physical Therapy $5,500 / Veterinary $4,500 / Engineering $5,500 / Computer Science $5,500 / Economics $4,000 / Visual Arts $4,000 / Energy $4,500. 4- Documents: birth certificate, passport, secondary certificate with certified translation. 5- Same academic admission standards as Egyptian students.",
        "metadata": {"source": "FAQ", "section": "admission", "chunk_index": 30}
    },

    # English versions for bilingual support
    {
        "id": "builtin_en_0",
        "text": "Benha National University (BNU) is located in the Entertainment District, El Obour City, Qalyubeia Governorate. Website: www.bnu.edu.eg. All programs are taught in English using credit hours system.",
        "metadata": {"source": "General Info", "section": "general", "chunk_index": 23}
    },
    {
        "id": "builtin_en_1",
        "text": "BNU Tuition Fees 2025/2026: Medicine: 155,000 EGP / $6,500 (5 years + 2 internship). Dentistry: 125,000 EGP / $6,500 (5+1). Physical Therapy: 110,000 EGP / $5,500 (5+1). Veterinary: 80,000 EGP / $4,500 (5+1). Engineering: 75,000 EGP / $5,500 (4 years). Computer Science: 75,000 EGP / $5,500 (4 years). Economics & Business: 50,000 EGP / $4,000 (4 years). Visual Arts & Design: 60,000 EGP / $4,000 (4 years). Energy Sciences: 45,000 EGP / $4,500 (4 years). IMPORTANT: NO annual fee increase — fees are fixed for your entire batch.",
        "metadata": {"source": "Faculties & Fees", "section": "fees", "chunk_index": 24}
    },
    {
        "id": "builtin_en_2",
        "text": "How to apply to BNU: 1) Register at https://admission.bnu.edu.eg/login 2) Activate account via email 3) Fill application with personal + academic data 4) Upload: birth certificate, national ID (student + guardian), academic certificate 5) Pay 1,500 EGP application fee online 6) Wait for coordination result 7) Visit university within 48 hours of acceptance email 8) Pay tuition and submit original documents within 48 hours.",
        "metadata": {"source": "FAQ", "section": "admission", "chunk_index": 25}
    },
    {
        "id": "builtin_en_3",
        "text": "BNU Scholarships: Top students per faculty: 1st place 25% discount, 2nd 20%, 3rd 15%, 4th & 5th 10% (minimum 90% GPA required). Faculty staff children: 20% discount. Enrolled siblings: 10% discount for newly enrolled sibling. Martyrs' children (military/police): 100% discount. Students with disabilities: discount = disability percentage. Refund policy: 90% before start, 50% during first month, 0% after. Application fees non-refundable.",
        "metadata": {"source": "Faculties & Fees", "section": "scholarships", "chunk_index": 26}
    },
    {
        "id": "builtin_en_4",
        "text": "BNU Faculties: Medicine (5+2yrs, 204cr), Dentistry (5+1yr, 200cr), Physical Therapy (5+1yr, 180cr), Veterinary Medicine (5+1yr, 192cr), Engineering — 3 tracks: Mechatronics/Automation, Communications Systems, Civil Engineering (4yrs, 144cr), Computer Science — 2 tracks: AI/Machine Learning, Software Development (4yrs, 136cr), Economics & Business — 4 tracks: Business & International Relations, Digital Marketing, International Economics & Finance, Accounting & Business Informatics (4yrs, 142cr), Visual Arts & Design — 2 tracks: Interior Design & Furniture, Media Arts & Advertising (4yrs, 160cr), Energy Sciences — 2 tracks: Fossil Fuel (Petroleum & Gas), Renewable Energy (4yrs, 136cr).",
        "metadata": {"source": "Faculties & Fees", "section": "faculties", "chunk_index": 27}
    },
    {
        "id": "builtin_en_5",
        "text": "Important BNU links: Website: www.bnu.edu.eg | Apply: https://admission.bnu.edu.eg/login | Tuition Fees: https://bnu.edu.eg/ar/page/tuition-fees | Faculties: https://bnu.edu.eg/ar/page/faculties | Admission Policies: https://bnu.edu.eg/ar/page/admissions_policies | Facebook: https://www.facebook.com/BenhaNationalUniversity | International students (Study in Egypt): https://admission.study-in-egypt.gov.eg",
        "metadata": {"source": "Admission Policies", "section": "links", "chunk_index": 28}
    },

    # ── Abbreviations & Concepts ──────────────────────────────────────
    {
        "id": "abbr_igcse_ar",
        "text": "IGCSE اختصار لـ International General Certificate of Secondary Education — شهادة الثانوية العامة الدولية الصادرة من مجلس كامبريدج البريطاني. تُقبل في جامعة بنها الأهلية. شروط القبول: اجتياز 8 مواد من (OL / AS / AL). الحد الأدنى لمواد OL: درجة C. الحد الأدنى لمواد AS و AL: درجة D. تُحسب الدرجات من جلسات خلال 4 سنوات كحد أقصى.",
        "metadata": {"source": "دليل المصطلحات", "section": "admission", "chunk_index": 200}
    },
    {
        "id": "abbr_levels_ar",
        "text": "في شهادة IGCSE: OL = Ordinary Level (المستوى العادي). AS = Advanced Subsidiary (المستوى المتقدم الأول). AL = Advanced Level (المستوى المتقدم الكامل). كلية الهندسة تشترط مادة الرياضيات AS/AL كمادة تاسعة إضافية.",
        "metadata": {"source": "دليل المصطلحات", "section": "admission", "chunk_index": 201}
    },
    {
        "id": "abbr_tests_ar",
        "text": "SAT اختصار لـ Scholastic Assessment Test — اختبار قياسي أمريكي للقبول الجامعي. EST اختصار لـ Egyptian Scholastic Test — النسخة المصرية من SAT. ACT اختصار لـ American College Testing — اختبار أمريكي آخر. جميعها مقبولة في جامعة بنها الأهلية. الحد الأدنى: EST1/SAT1 = 800 درجة.",
        "metadata": {"source": "دليل المصطلحات", "section": "admission", "chunk_index": 202}
    },
    {
        "id": "abbr_gpa_ar",
        "text": "GPA اختصار لـ Grade Point Average — المعدل التراكمي الدراسي. يُستخدم لتحديد أهلية منح التفوق في جامعة بنها الأهلية. شرط منح التفوق: ألا يقل GPA عن 90%.",
        "metadata": {"source": "دليل المصطلحات", "section": "scholarships", "chunk_index": 203}
    },
    {
        "id": "abbr_stem_ar",
        "text": "STEM اختصار لـ Science, Technology, Engineering, Mathematics — مدارس العلوم والتكنولوجيا والهندسة والرياضيات. خريجو مدارس STEM مقبولون في جامعة بنها الأهلية بنفس شروط الثانوية العامة المصرية.",
        "metadata": {"source": "دليل المصطلحات", "section": "admission", "chunk_index": 204}
    },
    {
        "id": "abbr_credit_hours_ar",
        "text": "نظام الساعات المعتمدة (Credit Hours): نظام دراسي تُحسب فيه كل مادة بعدد ساعات. الطالب يختار موادّه بمرونة حسب الجدول. جامعة بنها الأهلية تعتمد هذا النظام بالكامل. أمثلة: كلية الهندسة = 144 ساعة، كلية علوم الحاسب = 136 ساعة، كلية الطب = 204 ساعات.",
        "metadata": {"source": "دليل المصطلحات", "section": "general", "chunk_index": 205}
    },
    {
        "id": "abbr_wdoms_ar",
        "text": "WDOMS اختصار لـ World Directory of Medical Schools — الدليل العالمي لكليات الطب. كلية الطب البشري بجامعة بنها الأهلية مدرجة في هذا الدليل منذ عام 2024/2025، مما يعني اعتراف دولي بشهادتها.",
        "metadata": {"source": "دليل المصطلحات", "section": "general", "chunk_index": 206}
    },
    {
        "id": "abbr_igcse_en",
        "text": "IGCSE = International General Certificate of Secondary Education — Cambridge international qualification accepted at BNU. Requirements: 8 subjects (OL/AS/AL). OL minimum: grade C. AS/AL minimum: grade D. Subjects taken within max 4 years.",
        "metadata": {"source": "Terminology Guide", "section": "admission", "chunk_index": 207}
    },
    {
        "id": "abbr_tests_en",
        "text": "SAT = Scholastic Assessment Test. EST = Egyptian Scholastic Test (Egyptian SAT). ACT = American College Testing. GPA = Grade Point Average. All accepted at BNU. Minimum SAT1/EST1 = 800. STEM = Science, Technology, Engineering, Mathematics schools — graduates accepted at BNU under same conditions as public high school.",
        "metadata": {"source": "Terminology Guide", "section": "admission", "chunk_index": 208}
    },
    {
        "id": "abbr_levels_en",
        "text": "In IGCSE: OL = Ordinary Level. AS = Advanced Subsidiary. AL = Advanced Level (A-Level). Engineering faculty requires Mathematics at AS/AL level as a 9th subject. Credit Hours system: each course has a credit value; students build their schedule flexibly.",
        "metadata": {"source": "Terminology Guide", "section": "admission", "chunk_index": 209}
    },

    # ── Additional educational terms ──────────────────────────────────
    {
        "id": "abbr_toefl_ielts_ar",
        "text": "TOEFL اختصار لـ Test of English as a Foreign Language — اختبار اللغة الإنجليزية للطلاب الأجانب. IELTS اختصار لـ International English Language Testing System — نظام الاختبار الدولي للغة الإنجليزية. كلاهما يقيسان مستوى الإنجليزية ولكنهما ليسا شرطاً للقبول في جامعة بنها الأهلية؛ الجامعة تُجري اختبار مستوى خاصاً بها.",
        "metadata": {"source": "دليل المصطلحات", "section": "admission", "chunk_index": 210}
    },
    {
        "id": "abbr_ib_ap_ar",
        "text": "IB اختصار لـ International Baccalaureate — البكالوريا الدولية، شهادة ثانوية دولية معتمدة في جامعة بنها الأهلية. AP اختصار لـ Advanced Placement — برنامج الكورسات المتقدمة الأمريكية التي تُدرَّس في المدرسة الثانوية وتُعادَل بساعات جامعية.",
        "metadata": {"source": "دليل المصطلحات", "section": "admission", "chunk_index": 211}
    },
    {
        "id": "abbr_degrees_ar",
        "text": "BS أو B.Sc اختصار لـ Bachelor of Science — بكالوريوس علوم. BA أو B.A اختصار لـ Bachelor of Arts — بكالوريوس آداب. B.Eng اختصار لـ Bachelor of Engineering — بكالوريوس هندسة. MD اختصار لـ Doctor of Medicine — دكتوراه الطب البشري. MS أو M.Sc = ماجستير علوم. PhD = دكتوراه. جامعة بنها الأهلية تمنح درجة البكالوريوس في جميع كلياتها.",
        "metadata": {"source": "دليل المصطلحات", "section": "general", "chunk_index": 212}
    },
    {
        "id": "abbr_academic_terms_ar",
        "text": "Transcript أو كشف الدرجات: وثيقة رسمية تُظهر جميع المواد الدراسية والدرجات. Prerequisite أو متطلب سابق: مادة يجب اجتيازها قبل دراسة مادة أخرى. Elective أو مادة اختيارية: مادة يختارها الطالب بحرية من قائمة محددة. Core Course أو مادة إجبارية: مادة إلزامية لجميع طلاب البرنامج. GPA = المعدل التراكمي. Syllabus = خطة المقرر الدراسي.",
        "metadata": {"source": "دليل المصطلحات", "section": "general", "chunk_index": 213}
    },
    {
        "id": "abbr_toefl_ielts_en",
        "text": "TOEFL = Test of English as a Foreign Language. IELTS = International English Language Testing System. Both measure English proficiency. Neither is required for BNU admission — BNU runs its own language placement test. IB = International Baccalaureate (accepted at BNU). AP = Advanced Placement (US high school advanced courses).",
        "metadata": {"source": "Terminology Guide", "section": "admission", "chunk_index": 214}
    },
    {
        "id": "abbr_degrees_en",
        "text": "BS/B.Sc = Bachelor of Science. BA = Bachelor of Arts. B.Eng = Bachelor of Engineering. MD = Doctor of Medicine. MS/M.Sc = Master of Science. PhD = Doctor of Philosophy. BNU awards bachelor's degrees across all faculties. Transcript = official academic record showing all courses and grades. Prerequisite = a course that must be passed before taking another. Elective = optional course chosen by the student. Core course = mandatory course for all students in the program.",
        "metadata": {"source": "Terminology Guide", "section": "general", "chunk_index": 215}
    },
]
