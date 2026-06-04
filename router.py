"""
router.py
Smart intent classifier — routes each message to the right handler.
Intents: greeting | acknowledgment | language_switch | university_rag | out_of_scope
"""

import re
from utils import is_greeting, detect_language


# ── Language switch patterns ──────────────────────────────────────────────────
LANGUAGE_SWITCH_EN = [
    r'\bin english\b', r'\banswer in english\b', r'\brespond in english\b',
    r'\btranslate to english\b', r'\btranslate it\b',
    r'بالانجليزي', r'بالإنجليزي', r'بالإنجليزية', r'باللغة الانجليزية',
    r'باللغة الإنجليزية', r'جاوب بالانجليزي', r'جاوب باللغة الانجليزية',
    r'رد بالانجليزي', r'رد باللغة الانجليزية', r'قولي بالانجليزي',
    r'ترجمه?\s*(ل|بال)?\s*انجليزي', r'ترجم\s*(ل)?\s*انجليزي',
    r'ترجم\s*(ل)?\s*الانجليزية', r'ترجم\s*(ل)?\s*الإنجليزية',
    r'قوله?\s*بالانجليزي', r'بالانجش', r'english please', r'say it in english',
]
LANGUAGE_SWITCH_AR = [
    r'\bin arabic\b', r'\banswer in arabic\b', r'\brespond in arabic\b',
    r'\btranslate to arabic\b',
    r'بالعربي', r'بالعربية', r'باللغة العربية',
    r'جاوب بالعربي', r'جاوب باللغة العربية',
    r'رد بالعربي', r'رد باللغة العربية',
    r'ترجم\s*(ل)?\s*عربي', r'ترجم\s*(ل)?\s*العربية',
]


def detect_language_switch(text: str):
    """Returns 'en', 'ar', or None if the message is a language-switch request."""
    t = text.lower().strip()
    for pattern in LANGUAGE_SWITCH_EN:
        if re.search(pattern, t, re.IGNORECASE):
            return 'en'
    for pattern in LANGUAGE_SWITCH_AR:
        if re.search(pattern, t, re.IGNORECASE):
            return 'ar'
    return None


# Intents that produce trivial/system replies — skip when looking for last real answer
_TRIVIAL_INTENTS = {'greeting', 'acknowledgment', 'out_of_scope', 'language_switch'}


def get_last_real_answer(history: list) -> str | None:
    """
    Walk conversation history backwards and return the last substantive
    RAG answer, skipping greetings, acknowledgments, and system replies.
    """
    for turn in reversed(history):
        if turn.get("intent") not in _TRIVIAL_INTENTS:
            return turn["assistant"]
    return None


# ── Acknowledgment words ──────────────────────────────────────────────────────
_ACK_WORDS = {
    # Arabic
    'طيب', 'حاضر', 'ماشي', 'تمام', 'أوكي', 'اوكي', 'اوكى', 'أوكى',
    'حسنا', 'حسناً', 'نعم', 'آه', 'أيوه', 'ايوه', 'يلا', 'هيا',
    'شكرا', 'شكراً', 'متشكر', 'ممتاز', 'كويس', 'تسلم', 'صح',
    'مظبوط', 'عال', 'عاش', 'برافو', 'جميل', 'رائع', 'عظيم', 'معلش',
    'يعني', 'بس', 'اه', 'اوه', 'واو',
    'مش', 'فاهم', 'فاهمة', 'واضح', 'مفهوم', 'تمام', 'عارف', 'عارفة',
    # English
    'ok', 'okay', 'sure', 'yes', 'yep', 'yeah', 'alright', 'fine',
    'thanks', 'thank', 'you', 'noted', 'understood', 'cool', 'great',
    'nice', 'good', 'right', 'perfect', 'awesome', 'wow', 'oh', 'ah',
    'i', 'see', 'got', 'it',
}


def _is_acknowledgment(text: str) -> bool:
    """True if the entire message is a filler/acknowledgment with no real question."""
    words = text.strip().split()
    if not words or len(words) > 6:
        return False
    # All significant words must be acknowledgment words
    significant = [w.strip('.,!؟?') for w in words if len(w.strip('.,!؟?')) > 1]
    return bool(significant) and all(w in _ACK_WORDS for w in significant)


# ── University keywords ───────────────────────────────────────────────────────
UNIVERSITY_KEYWORDS_AR = [
    'جامعة', 'كلية', 'مصاريف', 'رسوم', 'تكاليف', 'قبول', 'تقديم', 'تسجيل',
    'تنسيق', 'شروط', 'منحة', 'خصم', 'دراسة', 'برنامج', 'تخصص', 'ساعات',
    'طب', 'هندسة', 'حاسب', 'اقتصاد', 'فنون', 'طاقة', 'بيطري', 'علاج',
    'امتياز', 'درجات', 'مجموع', 'ثانوية', 'ألف', 'دولار', 'جنيه',
    'سكن', 'نقل', 'تأمين', 'اعتماد', 'معادلة', 'وافد', 'وافدين', 'مصري',
    'رابط', 'موقع', 'فيسبوك', 'التقديم', 'مستندات', 'سياسة', 'سحب',
    'استرداد', 'تدريب', 'مستشفى', 'بنها', 'العبور', 'أهلية',
    'متطلبات', 'الالتحاق', 'التحاق', 'طلاب', 'الطلاب', 'دولي', 'أجنبي',
    'التسجيل', 'القبول', 'الدراسة', 'الرسوم', 'المصاريف', 'التخصص',
    'مدة', 'سنوات', 'فصل', 'ترم', 'اكاديمي', 'أكاديمي', 'دبلوم',
    'IG', 'IGCSE', 'SAT', 'EST', 'ACT', 'STEM', 'GPA', 'IB', 'AP',
    'TOEFL', 'IELTS', 'transcript', 'prerequisite', 'elective', 'syllabus',
    'اختصار', 'معناه', 'ما معنى', 'ما هو',
    'عايز ادخل', 'عايز اتقدم', 'محتاج', 'ايه الكليات', 'ايه المصاريف',
    'اشتغل', 'مستقبل', 'تخرج', 'شهادة', 'اعتراف',
]

UNIVERSITY_KEYWORDS_EN = [
    'university', 'college', 'faculty', 'fees', 'tuition', 'cost', 'price',
    'admission', 'apply', 'application', 'register', 'enrollment', 'accept',
    'scholarship', 'discount', 'grant', 'program', 'major', 'degree',
    'medicine', 'engineering', 'computer', 'science', 'economics', 'arts',
    'energy', 'veterinary', 'therapy', 'dentistry', 'pharmacy',
    'minimum', 'gpa', 'grade', 'score', 'igcse', 'sat', 'est', 'act',
    'housing', 'transport', 'insurance', 'accreditation',
    'bnu', 'benha', 'obour', 'national',
    'link', 'website', 'facebook', 'document', 'refund',
    'training', 'hospital', 'internship', 'graduate', 'certificate',
    'toefl', 'ielts', 'ib', 'transcript', 'prerequisite', 'elective',
    'abbreviation', 'what does', 'what is', 'meaning of',
]

# Off-topic patterns — clearly not university-related
OUT_OF_SCOPE_PATTERNS = [
    r'(weather|طقس|حالة الجو)',
    r'(recipe|وصفة|طبخ|اطبخ)',
    r'(football|soccer|كرة\s*ال?قدم|الكورة|كورة|مباراة|ليفربول|الأهلي|الزمالك|بتشجع|بيشجع)',
    r'(movie|film|فيلم|مسلسل|نتفليكس|cinema)',
    r'\bjoke\b|نكتة',
    r'(stock market|بورصة)',
    r'(احسب|جدول ضرب|معادلة رياضية|calculate|math problem)',
    r'(اكتبلي كود|اكتب كود|برمجلي|write.*code|python code|javascript code)',
    r'(اعمل قصيدة|اكتبلي قصة|قصيدة|poem\b|write.*story)',
    r'(dating|علاقة عاطفية|حبيب|حبيبة)',
    r'(politics|سياسة|حكومة|رئيس\s+جمهورية)',
    r'(religion|دين|فتوى)',
    r'(بيتزا|برجر|برغر|كشري|فول|طعمية|شاورما|سندوتش|مطعم|وجبة|pizza|burger|food|restaurant)',
    r'(ما\s*را?يك\s+ف[ي|ى]|را?يك\s+ف[ي|ى]|بتحب\s+ايه|بتحب\s+إيه|تحب\s+ايه)',
]

OFFENSIVE_PATTERNS = [
    r'(خرا|زبالة|كلب\b|حمار\b|غبي|احمق|عبيط|منيك|كس\b|طيز|نيك\b|لعنة)',
    r'\b(fuck|shit|bitch|ass\b|damn\b|crap\b|idiot|stupid\b|dumb\b)',
]

CRISIS_PATTERNS = [
    r'(عايز اموت|عايزة اموت|اموت|انتحار|اقتل نفسي|مش عارف اعيش|تعبت من الحياة|مش قادر اكمل)',
    r'\b(want to die|kill myself|suicide|end my life|cant go on|can\'t go on)\b',
]

QUESTION_PATTERNS = [
    r'[?؟]',
    r'^(ما|ماذا|من|كيف|كيفية|هل|متى|أين|لماذا|كم|ماهي|ماهو)',
    r'^(what|how|when|where|why|who|can|is|are|do|does)\b',
    r'\b(ايه|فين|امتى|ازاي|ليه|كام|مين|إيه)\b',
    r'^(اخبرني|اخبر|وضح|اشرح|عرف|قولي|ممكن|محتاج)',
]


def _is_offensive(text: str) -> bool:
    for p in OFFENSIVE_PATTERNS:
        if re.search(p, text, re.IGNORECASE):
            return True
    return False


def _is_crisis(text: str) -> bool:
    for p in CRISIS_PATTERNS:
        if re.search(p, text, re.IGNORECASE):
            return True
    return False


def classify_intent(user_message: str) -> str:
    """
    Smart intent classifier. Priority order:
    1. language_switch  → user wants last answer in different language
    2. greeting         → hi / hello / مرحبا
    3. acknowledgment   → طيب / ok / تمام (filler, no real question)
    4. out_of_scope     → clearly non-university topic
    5. university_rag   → has university keywords or question patterns
    6. out_of_scope     → default for non-university messages
    """
    msg = user_message.lower().strip()

    # 1. Crisis detection — handle with empathy before anything else
    if _is_crisis(msg):
        return 'crisis'

    # 2. Offensive language
    if _is_offensive(msg):
        return 'offensive'

    # 3. Language switch
    if detect_language_switch(user_message):
        return 'language_switch'

    # 4. Greeting
    if is_greeting(msg):
        return 'greeting'

    # 5. Acknowledgment (filler words, no real question)
    if _is_acknowledgment(msg):
        return 'acknowledgment'

    # 6. Explicit out-of-scope topics
    for pattern in OUT_OF_SCOPE_PATTERNS:
        if re.search(pattern, msg, re.IGNORECASE):
            return 'out_of_scope'

    # 5. University keywords → send to RAG
    has_ar = any(kw in msg for kw in UNIVERSITY_KEYWORDS_AR)
    has_en = any(kw.lower() in msg for kw in UNIVERSITY_KEYWORDS_EN)
    if has_ar or has_en:
        return 'university_rag'

    # 6. Question patterns → try RAG (user is asking something)
    for pattern in QUESTION_PATTERNS:
        if re.search(pattern, msg, re.IGNORECASE):
            return 'university_rag'

    # 7. Any Arabic message with 2+ words that isn't explicitly out-of-scope → try RAG
    arabic_chars = sum(1 for c in msg if '؀' <= c <= 'ۿ')
    if arabic_chars >= 4 and len(msg.split()) >= 2:
        return 'university_rag'

    # 8. Default: clearly not a university question
    return 'out_of_scope'


# ── Response generators ───────────────────────────────────────────────────────

def get_greeting_response(language: str) -> str:
    if language == 'ar':
        return (
            "أهلاً وسهلاً! 🎓\n"
            "أنا المساعد الذكي لجامعة بنها الأهلية.\n"
            "يسعدني الإجابة على استفساراتك عن:\n"
            "• الكليات والبرامج الدراسية\n"
            "• المصروفات والمنح الدراسية\n"
            "• شروط وخطوات القبول\n"
            "• الروابط والمعلومات العامة\n\n"
            "اسألني أي سؤال! 😊"
        )
    return (
        "Welcome! 🎓\n"
        "I'm the BNU smart assistant for Benha National University.\n"
        "I can help you with:\n"
        "• Available faculties and programs\n"
        "• Tuition fees and scholarships\n"
        "• Admission requirements and steps\n"
        "• Links and general information\n\n"
        "Ask me anything about BNU! 😊"
    )


def get_acknowledgment_response(language: str, has_history: bool) -> str:
    if language == 'ar':
        if has_history:
            return "هل تريد الاستفسار عن شيء آخر؟ 😊\nأنا هنا لمساعدتك في أي سؤال عن جامعة بنها الأهلية."
        return "أهلاً! 🎓 اسألني عن الكليات، المصاريف، شروط القبول، أو أي شيء عن جامعة بنها الأهلية."
    if has_history:
        return "Is there anything else you'd like to know? 😊\nI'm here to help with any questions about BNU."
    return "Hello! 🎓 Ask me about BNU faculties, fees, admission requirements, or anything about the university."


def get_offensive_response(language: str) -> str:
    if language == 'ar':
        return "يرجى استخدام لغة محترمة. 🙏\nأنا هنا لمساعدتك في أسئلة جامعة بنها الأهلية."
    return "Please use respectful language. 🙏\nI'm here to help you with BNU questions."


def get_crisis_response(language: str) -> str:
    if language == 'ar':
        return (
            "أنا قلقان عليك. 💙\n"
            "إذا كنت تمر بوقت صعب، يرجى التواصل مع شخص تثق به أو طلب المساعدة المتخصصة.\n"
            "خط نجدة الطفل (مصر): 16000\n"
            "أنا هنا إذا أردت التحدث عن أي شيء يخص الجامعة."
        )
    return (
        "I'm concerned about you. 💙\n"
        "If you're going through a difficult time, please reach out to someone you trust or seek professional support.\n"
        "I'm here if you'd like to ask about anything related to BNU."
    )


def get_out_of_scope_response(language: str) -> str:
    if language == 'ar':
        return (
            "أنا متخصص فقط في الإجابة على أسئلة جامعة بنها الأهلية. 🎓\n"
            "يمكنني مساعدتك في:\n"
            "• المصروفات الدراسية والمنح\n"
            "• شروط القبول والتسجيل\n"
            "• الكليات والبرامج المتاحة\n"
            "هل لديك سؤال عن الجامعة؟"
        )
    return (
        "I'm specialized in answering questions about Benha National University only. 🎓\n"
        "I can help with:\n"
        "• Tuition fees and scholarships\n"
        "• Admission requirements and registration\n"
        "• Available faculties and programs\n"
        "Do you have a question about BNU?"
    )
