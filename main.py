"""
Challenge 2: Vyčítání dat ze souborů (Document Data Extraction)

Input:  OCR text from insurance contract documents (main contract + amendments)
Output: Structured CRM fields extracted from the documents
"""

import hashlib
import json
import os
import re
import threading
import time

import google.generativeai as genai
import psycopg2
from fastapi import FastAPI
import uvicorn

app = FastAPI(title="Challenge 2: Document Data Extraction")

DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://hackathon:hackathon@localhost:5432/hackathon"
)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")


class GeminiTracker:
    """Wrapper around Gemini that tracks token usage."""

    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash"):
        self.enabled = bool(api_key)
        if self.enabled:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(model_name)
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.request_count = 0
        self._lock = threading.Lock()

    def generate(self, prompt, **kwargs):
        if not self.enabled:
            raise RuntimeError("Gemini API key not configured")
        response = self.model.generate_content(prompt, **kwargs)
        with self._lock:
            self.request_count += 1
            meta = getattr(response, "usage_metadata", None)
            if meta:
                self.prompt_tokens += getattr(meta, "prompt_token_count", 0)
                self.completion_tokens += getattr(meta, "candidates_token_count", 0)
                self.total_tokens += getattr(meta, "total_token_count", 0)
        return response

    def get_metrics(self):
        with self._lock:
            return {
                "gemini_request_count": self.request_count,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.total_tokens,
            }

    def reset(self):
        with self._lock:
            self.prompt_tokens = 0
            self.completion_tokens = 0
            self.total_tokens = 0
            self.request_count = 0


gemini = GeminiTracker(GEMINI_API_KEY)


def get_db():
    return psycopg2.connect(DATABASE_URL)


@app.on_event("startup")
def init_db():
    for _ in range(15):
        try:
            conn = get_db()
            cur = conn.cursor()
            cur.execute(
                """CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    value JSONB,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                )"""
            )
            conn.commit()
            cur.close()
            conn.close()
            return
        except Exception:
            time.sleep(1)


@app.get("/")
def health():
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    return gemini.get_metrics()


@app.post("/metrics/reset")
def reset_metrics():
    gemini.reset()
    return {"status": "reset"}


# Sections to skip from OCR text before sending to Gemini.
# Add gradually, verify score doesn't drop after each addition.
SECTION_BLACKLIST = [
    # "prohlášení",
    # "zpracování osobních údajů",
    # "závěrečná ustanovení",
    # "infekční onemocnění",
    # "kontaminace",
    # "vaše povinnost informovat",
]


def _is_section_header(line: str) -> bool:
    """Detect if a line is a section header."""
    stripped = line.strip()
    if not stripped or len(stripped) > 120:
        return False
    # All caps line (at least 3 chars)
    if len(stripped) >= 3 and stripped == stripped.upper() and any(c.isalpha() for c in stripped):
        return True
    # Numbered sections: "1.", "1.1", "Článek 1", "Čl. 1", Roman numerals
    if re.match(r'^(\d+\.|\d+\.\d+|[IVXLC]+\.|článek\s+\d|čl\.\s*\d)', stripped, re.IGNORECASE):
        return True
    return False


def filter_sections(text: str) -> str:
    """Remove blacklisted sections from OCR text."""
    if not SECTION_BLACKLIST:
        return text

    lines = text.splitlines()
    result = []
    skip = False

    for line in lines:
        if _is_section_header(line):
            header_lower = line.strip().lower()
            skip = any(bl in header_lower for bl in SECTION_BLACKLIST)

        if not skip:
            result.append(line)

    return '\n'.join(result)


def extract_note(combined_text: str) -> str | None:
    """Extract note from 'Poznámka:' section without AI."""
    match = re.search(r'pozn[aá]mk[ay]\s*:', combined_text, re.IGNORECASE)
    if not match:
        return None

    after = combined_text[match.end():]
    lines = after.split('\n')

    note_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if note_lines:
                continue
            continue

        is_header = False
        if len(stripped) < 60 and stripped.endswith(':'):
            is_header = True
        if len(stripped) >= 3 and stripped == stripped.upper() and any(c.isalpha() for c in stripped):
            is_header = True
        if re.match(r'^\d+\.', stripped):
            is_header = True
        if len(stripped.split()) == 1 and len(stripped) < 30 and not stripped.endswith('.'):
            is_header = True

        if is_header:
            break

        note_lines.append(stripped)

    if note_lines:
        return ' '.join(note_lines)
    return None


def is_vpp_document(filename: str) -> bool:
    """Check if document is VPP/general conditions (no contract-specific data)."""
    name = filename.lower()
    return "vpp" in name or "podmínky" in name or "podminky" in name


def clean_ocr_text(text: str) -> str:
    """Preprocess OCR text to save tokens."""
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'(?im)^[\s]*strana\s+\d+\s*(z\s+\d+)?[\s]*$', '', text)
    text = re.sub(r'(?m)^[\s]*-\s*\d+\s*-[\s]*$', '', text)
    text = '\n'.join(line.rstrip() for line in text.splitlines())
    return text.strip()


def extract_endorsement_number(documents: list) -> str | None:
    """Extract the highest endorsement (dodatek) number from filenames and text."""
    numbers = []
    for doc in documents:
        filename = doc.get("filename", "")
        ocr_text = doc.get("ocr_text", "")
        for m in re.finditer(r'dodatek[_\s]*(\d+)', filename, re.IGNORECASE):
            numbers.append(int(m.group(1)))
        for m in re.finditer(r'dodatek\s+č[\.\s]*(\d+)', ocr_text, re.IGNORECASE):
            numbers.append(int(m.group(1)))
    if numbers:
        return str(max(numbers))
    return None


from extraction_rules import (
    FIELD_RULES,
    RULES_BY_NAME,
    ENUM_FIELDS,
    ENUM_DEFAULTS,
    VALID_NOTICE_PERIODS,
    VALID_INSTALLMENTS,
    VALID_PERIODS,
    build_extraction_prompt,
)

EXTRACTION_PROMPT = build_extraction_prompt()


def validate_date(value) -> str | None:
    """Validate and normalize date to DD.MM.YYYY format."""
    if not value or not isinstance(value, str):
        return None
    m = re.match(r'^(\d{1,2})\.(\d{1,2})\.(\d{4})$', value.strip())
    if m:
        return f"{int(m.group(1)):02d}.{int(m.group(2)):02d}.{m.group(3)}"
    return None


def parse_gemini_response(response_text: str) -> dict:
    """Parse and validate JSON response from Gemini using FIELD_RULES."""
    text = response_text.strip()
    # Strip markdown wrapper if present
    md_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if md_match:
        text = md_match.group(1)
    else:
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            text = json_match.group()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {}

    for rule in FIELD_RULES:
        if rule.name.startswith("premium."):
            continue  # Handle nested premium separately

        val = data.get(rule.name)

        if rule.type == "enum" and rule.allowed_values:
            if val not in rule.allowed_values:
                data[rule.name] = rule.fallback_on_invalid or rule.default

        elif rule.type == "date":
            data[rule.name] = validate_date(val)

        elif rule.type == "number" and rule.allowed_values:
            if val is None and rule.nullable:
                data[rule.name] = None
            elif val not in rule.allowed_values:
                data[rule.name] = rule.fallback_on_invalid if rule.fallback_on_invalid is not None else rule.default

    # Handle premium nested object
    premium = data.get("premium", {})
    if isinstance(premium, dict):
        currency = premium.get("currency", "czk")
        premium["currency"] = currency.lower() if isinstance(currency, str) else "czk"
        is_coll = premium.get("isCollection")
        premium["isCollection"] = is_coll if isinstance(is_coll, bool) else False
        data["premium"] = premium
    else:
        data["premium"] = {"currency": "czk", "isCollection": False}

    # Validate noticePeriod
    np_val = data.get("noticePeriod")
    if np_val and np_val not in VALID_NOTICE_PERIODS:
        data["noticePeriod"] = None

    return data


def get_cached_result(cache_key: str) -> dict | None:
    """Get cached result from PostgreSQL."""
    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute("SELECT value FROM cache WHERE key = %s", (cache_key,))
        row = cur.fetchone()
        cur.close()
        conn.close()
        if row:
            return row[0]
    except Exception:
        pass
    return None


def set_cached_result(cache_key: str, value: dict):
    """Save result to PostgreSQL cache."""
    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO cache (key, value) VALUES (%s, %s) ON CONFLICT (key) DO UPDATE SET value = %s",
            (cache_key, json.dumps(value), json.dumps(value)),
        )
        conn.commit()
        cur.close()
        conn.close()
    except Exception:
        pass


def extract_insurer_from_vpp(documents: list) -> str | None:
    """Extract insurer name from VPP documents before filtering them out."""
    for doc in documents:
        filename = doc.get("filename", "")
        if not is_vpp_document(filename):
            continue
        ocr_text = doc.get("ocr_text", "")
        header = ocr_text[:3000]
        m = re.search(
            r'[Pp]ojistitel\s+(?:znamená|je|znamena)\s+(.+?)(?:,\s*se\s+sídlem|,\s*se\s+sidlem|$)',
            header,
        )
        if m:
            return m.group(1).strip()
    return None


def extract_notice_period_from_vpp(documents: list) -> str | None:
    """Extract standard notice period from VPP general conditions."""
    notice_map = {
        r'šest\s+t[ýy]dn[ůu]': "six-weeks",
        r'6\s+t[ýy]dn[ůu]': "six-weeks",
        r'tř[ií]\s+měs[íi]c': "three-months",
        r'3\s+měs[íi]c': "three-months",
        r'dv[aou]\s+měs[íi]c': "two-months",
        r'2\s+měs[íi]c': "two-months",
        r'jed(?:en|no)\s+měs[íi]c': "one-month",
        r'1\s+měs[íi]c': "one-month",
        r'osm\s+dn[ůuí]': "eight-days",
        r'8\s+dn[ůuí]': "eight-days",
    }
    for doc in documents:
        filename = doc.get("filename", "")
        if not is_vpp_document(filename):
            continue
        text = doc.get("ocr_text", "")
        for pattern, value in notice_map.items():
            context_re = r'(?:výpověd|vypověd|výpověď|lhůt)[^\n]{0,80}' + pattern
            if re.search(context_re, text, re.IGNORECASE):
                return value
    return None


@app.post("/cache/clear")
def clear_cache():
    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute("DELETE FROM cache")
        conn.commit()
        cur.close()
        conn.close()
        return {"status": "cleared"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/solve")
def solve(payload: dict):
    documents = payload.get("documents", [])

    # Collect and clean OCR text (skip VPP documents)
    combined_text = ""
    for doc in documents:
        filename = doc.get("filename", "unknown")
        if is_vpp_document(filename):
            continue
        ocr_text = clean_ocr_text(doc.get("ocr_text", ""))
        ocr_text = filter_sections(ocr_text)
        combined_text += f"\n=== {filename} ===\n{ocr_text}\n"

    # Cache lookup
    cache_key = hashlib.sha256(combined_text.encode()).hexdigest()
    cached = get_cached_result(cache_key)
    if cached:
        return cached

    # Extract fields via regex (no AI needed)
    endorsement_number = extract_endorsement_number(documents)
    note = extract_note(combined_text)

    # Call Gemini
    prompt = EXTRACTION_PROMPT + combined_text
    response = gemini.generate(prompt)
    extracted = parse_gemini_response(response.text)

    # Build final result
    result = {
        "contractNumber": extracted.get("contractNumber"),
        "insurerName": extracted.get("insurerName"),
        "state": extracted.get("state", "accepted"),
        "assetType": extracted.get("assetType", "other"),
        "concludedAs": extracted.get("concludedAs", "broker"),
        "contractRegime": extracted.get("contractRegime", "individual"),
        "startAt": extracted.get("startAt"),
        "endAt": extracted.get("endAt"),
        "concludedAt": extracted.get("concludedAt"),
        "installmentNumberPerInsurancePeriod": extracted.get("installmentNumberPerInsurancePeriod", 1),
        "insurancePeriodMonths": extracted.get("insurancePeriodMonths"),
        "premium": extracted.get("premium", {"currency": "czk", "isCollection": False}),
        "actionOnInsurancePeriodTermination": extracted.get("actionOnInsurancePeriodTermination", "auto-renewal"),
        "noticePeriod": extracted.get("noticePeriod"),
        "regPlate": extracted.get("regPlate"),
        "latestEndorsementNumber": endorsement_number,
        "note": note,
    }

    # VPP fallbacks: extract data from filtered-out VPP docs if AI missed it
    if not result.get("insurerName"):
        vpp_insurer = extract_insurer_from_vpp(documents)
        if vpp_insurer:
            result["insurerName"] = vpp_insurer

    if not result.get("noticePeriod"):
        vpp_notice = extract_notice_period_from_vpp(documents)
        if vpp_notice:
            result["noticePeriod"] = vpp_notice

    # Cache result
    set_cached_result(cache_key, result)

    return result


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
