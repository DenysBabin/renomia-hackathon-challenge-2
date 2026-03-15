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


@app.post("/solve")
def solve(payload: dict):
    """
    Extract structured CRM fields from insurance contract documents.

    Input example:
    {
        "documents": [
            {
                "pdf_url": "https://storage.googleapis.com/.../smlouva.pdf",
                "filename": "smlouva_hlavni.pdf",
                "ocr_text": "... OCR extracted text of main contract ..."
            },
            {
                "pdf_url": "https://storage.googleapis.com/.../dodatek_1.pdf",
                "filename": "dodatek_1.pdf",
                "ocr_text": "... OCR text of amendment 1 ..."
            },
            {
                "pdf_url": "https://storage.googleapis.com/.../dodatek_2.pdf",
                "filename": "dodatek_2.pdf",
                "ocr_text": "... OCR text of amendment 2 ..."
            }
        ]
    }

def sections_to_text(sections: list[dict]) -> str:
    """Reassemble sections back into plain text."""
    parts = []
    for sec in sections:
        if sec["title"]:
            parts.append(sec["title"])
        parts.extend(sec["lines"])
    return "\n".join(parts)


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


def _parse_czech_number(s: str) -> float | None:
    """Parse Czech formatted number: '24 000', '24000', '24.000,50'."""
    s = s.strip().replace('\xa0', '').replace(' ', '')
    # "24.000,50" → "24000.50"
    if ',' in s:
        s = s.replace('.', '').replace(',', '.')
    else:
        # "24.000" with dot as thousands separator (no decimal)
        if s.count('.') > 1 or (s.count('.') == 1 and len(s.split('.')[-1]) == 3):
            s = s.replace('.', '')
    try:
        val = float(s)
        return val if val > 0 else None
    except ValueError:
        return None


def extract_insurance_period_from_premiums(text: str) -> int | None:
    """Calculate insurance period: 12 / (annual_premium / installment_amount)."""
    # Find annual/total premium
    annual_m = re.search(
        r'(?:roční|celkov[ée]|úhrn\w*)\s+pojistn[ée]\s*:?\s*([\d\s.,\xa0]+)',
        text, re.IGNORECASE
    )
    if not annual_m:
        return None
    annual = _parse_czech_number(annual_m.group(1))
    if not annual:
        return None

    # Find installment/periodic amount
    installment_m = re.search(
        r'(?:splátk[ay]|(?:pololetní|čtvrtletní|měsíční)\s+pojistn[ée])\s*:?\s*([\d\s.,\xa0]+)',
        text, re.IGNORECASE
    )
    if not installment_m:
        return None
    installment = _parse_czech_number(installment_m.group(1))
    if not installment:
        return None

    # Calculate
    num_payments = round(annual / installment)
    if num_payments <= 0:
        return None
    period = 12 / num_payments
    if period in (1, 3, 6, 12):
        return int(period)
    return None


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
            # Match "výpovědní lhůta" or "lhůta pro výpověď" near time pattern
            for ctx in [
                r'výpovědní\s+lhůt[^\n]{0,80}' + pattern,
                pattern + r'[^\n]{0,80}výpovědní\s+lhůt',
                r'lhůt[^\n]{0,40}výpověd[^\n]{0,40}' + pattern,
                r'výpověd[^\n]{0,60}' + pattern,
            ]:
                match = re.search(ctx, text, re.IGNORECASE)
                if match:
                    # Exclude renewal notification deadlines
                    context = match.group()
                    if re.search(r'před\s+uplynutím|před\s+koncem', context, re.IGNORECASE):
                        continue
                    return value
    return None


KNOWN_INSURERS = [
    # Longer/more specific names first to prefer best match
    "Kooperativa pojišťovna, a.s., Vienna Insurance Group",
    "Kooperativa pojišťovna",
    "Generali Česká pojišťovna a.s.",
    "Generali Česká pojišťovna",
    "Generali Ceska pojistovna",
    "INTER PARTNER ASSISTANCE, S.A.",
    "INTER PARTNER ASSISTANCE",
    "ČSOB Pojišťovna",
    "CSOB Pojistovna",
    "Hasičská vzájemná pojišťovna",
    "Pojišťovna VZP",
    "Slavia pojišťovna",
    "Direct pojišťovna",
    "Pillow pojišťovna",
    "Colonnade",
    "Kooperativa",
    "Allianz",
    "UNIQA",
    "Generali",
    "Česká pojišťovna",
    "AXA",
    "HDI",
    "Credendo",
]


def extract_insurer_by_name(text: str) -> str | None:
    """Search for known insurer names in OCR text (longest match first)."""
    text_lower = text.lower()
    for name in KNOWN_INSURERS:
        if name.lower() in text_lower:
            return name
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

    # 1. Preprocess: clean OCR, parse into sections, filter, reassemble
    combined_text = ""
    for doc in documents:
        filename = doc.get("filename", "unknown")
        if is_vpp_document(filename):
            continue
        cleaned = clean_ocr_text(doc.get("ocr_text", ""))
        sections = parse_sections(cleaned)
        sections = filter_sections(sections)
        text = sections_to_text(sections)
        combined_text += f"\n=== {filename} ===\n{text}\n"

    # 2. Cache lookup
    cache_key = hashlib.sha256(combined_text.encode()).hexdigest()
    cached = get_cached_result(cache_key)
    if cached:
        return cached

    # 3. Regex extractions (no AI)
    endorsement_number = extract_endorsement_number(documents)
    note = extract_note(combined_text)

    # 4. Gemini extraction
    prompt = EXTRACTION_PROMPT + combined_text
    response = gemini.generate(prompt)
    extracted = parse_gemini_response(response.text)

    # 5. Build result from extracted + regex fields
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

    # 6. Deterministic overrides
    if result.get("state") == "draft" and result.get("startAt"):
        result["state"] = "accepted"

    # insurancePeriodMonths fallback: calculate from premium amounts (only for auto-renewal)
    if (result.get("insurancePeriodMonths") is None
            and result.get("actionOnInsurancePeriodTermination") == "auto-renewal"):
        calculated_period = extract_insurance_period_from_premiums(combined_text)
        if calculated_period is not None:
            result["insurancePeriodMonths"] = calculated_period

    # 7. Fallbacks: VPP docs + known insurer names
    if not result.get("insurerName"):
        result["insurerName"] = (
            extract_insurer_from_vpp(documents)
            or extract_insurer_by_name(combined_text)
        )

    if not result.get("noticePeriod"):
        result["noticePeriod"] = extract_notice_period_from_vpp(documents)

    # 8. Cache and return
    set_cached_result(cache_key, result)
    return result


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
