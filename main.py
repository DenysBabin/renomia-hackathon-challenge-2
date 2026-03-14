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


def clean_ocr_text(text: str) -> str:
    """Предобработка OCR-текста для экономии токенов."""
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'(?im)^[\s]*strana\s+\d+\s*(z\s+\d+)?[\s]*$', '', text)
    text = re.sub(r'(?m)^[\s]*-\s*\d+\s*-[\s]*$', '', text)
    text = '\n'.join(line.rstrip() for line in text.splitlines())
    return text.strip()


def extract_endorsement_number(documents: list) -> str | None:
    """Извлечь максимальный номер дополнения (dodatek) из файлов и текста."""
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


EXTRACTION_PROMPT = """Jsi expert na extrakci dat z českých pojistných smluv. Z OCR textu dokumentů extrahuj následující pole do JSON.

POLE A PRAVIDLA:
- contractNumber: string — číslo pojistné smlouvy
- insurerName: string — název pojišťovny (pojistitel), NIKOLI pojistník
- state: "draft" | "accepted" | "cancelled" — stav smlouvy. Pokud je smlouva podepsána/platná → "accepted". Pokud je návrh → "draft". Pokud je vypovězena/zrušena → "cancelled". Default: "accepted"
- assetType: "other" | "vehicle" — "vehicle" pouze pokud jde o pojištění vozidla (havarijní, povinné ručení, SPZ). Jinak "other"
- concludedAs: "agent" | "broker" — "broker" pokud je zmíněn makléř (např. Renomia). "agent" pokud je zmíněn agent. Default: "broker"
- contractRegime: "individual" | "frame" | "fleet" | "coinsurance" — "frame" = rámcová smlouva, "fleet" = flotilová, "coinsurance" = soupojištění. Default: "individual"
- startAt: string DD.MM.YYYY — počátek pojištění
- endAt: string DD.MM.YYYY | null — konec pojištění. null = doba neurčitá
- concludedAt: string DD.MM.YYYY — datum uzavření/podpisu smlouvy
- installmentNumberPerInsurancePeriod: number — počet splátek: ročně=1, pololetně=2, čtvrtletně=4, měsíčně=12
- insurancePeriodMonths: number — délka pojistného období: roční=12, pololetní=6, čtvrtletní=3, měsíční=1
- premium.currency: string — měna pojistného, ISO 4217 lowercase (czk, eur, usd)
- premium.isCollection: boolean — true pokud je inkaso pojistného přes makléře/zprostředkovatele
- actionOnInsurancePeriodTermination: "auto-renewal" | "policy-termination" — "auto-renewal" pokud se smlouva automaticky prodlužuje. "policy-termination" pokud po konci období končí
- noticePeriod: string | null — výpovědní lhůta: "six-weeks", "three-months", "two-months", "one-month", "eight-days". null pokud není uvedena
- regPlate: string | null — SPZ/RZ vozidla. null pokud nejde o vozidlo
- note: string | null — zvláštní/nestandardní podmínky. null pokud žádné nejsou

DŮLEŽITÉ:
- Dodatky (amendments) PŘEPISUJÍ hodnoty ze základní smlouvy — použij POSLEDNÍ platnou hodnotu
- Datumy vždy ve formátu DD.MM.YYYY s nulami (01.03.2024, ne 1.3.2024)
- Pokud pole nelze určit z textu → null
- Odpověz POUZE validním JSON objektem, BEZ vysvětlení, BEZ markdown

TEXT DOKUMENTŮ:
"""


ENUM_DEFAULTS = {
    "state": ("draft", "accepted", "cancelled"),
    "assetType": ("other", "vehicle"),
    "concludedAs": ("agent", "broker"),
    "contractRegime": ("individual", "frame", "fleet", "coinsurance"),
    "actionOnInsurancePeriodTermination": ("auto-renewal", "policy-termination"),
}

ENUM_FALLBACKS = {
    "state": "accepted",
    "assetType": "other",
    "concludedAs": "broker",
    "contractRegime": "individual",
    "actionOnInsurancePeriodTermination": "auto-renewal",
}

VALID_NOTICE_PERIODS = {
    "six-weeks", "three-months", "two-months", "one-month", "eight-days",
}


def validate_date(value) -> str | None:
    """Валидация и нормализация даты в формат DD.MM.YYYY."""
    if not value or not isinstance(value, str):
        return None
    m = re.match(r'^(\d{1,2})\.(\d{1,2})\.(\d{4})$', value.strip())
    if m:
        return f"{int(m.group(1)):02d}.{int(m.group(2)):02d}.{m.group(3)}"
    return None


def parse_gemini_response(response_text: str) -> dict:
    """Парсинг и валидация JSON-ответа от Gemini."""
    text = response_text.strip()
    # Убрать markdown-обёртку если есть
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

    # Валидация enum-полей
    for field, valid_values in ENUM_DEFAULTS.items():
        val = data.get(field)
        if val not in valid_values:
            data[field] = ENUM_FALLBACKS[field]

    # Валидация дат
    for date_field in ("startAt", "endAt", "concludedAt"):
        data[date_field] = validate_date(data.get(date_field))

    # currency → lowercase
    premium = data.get("premium", {})
    if isinstance(premium, dict):
        currency = premium.get("currency", "czk")
        if isinstance(currency, str):
            premium["currency"] = currency.lower()
        else:
            premium["currency"] = "czk"
        if not isinstance(premium.get("isCollection"), bool):
            premium["isCollection"] = False
        data["premium"] = premium
    else:
        data["premium"] = {"currency": "czk", "isCollection": False}

    # noticePeriod валидация
    np = data.get("noticePeriod")
    if np and np not in VALID_NOTICE_PERIODS:
        data["noticePeriod"] = None

    # installment и period — числа
    inst = data.get("installmentNumberPerInsurancePeriod")
    if inst not in (1, 2, 4, 12):
        data["installmentNumberPerInsurancePeriod"] = 1
    period = data.get("insurancePeriodMonths")
    if period not in (1, 3, 6, 12):
        data["insurancePeriodMonths"] = 12

    return data


def get_cached_result(cache_key: str) -> dict | None:
    """Получить результат из кэша PostgreSQL."""
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
    """Сохранить результат в кэш PostgreSQL."""
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


@app.post("/solve")
def solve(payload: dict):
    documents = payload.get("documents", [])

    # Собираем и чистим OCR-текст
    combined_text = ""
    for doc in documents:
        filename = doc.get("filename", "unknown")
        ocr_text = clean_ocr_text(doc.get("ocr_text", ""))
        combined_text += f"\n=== {filename} ===\n{ocr_text}\n"

    # Кэширование
    cache_key = hashlib.sha256(combined_text.encode()).hexdigest()
    cached = get_cached_result(cache_key)
    if cached:
        return cached

    # Regex-извлечение endorsement number
    endorsement_number = extract_endorsement_number(documents)

    # Запрос к Gemini
    prompt = EXTRACTION_PROMPT + combined_text
    response = gemini.generate(prompt)
    extracted = parse_gemini_response(response.text)

    # Собираем финальный результат
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
        "insurancePeriodMonths": extracted.get("insurancePeriodMonths", 12),
        "premium": extracted.get("premium", {"currency": "czk", "isCollection": False}),
        "actionOnInsurancePeriodTermination": extracted.get("actionOnInsurancePeriodTermination", "auto-renewal"),
        "noticePeriod": extracted.get("noticePeriod"),
        "regPlate": extracted.get("regPlate"),
        "latestEndorsementNumber": endorsement_number,
        "note": extracted.get("note"),
    }

    # Кэшируем результат
    set_cached_result(cache_key, result)

    return result


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
