# Challenge 2: Vyčítání dat ze souborů (Document Data Extraction)

Extract structured CRM fields from OCR text of insurance contracts and their amendments.

## What you need to do

Implement the `solve()` function in `main.py`. Your endpoint receives OCR text from insurance contract documents (a main contract and zero or more amendments) and must extract all required CRM fields.

Key rule: **amendments override base contract values**. If an amendment changes a field, use the amended value.

## Input format

```json
POST /solve
{
  "documents": [
    {
      "filename": "smlouva_hlavni.pdf",
      "ocr_text": "POJISTNÁ SMLOUVA č. POJ-2024-12345\n\nPojistitel: Generali Česká pojišťovna a.s.\nPojistník: ACME s.r.o.\n\nDatum uzavření: 15.12.2023\nPočátek pojištění: 01.01.2024\nDoba trvání: neurčitá\n\nSplátky: čtvrtletně\nPojistné období: 12 měsíců\nMěna: CZK\n\nPo uplynutí pojistného období se smlouva automaticky prodlužuje.\nVýpovědní lhůta: 6 týdnů před koncem pojistného období."
    },
    {
      "filename": "dodatek_1.pdf",
      "ocr_text": "DODATEK č. 1 k pojistné smlouvě POJ-2024-12345\n\nS účinností od 01.04.2024 se mění:\n- Splátky: pololetně"
    },
    {
      "filename": "dodatek_3.pdf",
      "ocr_text": "DODATEK č. 3 k pojistné smlouvě POJ-2024-12345\n\nS účinností od 01.10.2024 se mění:\n- Pojistné období: 6 měsíců"
    }
  ]
}
```

## Expected output

```json
{
  "contractNumber": "POJ-2024-12345",
  "insurerName": "Generali Česká pojišťovna a.s.",
  "state": "accepted",
  "assetType": "other",
  "concludedAs": "broker",
  "contractRegime": "individual",
  "startAt": "01.01.2024",
  "endAt": null,
  "concludedAt": "15.12.2023",
  "installmentNumberPerInsurancePeriod": 2,
  "insurancePeriodMonths": 6,
  "premium": {
    "currency": "czk",
    "isCollection": false
  },
  "actionOnInsurancePeriodTermination": "auto-renewal",
  "noticePeriod": "six-weeks",
  "regPlate": null,
  "latestEndorsementNumber": "3",
  "note": null
}
```

### Field reference

| Field | Type | Values / Format | Description |
|-------|------|-----------------|-------------|
| `contractNumber` | string | — | Číslo smlouvy |
| `insurerName` | string | — | Název pojišťovny |
| `state` | enum | `draft` \| `accepted` \| `cancelled` | Stav smlouvy |
| `assetType` | enum | `other` \| `vehicle` | Typ předmětu pojištění |
| `concludedAs` | enum | `agent` \| `broker` | Sjednáno jako agent/makléř |
| `contractRegime` | enum | `individual` \| `frame` \| `fleet` \| `coinsurance` | Režim smlouvy |
| `startAt` | string | `DD.MM.YYYY` | Počátek pojištění |
| `endAt` | string/null | `DD.MM.YYYY` or `null` | Konec pojištění (null = doba neurčitá) |
| `concludedAt` | string | `DD.MM.YYYY` | Datum uzavření smlouvy |
| `installmentNumberPerInsurancePeriod` | number | `1`=yearly, `2`=semi, `4`=quarterly, `12`=monthly | Počet splátek za pojistné období |
| `insurancePeriodMonths` | number | `12`=yearly, `6`=semi, `3`=quarterly, `1`=monthly | Délka pojistného období v měsících |
| `premium.currency` | string | ISO 4217 lowercase (e.g. `czk`) | Měna pojistného |
| `premium.isCollection` | boolean | — | Inkaso přes makléře |
| `actionOnInsurancePeriodTermination` | enum | `auto-renewal` \| `policy-termination` | Co se stane po skončení pojistného období |
| `noticePeriod` | string/null | e.g. `six-weeks` | Výpovědní lhůta |
| `regPlate` | string/null | — | SPZ (only for vehicle insurance) |
| `latestEndorsementNumber` | string/null | — | Číslo posledního dodatku |
| `note` | string/null | — | Zvláštní podmínky |

## Scoring

Each field is scored individually and averaged. Scoring methods per field type:

- **Enum fields**: exact match (1.0 or 0.0)
- **String fields**: fuzzy match (similarity ratio)
- **Number fields**: ±10% tolerance for full score
- **Boolean fields**: exact match
- **Dates**: exact string match (`DD.MM.YYYY`)
- **Null handling**: correctly returning `null` when expected counts as correct

## Local development

```bash
# Start the app + sidecar database
docker compose up --build

# Test your endpoint
curl -X POST http://localhost:8080/solve \
  -H "Content-Type: application/json" \
  -d '{
    "documents": [
      {
        "filename": "smlouva.pdf",
        "ocr_text": "POJISTNÁ SMLOUVA č. ABC-123\nPojistitel: Allianz pojišťovna a.s.\nDatum uzavření: 01.03.2024\nPočátek pojištění: 01.04.2024\nDoba trvání: neurčitá\nSplátky: ročně\nPojistné období: 12 měsíců\nMěna: CZK"
      }
    ]
  }'

# Check health
curl http://localhost:8080/

# Check token usage
curl http://localhost:8080/metrics
```

## Available tools

- **Gemini API** — use the pre-configured `gemini` object: `response = gemini.generate("your prompt")`. Token usage is tracked automatically.
- **PostgreSQL sidecar** — available at `DATABASE_URL` for caching. A `cache` table (key TEXT, value JSONB) is created on startup.

## Deployment

Push to your GitHub repo — Cloud Build will automatically build and deploy to Cloud Run.

## Tips

- Process all documents together — amendments modify the base contract
- `latestEndorsementNumber` should be the highest "dodatek" number found across all documents (as a string)
- "doba neurčitá" / no end date means `endAt: null`
- Installment frequency: "ročně"=1, "pololetně"=2, "čtvrtletně"=4, "měsíčně"=12
- The `concludedAs` field is typically "broker" for Renomia contracts (makléř)
- Date format must be `DD.MM.YYYY` — zero-padded
- Use the sidecar DB to cache extraction results
- Return `null` for fields that aren't present in the documents rather than guessing
