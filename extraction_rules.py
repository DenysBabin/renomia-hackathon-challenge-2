"""
Extraction rules for Czech insurance document field extraction.

Each field has a FieldRule definition that drives:
1. Prompt generation (extraction_rule text is included in the AI prompt)
2. Response validation (type, allowed_values, default)
3. Post-processing normalization
"""
from dataclasses import dataclass, field
from typing import Any


@dataclass
class FieldRule:
    """Definition for a single extraction field."""
    name: str                        # JSON field name (e.g., "contractNumber")
    type: str                        # "string", "enum", "number", "date", "boolean"
    description: str                 # Short English description
    extraction_rule: str             # Detailed instructions for the AI
    allowed_values: tuple = ()       # For enums/numbers: tuple of valid values
    default: Any = None              # Default if field missing or invalid
    nullable: bool = True            # Whether null is a valid output
    fallback_on_invalid: Any = None  # Value to use if AI returns invalid value


FIELD_RULES: list[FieldRule] = [
    FieldRule(
        name="contractNumber",
        type="string",
        description="Insurance contract number",
        extraction_rule=(
            "Extract the contract number (cislo smlouvy / cislo pojistne smlouvy). "
            "It often appears at the top of the document after 'Pojistna smlouva c.' or 'PS c.'. "
            "Preserve the original formatting including spaces (e.g., '3301 0150 23'). "
            "Return null only if truly not found anywhere in the documents."
        ),
        nullable=True,
    ),
    FieldRule(
        name="insurerName",
        type="string",
        description="Name of the insurance company (insurer/pojistitel)",
        extraction_rule=(
            "Extract the insurer name (pojistitel). This is the INSURANCE COMPANY, "
            "NOT the policyholder (pojistnik) and NOT the broker (makler). "
            "Look for it after 'Pojistitel:' label, in document headers, watermarks "
            "(e.g., 'Colonnade Confidential' means insurer is 'Colonnade'), "
            "or in the company identification section. "
            "Common Czech insurers: Allianz, Colonnade, Generali Ceska pojistovna, "
            "Kooperativa, UNIQA, CSOB Pojistovna, INTER PARTNER ASSISTANCE. "
            "If the insurer name field is blank/redacted, look for company names "
            "in watermarks, footers, or confidentiality notices. "
            "Return null only if truly impossible to determine."
        ),
        nullable=True,
    ),
    FieldRule(
        name="state",
        type="enum",
        description="Contract state: draft, accepted, or cancelled",
        extraction_rule=(
            "Determine the contract state. IMPORTANT DEFAULT RULE: "
            "Most insurance contracts that exist as documents are 'accepted' (signed/effective). "
            "Use 'accepted' as the DEFAULT unless there is EXPLICIT evidence otherwise. "
            "Use 'draft' ONLY if the document explicitly says it is a 'navrh' (proposal) "
            "or 'navrh pojistne smlouvy' and has NOT been accepted yet. "
            "Use 'cancelled' ONLY if there is explicit mention of cancellation "
            "(vypoved, storno, zruseni) that terminated the contract. "
            "DO NOT use 'draft' just because you cannot see signatures — "
            "OCR text almost never preserves signature information."
        ),
        allowed_values=("draft", "accepted", "cancelled"),
        default="accepted",
        nullable=False,
        fallback_on_invalid="accepted",
    ),
    FieldRule(
        name="assetType",
        type="enum",
        description="Type of insured asset",
        extraction_rule=(
            "Determine the type of insured asset. "
            "Use 'vehicle' ONLY for vehicle insurance (havarijni pojisteni, "
            "povinne ruceni, autopojisteni, pojisteni vozidla, when SPZ/RZ is mentioned). "
            "Use 'other' for everything else (property, liability, travel, health, etc.)."
        ),
        allowed_values=("other", "vehicle"),
        default="other",
        nullable=False,
        fallback_on_invalid="other",
    ),
    FieldRule(
        name="concludedAs",
        type="enum",
        description="Whether concluded as agent or broker",
        extraction_rule=(
            "Determine if the contract was concluded through an agent or broker. "
            "STRONG DEFAULT: 'broker'. Use 'broker' in most cases, especially when "
            "Renomia (a brokerage company) is mentioned anywhere. "
            "Use 'agent' ONLY if the document EXPLICITLY mentions the intermediary role "
            "as 'agent' or 'pojistovaci agent', or if the insurance company itself "
            "arranged the contract directly without any broker. "
            "If there is NO mention of broker or agent at all, return 'broker'. "
            "DO NOT return 'agent' just because you see no broker mentioned."
        ),
        allowed_values=("agent", "broker"),
        default="broker",
        nullable=False,
        fallback_on_invalid="broker",
    ),
    FieldRule(
        name="contractRegime",
        type="enum",
        description="Contract regime type",
        extraction_rule=(
            "Determine the contract regime. "
            "'frame' = ramcova smlouva (framework agreement). "
            "'fleet' = flotilova smlouva (fleet insurance). "
            "'coinsurance' = soupojisteni. "
            "'individual' = standard individual contract (DEFAULT). "
            "Use 'individual' unless the document explicitly indicates another regime."
        ),
        allowed_values=("individual", "frame", "fleet", "coinsurance"),
        default="individual",
        nullable=False,
        fallback_on_invalid="individual",
    ),
    FieldRule(
        name="startAt",
        type="date",
        description="Insurance start date (DD.MM.YYYY)",
        extraction_rule=(
            "Extract the insurance start date (pocatek pojisteni / pojisteni od / "
            "ucinnost od). Format: DD.MM.YYYY with zero-padding. "
            "If an amendment changes the start date, use the amended value."
        ),
        nullable=True,
    ),
    FieldRule(
        name="endAt",
        type="date",
        description="Insurance end date (DD.MM.YYYY) or null for indefinite",
        extraction_rule=(
            "Extract the insurance end date ONLY if EXPLICITLY stated in the contract. "
            "CRITICAL RULES: "
            "1. 'doba neurcita' (indefinite period) = return null. "
            "2. If the contract says it auto-renews (automaticky se prodluzuje) "
            "and there is no FIXED end date = return null. "
            "3. DO NOT CALCULATE end date from start date + duration. "
            "4. Only return a date if the contract EXPLICITLY states "
            "'konec pojisteni', 'pojisteni do', 'pojistna doba konci dnem', "
            "or has a specific end date clearly written. "
            "5. For fixed-term contracts (e.g., travel insurance with explicit end date), "
            "return the stated end date. "
            "6. If the contract has both a fixed initial period AND auto-renewal to "
            "indefinite, return null."
        ),
        nullable=True,
    ),
    FieldRule(
        name="concludedAt",
        type="date",
        description="Date contract was signed/concluded (DD.MM.YYYY)",
        extraction_rule=(
            "Extract the date the contract was signed or concluded "
            "(datum uzavreni / datum podpisu / datum sjednani / V ... dne ...). "
            "Format: DD.MM.YYYY. Return null if not found."
        ),
        nullable=True,
    ),
    FieldRule(
        name="installmentNumberPerInsurancePeriod",
        type="number",
        description="Number of premium installments per insurance period",
        extraction_rule=(
            "Determine how many premium installments per insurance period. "
            "Values: 1=annually (rocne), 2=semi-annually (pololetne), "
            "4=quarterly (ctvrtletne), 12=monthly (mesicne). "
            "For single-payment (jednorazove) contracts, use 1. "
            "Default: 1 if not specified."
        ),
        allowed_values=(1, 2, 4, 12),
        default=1,
        nullable=False,
        fallback_on_invalid=1,
    ),
    FieldRule(
        name="insurancePeriodMonths",
        type="number",
        description="Length of insurance period in months",
        extraction_rule=(
            "Determine the insurance period length (pojistne obdobi) in months. "
            "Values: 12=annual, 6=semi-annual, 3=quarterly, 1=monthly. "
            "CRITICAL: This is the BILLING/RENEWAL CYCLE length, NOT the total contract duration. "
            "DO NOT confuse contract duration (doba trvani, e.g., '3x1 rok', '3 roky') with "
            "the insurance period (pojistne obdobi). "
            "For single-payment (jednorazove) short-term insurance "
            "(e.g., travel insurance lasting weeks or a few months), use 1. "
            "Return null if the insurance period is not explicitly mentioned "
            "and cannot be reliably determined. "
            "DO NOT default to 12 — return null if unsure."
        ),
        allowed_values=(1, 3, 6, 12),
        default=None,
        nullable=True,
        fallback_on_invalid=None,
    ),
    FieldRule(
        name="premium.currency",
        type="string",
        description="Premium currency in ISO 4217 lowercase",
        extraction_rule=(
            "Extract the currency of the premium. "
            "Use ISO 4217 lowercase: czk, eur, usd. "
            "Default: 'czk' for Czech contracts."
        ),
        default="czk",
        nullable=False,
    ),
    FieldRule(
        name="premium.isCollection",
        type="boolean",
        description="Whether premium is collected through broker",
        extraction_rule=(
            "Determine if premium is collected (inkasovano) through a broker/intermediary. "
            "true = premium is collected via broker (inkaso pojistneho pres maklere). "
            "false = premium is paid directly to insurer or method not specified. "
            "IMPORTANT: This must always be true or false, never null. "
            "If there is no explicit mention of collection through broker, return false."
        ),
        default=False,
        nullable=False,
    ),
    FieldRule(
        name="actionOnInsurancePeriodTermination",
        type="enum",
        description="What happens when insurance period ends",
        extraction_rule=(
            "Determine what happens at the end of the insurance period. "
            "'auto-renewal' = contract auto-renews (automaticky se prodluzuje / "
            "pojisteni se obnovuje / doba neurcita). "
            "'policy-termination' = contract ends (pojisteni zanika / konci / jednorazove). "
            "Default: 'auto-renewal' if not specified. "
            "For fixed-term/single-payment contracts (jednorazove pojisteni), "
            "use 'policy-termination'."
        ),
        allowed_values=("auto-renewal", "policy-termination"),
        default="auto-renewal",
        nullable=False,
        fallback_on_invalid="auto-renewal",
    ),
    FieldRule(
        name="noticePeriod",
        type="string",
        description="Notice period for contract termination",
        extraction_rule=(
            "Extract the notice period (vypovedni lhuta) for terminating the contract. "
            "Valid values: 'six-weeks', 'three-months', 'two-months', 'one-month', 'eight-days'. "
            "CRITICAL RULES: "
            "1. Extract ONLY if the notice period is EXPLICITLY stated in the contract text. "
            "2. Common Czech phrases: '6 tydnu'/'sest tydnu' = 'six-weeks', "
            "'3 mesice'/'tri mesice' = 'three-months', '2 mesice' = 'two-months', "
            "'1 mesic' = 'one-month', '8 dnu'/'osm dnu' = 'eight-days'. "
            "3. DO NOT invent or assume a notice period that is not written in the text. "
            "4. Return null if no notice period is explicitly mentioned."
        ),
        allowed_values=("six-weeks", "three-months", "two-months", "one-month", "eight-days"),
        default=None,
        nullable=True,
    ),
    FieldRule(
        name="regPlate",
        type="string",
        description="Vehicle registration plate (SPZ/RZ)",
        extraction_rule=(
            "Extract the vehicle registration plate (SPZ / RZ / registracni znacka). "
            "Only applicable for vehicle insurance. Return null for non-vehicle insurance."
        ),
        nullable=True,
    ),
    FieldRule(
        name="note",
        type="string",
        description="Special conditions or notes",
        extraction_rule=(
            "ALWAYS return null. Notes are extracted separately by a different method."
        ),
        default=None,
        nullable=True,
    ),
]

# --- Convenience lookups derived from FIELD_RULES ---

RULES_BY_NAME: dict[str, FieldRule] = {r.name: r for r in FIELD_RULES}

ENUM_FIELDS: dict[str, tuple] = {
    r.name: r.allowed_values for r in FIELD_RULES if r.type == "enum"
}

ENUM_DEFAULTS: dict[str, Any] = {
    r.name: r.default for r in FIELD_RULES if r.type == "enum"
}

VALID_NOTICE_PERIODS: set[str] = set(
    RULES_BY_NAME["noticePeriod"].allowed_values
)

VALID_INSTALLMENTS: tuple = RULES_BY_NAME[
    "installmentNumberPerInsurancePeriod"
].allowed_values

VALID_PERIODS: tuple = RULES_BY_NAME["insurancePeriodMonths"].allowed_values


def build_extraction_prompt() -> str:
    """Generate the Gemini extraction prompt from field rules."""
    lines = [
        "You are an expert at extracting structured data from Czech insurance contracts (pojistne smlouvy).",
        "Extract the following fields from the OCR document text into a JSON object.",
        "",
        "FIELDS AND RULES:",
    ]

    for rule in FIELD_RULES:
        type_info = rule.type
        if rule.allowed_values:
            type_info = " | ".join(f'"{v}"' for v in rule.allowed_values)
        nullable_str = " | null" if rule.nullable else ""

        lines.append(f"- {rule.name}: {type_info}{nullable_str}")
        lines.append(f"  Rule: {rule.extraction_rule}")
        lines.append("")

    lines.extend([
        "IMPORTANT:",
        "- Amendments (dodatky) OVERRIDE base contract values — use the LATEST valid value.",
        "- Dates must be DD.MM.YYYY with zero-padding (01.03.2024, not 1.3.2024).",
        "- If a field cannot be determined from the text, return null (unless the field has a required default).",
        "- Respond with ONLY a valid JSON object. No explanations, no markdown wrapping.",
        "",
        "DOCUMENT TEXT:",
    ])

    return "\n".join(lines)
