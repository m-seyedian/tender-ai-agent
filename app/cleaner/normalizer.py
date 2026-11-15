import html
import re
from datetime import datetime
from typing import Optional


def clean_text(raw_text: Optional[str]) -> str:
    """Nettoie un texte brut (entités HTML, balises, retours à la ligne et espaces multiples)."""
    if not raw_text:
        return ""
    text = html.unescape(str(raw_text))
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_date(date_str: Optional[str]) -> Optional[str]:
    """Convertit différents formats de date (ISO, français DD/MM/YYYY) vers YYYY-MM-DD."""
    if not date_str:
        return None
    cleaned = str(date_str).strip()

    iso_match = re.match(r"^(\d{4}-\d{2}-\d{2})", cleaned)
    if iso_match:
        try:
            datetime.strptime(iso_match.group(1), "%Y-%m-%d")
            return iso_match.group(1)
        except ValueError:
            return None

    fr_match = re.match(r"^(\d{2})[/\-](\d{2})[/\-](\d{4})", cleaned)
    if fr_match:
        day, month, year = fr_match.groups()
        try:
            dt = datetime(int(year), int(month), int(day))
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            return None

    return None


def extract_amount_eur(text: Optional[str]) -> Optional[float]:
    """Extrait le premier montant financier en euros depuis un texte (ex: '150 000,50 € HT' -> 150000.5)."""
    if not text:
        return None

    pattern = r"(\d[\d\s\u00a0.]*(?:,\d{1,2})?)\s*(?:€|euros?|EUR)(?!\w)"
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if not match:
        return None

    raw_number = match.group(1)
    normalized = raw_number.replace(" ", "").replace("\u00a0", "").replace(".", "")
    normalized = normalized.replace(",", ".")

    try:
        return float(normalized)
    except ValueError:
        return None
