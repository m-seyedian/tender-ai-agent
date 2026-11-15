import re
from typing import Dict, List, Optional


def extract_cpv_codes(text: Optional[str]) -> List[str]:
    """Extrait tous les codes CPV à 8 chiffres uniques depuis un texte (ex: '90511000-2' -> '90511000')."""
    if not text:
        return []

    # Les codes CPV européens comportent 8 chiffres, parfois suivis d'un tiret et d'une clé de contrôle
    matches = re.findall(r"\b(\d{8})(?:-\d)?\b", text)

    # Déduplication tout en conservant l'ordre d'apparition
    seen = set()
    unique_codes: List[str] = []
    for code in matches:
        if code not in seen:
            seen.add(code)
            unique_codes.append(code)

    return unique_codes


def count_keyword_matches(text: Optional[str], keywords: List[str]) -> Dict[str, int]:
    """Compte les occurrences exactes (insensibles à la casse) de chaque mot-clé dans le texte."""
    if not text or not keywords:
        return {}

    results: Dict[str, int] = {}
    for kw in keywords:
        clean_kw = kw.strip()
        if not clean_kw:
            continue
        # Recherche par mot entier pour éviter les faux positifs partiels
        pattern = rf"\b{re.escape(clean_kw)}\b"
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            results[clean_kw.lower()] = len(matches)

    return results
