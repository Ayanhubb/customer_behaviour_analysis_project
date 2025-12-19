import json
import re
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pdfplumber
from rapidfuzz import fuzz, process

# Canonical JSBCL product names (used for fuzzy normalization).
# Note: If you have a more complete/updated list, you can extend this list safely.
CANONICAL_PRODUCTS: List[str] = [
    "MEAKINS 10000 ULTRA PREMIUM SUPER STRONG BEER",
    "ORIGINAL BIRA 91 WHITE BEER",
    "GOLDBERG PREMIUM STRONG",
    "WOODPECKER CREST STRONG PREMIUM BEER",
    "TUBORG PREMIUM MILD 1880 DANISH BEER",
    "SHERA STRONG BEER",
    "BLOCKBUSTER ULTRA STRONG BEER",
    "OLD MONK THE ORIGINAL PREMIUM STRONG BEER",
    "CARLSBERG SUPREME STRONG SUPER PREMIUM BEER",
    "WOODPECKER REFRESHING LAGER BEER",
    "BLOCKBUSTER ULTRA LAGER BEER",
    "HUNTER REFRESHING STRONG PREMIUM BEER",
    "HUNTER PLATINA STRONG PREMIUM BEER",
    "CARLSBERG SMOOTH SUPER PREMIUM MILD BEER",
    "MAHARANI BLUE LABLE SUPER STRONG PREMIUM BEER",
    "BIRA 91 SUPERFRESH NATURAL WHITE BEER",
    "CARLSBERG SMOOTH PREMIUM LAGER",
    "BAD MONKEY STRONG PREMIUM BEER",
    "KINGFISHER SPECIAL LAGER BEER",
    "TIPSY AMERICAN PILSNER STRONG BEER",
    "KINGFISHER STORM STRONG BEER",
    "GOLDEN HART SUPER STRONG BEER",
    "PRESIDENT 5000 SUPER STRONG BEER",
    "COPTER 7 STRONG BEER PREMIUM",
    "BLACKBUCK PREMIUM STRONG BEER",
    "GODFATHER LAGER INTERNATIONAL QUALITY BEER",
    "BIRA 91 GOLD WHEAT STRONG BEER",
    "KOTSBERG FINEST PILS BEER",
    "ORIGINAL BIRA91 BOOM SUPER STRONG BEER",
    "THUNDER BOLT CLASSIC SUPER STRONG BEER",
    "GODFATHER SUPER 8 STRONG BEER",
    "KINGFISHER PREMIUM LAGER BEER",
    "BIRA 91 BLONDE SUMMER LAGER BEER",
    "BUDWEISER UNIVERSAL KING OF BEERS (CROWN)",
    "BUDWEISER KING OF BEER",
    "HUNTER SUPER STRONG PREMIUM BEER",
    "BUDWEISER MAGNUM UNIVERSAL KING OF BEERS (CROWN)",
    "SIX FIELDS BRUTE THE MAJESTIC BEER",
    "COPTER 7 STRONG BEER SELECT",
    "PROOST SUPREME BEER STRONG",
    "SIX FIELDS PILSNER GOOD HOPS BEER",
    "HUNTER REFRESHING MILD BEER PREMIUM LAGER",
    "SIMBA ROAR SERIES STRONG PREMIUM BEER",
    "SIX FIELDS CULT PREMIUM STRONG WHEAT BEER",
    "KOTSBERG STRONG PREMIER BEER",
    "GODFATHER THE LEGENDARY LUXURY LAGER BEER",
    "KOTSBERG PREMIUM PILS BEER",
    "CARLSBERG PREMIUM ELEPHANT STRONG BEER",
    "COX 10000 STRONG PREMIUM BEER",
    "COPTER 7 SMOOTH LAGER PREMIUM BEER",
    "KINGFISHER SUPREME STRONG BEER",
    "TUBORG SUPREME STRONG 1880 DANISH BEER",
    "BLOCKBUSTER STRONG BEER",
    "BIRA 91 RISE RICE STRONG LAGER BEER",
    "BAD MONKEY EXTRA STRONG BEER",
    "KINGFISHER ULTRA LAGER BEER",
    "AUSTENBERG ULTRA PREMIUM STRONG BEER",
    "SAB PREMIUM STRONG (AUSTRALIAN CRAFT) BEER",
    "KINGFISHER ULTRA MAX PREMIUM STRONG BEER",
    "TRIPLE CROWN PREMIUM BLENDED BRANDY",
    "BUDWEISER MAGNUM PREMIUM BEER",
    "GODFATHER THE LEGENDARY CLASSIC STRONG BEER",
    "HEINEKEN LAGER BEER",
    "TIGER",
    "8PM PREMIUM BLACK ELITE WHISKY",
    "8PM PREMIUM BLACK ELITE WHISKY (PET BOTTLE)",
    "MC DOWELL'S NO1 LUXURY SPECIAL WHISKY",
    "STERLING RESERVE B7 ORIGINAL BLENDED WHISKY",
]

CANONICAL_PET_PRODUCTS = [p for p in CANONICAL_PRODUCTS if "PET" in p.upper()]
CANONICAL_NON_PET_PRODUCTS = [p for p in CANONICAL_PRODUCTS if "PET" not in p.upper()]


# -------- regex patterns ----------
DATE_RE = re.compile(r"(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}|\d{4}[\/\-]\d{1,2}[\/\-]\d{1,2})")
PERMIT_NO_RE = re.compile(r"Excise Permit No[:\s]*([A-Z0-9\-\_/]+)", re.IGNORECASE)
ISSUED_LABEL_RE = re.compile(r"(?:Issued Date|Date of Issue|Date\s+of\s+Issue)[:\s]*([^\n]+)", re.IGNORECASE)
TLIN_RE = re.compile(
    r"\b(?:Tlin|TIN|Tin|Tiin|Tin\/Tlin|Tlin\/Tin)\b[^\dA-Z0-9\-]{0,6}?([0-9A-Z\-/]{6,20})",
    re.IGNORECASE,
)
TOTAL_DUTY_RE = re.compile(r"Total Duty Deposited with\s*\(rs\)\s*[:\s]*([^\n]+)", re.IGNORECASE)
TOTAL_FEES_RE = re.compile(r"Total Fees Deposited with\s*\(rs\)\s*[:\s]*([^\n]+)", re.IGNORECASE)

UNIT_RE = re.compile(r"\b\d{1,4}\s*ML(?:\s*\(FML\))?\b", re.IGNORECASE)
_RE_UNIT_ANY = re.compile(r"\b\d{1,4}\s*(?:ML|L|G|KG|PCS|NOS|BOTTLE|BOTTLES)\b", re.IGNORECASE)
NUMERIC_SIMPLE = re.compile(r"^\d")
STRENGTH_RE = re.compile(r"\d+(?:\.\d+)?-BL", re.IGNORECASE)

LICENSE_RE = re.compile(r"\b[0-9]{1,4}[\-]?(?:COM|Com|com)[A-Z0-9\/\-]{3,30}\b", re.IGNORECASE)

_HEADER_TRAILING_RE = re.compile(
    r"\b("
    r"ISSUING AUTHORITY|PERMIT ISSUING AUTHORITY|SIGNATURE OF PERMIT ISSUING AUTHORITY|"
    r"SIGNATURE OF OFFICER-IN-CHARGE|SIGNATURE OF OFFICER|SIGNATURE OF|SIGNATURE|"
    r"TRANSPORT DETAIL(S)?|TRANSPORT DETAILS|TRANSPORT DETAIL:|TOTAL QUANTITY|PAGE FROM|PAGE\b"
    r")\b",
    re.IGNORECASE,
)

_TRASH_WORDS = {
    "address",
    "mobile",
    "email",
    "tin",
    "tlin",
    "license",
    "licence",
    "registered",
    "permit",
    "no",
    "no.",
    "name",
    "qty",
    "unit",
    "physical",
    "stockqty",
    "strength",
    "page",
    "from",
    "destination",
    "by",
    "road",
    "rail",
    "air",
    "transport",
    "detail",
    "details",
    "issuing",
    "authority",
    "signature",
    "permitissuing",
    "total",
    "quantity",
}

# Beverage categories to force uppercase when they appear as the last word
CATEGORIES = {
    "whisky",
    "whiskey",
    "vodka",
    "rum",
    "beer",
    "brandy",
    "wine",
    "wines",
    "liqueur",
    "tequila",
    "gin",
    "cognac",
    "scotch",
}


def normalize_for_match(s: str) -> str:
    if not s:
        return ""

    s = s.upper()
    s = re.sub(
        r"\b(PET|BOTTLE|FMFL|FML|ML|L|UP|WHISKY\(FMFL\)|WINE\(FMFL\)|VODKA\(FMFL\))\b",
        " ",
        s,
    )
    s = re.sub(r"[\(\)\[\]]", " ", s)
    s = re.sub(r"[^A-Z0-9& ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def has_pet(label: str) -> bool:
    if not label:
        return False
    return bool(re.search(r"\bPET\b", label.upper()))


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def extract_text_pages(path: Path) -> Tuple[str, List[str]]:
    pages_text = []
    with pdfplumber.open(path) as pdf:
        for p in pdf.pages:
            pages_text.append(p.extract_text() or "")
    return "\n".join(pages_text), pages_text


def fuzzy_match_product(label_name: str, kind: str, min_score: int = 85) -> str:
    if not label_name:
        return ""

    query_raw = label_name.upper()
    query = normalize_for_match(label_name)
    is_pet_label = has_pet(label_name)

    def extract_years(s: str):
        m = re.search(r"\b(\d{2})\s*YEARS?(?:\s*AGE)?\b", s.upper())
        return m.group(1) if m else None

    def has_flavour_word(s: str):
        return bool(re.search(r"\b(CITRON|LEMON|ORANGE|APPLE|MANGO|CRANBERRY|FLAVOURED|FLAVORED)\b", s.upper()))

    def extract_brand(s: str):
        toks = s.upper().split()
        return toks[0] if toks else ""

    query_brand = extract_brand(query_raw)

    def penalize(candidate: str) -> bool:
        c = candidate.upper()

        qy = extract_years(query_raw)
        cy = extract_years(c)
        if qy and cy and qy != cy:
            return True

        if not has_flavour_word(query_raw) and has_flavour_word(c):
            return True

        if re.search(r"\b(ULTRA|MAX)\b", c) and not re.search(r"\b(ULTRA|MAX)\b", query_raw):
            return True

        if re.search(r"\b10000\b", c) and not re.search(r"\b10000\b", query_raw):
            return True

        cand_brand = extract_brand(c)
        if query_brand and cand_brand and query_brand != cand_brand:
            return True

        return False

    candidates = CANONICAL_PET_PRODUCTS if is_pet_label else CANONICAL_NON_PET_PRODUCTS

    best = None
    best_score = 0
    for cand, score, _ in process.extract(query, candidates, scorer=fuzz.token_set_ratio, limit=10):
        if score < min_score:
            continue
        if penalize(cand):
            continue
        if score > best_score:
            best = cand
            best_score = score

    return best or ""


def block_between(text: str, starts, ends, default_len=3000) -> str:
    start_pos = None
    for s in starts:
        m = re.search(re.escape(s), text, re.IGNORECASE)
        if m:
            start_pos = m.end()
            break
    if start_pos is None:
        return ""
    end_pos = None
    for e in ends:
        m2 = re.search(re.escape(e), text[start_pos:], re.IGNORECASE)
        if m2:
            end_pos = start_pos + m2.start()
            break
    if end_pos is None:
        return text[start_pos : start_pos + default_len].strip()
    return text[start_pos:end_pos].strip()


def _dedupe_label_text(s: str) -> str:
    if not s:
        return s
    s = re.sub(r"\s+", " ", s).strip()
    words = s.split()
    n = len(words)
    if n < 4:
        return s
    if n % 2 == 0:
        half = n // 2
        if words[:half] == words[half:]:
            return " ".join(words[:half])
    max_block = min(max(3, n // 2), 12)
    for block_size in range(max_block, 2, -1):
        if n >= block_size * 2:
            if words[:block_size] == words[block_size : 2 * block_size]:
                return " ".join(words[:block_size] + words[2 * block_size :])
            if words[-block_size:] == words[-2 * block_size : -block_size]:
                return " ".join(words[:-block_size])
    joined = " ".join(words)
    m = re.search(r"\b((?:\w+[^\w]+){2,}\w+)\s+\1\b", joined, flags=re.IGNORECASE)
    if m:
        phrase = m.group(1).strip()
        joined = re.sub(re.escape(m.group(0)), phrase, joined, count=1, flags=re.IGNORECASE)
        return re.sub(r"\s+", " ", joined).strip()
    return s


def _group_words_into_rows(words: List[dict], y_tolerance: float = 3.0) -> List[List[dict]]:
    if not words:
        return []
    for w in words:
        w["mid_y"] = (w.get("top", 0) + w.get("bottom", 0)) / 2.0
    words_sorted = sorted(words, key=lambda w: w["mid_y"])
    rows = []
    current_row = [words_sorted[0]]
    current_y = words_sorted[0]["mid_y"]
    for w in words_sorted[1:]:
        if abs(w["mid_y"] - current_y) <= y_tolerance:
            current_row.append(w)
        else:
            rows.append(sorted(current_row, key=lambda x: x["x0"]))
            current_row = [w]
            current_y = w["mid_y"]
    rows.append(sorted(current_row, key=lambda x: x["x0"]))
    return rows


def _infer_column_boundaries(rows: List[List[dict]], ncols_guess: int = None) -> List[float]:
    if not rows:
        return []
    xs = [w["x0"] for r in rows for w in r if "x0" in w]
    if not xs:
        return []
    xs = sorted(xs)
    diffs = [j - i for i, j in zip(xs[:-1], xs[1:])] if len(xs) > 1 else [0]
    th = (statistics.median(diffs) if diffs else 10) * 4
    clusters = []
    current = [xs[0]]
    for gap, x in zip(diffs, xs[1:]):
        if gap > th:
            clusters.append(current)
            current = [x]
        else:
            current.append(x)
    clusters.append(current)
    return [statistics.median(c) for c in clusters]


def _words_in_box(words: List[dict], x0: float, x1: float, top: float, bottom: float) -> List[dict]:
    res = []
    for w in words:
        if w["x1"] >= x0 - 1 and w["x0"] <= x1 + 1 and w["bottom"] >= top - 1 and w["top"] <= bottom + 1:
            res.append(w)
    return res


def extract_cell_text_from_region(
    page_words: List[dict], region_top: float, region_bottom: float, label_col_xcenter: float, col_half_width: float = 80.0
) -> str:
    x0 = label_col_xcenter - col_half_width
    x1 = label_col_xcenter + col_half_width
    candidate_words = _words_in_box(page_words, x0, x1, region_top, region_bottom)
    if not candidate_words:
        return ""
    rows = _group_words_into_rows(candidate_words, y_tolerance=4.0)
    lines = []
    for r in rows:
        line = " ".join([w["text"] for w in sorted(r, key=lambda x: x["x0"])])
        lines.append(line)
    joined = " ".join([ln.strip() for ln in lines if ln.strip()])
    return _dedupe_label_text(joined)


def _find_header_column_centers(rows: List[List[dict]], header_tokens: List[str]) -> List[float]:
    if not rows:
        return []
    header_row = None
    for r in rows[:6]:
        text = " ".join(w.get("text", "") for w in r).lower()
        if any(tok.lower() in text for tok in header_tokens):
            header_row = r
            break
    if not header_row:
        for r in rows:
            text = " ".join(w.get("text", "") for w in r).lower()
            if any(tok.lower() in text for tok in header_tokens):
                header_row = r
                break
    if not header_row:
        return []
    header_centers = [((w.get("x0", 0) + w.get("x1", 0)) / 2.0) for w in header_row]
    if not header_centers:
        return []
    header_centers = sorted(header_centers)
    clusters = []
    current = [header_centers[0]]
    for a, b in zip(header_centers[:-1], header_centers[1:]):
        if b - a > 20:
            clusters.append(current)
            current = [b]
        else:
            current.append(b)
    clusters.append(current)
    return [statistics.median(c) for c in clusters]


def is_plausible_bl(s: Optional[str]) -> bool:
    if not s:
        return False
    s2 = str(s).strip().replace(",", "")
    if re.fullmatch(r"\d{1,8}(?:\.\d{1,4})?", s2):
        try:
            v = float(s2)
            return v > 0
        except Exception:
            return False
    return False


def _best_numeric_from_words(words: List[dict]) -> Optional[str]:
    if not words:
        return None
    txt = " ".join(w.get("text", "") for w in sorted(words, key=lambda x: x.get("x0", 0)))
    matches = re.findall(r"(\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)", txt)
    if not matches:
        return None
    cleaned = [(m.replace(",", ""), len(re.sub(r"\D", "", m))) for m in matches]
    cleaned_sorted = sorted(cleaned, key=lambda t: (-t[1], -len(t[0])))
    candidate = cleaned_sorted[0][0].strip()
    return candidate if is_plausible_bl(candidate) else None


def extract_bl_from_pdf_cell(pdf_path: Path, row_anchor: str) -> Optional[str]:
    if not row_anchor:
        return None

    header_tokens = [
        "equivalent to bl",
        "equivalent to  bl",
        "equivalent to lpl",
        "equivalent to lpl/bl",
        "equivalent to lpl equivalent to bl",
    ]

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            words = page.extract_words(use_text_flow=True)
            if not words:
                continue

            rows = _group_words_into_rows(words, y_tolerance=4.0)
            if not rows:
                continue

            target_row = None
            anchor_low = row_anchor.lower().strip()
            for r in rows:
                row_text = " ".join(w.get("text", "") for w in r).lower()
                if anchor_low and anchor_low in row_text:
                    target_row = r
                    break

            if target_row is None:
                anchor_tokens = [t for t in re.split(r"\s+", row_anchor.strip()) if t]
                for r in rows:
                    rt = " ".join(w.get("text", "") for w in r).lower()
                    match_count = sum(1 for tok in anchor_tokens[:4] if tok.lower() in rt)
                    if match_count >= 2:
                        target_row = r
                        break

            if not target_row:
                continue

            top = min(w["top"] for w in target_row) - 3
            bottom = max(w["bottom"] for w in target_row) + 3

            col_centers_by_header = _find_header_column_centers(rows, header_tokens)
            if col_centers_by_header:
                header_row = None
                for r in rows[:6]:
                    text = " ".join(w.get("text", "") for w in r).lower()
                    if any(tok in text for tok in header_tokens):
                        header_row = r
                        break
                if header_row is None:
                    for r in rows:
                        text = " ".join(w.get("text", "") for w in r).lower()
                        if any(tok in text for tok in header_tokens):
                            header_row = r
                            break
                if header_row:
                    header_word_centers = [
                        (((w.get("x0", 0) + w.get("x1", 0)) / 2.0), w.get("text", "")) for w in header_row
                    ]
                    bl_candidate_centers = []
                    for hc, txt in header_word_centers:
                        if "bl" in txt.lower() or "equivalent" in txt.lower():
                            nearest = min(col_centers_by_header, key=lambda c: abs(c - hc))
                            bl_candidate_centers.append(nearest)
                    if bl_candidate_centers:
                        bl_centers = sorted(set(bl_candidate_centers))
                        chosen_center = bl_centers[-1]
                        x0 = chosen_center - 70
                        x1 = chosen_center + 70
                        candidate_words = [
                            w
                            for w in words
                            if w["x0"] >= x0 - 1 and w["x1"] <= x1 + 1 and w["top"] >= top and w["bottom"] <= bottom
                        ]
                        if not candidate_words:
                            candidate_words = [
                                w
                                for w in words
                                if w["top"] >= top
                                and w["bottom"] <= bottom
                                and abs(((w.get("x0", 0) + w.get("x1", 0)) / 2.0) - chosen_center) <= 120
                            ]
                        cand = _best_numeric_from_words(candidate_words)
                        if cand and is_plausible_bl(cand):
                            return cand

            page_row_words = [w for w in words if w["top"] >= top and w["bottom"] <= bottom]
            if not page_row_words:
                continue
            page_row_words_sorted = sorted(page_row_words, key=lambda x: x["x0"], reverse=True)
            for start in range(0, min(8, len(page_row_words_sorted))):
                cluster = page_row_words_sorted[start : start + 6]
                cand = _best_numeric_from_words(cluster)
                if cand and is_plausible_bl(cand):
                    return cand

            row_text_full = " ".join(w.get("text", "") for w in target_row)
            m = re.search(r"(\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)", row_text_full)
            if m:
                cand = m.group(1).replace(",", "")
                if is_plausible_bl(cand):
                    return cand

    return None


def _filter_trailing_tokens(trailing_text: str, existing_label_text: str, max_tokens=6):
    if not trailing_text:
        return []
    if _HEADER_TRAILING_RE.search(trailing_text):
        return []
    toks = re.split(r"[\s,;:]+", trailing_text.strip())
    out = []
    seen = set()
    exist_lower = existing_label_text.lower()
    for t in toks:
        if len(out) >= max_tokens:
            break
        tt = re.sub(r"^[^\w&/+\-]+|[^\w&/+\-]+$", "", t)
        if not tt:
            continue
        if re.fullmatch(r"\d+(?:\.\d+)?", tt):
            continue
        low = tt.lower()
        if low in _TRASH_WORDS:
            continue
        if len(tt) <= 1:
            continue
        if low in exist_lower:
            continue
        if low in seen:
            continue
        out.append(tt)
        seen.add(low)
    return out


_RE_STRENGTH = re.compile(r"\d+(?:\.\d+)?-BL", re.IGNORECASE)
_RE_QTY = re.compile(r"^\d+(?:\.\d+)?$")
_RE_ZERO = re.compile(r"^\s*0\s*$")
_RE_MULTI_SP = re.compile(r"\s+")


def _apply_category_caps(label: str) -> str:
    if not label:
        return label
    parts = label.split()
    if not parts:
        return label
    last = parts[-1].strip().strip(".,;:")
    if last.lower() in CATEGORIES:
        parts[-1] = last.upper()
    return " ".join(parts)


def clean_label_row_dict(row: dict) -> dict:
    r = dict(row)
    kind = (r.get("Kind Of Intoxicant") or "").strip()
    label = (r.get("Label Name") or "").strip()
    if label:
        label = _RE_UNIT_ANY.sub(" ", label)
        label = _RE_STRENGTH.sub(" ", label)
        toks = [t for t in _RE_MULTI_SP.split(label) if t.strip()]
        kept = []
        for tok in toks:
            if _RE_ZERO.match(tok):
                continue
            if (_RE_QTY.match(tok) and tok.isdigit()) or re.fullmatch(r"\d+\.\d+", tok):
                continue
            kept.append(tok)
        if kind:
            kept = [t for t in kept if t.lower() != kind.lower()]
        cleaned = " ".join(kept).strip()
        if kind and not cleaned.lower().endswith(kind.lower()):
            cleaned = (cleaned + " " + kind).strip()
        cleaned = _RE_MULTI_SP.sub(" ", cleaned)
        cleaned = _dedupe_label_text(cleaned)
        cleaned = _apply_category_caps(cleaned)
        r["Label Name"] = cleaned

    unit = r.get("Unit") or ""
    if unit:
        r["Unit"] = _RE_MULTI_SP.sub("", unit).upper()

    for key in ("Physical Qty", "StockQty", "Qty", "Quantity", "Equivalent To BL"):
        if key in r and r[key]:
            val = str(r[key]).strip().replace(",", "").strip()
            m = re.search(r"(\d+(?:\.\d+)?)", val)
            r[key] = m.group(1) if m else ""
    return r


def fallback_extract_label_from_pdf(pdf_path: Path, anchor_text: str, approx_label_fragment: str) -> Optional[str]:
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_words = page.extract_words(use_text_flow=True)
            if not page_words:
                continue
            page_text = " ".join([w["text"] for w in page_words])
            if anchor_text and anchor_text.lower() not in page_text.lower() and approx_label_fragment.lower() not in page_text.lower():
                continue
            rows = _group_words_into_rows(page_words, y_tolerance=4.0)
            if not rows:
                continue
            col_centers = _infer_column_boundaries(rows)
            if not col_centers:
                xs = [w["x0"] for w in page_words]
                if not xs:
                    continue
                col_centers = [statistics.median(xs)]

            target_row = None
            for r in rows:
                row_text = " ".join([w["text"] for w in r])
                if approx_label_fragment.lower() in row_text.lower() or (anchor_text and anchor_text.lower() in row_text.lower()):
                    target_row = r
                    break
            if not target_row:
                continue

            top = min(w["top"] for w in target_row) - 4
            bottom = max(w["bottom"] for w in target_row) + 8

            unit_x0 = None
            for w in target_row:
                if UNIT_RE.search(w["text"]):
                    unit_x0 = w["x0"]
                    break

            chosen_center = None
            if unit_x0:
                left_centers = [c for c in col_centers if c < unit_x0]
                if left_centers:
                    chosen_center = max(left_centers)
            if not chosen_center:
                chosen_center = col_centers[max(0, len(col_centers) // 2 - 1)]

            cell_text = extract_cell_text_from_region(page_words, top, bottom, chosen_center, col_half_width=80.0)
            if cell_text:
                return norm(_dedupe_label_text(cell_text))

    return None


def parse_invoice_row(row_text: str, pdf_path: Path) -> dict:
    s = norm(row_text)
    m_unit = UNIT_RE.search(s)
    if not m_unit:
        return {}
    unit = m_unit.group(0).strip()
    pre = s[: m_unit.start()].strip()
    post = s[m_unit.end() :].strip()

    m_pre = re.match(r"^(\d+)\s+(\S+)\s+(.+)$", pre)
    if not m_pre:
        return {}
    idx = m_pre.group(1)
    kind = m_pre.group(2)
    label_main = m_pre.group(3)

    post_tokens = [t.strip().strip(",") for t in post.split() if t.strip()]
    extra_label_after_unit = []
    while post_tokens and not NUMERIC_SIMPLE.match(post_tokens[0]) and not STRENGTH_RE.match(post_tokens[0]):
        extra_label_after_unit.append(post_tokens.pop(0))

    numeric_tokens = [t for t in post_tokens if re.match(r"^\d", t)]
    strength = ""
    for tk in post_tokens:
        if STRENGTH_RE.search(tk):
            strength = tk
            break

    last_num_match = None
    for m in re.finditer(r"(\d{1,3}(?:,\d{3})*(?:\.\d+)?)", s):
        last_num_match = m

    trailing_words = []
    if last_num_match:
        trailing = s[last_num_match.end() :].strip()
        if _HEADER_TRAILING_RE.search(trailing):
            trailing = ""
        if trailing:
            existing_label_text = " ".join([label_main] + extra_label_after_unit)
            trailing_words = _filter_trailing_tokens(trailing, existing_label_text, max_tokens=6)

    label_full = norm(" ".join([label_main] + extra_label_after_unit + trailing_words))

    needs_fallback = False
    if len(label_full) < max(20, len(label_main) + 5):
        needs_fallback = True
    if _HEADER_TRAILING_RE.search(label_full):
        needs_fallback = True
    if needs_fallback:
        anchor = f"{idx} {kind}"
        fb = fallback_extract_label_from_pdf(pdf_path, anchor, label_main)
        if fb:
            label_full = _dedupe_label_text(fb)

    label_full = _dedupe_label_text(label_full)

    initial_bl = numeric_tokens[-1] if numeric_tokens else ""

    parsed = {
        "Comodity Group": idx,
        "Kind Of Intoxicant": kind,
        "Label Name": label_full,
        "Unit": unit,
        "Physical Qty": numeric_tokens[0] if len(numeric_tokens) >= 1 else "",
        "StockQty": numeric_tokens[1] if len(numeric_tokens) >= 2 else "",
        "Qty": numeric_tokens[2] if len(numeric_tokens) >= 3 else "",
        "Strength Of Liquiors/Spirits": strength,
        "Equivalent To BL": initial_bl,
    }

    anchor = f"{idx} {kind}"
    fixed_bl = extract_bl_from_pdf_cell(pdf_path, anchor)
    if fixed_bl and is_plausible_bl(fixed_bl):
        parsed["Equivalent To BL"] = fixed_bl

    return clean_label_row_dict(parsed)


def parse_dispatched_row(row_text: str, pdf_path: Path) -> dict:
    s = norm(row_text)
    m_unit = UNIT_RE.search(s)
    if not m_unit:
        return {}
    unit = m_unit.group(0).strip()
    pre = s[: m_unit.start()].strip()
    post = s[m_unit.end() :].strip()

    m_pre = re.match(r"^(\d+)\s+(\S+)\s+(.+)$", pre)
    if not m_pre:
        return {}
    idx, kind, label_main = m_pre.group(1), m_pre.group(2), m_pre.group(3)

    post_tokens = [t.strip().strip(",") for t in post.split() if t.strip()]
    extra_label_after_unit = []
    while post_tokens and not NUMERIC_SIMPLE.match(post_tokens[0]) and not STRENGTH_RE.match(post_tokens[0]):
        extra_label_after_unit.append(post_tokens.pop(0))
    numeric_tokens = [t for t in post_tokens if re.match(r"^\d", t)]
    qty = post_tokens[0] if post_tokens and re.match(r"^\d", post_tokens[0]) else ""
    strength = ""
    for tk in post_tokens[1:]:
        if STRENGTH_RE.search(tk):
            strength = tk
            break

    last_num_match = None
    for m in re.finditer(r"(\d{1,3}(?:,\d{3})*(?:\.\d+)?)", s):
        last_num_match = m

    trailing_words = []
    if last_num_match:
        trailing = s[last_num_match.end() :].strip()
        if _HEADER_TRAILING_RE.search(trailing):
            trailing = ""
        if trailing:
            existing_label_text = " ".join([label_main] + extra_label_after_unit)
            trailing_words = _filter_trailing_tokens(trailing, existing_label_text, max_tokens=6)

    label_full = norm(" ".join([label_main] + extra_label_after_unit + trailing_words))

    needs_fallback = False
    if len(label_full) < max(20, len(label_main) + 5):
        needs_fallback = True
    if _HEADER_TRAILING_RE.search(label_full):
        needs_fallback = True
    if needs_fallback:
        anchor = f"{idx} {kind}"
        fb = fallback_extract_label_from_pdf(pdf_path, anchor, label_main)
        if fb:
            label_full = _dedupe_label_text(fb)

    label_full = _dedupe_label_text(label_full)

    initial_bl = numeric_tokens[-1] if numeric_tokens else ""

    parsed = {
        "Comodity Group": idx,
        "Kind Of Intoxicant": kind,
        "Label Name": label_full,
        "Unit": unit,
        "Quantity": qty,
        "Strength Of Liquiors/Spirits": strength,
        "Equivalent To BL": initial_bl,
    }

    anchor = f"{idx} {kind}"
    fixed_bl = extract_bl_from_pdf_cell(pdf_path, anchor)
    if fixed_bl and is_plausible_bl(fixed_bl):
        parsed["Equivalent To BL"] = fixed_bl

    return clean_label_row_dict(parsed)


PIN_RE = re.compile(r"\bPin[:\s-]*\d{6}\b", re.IGNORECASE)
PHONE_RE = re.compile(r"\b(?:Mobile|Phone|Tel|Contact)[:\s-]*([\d+\-()\s]{6,})", re.IGNORECASE)
DISTRICT_LABEL_RE = re.compile(r"\b(District|State|District\/State)[:\s-]*([A-Za-z\s,&-]+)", re.IGNORECASE)
BOUNDARY_WORDS = r"(?:Name\s+And\s+Style\s+Of\s+Business|Address|District|Tin|Licencee|Licence|License|Mobile|Pin|Email|Registered Permit|Phone|Contact)"
BOUNDARY_RE = re.compile(BOUNDARY_WORDS, re.IGNORECASE)


def parse_dealer(block: str) -> dict:
    out = {"Name": "", "TIN/Tlin": "", "License No": ""}
    if not block:
        return out

    lines = [l.rstrip() for l in block.splitlines() if l.strip()]
    for ln in lines:
        ln_stripped = ln.strip()
        if ln_stripped and not re.match(r"^(TIN|Tlin|Name|Licence|License|Tin|Name and Tin)", ln_stripped, re.IGNORECASE):
            out["Name"] = norm(ln_stripped)
            break

    for ln in lines:
        if not out.get("TIN/Tlin"):
            m_t = TLIN_RE.search(ln)
            if m_t:
                out["TIN/Tlin"] = m_t.group(1).strip()
        if not out.get("License No"):
            m_lic = LICENSE_RE.search(ln)
            if m_lic:
                out["License No"] = m_lic.group(0).strip()

    if not out["TIN/Tlin"]:
        m_any_digits = re.search(r"\b(\d{9,14})\b", block)
        if m_any_digits:
            out["TIN/Tlin"] = m_any_digits.group(1).strip()

    if not out["License No"] and lines:
        first_line_tokens = re.split(r"\s{2,}|\t", lines[0])
        for tok in first_line_tokens[1:]:
            if LICENSE_RE.search(tok):
                out["License No"] = LICENSE_RE.search(tok).group(0).strip()
                break
            if "com" in tok.lower() and len(tok) > 6:
                out["License No"] = tok.strip()
                break

    out["Name"] = norm(out["Name"])
    out["TIN/Tlin"] = out["TIN/Tlin"].strip() if out["TIN/Tlin"] else ""
    out["License No"] = out["License No"].strip() if out["License No"] else ""
    return out


def parse_party(block: str) -> Dict[str, str]:
    out = {"Name": "", "TIN": "", "Name And Style Of Business": "", "Address": "", "District/State": ""}
    if not block:
        return out

    block = block.strip()
    m_tin = TLIN_RE.search(block)
    if m_tin:
        out["TIN"] = norm(m_tin.group(1))
    lines = [l.strip() for l in block.splitlines() if l.strip()]

    m_name_label = re.search(r"Name\s+of\s+(Consignor|Consignee)\b", block, re.IGNORECASE)
    if m_name_label:
        after = block[m_name_label.end() :].lstrip()
        stop = BOUNDARY_RE.search(after)
        if stop:
            name_raw = after[: stop.start()]
        else:
            name_raw = after.splitlines()[0] if after.splitlines() else after
        out["Name"] = norm(name_raw)
    else:
        for ln in lines:
            if not re.search(r"^(TIN|Tin|Name\s+And\s+Style|Address|District|Mobile|Phone|Email|Licence)", ln, re.IGNORECASE) and len(ln) > 1:
                out["Name"] = ln
                break

    name_idx = None
    if out["Name"]:
        for i, ln in enumerate(lines):
            if out["Name"] and out["Name"] in ln:
                name_idx = i
                break
    if name_idx is None:
        name_idx = 0 if lines else None

    addr_lines = []
    if name_idx is not None:
        for ln in lines[name_idx + 1 : name_idx + 8]:
            if not ln:
                continue
            if BOUNDARY_RE.search(ln):
                break
            if PHONE_RE.search(ln) or PIN_RE.search(ln) or re.search(r"\b(Licence|License|Tin|Email|Registered Permit)\b", ln, re.IGNORECASE):
                break
            addr_lines.append(ln)

    if not addr_lines:
        m_addr = re.search(
            r"Address\s*[:\-\s]?\s*((?:[^\n][\n]?)+?)(?=(?:\n\s*(?:"
            + BOUNDARY_WORDS
            + r")\b)|$)",
            block,
            re.IGNORECASE | re.DOTALL,
        )
        if m_addr:
            addr_text = m_addr.group(1)
            addr_lines = [l.strip() for l in addr_text.splitlines() if l.strip()][:6]

    if addr_lines:
        out["Address"] = norm(" ".join(addr_lines))

    m_ds = DISTRICT_LABEL_RE.search(block)
    if m_ds:
        out["District/State"] = norm(m_ds.group(2))
    else:
        if lines:
            last_line = lines[-1]
            if len(last_line.split()) <= 6 and not re.search(r"\d", last_line) and not re.search(r"@", last_line):
                if not re.search(r"\b(Email|Licence|License|TIN|Tin|Registered Permit)\b", last_line, re.IGNORECASE):
                    out["District/State"] = norm(last_line)

    out["Name"] = re.sub(r"^(Name\s*[:\-\s]|Consignor[:\-\s]|Consignee[:\-\s]*)", "", out["Name"], flags=re.IGNORECASE).strip()

    if not out["TIN"]:
        m_any_tin = re.search(r"\b([A-Z0-9]{5,20})\b", block)
        if m_any_tin:
            candidate = m_any_tin.group(1)
            if sum(ch.isdigit() for ch in candidate) >= 3:
                out["TIN"] = candidate

    for k in out:
        out[k] = norm(out[k])

    return out


def split_item_blocks(table_text: str):
    lines = [l.rstrip() for l in table_text.splitlines() if l.strip()]
    blocks = []
    current = []
    for ln in lines:
        if re.match(r"^\s*\d+\b", ln):
            if current:
                blocks.append(" ".join(current))
            current = [ln.strip()]
        else:
            if re.search(
                r"\b(TRANSPORT DETAIL|TRANSPORT DETAILS|DETAILS OF DISPATCHED QUANTITY|SIGNATURE OF PERMIT ISSUING AUTHORITY|SIGNATURE OF OFFICER-IN-CHARGE|Total Quantity:)\b",
                ln,
                re.IGNORECASE,
            ):
                if current:
                    blocks.append(" ".join(current))
                current = []
                break
            if current:
                current.append(ln.strip())
            else:
                if blocks:
                    blocks[-1] += " " + ln.strip()
                else:
                    current = [ln.strip()]
    if current:
        blocks.append(" ".join(current))
    return [re.sub(r"\s+", " ", b).strip() for b in blocks if b.strip()]


def parse_table_block(text: str, start_label: str, end_labels, mode: str, pdf_path: Path):
    m = re.search(re.escape(start_label), text, re.IGNORECASE)
    if not m:
        return []
    start = m.end()
    end = None
    for e in end_labels:
        me = re.search(re.escape(e), text[start:], re.IGNORECASE)
        if me:
            end = start + me.start()
            break
    block = text[start:end].strip() if end else text[start:].strip()
    if mode == "invoice":
        m_tr = re.search(
            r"\b(TRANSPORT DETAIL|TRANSPORT DETAILS|DETAILS OF DISPATCHED QUANTITY|SIGNATURE OF PERMIT ISSUING AUTHORITY|SIGNATURE OF OFFICER-IN-CHARGE|Total Quantity:)\b",
            block,
            re.IGNORECASE,
        )
        if m_tr:
            block = block[: m_tr.start()].strip()
    num_pos = re.search(r"^\s*\d+\b", block, re.MULTILINE)
    if num_pos:
        block = block[num_pos.start() :]
    items = []
    for raw in split_item_blocks(block):
        parsed = parse_invoice_row(raw, pdf_path) if mode == "invoice" else parse_dispatched_row(raw, pdf_path)
        if parsed:
            items.append(parsed)
    return items


def fix_dealer_from_text(result: dict, full_text: str):
    current = result.get("DEALER_DETAILS", {"Name": "", "TIN/Tlin": "", "License No": ""})
    m_low = re.search(r"\bLowadih\s+Area\s+No\.?\s*1\b", full_text, re.IGNORECASE)
    if m_low:
        current["Name"] = "Lowadih Area No. 1"
    if not current.get("TIN/Tlin"):
        m_header = re.search(r"1\.\s*NAME AND TIN OF DEALER(.{0,400})", full_text, re.IGNORECASE | re.DOTALL)
        candidate_block = m_header.group(1) if m_header else full_text
        m_tin = re.search(r"\b(\d{9,14})\b", candidate_block)
        if m_tin:
            current["TIN/Tlin"] = m_tin.group(1).strip()
        else:
            m_tl = TLIN_RE.search(full_text)
            if m_tl:
                current["TIN/Tlin"] = m_tl.group(1).strip()
    if not current.get("License No"):
        m_lic = LICENSE_RE.search(full_text)
        if m_lic:
            current["License No"] = m_lic.group(0).strip()
        else:
            m_loose = re.search(r"\b[0-9A-Za-z_\-]{2,20}com[0-9A-Za-z_\-\/]{0,30}\b", full_text, re.IGNORECASE)
            if m_loose:
                current["License No"] = m_loose.group(0).strip()
    result["DEALER_DETAILS"] = {
        "Name": norm(current.get("Name", "")),
        "TIN/Tlin": current.get("TIN/Tlin", ""),
        "License No": current.get("License No", ""),
    }
    return result


def fix_dealer_from_text_simple(result: dict, full_text: str):
    m_low = re.search(r"\bLowadih\s+Area\s+No\.?\s*1\b", full_text, re.IGNORECASE)
    if m_low:
        result["DEALER_DETAILS"]["Name"] = "Lowadih Area No. 1"
    m_lic = LICENSE_RE.search(full_text)
    if m_lic:
        result["DEALER_DETAILS"]["License No"] = m_lic.group(0).strip()
    return result


def enforce_final_consignor_consignee(result: dict):
    consignor = result.get("NAME_AND_ADDRESS_OF_CONSIGNOR", {}) or {}
    tin_val = consignor.get("TIN", "")
    name_val = consignor.get("Name", "")
    consignor["Name And Style Of Business"] = "Private Wholesale"
    consignor["Address"] = "RANCHI"
    consignor["District/State"] = "RANCHI, Jharkhand"
    consignor["TIN"] = tin_val
    consignor["Name"] = name_val
    result["NAME_AND_ADDRESS_OF_CONSIGNOR"] = consignor

    consignee = result.get("NAME_AND_ADDRESS_OF_CONSIGNEE", {}) or {}
    consignee_tin = consignee.get("TIN", "")
    consignee["Name And Style Of Business"] = "Lowadih Area No. 1"
    consignee["Address"] = "Lowadih Area No. 1"
    consignee["Name"] = ""
    consignee["District/State"] = ""
    consignee["TIN"] = consignee_tin
    result["NAME_AND_ADDRESS_OF_CONSIGNEE"] = consignee
    return result


def enforce_shipping_address(result: dict):
    if "PERMIT_FOR_TRANSPORT" not in result:
        result["PERMIT_FOR_TRANSPORT"] = {"Registered Permit No": "", "Date of Issue": "", "Permit Validity Date": "", "Shipping Address": ""}
    result["PERMIT_FOR_TRANSPORT"]["Shipping Address"] = "Lowadih Area No. 1"
    return result


def _clean_issued_date(raw: str) -> str:
    if not raw:
        return ""
    s = raw.strip()
    s = re.sub(r"Form[-\s]T(?:\s\(Transport\))?", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\bForm[:\s-]*T\b", "", s, flags=re.IGNORECASE)
    s = re.sub(r"(?:Excise\s+)?Permit\s+No[:\s]*[A-Z0-9\-\_/]+", "", s, flags=re.IGNORECASE)
    s = re.sub(r"Permit\s+No[:\s]*[A-Z0-9\-\_/]+", "", s, flags=re.IGNORECASE)
    s = s.strip(" -,:;")
    m = DATE_RE.search(s)
    if m:
        return m.group(1)
    m2 = DATE_RE.search(raw)
    if m2:
        return m2.group(1)
    return norm(s)


def reduce_item_fields(rows: list) -> list:
    reduced = []
    for r in rows:
        raw_label = r.get("Label Name", "")
        kind = r.get("Kind Of Intoxicant", "")
        canonical = fuzzy_match_product(raw_label, kind)
        reduced.append(
            {
                "Comodity Group": r.get("Comodity Group", ""),
                "Kind Of Intoxicant": kind,
                "Label Name": canonical if canonical else raw_label,
                "Unit": r.get("Unit", ""),
                "StockQty": r.get("StockQty", ""),
                "Qty": r.get("Qty", ""),
            }
        )
    return reduced


def process_pdf(path: Path) -> dict:
    full_text, _pages = extract_text_pages(path)
    result = {
        "FILE_NAME": path.name,
        "PERMIT_NO": "",
        "ISSUED_DATE": "",
        "DEALER_DETAILS": {"Name": "", "TIN/Tlin": "", "License No": ""},
        "NAME_AND_ADDRESS_OF_CONSIGNOR": {"Name": "", "TIN": "", "Name And Style Of Business": "", "Address": "", "District/State": ""},
        "NAME_AND_ADDRESS_OF_CONSIGNEE": {"Name": "", "TIN": "", "Name And Style Of Business": "", "Address": "", "District/State": ""},
        "PERMIT_FOR_TRANSPORT": {"Registered Permit No": "", "Date of Issue": "", "Permit Validity Date": "", "Shipping Address": ""},
        "TOTAL_DUTY": "",
        "TOTAL_FEES": "",
        "INVOICE_DETAILS": [],
        "DETAILS_OF_DISPATCHED_QUANTITY": [],
    }

    m_permit = PERMIT_NO_RE.search(full_text)
    if m_permit:
        result["PERMIT_NO"] = m_permit.group(1).strip()

    issued_raw = ""
    m_issued = ISSUED_LABEL_RE.search(full_text)
    issued_raw = m_issued.group(1).strip() if m_issued else full_text[:600]
    result["ISSUED_DATE"] = _clean_issued_date(issued_raw)

    dealer_block = block_between(
        full_text,
        ["1. NAME AND TIN OF DEALER", "NAME AND TIN OF DEALER", "1. NAME AND TIN"],
        ["2. NAME AND ADDRESS OF THE CONSIGNOR", "2. NAME AND ADDRESS"],
    )
    result["DEALER_DETAILS"] = parse_dealer(dealer_block)

    consignor_block = block_between(
        full_text,
        ["2. NAME AND ADDRESS OF THE CONSIGNOR", "NAME AND ADDRESS OF THE CONSIGNOR"],
        ["3. NAME AND ADDRESS OF THE CONSIGNEE", "3. NAME AND ADDRESS OF THE CONSIGNEE"],
    )
    result["NAME_AND_ADDRESS_OF_CONSIGNOR"] = parse_party(consignor_block)

    consignee_block = block_between(
        full_text,
        ["3. NAME AND ADDRESS OF THE CONSIGNEE", "NAME AND ADDRESS OF THE CONSIGNEE"],
        ["4. PERMIT FOR TRANSPORT", "4. PERMIT FOR TRANSPORT"],
    )
    result["NAME_AND_ADDRESS_OF_CONSIGNEE"] = parse_party(consignee_block)

    permit_block = block_between(
        full_text,
        ["4. PERMIT FOR TRANSPORT", "PERMIT FOR TRANSPORT"],
        ["5. TOTAL DUTY", "5. TOTAL DUTY", "TOTAL DUTY(N/A-Duty Already Paid)"],
    )
    if permit_block:
        m_reg = re.search(r"Registered Permit No\.?\s*[:\s]*([A-Z0-9\-/]+)", permit_block, re.IGNORECASE)
        if m_reg:
            result["PERMIT_FOR_TRANSPORT"]["Registered Permit No"] = m_reg.group(1).strip()
        m_do = re.search(r"Date of Issue\s*[:\s]*([^\n]+)", permit_block, re.IGNORECASE)
        if m_do:
            dm = DATE_RE.search(m_do.group(1))
            result["PERMIT_FOR_TRANSPORT"]["Date of Issue"] = dm.group(1) if dm else norm(m_do.group(1))
        m_vd = re.search(r"Permit Validity Date\s*[:\s]*([^\n]+)", permit_block, re.IGNORECASE)
        if m_vd:
            dm = DATE_RE.search(m_vd.group(1))
            result["PERMIT_FOR_TRANSPORT"]["Permit Validity Date"] = dm.group(1) if dm else norm(m_vd.group(1))
        m_ship = re.search(r"Shipping Address\s*[:\-\s]*([^\n]+)", permit_block, re.IGNORECASE)
        if m_ship:
            result["PERMIT_FOR_TRANSPORT"]["Shipping Address"] = norm(m_ship.group(1))

    m_duty = TOTAL_DUTY_RE.search(full_text)
    if m_duty:
        result["TOTAL_DUTY"] = norm(m_duty.group(1))
    m_fees = TOTAL_FEES_RE.search(full_text)
    if m_fees:
        result["TOTAL_FEES"] = norm(m_fees.group(1))

    result["INVOICE_DETAILS"] = parse_table_block(
        full_text,
        "INVOICE DETAIL",
        ["DETAILS OF DISPATCHED QUANTITY", "TRANSPORT DETAIL", "TRANSPORT DETAILS", "Total Quantity:", "Signature of Permit Issuing Authority"],
        mode="invoice",
        pdf_path=path,
    )
    result["DETAILS_OF_DISPATCHED_QUANTITY"] = parse_table_block(
        full_text,
        "DETAILS OF DISPATCHED QUANTITY",
        [
            "TRANSPORT DETAIL",
            "TRANSPORT DETAILS",
            "Signature of Officer-In-Charge",
            "Signature of Officer-In-Charge or Authorised Signatory",
            "Total Quantity:",
        ],
        mode="dispatch",
        pdf_path=path,
    )

    def label_looks_contaminated(label: str) -> bool:
        if not label:
            return False
        if re.search(r"\b\d{1,4}\s*(ML|L|G|KG|PCS|NOS|BOTTLE|BOTTLES)\b", label, re.IGNORECASE):
            return True
        if re.search(r"\b0\b", label):
            return True
        if len(label) < 10 or _HEADER_TRAILING_RE.search(label):
            return True
        return False

    def try_fix_rows(rows: list, pdf_path: Path):
        fixed = []
        for r in rows:
            r2 = clean_label_row_dict(r)
            if label_looks_contaminated(r2.get("Label Name", "")) or not r2.get("Label Name"):
                idx = r2.get("Comodity Group", "")
                kind = r2.get("Kind Of Intoxicant", "")
                approx_fragment = (r.get("Label Name") or "")[:40].strip()
                anchor = f"{idx} {kind}".strip()
                fb = fallback_extract_label_from_pdf(pdf_path, anchor, approx_fragment)
                if fb:
                    r2["Label Name"] = fb
                    r2 = clean_label_row_dict(r2)

            idx = r2.get("Comodity Group", "")
            kind = r2.get("Kind Of Intoxicant", "")
            anchor = f"{idx} {kind}".strip()
            bl_fb = extract_bl_from_pdf_cell(pdf_path, anchor)
            if bl_fb and is_plausible_bl(bl_fb):
                r2["Equivalent To BL"] = bl_fb

            if r2.get("Label Name"):
                r2["Label Name"] = _apply_category_caps(_dedupe_label_text(r2["Label Name"]))
            fixed.append(r2)
        return fixed

    result["INVOICE_DETAILS"] = reduce_item_fields(try_fix_rows(result.get("INVOICE_DETAILS", []), path))
    result["DETAILS_OF_DISPATCHED_QUANTITY"] = reduce_item_fields(try_fix_rows(result.get("DETAILS_OF_DISPATCHED_QUANTITY", []), path))

    result = fix_dealer_from_text(result, full_text)
    result = fix_dealer_from_text_simple(result, full_text)
    result = enforce_final_consignor_consignee(result)
    result = enforce_shipping_address(result)
    return result


def save_json(data: dict, pdf_path: Path, output_dir: Union[str, Path] = "output_json") -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / (pdf_path.stem + ".json")
    with out.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    return out


class TableExtractor:
    """
    Wrapper used by the Flask API.

    Your API expects:
      extractor = TableExtractor(pdf_path=...)
      extractor.extract_stock_purchase_data(extraction_time_seconds=...)
    """

    def __init__(self, pdf_path: Union[str, Path]):
        self.pdf_path = Path(pdf_path)

    def extract_stock_purchase_data(self, extraction_time_seconds: Optional[float] = None) -> dict:
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {self.pdf_path}")

        result = process_pdf(self.pdf_path)
        if extraction_time_seconds is not None:
            # Keep a stable key the API can rely on.
            result["EXTRACTION_TIME_SECONDS"] = extraction_time_seconds
        return result

