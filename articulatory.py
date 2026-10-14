"""Devanagari articulatory taxonomy, text normalisation and character alignment.

Grapheme-level proxy for phonetic classes: Devanagari is largely phonemic, so
characters are a reasonable stand-in for phones. Known limitation: spelling
variants (nukta, anusvara vs nasal conjunct) show up as "errors".
"""
import unicodedata
import numpy as np


def normalize_hi(t: str) -> str:
    """NFC, drop punctuation/symbols (incl. danda), collapse whitespace.
    Uses Unicode categories because Python's \\w drops Devanagari matras."""
    t = unicodedata.normalize("NFC", t)
    t = "".join(c for c in t if not unicodedata.category(c).startswith(("P", "S")))
    return " ".join(t.split())


# order inside each row: voiceless, voiceless-aspirated, voiced, voiced-aspirated
STOPS = {"velar": "कखगघ", "palatal": "चछजझ", "retroflex": "टठडढ",
         "dental": "तथदध", "labial": "पफबभ"}

DIRECTED = {k: {} for k in [
    "unaspirated→aspirated", "aspirated→unaspirated",
    "voiceless→voiced", "voiced→voiceless",
    "retroflex→dental", "dental→retroflex",
    "short→long vowel", "long→short vowel"]}

for k, kh, g, gh in STOPS.values():
    for a, b in [(k, kh), (g, gh)]:
        DIRECTED["unaspirated→aspirated"][a] = b
        DIRECTED["aspirated→unaspirated"][b] = a
    for a, b in [(k, g), (kh, gh)]:
        DIRECTED["voiceless→voiced"][a] = b
        DIRECTED["voiced→voiceless"][b] = a
for r, d in list(zip(STOPS["retroflex"], STOPS["dental"])) + [("ण", "न")]:
    DIRECTED["retroflex→dental"][r] = d
    DIRECTED["dental→retroflex"][d] = r
for s, l in [("इ", "ई"), ("उ", "ऊ"), ("ि", "ी"), ("ु", "ू")]:
    DIRECTED["short→long vowel"][s] = l
    DIRECTED["long→short vowel"][l] = s

# (forward, reverse) pairs used for asymmetry analysis
ASYMMETRY_PAIRS = [
    ("aspirated→unaspirated", "unaspirated→aspirated"),
    ("retroflex→dental", "dental→retroflex"),
    ("voiced→voiceless", "voiceless→voiced"),
    ("long→short vowel", "short→long vowel"),
]

CLASSES = {
    "aspirated stops": "खघछझठढथधफभ",
    "unaspirated stops": "कगचजटडतदपब",
    "voiceless stops": "कखचछटठतथपफ",
    "voiced stops": "गघजझडढदधबभ",
    "retroflex (ट ठ ड ढ ण)": "टठडढण",
    "dental (त थ द ध न)": "तथदधन",
    "velar stops": "कखगघ", "palatal affricates": "चछजझ", "labial stops": "पफबभ",
    "nasals": "ङञणनम", "fricatives": "शषसह", "semivowels/liquids": "यरलव",
    "short vowels": "अइउऋिुृ", "long vowels": "आईऊएऐओऔाीूेैोौ",
    "nasalisation/nukta/virama": "ंँः़्",
}
CLASSES = {k: set(v) for k, v in CLASSES.items()}


def align_chars(ref: str, hyp: str):
    """Levenshtein alignment -> list of (op, ref_char, hyp_char); op in C/S/D/I."""
    n, m = len(ref), len(hyp)
    D = np.zeros((n + 1, m + 1), dtype=np.int32)
    D[:, 0] = np.arange(n + 1)
    D[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i, j] = min(D[i-1, j] + 1, D[i, j-1] + 1,
                          D[i-1, j-1] + (ref[i-1] != hyp[j-1]))
    ops, i, j = [], n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and D[i, j] == D[i-1, j-1] + (ref[i-1] != hyp[j-1]):
            ops.append(("C" if ref[i-1] == hyp[j-1] else "S", ref[i-1], hyp[j-1]))
            i -= 1; j -= 1
        elif i > 0 and D[i, j] == D[i-1, j] + 1:
            ops.append(("D", ref[i-1], None)); i -= 1
        else:
            ops.append(("I", None, hyp[j-1])); j -= 1
    return ops[::-1]
