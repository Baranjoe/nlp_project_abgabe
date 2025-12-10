"""
wer_train.py
------------

TRAINING SCRIPT: Lernt Korrekturregeln aus einem Trainingsdatensatz
und speichert sie als "Modell" (JSON-Datei).

Usage:
    python wer_train.py --train data/train.csv --output model.json
    python 02_rules/claude_train.py --train 02_rules/data_prepared/strict/emb_scores_clean_strict_train.csv --output 02_rules/model_rules_strict_train_base.json
    python 02_rules/claude_train.py --train 02_rules/data_prepared/special/eval_all_models_strict_train.csv --hyp-col ger-spellcorr-base_strict_out --output 02_rules/model_rules_strict_train_ger-spellcorr-base_strict.json

Das gespeicherte Modell enthält:
- Phrase Rules
- Single-Token Rules
- Verb Transformationen (statisch)
- Normalisierungsparameter
"""

from __future__ import annotations
import argparse
import json
import re
from collections import Counter, defaultdict
from typing import List, Tuple, Dict, Optional
from pathlib import Path

import pandas as pd

# ===============================================================
# DEFAULT CONFIG
# ===============================================================

DEFAULT_CONFIG = {
    "min_count": 2,
    "dominance": 0.55,
    "error_ratio": 0.60,
    "min_phrase_count": 2,
    "min_phrase_dominance": 0.50,
    "max_phrase_length": 3,
    "max_ngram_size": 4,
}

# ===============================================================
# STATIC RULES (Linguistisch motiviert)
# ===============================================================

VERB_TRANSFORMS = [
    ("hat", "zurückgewiesen", "wies zurück"),
    ("hat", "gesprochen", "sprach"),
    ("hat", "geredet", "redete"),
    ("hat", "gesagt", "sagte"),
    ("hat", "gemacht", "machte"),
    ("hat", "gehabt", "hatte"),
    ("hat", "gegeben", "gab"),
    ("hat", "genommen", "nahm"),
    ("hat", "gesehen", "sah"),
    ("hat", "gefunden", "fand"),
    ("hat", "gehalten", "hielt"),
    ("hat", "gelassen", "liess"),
    ("hat", "geschrieben", "schrieb"),
    ("hat", "gelesen", "las"),
    ("hat", "getroffen", "traf"),
    ("hat", "begonnen", "begann"),
    ("hat", "gewonnen", "gewann"),
    ("hat", "verloren", "verlor"),
    ("hat", "verstanden", "verstand"),
    ("hat", "gestanden", "stand"),
    ("hat", "gesessen", "sass"),
    ("hat", "gelegen", "lag"),
    ("hat", "geschlafen", "schlief"),
    ("hat", "gerufen", "rief"),
    ("hat", "getragen", "trug"),
    ("hat", "geworfen", "warf"),
    ("hat", "geholfen", "half"),
    ("hat", "gebracht", "brachte"),
    ("hat", "gedacht", "dachte"),
    ("hat", "gewusst", "wusste"),
    ("hat", "gekannt", "kannte"),
    ("hat", "genannt", "nannte"),
    ("hat", "gewollt", "wollte"),
    ("hat", "gekonnt", "konnte"),
    ("hat", "gemusst", "musste"),
    ("hat", "gedurft", "durfte"),
    ("hat", "gesollt", "sollte"),
    ("hat", "bekommen", "bekam"),
    ("hat", "vergessen", "vergass"),
    ("hat", "beschlossen", "beschloss"),
    ("hat", "geschlossen", "schloss"),
    ("hat", "gezogen", "zog"),
    ("hat", "geflogen", "flog"),
    ("hat", "gehoben", "hob"),
    ("hat", "geschoben", "schob"),
    ("hat", "geboten", "bot"),
    ("hat", "entschieden", "entschied"),
    ("hat", "geschnitten", "schnitt"),
    ("hat", "gegriffen", "griff"),
    ("hat", "gebrochen", "brach"),
    ("hat", "gestochen", "stach"),
    ("hat", "empfohlen", "empfahl"),
    ("hat", "gestohlen", "stahl"),
    ("hat", "geschworen", "schwor"),
    ("hat", "gefroren", "fror"),
    ("hat", "geschossen", "schoss"),
    ("hat", "genossen", "genoss"),
    ("hat", "gegossen", "goss"),
    ("hat", "geklungen", "klang"),
    ("hat", "gesungen", "sang"),
    ("hat", "gesprungen", "sprang"),
    ("hat", "getrunken", "trank"),
    ("hat", "gesunken", "sank"),
    ("hat", "gebunden", "band"),
    ("hat", "verschwunden", "verschwand"),
    ("hat", "abgesagt", "sagte ab"),
    ("hat", "reagiert", "reagierte"),
    ("hat", "gesucht", "suchte"),
    ("hat", "versucht", "versuchte"),
    ("hat", "gebraucht", "brauchte"),
    ("hat", "gedauert", "dauerte"),
    ("hat", "geführt", "führte"),
    ("hat", "gehört", "hörte"),
    ("hat", "erklärt", "erklärte"),
    ("hat", "geändert", "änderte"),
    ("hat", "gefordert", "forderte"),
    ("hat", "gefördert", "förderte"),
    ("hat", "gefeiert", "feierte"),
    ("hat", "passiert", "passierte"),
    ("hat", "verletzt", "verletzte"),
    ("hat", "gesetzt", "setzte"),
    ("hat", "genutzt", "nutzte"),
    ("hat", "gezeigt", "zeigte"),
    ("hat", "bewiesen", "bewies"),
    ("hat", "vereinbart", "vereinbarte"),
    ("hat", "gearbeitet", "arbeitete"),
    ("hat", "erwartet", "erwartete"),
    ("hat", "gewartet", "wartete"),
    ("hat", "berichtet", "berichtete"),
    ("hat", "gedroht", "drohte"),
    ("hat", "gewohnt", "wohnte"),
    ("hat", "betont", "betonte"),
    ("hat", "verhindert", "verhinderte"),
    ("hat", "verbracht", "verbrachte"),
    ("hat", "überrascht", "überraschte"),
    ("hat", "gewählt", "wählte"),
    ("hat", "erzählt", "erzählte"),
    ("hat", "gezählt", "zählte"),
    ("hat", "bezahlt", "bezahlte"),
    ("hat", "gefehlt", "fehlte"),
    ("hat", "gestellt", "stellte"),
    ("hat", "festgestellt", "stellte fest"),
    ("hat", "vorgestellt", "stellte vor"),
    ("hat", "gekauft", "kaufte"),
    ("hat", "verkauft", "verkaufte"),
    ("hat", "geglaubt", "glaubte"),
    ("hat", "gebaut", "baute"),
    ("hat", "geschaut", "schaute"),
    ("hat", "gestartet", "startete"),
    ("hat", "gelebt", "lebte"),
    ("hat", "erlebt", "erlebte"),
    ("hat", "geliebt", "liebte"),
    ("hat", "geleitet", "leitete"),
    ("hat", "bedeutet", "bedeutete"),
    ("hat", "geöffnet", "öffnete"),
    ("hat", "gerechnet", "rechnete"),
    ("ist", "gewesen", "war"),
    ("ist", "geworden", "wurde"),
    ("ist", "gekommen", "kam"),
    ("ist", "gegangen", "ging"),
    ("ist", "gefahren", "fuhr"),
    ("ist", "geflogen", "flog"),
    ("ist", "gelaufen", "lief"),
    ("ist", "geschwommen", "schwamm"),
    ("ist", "gefallen", "fiel"),
    ("ist", "gestorben", "starb"),
    ("ist", "geblieben", "blieb"),
    ("ist", "erschienen", "erschien"),
    ("ist", "verschwunden", "verschwand"),
    ("ist", "entstanden", "entstand"),
    ("ist", "gelungen", "gelang"),
    ("ist", "gesprungen", "sprang"),
    ("ist", "gestiegen", "stieg"),
    ("ist", "gesunken", "sank"),
    ("ist", "gewachsen", "wuchs"),
    ("ist", "gebrochen", "brach"),
    ("ist", "getreten", "trat"),
    ("ist", "gerannt", "rannte"),
    ("ist", "angekommen", "kam an"),
    ("ist", "losgegangen", "ging los"),
    ("ist", "ausgegangen", "ging aus"),
    ("ist", "zurückgegangen", "ging zurück"),
    ("ist", "abgefahren", "fuhr ab"),
    ("ist", "zurückgefahren", "fuhr zurück"),
    ("ist", "aufgeflogen", "flog auf"),
    ("ist", "zusammengeflogen", "flog zusammen"),
    ("hat", "sich gesetzt", "setzte sich"),
    ("hat", "sich gezeigt", "zeigte sich"),
    ("hat", "sich geändert", "änderte sich"),
    ("hat", "sich entwickelt", "entwickelte sich"),
    ("hat", "sich ergeben", "ergab sich"),
    ("hat", "sich entschieden", "entschied sich"),
    ("hat", "sich gewehrt", "wehrte sich"),
    ("hat", "sich abgespielt", "spielte sich ab"),
]

KONJUNKTIV_MAP = {
    "habe": "hat",
    "sei": "ist",
    "werde": "wird",
    "könne": "kann",
    "müsse": "muss",
    "wolle": "will",
    "solle": "soll",
    "dürfe": "darf",
}

WORD_FIXES = {
    "aussen": "auch so",
    "außen": "auch so",
    "wald": "welt",
    "gramm": "rappen",
    "blumenstärke": "lumenstärke",
    "blumenstarke": "lumenstarke",
    "geredet": "gesprochen",
    "erhalten": "halten",
    "auslösung": "auslosung",
    "zusammengeflogen": "zusammen",
    "kiesinger": "kissinger",
    "maier": "meier",
    "moda": "mode",
    "apfer": "opfer",
    "fachung": "entkommen",
}

PHRASE_FIXES = {
    "von summen": "der summe",
    "zur höhe von summen": "zur höhe der summe",
    "hundert trump auf": "kommt trump auf",
    "in seiner partei hundert trump": "in seiner partei kommt trump",
    "vom früheren aussenminister": "des früheren aussenministers",
    "der wenger": "wenger",
    "das isis": "die isis",
    "ein cannabislegalisierung": "eine cannabislegalisierung",
    "vor wald": "der welt",
}


# ===============================================================
# WER & ALIGNMENT
# ===============================================================

def wer(ref: str, hyp: str) -> float:
    r = ref.split()
    h = hyp.split()
    R, H = len(r), len(h)
    if R == 0:
        return 1.0 if H > 0 else 0.0
    prev = list(range(H + 1))
    curr = [0] * (H + 1)
    for i in range(1, R + 1):
        curr[0] = i
        for j in range(1, H + 1):
            if r[i - 1] == h[j - 1]:
                curr[j] = prev[j - 1]
            else:
                curr[j] = 1 + min(prev[j], curr[j - 1], prev[j - 1])
        prev, curr = curr, prev
    return prev[H] / R


def mean_wer(refs: List[str], hyps: List[str]) -> float:
    if not refs:
        return 0.0
    return sum(wer(r, h) for r, h in zip(refs, hyps)) / len(refs)


def wer_pair(hyp: str, ref: str) -> float:
    if len(ref.split()) == 0:
        return 1.0 if len(hyp.split()) > 0 else 0.0
    return wer(ref, hyp)


def align_tokens(ref_tokens: List[str], hyp_tokens: List[str]) -> List[Tuple[str, Optional[str], Optional[str]]]:
    n, m = len(ref_tokens), len(hyp_tokens)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    bt = [[None] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        dp[i][0] = i
        bt[i][0] = ("D", i - 1, 0)
    for j in range(1, m + 1):
        dp[0][j] = j
        bt[0][j] = ("I", 0, j - 1)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref_tokens[i - 1] == hyp_tokens[j - 1] else 1
            del_cost = dp[i - 1][j] + 1
            ins_cost = dp[i][j - 1] + 1
            sub_cost = dp[i - 1][j - 1] + cost
            best = min(del_cost, ins_cost, sub_cost)
            dp[i][j] = best
            if best == sub_cost:
                bt[i][j] = ("=" if cost == 0 else "S", i - 1, j - 1)
            elif best == del_cost:
                bt[i][j] = ("D", i - 1, j)
            else:
                bt[i][j] = ("I", i, j - 1)

    ops = []
    i, j = n, m
    while i > 0 or j > 0:
        op, pi, pj = bt[i][j]
        if op in ("=", "S"):
            ops.append((op, ref_tokens[pi], hyp_tokens[pj]))
        elif op == "D":
            ops.append((op, ref_tokens[pi], None))
        else:
            ops.append((op, None, hyp_tokens[pj]))
        i, j = pi, pj
    return ops[::-1]


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    prev = list(range(lb + 1))
    curr = [0] * (lb + 1)
    for i in range(1, la + 1):
        curr[0] = i
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev, curr = curr, prev
    return prev[lb]


def is_valid_correction(source: str, target: str) -> bool:
    if source == target:
        return False
    if source.isdigit() or target.isdigit():
        return True
    if abs(len(source) - len(target)) > 5:
        return False
    return levenshtein(source, target) <= 3


# ===============================================================
# NORMALIZATION
# ===============================================================

def normalize(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"[.,;:!?()\[\]{}«»\"'„\"‚'–—]", " ", text)
    text = text.replace("ß", "ss")

    number_words = {
        "null": "0", "eins": "1", "zwei": "2", "drei": "3", "vier": "4",
        "fünf": "5", "fuenf": "5", "sechs": "6", "sieben": "7", "acht": "8",
        "neun": "9", "zehn": "10", "elf": "11", "zwölf": "12", "zwoelf": "12",
    }
    for word, digit in number_words.items():
        text = re.sub(rf"\b{word}\b", digit, text)

    text = re.sub(r"\s+", " ", text).strip()
    return text


# ===============================================================
# RULE LEARNING
# ===============================================================

def extract_ngram_substitutions(refs: List[str], hyps: List[str], max_n: int = 4) -> Counter:
    """Extrahiere Multi-Word Substitutionen."""
    ngram_subs: Counter = Counter()

    for ref, hyp in zip(refs, hyps):
        ref_tokens = ref.split()
        hyp_tokens = hyp.split()
        ops = align_tokens(ref_tokens, hyp_tokens)

        i = 0
        while i < len(ops):
            if ops[i][0] in ("S", "D", "I"):
                j = i
                ref_phrase = []
                hyp_phrase = []

                while j < len(ops) and j - i < max_n * 2:
                    op, r_tok, h_tok = ops[j]
                    if op == "=":
                        break
                    if r_tok:
                        ref_phrase.append(r_tok)
                    if h_tok:
                        hyp_phrase.append(h_tok)
                    j += 1

                if ref_phrase and hyp_phrase:
                    ref_str = " ".join(ref_phrase)
                    hyp_str = " ".join(hyp_phrase)
                    if ref_str != hyp_str and len(ref_phrase) <= max_n and len(hyp_phrase) <= max_n:
                        ngram_subs[(ref_str, hyp_str)] += 1

                i = j
            else:
                i += 1

    return ngram_subs


def build_phrase_rules(ngram_subs: Counter, config: dict) -> Dict[str, str]:
    """Baue Phrase-Regeln."""
    by_hyp: Dict[str, Counter] = defaultdict(Counter)
    for (ref_phrase, hyp_phrase), count in ngram_subs.items():
        by_hyp[hyp_phrase][ref_phrase] += count

    rules = {}
    for hyp_phrase, ref_counts in by_hyp.items():
        total = sum(ref_counts.values())
        if total < config["min_phrase_count"]:
            continue

        ref_phrase, best_count = ref_counts.most_common(1)[0]
        if hyp_phrase == ref_phrase:
            continue

        if best_count / total < config["min_phrase_dominance"]:
            continue

        if len(hyp_phrase.split()) > config["max_phrase_length"]:
            continue
        if len(ref_phrase.split()) > config["max_phrase_length"]:
            continue

        rules[hyp_phrase] = ref_phrase

    return rules


def learn_single_token_rules(refs: List[str], hyps: List[str]) -> Tuple[Counter, Counter]:
    """Lerne Single-Token Substitutionsstatistiken."""
    sub_counter: Counter = Counter()
    match_counter: Counter = Counter()

    for ref, hyp in zip(refs, hyps):
        ops = align_tokens(ref.split(), hyp.split())
        for op, r_tok, h_tok in ops:
            if op == "S" and r_tok and h_tok and r_tok != h_tok:
                sub_counter[(r_tok, h_tok)] += 1
            elif op == "=" and r_tok and h_tok:
                match_counter[h_tok] += 1

    return sub_counter, match_counter


def build_single_token_rules(sub_counter: Counter, match_counter: Counter, config: dict) -> Dict[str, str]:
    """Baue Single-Token Regeln."""
    by_source: Dict[str, Counter] = defaultdict(Counter)
    for (ref_tok, hyp_tok), count in sub_counter.items():
        by_source[hyp_tok][ref_tok] += count

    rules = {}
    for source, target_counts in by_source.items():
        total_errors = sum(target_counts.values())
        if total_errors < config["min_count"]:
            continue

        target, best_count = target_counts.most_common(1)[0]
        matches = match_counter.get(source, 0)
        total_occ = total_errors + matches

        dom = best_count / total_errors
        err_rate = best_count / total_occ if total_occ > 0 else 0

        if dom < config["dominance"] or err_rate < config["error_ratio"]:
            continue
        if not is_valid_correction(source, target):
            continue

        rules[source] = target

    return rules


def apply_rule(hyps: List[str], source: str, target: str) -> List[str]:
    pattern = re.compile(rf"\b{re.escape(source)}\b")
    return [pattern.sub(target, h) for h in hyps]


def greedy_select_rules(
        hyps: List[str],
        refs: List[str],
        rules: Dict[str, str],
        sub_counter: Counter,
        verbose: bool = True
) -> Dict[str, str]:
    """Wähle nur Regeln die WER verbessern."""
    current = hyps
    current_wer = mean_wer(refs, current)
    selected = {}

    sorted_rules = sorted(rules.items(), key=lambda x: sub_counter.get((x[1], x[0]), 0), reverse=True)

    for source, target in sorted_rules:
        test = apply_rule(current, source, target)
        test_wer = mean_wer(refs, test)
        if test_wer < current_wer - 1e-9:
            current = test
            current_wer = test_wer
            selected[source] = target
            if verbose:
                cnt = sub_counter.get((target, source), 0)
                print(f"  [OK] {source!r:>18} -> {target!r:<20} (n={cnt}) WER: {current_wer:.4f}")

    return selected


def apply_phrase_rules_for_training(
        hyps: List[str],
        refs: List[str],
        rules: Dict[str, str]
) -> Tuple[List[str], Dict[str, str]]:
    """Wende Phrase-Regeln an und tracke welche effektiv waren."""
    result = list(hyps)
    effective_rules = {}

    sorted_rules = sorted(rules.items(), key=lambda x: len(x[0].split()), reverse=True)

    for hyp_phrase, ref_phrase in sorted_rules:
        applied = False
        new_result = []
        for hyp, ref in zip(result, refs):
            if hyp_phrase in hyp:
                candidate = hyp.replace(hyp_phrase, ref_phrase)
                if wer_pair(candidate, ref) < wer_pair(hyp, ref):
                    new_result.append(candidate)
                    applied = True
                else:
                    new_result.append(hyp)
            else:
                new_result.append(hyp)

        if applied:
            effective_rules[hyp_phrase] = ref_phrase
        result = new_result

    return result, effective_rules


# ===============================================================
# TRAINING PIPELINE
# ===============================================================

def train(
        train_csv: str,
        ref_col: str = "sentence_norm",
        hyp_col: str = "transcript_norm",
        config: dict = None,
        verbose: bool = True
) -> dict:
    """
    Trainiere Korrekturregeln aus einem Datensatz.

    Returns:
        dict: Das "Modell" mit allen gelernten Regeln
    """
    if config is None:
        config = DEFAULT_CONFIG.copy()

    print("=" * 70)
    print("WER OPTIMIZATION - TRAINING")
    print("=" * 70)

    # Daten laden
    df = pd.read_csv(train_csv)
    df = df.dropna(subset=[ref_col, hyp_col]).reset_index(drop=True)

    refs_raw = df[ref_col].astype(str).tolist()
    hyps_raw = df[hyp_col].astype(str).tolist()

    print(f"Training samples: {len(df)}")
    print("=" * 70)

    # Normalisieren
    refs_norm = [normalize(t) for t in refs_raw]
    hyps_norm = [normalize(t) for t in hyps_raw]

    wer_initial = mean_wer(refs_norm, hyps_norm)
    print(f"[1] WER after normalization: {wer_initial:.4f}")

    # Phrase Rules lernen
    print("\n[2] Learning phrase rules...")
    ngram_subs = extract_ngram_substitutions(refs_norm, hyps_norm, config["max_ngram_size"])
    phrase_rules_candidates = build_phrase_rules(ngram_subs, config)
    print(f"    Candidate phrase rules: {len(phrase_rules_candidates)}")

    # Phrase Rules anwenden und effektive filtern
    hyps_after_phrase, effective_phrase_rules = apply_phrase_rules_for_training(
        hyps_norm, refs_norm, phrase_rules_candidates
    )
    wer_after_phrase = mean_wer(refs_norm, hyps_after_phrase)
    print(f"    Effective phrase rules: {len(effective_phrase_rules)}")
    print(f"    WER after phrase rules: {wer_after_phrase:.4f}")

    # Single-Token Rules lernen
    print("\n[3] Learning single-token rules...")
    sub_counter, match_counter = learn_single_token_rules(refs_norm, hyps_after_phrase)
    token_rules_candidates = build_single_token_rules(sub_counter, match_counter, config)
    print(f"    Candidate token rules: {len(token_rules_candidates)}")

    # Greedy Selection
    effective_token_rules = greedy_select_rules(
        hyps_after_phrase, refs_norm, token_rules_candidates, sub_counter, verbose
    )
    print(f"    Effective token rules: {len(effective_token_rules)}")

    # Finale WER
    hyps_final = hyps_after_phrase
    for source, target in effective_token_rules.items():
        hyps_final = apply_rule(hyps_final, source, target)
    wer_final = mean_wer(refs_norm, hyps_final)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"WER initial:  {wer_initial:.4f}")
    print(f"WER final:    {wer_final:.4f}")
    print(f"Improvement:  {wer_initial - wer_final:.4f} ({(wer_initial - wer_final) / wer_initial * 100:.2f}%)")
    print("=" * 70)

    # Modell zusammenstellen
    model = {
        "version": "v4",
        "config": config,
        "training_stats": {
            "num_samples": len(df),
            "wer_initial": wer_initial,
            "wer_final": wer_final,
        },
        "static_rules": {
            "verb_transforms": VERB_TRANSFORMS,
            "konjunktiv_map": KONJUNKTIV_MAP,
            "word_fixes": WORD_FIXES,
            "phrase_fixes": PHRASE_FIXES,
        },
        "learned_rules": {
            "phrase_rules": effective_phrase_rules,
            "token_rules": effective_token_rules,
        }
    }

    return model


def save_model(model: dict, output_path: str):
    """Speichere das Modell als JSON."""
    # Convert tuples to lists for JSON serialization
    model_serializable = model.copy()
    model_serializable["static_rules"] = model["static_rules"].copy()
    model_serializable["static_rules"]["verb_transforms"] = [
        list(t) for t in model["static_rules"]["verb_transforms"]
    ]

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(model_serializable, f, ensure_ascii=False, indent=2)

    print(f"\nModel saved to: {output_path}")


# ===============================================================
# MAIN
# ===============================================================

def main():
    parser = argparse.ArgumentParser(description="Train WER correction model")
    parser.add_argument("--train", required=True, help="Path to training CSV")
    parser.add_argument("--output", default="wer_model.json", help="Output model path")
    parser.add_argument("--ref-col", default="sentence_norm", help="Reference column name")
    parser.add_argument("--hyp-col", default="transcript_norm", help="Hypothesis column name")
    parser.add_argument("--min-count", type=int, default=2, help="Min count for rules")
    parser.add_argument("--dominance", type=float, default=0.55, help="Dominance threshold")
    parser.add_argument("--error-ratio", type=float, default=0.60, help="Error ratio threshold")

    args = parser.parse_args()

    config = DEFAULT_CONFIG.copy()
    config["min_count"] = args.min_count
    config["dominance"] = args.dominance
    config["error_ratio"] = args.error_ratio

    model = train(
        train_csv=args.train,
        ref_col=args.ref_col,
        hyp_col=args.hyp_col,
        config=config
    )

    save_model(model, args.output)


if __name__ == "__main__":
    main()
