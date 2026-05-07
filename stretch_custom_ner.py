"""
Stretch assignment: custom climate NER rules with spaCy EntityRuler.

Run:
    ./venv/bin/python stretch_custom_ner.py
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import spacy


DATA_PATH = Path("data/climate_articles.csv")
GOLD_PATH = Path("data/gold_entities.csv")
OUTPUT_PATH = Path("stretch_comparison.json")

STANDARD_LABELS = {
    "ORG",
    "GPE",
    "DATE",
    "LAW",
    "MONEY",
    "PERSON",
    "QUANTITY",
    "LOC",
    "EVENT",
    "WORK_OF_ART",
}

CLIMATE_PATTERNS = [
    {"label": "LAW", "pattern": "Paris Agreement"},
    {"label": "EVENT", "pattern": "Bonn Climate Change Conference"},
    {"label": "EVENT", "pattern": "Climate Ambition Summit"},
    {"label": "ORG", "pattern": "COP28"},
    {"label": "ORG", "pattern": "COP27"},
    {"label": "ORG", "pattern": "COP26"},
    {"label": "WORK_OF_ART", "pattern": "Sixth Assessment Report"},
    {"label": "POLICY", "pattern": "Global Stocktake"},
    {"label": "POLICY", "pattern": "Loss and Damage Fund"},
    {"label": "POLICY", "pattern": "Global Methane Pledge"},
    {"label": "REPORT", "pattern": "Emissions Gap Report 2023"},
    {"label": "REPORT", "pattern": "Net Zero by 2050 roadmap"},
    {
        "label": "THRESHOLD",
        "pattern": [
            {"TEXT": {"REGEX": r"1\.5|2(?:\.0)?"}},
            {"LOWER": {"IN": ["degree", "degrees"]}},
            {"LOWER": {"IN": ["celsius", "warming"]}, "OP": "?"},
        ],
    },
    {
        "label": "THRESHOLD",
        "pattern": [
            {"TEXT": {"REGEX": r"1\.5"}},
            {"TEXT": "-"},
            {"LOWER": "degree"},
            {"LOWER": "pathway"},
        ],
    },
]


def load_articles() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    return df[df["language"] == "en"].copy()


def load_gold() -> pd.DataFrame:
    return pd.read_csv(GOLD_PATH)


def build_pipeline(position: str) -> spacy.Language:
    nlp = spacy.load("en_core_web_sm")
    if position == "before":
        ruler = nlp.add_pipe("entity_ruler", before="ner", config={"phrase_matcher_attr": "LOWER"})
    elif position == "after":
        ruler = nlp.add_pipe("entity_ruler", after="ner", config={"phrase_matcher_attr": "LOWER"})
    else:
        raise ValueError(f"Unsupported position: {position}")

    ruler.add_patterns(CLIMATE_PATTERNS)
    return nlp


def extract_entities(df: pd.DataFrame, nlp: spacy.Language) -> pd.DataFrame:
    rows = []
    for doc, text_id in zip(nlp.pipe(df["text"], batch_size=16), df["id"]):
        for ent in doc.ents:
            rows.append(
                {
                    "text_id": int(text_id),
                    "entity_text": ent.text,
                    "entity_label": ent.label_,
                    "start_char": ent.start_char,
                    "end_char": ent.end_char,
                }
            )
    return pd.DataFrame(rows)


def summarize_entities(entity_df: pd.DataFrame) -> dict:
    counts = Counter(entity_df["entity_label"])
    return {
        "total_entities": int(len(entity_df)),
        "label_counts": dict(sorted(counts.items())),
    }


def evaluate_standard_labels(predicted_df: pd.DataFrame, gold_df: pd.DataFrame) -> dict:
    filtered_pred = predicted_df[predicted_df["entity_label"].isin(STANDARD_LABELS)].copy()
    filtered_gold = gold_df[gold_df["entity_label"].isin(STANDARD_LABELS)].copy()

    pred_set = {
        (int(row.text_id), str(row.entity_text), str(row.entity_label))
        for row in filtered_pred.itertuples(index=False)
    }
    gold_set = {
        (int(row.text_id), str(row.entity_text), str(row.entity_label))
        for row in filtered_gold.itertuples(index=False)
    }

    true_positives = len(pred_set & gold_set)
    false_positives = len(pred_set - gold_set)
    false_negatives = len(gold_set - pred_set)

    precision = true_positives / (true_positives + false_positives) if pred_set else 0.0
    recall = true_positives / (true_positives + false_negatives) if gold_set else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "predicted_standard_entities": len(pred_set),
        "gold_standard_entities": len(gold_set),
    }


def gather_rule_examples(df: pd.DataFrame, entity_df: pd.DataFrame) -> dict:
    custom_labels = {"POLICY", "REPORT", "THRESHOLD"}
    merged = entity_df.merge(df[["id", "text", "source"]], left_on="text_id", right_on="id", how="left")
    examples: dict[str, list[dict]] = defaultdict(list)

    for row in merged.itertuples(index=False):
        if row.entity_label not in custom_labels:
            continue
        if len(examples[row.entity_label]) >= 3:
            continue
        examples[row.entity_label].append(
            {
                "text_id": int(row.text_id),
                "source": row.source,
                "entity_text": row.entity_text,
                "context": row.text,
            }
        )
    return dict(examples)


def inspect_target_labels(df: pd.DataFrame, pipelines: dict[str, spacy.Language]) -> dict:
    target_terms = [
        "Paris Agreement",
        "Bonn Climate Change Conference",
        "Climate Ambition Summit",
        "COP28",
        "Sixth Assessment Report",
    ]
    examples = {}

    for text_id in [1, 2, 5, 7, 170]:
        text = df.loc[df["id"] == text_id, "text"].iloc[0]
        per_pipeline = {}
        for name, nlp in pipelines.items():
            doc = nlp(text)
            seen = {}
            for ent in doc.ents:
                if ent.text in target_terms:
                    seen[ent.text] = ent.label_
            per_pipeline[name] = seen
        examples[str(text_id)] = per_pipeline
    return examples


def main() -> None:
    articles = load_articles()
    gold = load_gold()

    base_nlp = spacy.load("en_core_web_sm")
    before_nlp = build_pipeline("before")
    after_nlp = build_pipeline("after")

    pipelines = {
        "base": base_nlp,
        "ruler_before_ner": before_nlp,
        "ruler_after_ner": after_nlp,
    }

    extracted = {name: extract_entities(articles, nlp) for name, nlp in pipelines.items()}

    results = {
        "pattern_count": len(CLIMATE_PATTERNS),
        "custom_labels": ["POLICY", "REPORT", "THRESHOLD"],
        "summaries": {name: summarize_entities(df) for name, df in extracted.items()},
        "evaluation_standard_labels_only": {
            name: evaluate_standard_labels(df, gold) for name, df in extracted.items()
        },
        "position_behavior_examples": inspect_target_labels(
            articles,
            {
                "base": base_nlp,
                "before": before_nlp,
                "after": after_nlp,
            },
        ),
        "custom_rule_examples": gather_rule_examples(articles, extracted["ruler_before_ner"]),
    }

    OUTPUT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"Wrote results to {OUTPUT_PATH}")
    for name, metrics in results["evaluation_standard_labels_only"].items():
        print(
            f"{name}: precision={metrics['precision']:.4f} "
            f"recall={metrics['recall']:.4f} f1={metrics['f1']:.4f}"
        )


if __name__ == "__main__":
    main()
