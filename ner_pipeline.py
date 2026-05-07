"""
Module 6 Week A — Lab: NER Pipeline

Build and compare Named Entity Recognition pipelines using spaCy
and Hugging Face on climate-related text data.

Run: python ner_pipeline.py
"""
import unicodedata
import string
import pandas as pd
import numpy as np
import spacy
from transformers import pipeline as hf_pipeline


def load_data(filepath="data/climate_articles.csv"):
    """Load the climate articles dataset.

    Args:
        filepath: Path to the CSV file.

    Returns:
        DataFrame with columns: id, text, source, language, category.
    """
    # TODO: Load the CSV and return the DataFrame
    df = pd.read_csv(filepath)
    return df


def explore_data(df):
    """Summarize basic corpus statistics.

    Args:
        df: DataFrame returned by load_data.

    Returns:
        Dictionary with keys:
          'shape': tuple (n_rows, n_cols)
          'lang_counts': dict mapping language code -> row count
          'category_counts': dict mapping category -> row count
          'text_length_stats': dict with 'mean', 'min', 'max' word counts
    """
    # TODO: Compute shape, language/category value_counts, and word-count
    #       statistics on df['text']
    shape = df.shape
    lang_counts = df['language'].value_counts().to_dict()
    category_counts = df['category'].value_counts().to_dict()
    
    # Calculate word counts for each text
    word_counts = df['text'].str.split().str.len()
    text_length_stats = {
        'mean': word_counts.mean(),
        'min': word_counts.min(),
        'max': word_counts.max()
    }
    
    return {
        'shape': shape,
        'lang_counts': lang_counts,
        'category_counts': category_counts,
        'text_length_stats': text_length_stats
    }


def preprocess_text(text, nlp):
    """Preprocess a single text string for NLP analysis.

    Normalize Unicode, lowercase, remove punctuation, tokenize,
    and lemmatize using the injected spaCy pipeline.

    Args:
        text: Raw text string.
        nlp: A loaded spaCy Language object (e.g., en_core_web_sm).

    Returns:
        List of cleaned, lemmatized token strings.
    """
    # TODO: NFC-normalize the text, run it through nlp(), drop
    #       punctuation/whitespace tokens, return lowercased lemmas
    text = unicodedata.normalize('NFC', text)
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = nlp(text)
    return [token.lemma_ for token in text if token.is_alpha]


def extract_spacy_entities(df, nlp):
    """Extract named entities from English texts using spaCy NER.

    Args:
        df: DataFrame with columns id, text, language, ...
        nlp: A loaded spaCy Language object.

    Returns:
        DataFrame with columns: text_id, entity_text, entity_label,
        start_char, end_char.
    """
    # TODO: Filter df to English rows, process each text with nlp,
    #       collect entities into rows, return as a DataFrame
    df = df[df['language'] == 'en'].copy()
    entities_data = []
    for _, row in df.iterrows():
        doc = nlp(row['text'])
        for ent in doc.ents:
            entities_data.append({
                'text_id': row['id'],
                'entity_text': ent.text,
                'entity_label': ent.label_,
                'start_char': ent.start_char,
                'end_char': ent.end_char
            })
    return pd.DataFrame(entities_data)


def extract_hf_entities(df, ner_pipeline):
    """Extract named entities from English texts using Hugging Face NER.

    Uses the injected HF pipeline (expected: dslim/bert-base-NER).

    Args:
        df: DataFrame with columns id, text, language, ...
        ner_pipeline: A loaded Hugging Face `pipeline('ner', ...)` object.

    Returns:
        DataFrame with columns: text_id, entity_text, entity_label,
        start_char, end_char.
    """
    # TODO: Filter df to English rows, run each text through
    #       ner_pipeline, merge ## subword tokens, strip B-/I- prefix
    #       from labels (IOB format), return as a DataFrame
    df = df[df['language'] == 'en'].copy()
    entities_data = []
    for _, row in df.iterrows():
        results = ner_pipeline(row['text'])
        merged_entities = []
        
        for res in results:
            word = res['word']
            # res['entity'] could be 'B-PER', 'I-PER', or just 'PER' depending on model
            label_parts = res['entity'].split('-')
            label = label_parts[-1]
            prefix = label_parts[0] if len(label_parts) > 1 else ""
            
            # Decide if this token should be merged with the previous one
            # It should be merged if:
            # 1. It's a subword (starts with ##)
            # 2. It has an 'I-' prefix and the label matches the previous entity
            is_subword = word.startswith("##")
            is_continuation = prefix == "I"
            
            if (is_subword or is_continuation) and merged_entities and merged_entities[-1]['entity_label'] == label:
                if is_subword:
                    merged_entities[-1]['entity_text'] += word[2:]
                else:
                    # For I- tokens, add a space if there's a gap in characters
                    if res['start'] > merged_entities[-1]['end_char']:
                        merged_entities[-1]['entity_text'] += " " + word
                    else:
                        merged_entities[-1]['entity_text'] += word
                merged_entities[-1]['end_char'] = res['end']
            else:
                # New entity
                merged_entities.append({
                    'text_id': row['id'],
                    'entity_text': word.lstrip("##"),
                    'entity_label': label,
                    'start_char': res['start'],
                    'end_char': res['end']
                })
        entities_data.extend(merged_entities)
    
    return pd.DataFrame(entities_data)


def compare_ner_outputs(spacy_df, hf_df):
    """Compare entity extraction results from spaCy and Hugging Face.

    Args:
        spacy_df: DataFrame of spaCy entities (from extract_spacy_entities).
        hf_df: DataFrame of HF entities (from extract_hf_entities).

    Returns:
        Dictionary with keys:
          'spacy_counts': dict of entity_label -> count for spaCy
          'hf_counts': dict of entity_label -> count for HF
          'total_spacy': int total entities from spaCy
          'total_hf': int total entities from HF
          'both': set of (text_id, entity_text) tuples found by both systems
          'spacy_only': set of (text_id, entity_text) tuples found only by spaCy
          'hf_only': set of (text_id, entity_text) tuples found only by HF
    """
    # TODO: Count entities per label for each system, compute totals,
    #       and derive the three overlap sets by matching on
    #       (text_id, entity_text)
    spacy_counts = spacy_df['entity_label'].value_counts()
    hf_counts = hf_df['entity_label'].value_counts()
    total_spacy = len(spacy_df)
    total_hf = len(hf_df)
    both = set(zip(spacy_df['text_id'], spacy_df['entity_text'])) & set(zip(hf_df['text_id'], hf_df['entity_text']))
    spacy_only = set(zip(spacy_df['text_id'], spacy_df['entity_text'])) - both
    hf_only = set(zip(hf_df['text_id'], hf_df['entity_text'])) - both
    return {
        'spacy_counts': spacy_counts,
        'hf_counts': hf_counts,
        'total_spacy': total_spacy,
        'total_hf': total_hf,
        'both': both,
        'spacy_only': spacy_only,
        'hf_only': hf_only
    }


def evaluate_ner(predicted_df, gold_df):
    """Evaluate NER predictions against gold-standard annotations.

    Computes entity-level precision, recall, and F1. An entity is a
    true positive if both the entity text and label match a gold entry
    for the same text_id.

    Args:
        predicted_df: DataFrame with columns text_id, entity_text,
                      entity_label.
        gold_df: DataFrame with columns text_id, entity_text,
                 entity_label.

    Returns:
        Dictionary with keys: 'precision', 'recall', 'f1' (floats 0-1).
    """
    # Match predicted entities to gold entities by text_id + entity_text + entity_label
    # We use inner merge to find the intersection (True Positives)
    tp_df = predicted_df.merge(
        gold_df, 
        on=['text_id', 'entity_text', 'entity_label'], 
        how='inner'
    )
    
    tp_count = len(tp_df)
    pred_count = len(predicted_df)
    gold_count = len(gold_df)
    
    precision = tp_count / pred_count if pred_count > 0 else 0
    recall = tp_count / gold_count if gold_count > 0 else 0
    
    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0
        
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


if __name__ == "__main__":
    # Load spaCy and HF models once, reuse across functions
    nlp = spacy.load("en_core_web_sm")
    hf_ner = hf_pipeline("ner", model="dslim/bert-base-NER")

    # Load and explore
    df = load_data()
    if df is not None:
        summary = explore_data(df)
        if summary is not None:
            print(f"Shape: {summary['shape']}")
            print(f"Languages: {summary['lang_counts']}")
            print(f"Categories: {summary['category_counts']}")
            print(f"Text length (words): {summary['text_length_stats']}")

        # Preprocess a sample to verify your function
        sample_row = df[df["language"] == "en"].iloc[0]
        sample_tokens = preprocess_text(sample_row["text"], nlp)
        if sample_tokens is not None:
            print(f"\nSample preprocessed tokens: {sample_tokens[:10]}")

        # spaCy NER across the English corpus
        try:
            spacy_entities = extract_spacy_entities(df, nlp)
            if spacy_entities is not None:
                print(f"\nspaCy entities: {len(spacy_entities)} total")
        except Exception as e:
            print(f"Error during spaCy entity extraction: {e}")
            spacy_entities = None
       

        # HF NER across the English corpus
        hf_entities = extract_hf_entities(df, hf_ner)
        if hf_entities is not None:
            print(f"HF entities: {len(hf_entities)} total")

        # Compare the two systems
        if spacy_entities is not None and hf_entities is not None:
            comparison = compare_ner_outputs(spacy_entities, hf_entities)
            if comparison is not None:
                print(f"\nBoth systems agreed on {len(comparison['both'])} entities")
                print(f"spaCy-only: {len(comparison['spacy_only'])}")
                print(f"HF-only: {len(comparison['hf_only'])}")

        # Evaluate against gold standard
        gold = pd.read_csv("data/gold_entities.csv")
        if spacy_entities is not None:
            metrics = evaluate_ner(spacy_entities, gold)
            if metrics is not None:
                print(f"\nspaCy evaluation: {metrics}")
