
import pandas as pd
import spacy
from ner_pipeline import extract_spacy_entities

nlp = spacy.load("en_core_web_sm")
df = pd.DataFrame({
    'id': [1],
    'text': [['this', 'is', 'a', 'list']],
    'language': ['en']
})

try:
    extract_spacy_entities(df, nlp)
except Exception as e:
    print(f"Caught expected error: {e}")
