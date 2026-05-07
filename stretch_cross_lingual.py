import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def get_embeddings(texts, model, tokenizer):
    """
    Extracts embeddings using mean pooling over the last hidden states.
    """
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Last hidden state has shape [batch_size, seq_len, hidden_size]
    last_hidden_state = outputs.last_hidden_state
    
    # Attention mask has shape [batch_size, seq_len]
    attention_mask = inputs['attention_mask']
    
    # Expand attention mask to match last_hidden_state shape
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    
    # Sum embeddings while masking out padding tokens
    sum_embeddings = torch.sum(last_hidden_state * mask, 1)
    
    # Count non-padding tokens
    sum_mask = torch.clamp(mask.sum(1), min=1e-9)
    
    # Mean pooling
    mean_embeddings = sum_embeddings / sum_mask
    return mean_embeddings.numpy()

def main():
    # Load data
    df = pd.read_csv('data/climate_articles.csv')
    
    # Define English and Arabic IDs for comparison
    # Choosing pairs that are related
    en_ids = [1, 9, 123, 129, 127, 133, 139, 144, 147, 148]
    ar_ids = [119, 116, 145, 146, 141, 135, 115, 120, 110, 103]
    
    en_df = df[df['id'].isin(en_ids)].copy()
    ar_df = df[df['id'].isin(ar_ids)].copy()
    
    # Ensure they are in a specific order if we want to see diagonal matches
    # Match id 9 (EN) with 119 (AR)
    # Match id 123 (EN) with 116 (AR)
    # Match id 129 (EN) with 145 (AR)
    
    # Re-ordering to make the heatmap more readable with matches on diagonal or near-diagonal
    en_selection = [
        df[df['id'] == 9].iloc[0],   # Green Climate Fund
        df[df['id'] == 123].iloc[0], # Methane
        df[df['id'] == 129].iloc[0], # Earth's Future (precip)
        df[df['id'] == 127].iloc[0], # Jordan Water Strategy
        df[df['id'] == 148].iloc[0], # Climate-smart agriculture
        df[df['id'] == 139].iloc[0], # Jordan River
        df[df['id'] == 133].iloc[0], # Dead Sea
        df[df['id'] == 147].iloc[0], # Glaciers
        df[df['id'] == 1].iloc[0],   # IPCC
        df[df['id'] == 144].iloc[0], # Red Sea Corals
    ]
    
    ar_selection = [
        df[df['id'] == 119].iloc[0], # Green Climate Fund
        df[df['id'] == 116].iloc[0], # Methane
        df[df['id'] == 145].iloc[0], # Earth's Future
        df[df['id'] == 146].iloc[0], # Jordan Water conservation
        df[df['id'] == 103].iloc[0], # Mafraq solar (Jordan adaptation)
        df[df['id'] == 141].iloc[0], # Jordan Groundwater
        df[df['id'] == 135].iloc[0], # MENA Warming
        df[df['id'] == 115].iloc[0], # Arctic ice
        df[df['id'] == 122].iloc[0], # Saudi renewable
        df[df['id'] == 110].iloc[0], # Noor Ouarzazate
    ]
    
    en_texts = [x['text'] for x in en_selection]
    ar_texts = [x['text'] for x in ar_selection]
    
    en_labels = [x['text'][:40] + "..." for x in en_selection]
    ar_labels = [x['text'][:40] + "..." for x in ar_selection]
    
    # Load model and tokenizer
    print("Loading bert-base-multilingual-cased...")
    model_name = "bert-base-multilingual-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    # Get embeddings
    print("Extracting English embeddings...")
    en_embeddings = get_embeddings(en_texts, model, tokenizer)
    print("Extracting Arabic embeddings...")
    ar_embeddings = get_embeddings(ar_texts, model, tokenizer)
    
    # Compute similarity matrix
    # We'll compute a 20x20 matrix as requested (all English and Arabic texts)
    all_texts = en_texts + ar_texts
    all_labels = en_labels + ar_labels
    all_embeddings = np.vstack([en_embeddings, ar_embeddings])
    
    similarity_matrix = cosine_similarity(all_embeddings)
    
    # Create heatmap
    plt.figure(figsize=(16, 12))
    sns.heatmap(similarity_matrix, annot=True, fmt=".2f", cmap="YlGnBu",
                xticklabels=all_labels, yticklabels=all_labels)
    plt.title("Cross-Lingual Embedding Similarity (Multilingual BERT)")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Save heatmap
    plt.savefig('cross_lingual_similarity_heatmap.png')
    print("Heatmap saved as cross_lingual_similarity_heatmap.png")
    
    # Also print some interesting findings
    print("\nSpecific Cross-Lingual Matches:")
    # Match id 9 (EN) and 119 (AR) are at indices 0 and 10
    print(f"EN: {en_labels[0]} vs AR: {ar_labels[0]} -> Similarity: {similarity_matrix[0, 10]:.4f}")
    # Match id 123 (EN) and 116 (AR) are at indices 1 and 11
    print(f"EN: {en_labels[1]} vs AR: {ar_labels[1]} -> Similarity: {similarity_matrix[1, 11]:.4f}")
    # Match id 129 (EN) and 145 (AR) are at indices 2 and 12
    print(f"EN: {en_labels[2]} vs AR: {ar_labels[2]} -> Similarity: {similarity_matrix[2, 12]:.4f}")

    # Random within-language comparison
    print("\nWithin-Language Examples:")
    print(f"EN: {en_labels[0]} vs EN: {en_labels[1]} -> Similarity: {similarity_matrix[0, 1]:.4f}")
    print(f"AR: {ar_labels[0]} vs AR: {ar_labels[1]} -> Similarity: {similarity_matrix[10, 11]:.4f}")

if __name__ == "__main__":
    main()
