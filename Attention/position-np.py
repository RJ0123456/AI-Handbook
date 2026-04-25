import numpy as np

def get_positional_encoding(seq_len, d_model):
    """
    Generates a sinusoidal positional encoding matrix.
    Args:
        seq_len: Number of words (tokens) in the sentence.
        d_model: Dimensionality of the word vectors (e.g., 512).
    """
    # Create an empty matrix (shape: sequence_length x dimensions)
    pos_enc = np.zeros((seq_len, d_model))
    
    for pos in range(seq_len):
        for i in range(0, d_model, 2):
            # The 'division term' controls the frequency of the waves
            div_term = np.exp(i * -np.log(10000.0) / d_model)
            
            # Apply sine to even indices, cosine to odd indices
            pos_enc[pos, i] = np.sin(pos * div_term)
            if i + 1 < d_model:
                pos_enc[pos, i + 1] = np.cos(pos * div_term)
                
    return pos_enc

# --- Setup Example ---
seq_len = 4   # e.g., "The cat sat down"
d_model = 6   # Using 6 dimensions for a clear view

# 1. Word Embeddings (Semantic meaning learned during training)
word_embeddings = np.random.randn(seq_len, d_model)

# 2. Positional Encodings (Fixed mathematical pattern for word order)
pos_encodings = get_positional_encoding(seq_len, d_model)

# 3. ADDITION: The key step where order meets meaning
final_input = word_embeddings + pos_encodings

print("Word Embeddings:\n", np.round(word_embeddings, 2))
print("\nPositional Encodings:\n", np.round(pos_encodings, 2))
print("\nFinal Input to Transformer:\n", np.round(final_input, 2))

#L = np.dot(pos_encodings, pos_encodings.T)
#print("\nDot Product of Positional Encodings:\n", np.round(L, 2))