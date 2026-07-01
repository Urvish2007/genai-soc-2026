from sentence_transformers import SentenceTransformer, util

def calculate_similarity(sentence_a, sentence_b):
 
    print("Loading model... (this may take a few seconds on the first run)")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    embedding_a = model.encode(sentence_a)
    embedding_b = model.encode(sentence_b)

    cosine_score = util.cos_sim(embedding_a, embedding_b)

    score = cosine_score.item()
    
    print("\n--- Results ---")
    print(f"Sentence A: '{sentence_a}'")
    print(f"Sentence B: '{sentence_b}'")
    print(f"Cosine Similarity: {score:.4f}")

if __name__ == "__main__":
    sentence_1 = "The cat sleeps on the mat."
    sentence_2 = "A cat is resting on the rug."
    
    calculate_similarity(sentence_1, sentence_2)
