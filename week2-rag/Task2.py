from sentence_transformers import SentenceTransformer, util

class SimpleVectorStore:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """Initializes the embedding model and our storage lists."""
        print(f"Loading embedding model '{model_name}'...")
        self.model = SentenceTransformer(model_name)
        
        self.documents = []
        self.embeddings = []

    def add(self, document: str):
        """Generates an embedding for the document and stores both."""
    
        emb = self.model.encode(document)
        
        self.documents.append(document)
        self.embeddings.append(emb)
        print(f"Added: '{document}'")

    def search(self, query: str, k: int = 3):
        """Searches for the top-k most similar documents to the query."""
        if not self.documents:
            return []

        query_emb = self.model.encode(query)

        results = []

        for i in range(len(self.documents)):
            score = util.cos_sim(query_emb, self.embeddings[i]).item()
            
            results.append({
                "document": self.documents[i],
                "score": score
            })

        results.sort(key=lambda x: x["score"], reverse=True)

        return results[:k]

if __name__ == "__main__":
    vector_db = SimpleVectorStore()
    
    print("\n--- Indexing Phase ---")
    vector_db.add("Python is a versatile programming language.")
    vector_db.add("LLMs are trained on massive amounts of text data.")
    vector_db.add("RAG combines information retrieval with text generation.")
    vector_db.add("Embeddings map human text to mathematical vectors.")
    vector_db.add("Groq provides incredibly fast inference for AI models.")

    print("\n--- Querying Phase ---")
    user_query = "How do we represent text for search?"
    print(f"Query: '{user_query}'\n")
    
    top_matches = vector_db.search(user_query, k=3)
    
    for rank, match in enumerate(top_matches, start=1):
        print(f"Rank {rank} (Score: {match['score']:.4f})")
        print(f"Document: {match['document']}\n")
