from src.embedder import Embedder
from src.retriever import Retriever
from src.vector_store import VectorStore


# -------------------------
# 1. Load vector store
# -------------------------

vector_store = VectorStore(
    "data/vector_store/store.json"
)

chunks, embeddings = vector_store.load()

print(f"Loaded {len(chunks)} chunks")


# -------------------------
# 2. Create embedding model
# -------------------------

embedder = Embedder()


# -------------------------
# 3. Create retriever
# -------------------------

retriever = Retriever(
    chunks,
    embeddings
)


# -------------------------
# 4. Ask question
# -------------------------

question = input("\nAsk a question: ")


# -------------------------
# 5. Embed question
# -------------------------

question_embedding = embedder.embed_text(
    question
)


# -------------------------
# 6. Search
# -------------------------

results = retriever.search(
    question_embedding,
    top_k=3,
    threshold=0.3
)


# -------------------------
# 7. Display results
# -------------------------

if not results:

    print("\nNo relevant information found.")

else:

    print("\nRelevant information:\n")

    for result in results:

        print(
            f"Similarity: {result['score']:.4f}"
        )

        print(
            result["chunk"]
        )

        print("-" * 60)