from src.chunker import create_chunks
from src.embedder import Embedder
from src.vector_store import VectorStore


# -------------------------
# 1. Load document
# -------------------------

with open("documents/gradex.txt", "r", encoding="utf-8") as file:
    text = file.read()


# -------------------------
# 2. Create chunks
# -------------------------

chunks = create_chunks(
    text,
    chunk_size=500,
    overlap=100
)

print(f"Created {len(chunks)} chunks")


# -------------------------
# 3. Create embeddings
# -------------------------

embedder = Embedder()

embeddings = embedder.embed_chunks(chunks)

print("Created embeddings")


# -------------------------
# 4. Save vector store
# -------------------------

vector_store = VectorStore(
    "data/vector_store/store.json"
)

vector_store.save(
    chunks,
    embeddings
)

print("Vector store created successfully")