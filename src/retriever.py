import numpy as np


class Retriever:

    def __init__(self, chunks, embeddings):
        self.chunks = chunks
        self.embeddings = embeddings

    def similarity(self, query_embedding, chunk_embedding):

        return np.dot(query_embedding, chunk_embedding) / (
            np.linalg.norm(query_embedding)
            * np.linalg.norm(chunk_embedding)
        )

    def search(self, query_embedding, top_k=3, threshold=0.3):

        scores = []

        for i, embedding in enumerate(self.embeddings):

            score = self.similarity(
                query_embedding,
                embedding
            )

            if score >= threshold:
                scores.append((i, score))

        scores.sort(
            key=lambda x: x[1],
            reverse=True
        )

        results = []

        for index, score in scores[:top_k]:

            results.append({
                "chunk": self.chunks[index],
                "score": float(score),
                "index": index
            })

        return results