import json
import numpy as np


class VectorStore:

    def __init__(self, file_path):
        self.file_path = file_path

    def save(self, chunks, embeddings):

        data = {
            "chunks": chunks,
            "embeddings": embeddings.tolist()
        }

        with open(self.file_path, "w", encoding="utf-8") as file:
            json.dump(data, file)

        print(f"Saved {len(chunks)} chunks")

    def load(self):

        with open(self.file_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        chunks = data["chunks"]

        embeddings = np.array(
            data["embeddings"],
            dtype=np.float32
        )

        return chunks, embeddings