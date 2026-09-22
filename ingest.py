with open("documents/gradex.txt", "r") as file:
    text = file.read()


def create_chunks(text, chunk_size=200):
    chunks = []

    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size]
        chunks.append(chunk)

    return chunks


chunks = create_chunks(text)

for i, chunk in enumerate(chunks):
    print(f"\n--- CHUNK {i} ---")
    print(chunk)