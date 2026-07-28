from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load embedding model (Hugging Face – offline after first load)
model = SentenceTransformer("all-MiniLM-L6-v2")

# Read file
with open("data.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Simple chunking (manual, easy)
chunks = text.split("\n")

# Create embeddings
embeddings = model.encode(chunks)

# Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

# Save index
faiss.write_index(index, "vector.index")

# Save chunks
with open("chunks.txt", "w", encoding="utf-8") as f:
    for chunk in chunks:
        f.write(chunk + "\n")

print("✅ Vector database created")