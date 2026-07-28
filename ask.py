from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load FAISS index
index = faiss.read_index("vector.index")

# Load chunks
with open("chunks.txt", "r", encoding="utf-8") as f:
    chunks = f.readlines()

def ask_question(question):
    q_embedding = model.encode([question])
    D, I = index.searchr̥r̥r̥r̥r̥r̥r̥r̥r̥r̥r̥(np.array(q_embedding), k=1)
    return chunks[I[0][0]]

while True:
    q = input("\nAsk a question (type exit): ")
    if q.lower() == "exit":
        break
    answer = ask_question(q)
    print("📌 Answer from file:\n", answer)