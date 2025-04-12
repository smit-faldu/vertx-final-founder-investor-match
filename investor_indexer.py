#investor_indexer.py
import json
import faiss
import pickle
import numpy as np
from langchain.embeddings import HuggingFaceEmbeddings

print("🔹 Loading investor data...")
with open("Combined file.xlsx - Sheet1.json", "r", encoding="utf-8") as f:
    investors = json.load(f)

# ✅ Use basic embedding model: all-MiniLM-L6-v2
print("🔹 Initializing all-MiniLM-L6-v2 embedding model...")
embedding_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}  # 🔥 Important for cosine similarity
)

# ✅ Convert investor metadata into searchable text
print("🔹 Preparing investor documents for embedding...")
texts = [
    f"{inv.get('Name', '')}. {inv.get('Overview', '')} "
    f"Industry: {inv.get('Industry', '')}. Stage: {inv.get('Stage', '')}. "
    f"Type: {inv.get('Type', '')}. Countries: {inv.get('Countries', '')}."
    for inv in investors
]

# ✅ Embed documents
print("🔹 Generating embeddings...")
embeddings = embedding_model.embed_documents(texts)
embeddings_np = np.array(embeddings).astype("float32")

# ✅ Use cosine similarity (IndexFlatIP)
print("🔹 Building FAISS index...")
dimension = embeddings_np.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(embeddings_np)

# ✅ Save index and investor metadata
print("🔹 Saving index and metadata...")
faiss.write_index(index, "investors.index")
with open("investors.pkl", "wb") as f:
    pickle.dump(investors, f)

print(f"✅ Indexed {len(investors)} investors using all-MiniLM-L6-v2 into FAISS (cosine similarity).")
