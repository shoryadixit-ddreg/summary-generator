from utils import load_and_chunk_pdf
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cuda"},
    encode_kwargs={"normalize_embeddings": True},
)

DATASET_PATH = "pharma_dataset"

chunks = []

for pdf in Path(DATASET_PATH).rglob("*.pdf"):
    print(f"Processing {pdf}")

    pdf_chunks = load_and_chunk_pdf(str(pdf))

    for chunk in pdf_chunks:
        chunk.metadata["source"] = str(pdf)
        chunk.metadata["filename"] = pdf.name

    chunks.extend(pdf_chunks)

print("=" * 80)
print(chunks[0].page_content[:1000])
print("=" * 80)
print(chunks[0].metadata)

print(f"\nTotal chunks: {len(chunks)}")

from langchain_community.vectorstores import FAISS

print("Creating FAISS index...")

vectorstore = FAISS.from_documents(chunks, embeddings)

print("FAISS created.")

vectorstore.save_local("knowledge_base")

db = FAISS.load_local(
    "knowledge_base",
    embeddings,
    allow_dangerous_deserialization=True,
)

print("Knowledge base loaded.")
