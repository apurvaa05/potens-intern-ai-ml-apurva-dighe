# test.py

from utils.pdf_loader import load_pdfs
from utils.chunker import chunk_text
from utils.embeddings import create_embeddings, model
from utils.retriever import build_faiss_index, retrieve

docs = load_pdfs("documents")

all_chunks = []
chunks_metadata = []

for doc in docs:

    chunks = chunk_text(doc["text"])

    for idx, chunk in enumerate(chunks):

        all_chunks.append(chunk)

        chunks_metadata.append({
            "file_name": doc["file_name"],
            "chunk_id": idx,
            "text": chunk
        })

embeddings = create_embeddings(all_chunks)

build_faiss_index(embeddings, chunks_metadata)

query = "What is human resource policy?"

query_embedding = model.encode(query)

results = retrieve(query_embedding)

print("\nTOP RESULTS:\n")

for result in results:

    print("FILE:", result["file_name"])

    print("CHUNK:", result["chunk_id"])

    print("SCORE:", result["score"])

    print("\nTEXT:\n")

    print(result["text"][:500])

    print("\n--------------------------\n")