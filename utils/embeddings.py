from sentence_transformers import SentenceTransformer

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

def create_embeddings(chunks):

    embeddings = model.encode(chunks)

    return embeddings