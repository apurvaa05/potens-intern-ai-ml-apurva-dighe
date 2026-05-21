import faiss
import numpy as np

index = None
stored_chunks = []


def build_faiss_index(embeddings, chunks_metadata):

    global index
    global stored_chunks

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(np.array(embeddings))

    stored_chunks = chunks_metadata


def retrieve(query_embedding, top_k=6):

    distances, indices = index.search(
        np.array([query_embedding]),
        top_k
    )

    results = []

    for idx, distance in zip(indices[0], distances[0]):

        result = stored_chunks[idx].copy()

        confidence = max(
            0,
            min(
                100,
                round(
                    (1 / (1 + float(distance))) * 100,
                    2
                )
            )
        )

        result["score"] = float(distance)

        result["confidence"] = confidence

        results.append(result)

    return results


def rerank_results(results, query):

    query_words = set(
        query.lower().split()
    )

    for result in results:

        chunk_words = set(
            result["text"].lower().split()
        )

        overlap = len(
            query_words.intersection(chunk_words)
        )

        result["rerank_score"] = overlap

    results.sort(
        key=lambda x: x["rerank_score"],
        reverse=True
    )

    return results