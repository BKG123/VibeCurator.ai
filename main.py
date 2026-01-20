from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, Filter, PointStruct, VectorParams
import os
import uuid

model = SentenceTransformer("all-MiniLM-L6-v2")
client = QdrantClient(
    host=os.getenv("QDRANT_HOST", "localhost"),
    port=int(os.getenv("QDRANT_PORT", "6333")),
)

COLLECTION_NAME = "songs"


def get_batches(iterable, size):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def main(ds):
    # Get embedding dimension from model
    embedding_dim = model.get_sentence_embedding_dimension()

    # Create collection if it doesn't exist
    try:
        client.get_collection(COLLECTION_NAME)
    except Exception:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE),
        )

    # Process and Upsert
    for batch in get_batches(ds, size=100):
        # Create the rich text to be embedded
        texts = [
            f"Artist: {x['artist']} Song: {x['song']} Lyrics: {x['text'][:500]}"
            for x in batch
        ]
        embeddings = model.encode(texts)

        # Payload contains metadata for the final playlist
        payloads = [
            {
                "artist": x.get("artist"),
                "song": x.get("song"),
                "link": x.get("link"),
                "text_preview": (x.get("text") or "")[:200],
            }
            for x in batch
        ]

        # Create points for upsert
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding.tolist(),
                payload=payload,
            )
            for idx, (embedding, payload) in enumerate(zip(embeddings, payloads))
        ]

        # Upsert points to collection
        client.upsert(collection_name=COLLECTION_NAME, points=points)

        print(f"Upserted {len(batch)} points to collection")


def retrieve_similar(
    query: str,
    *,
    top_k: int = 10,
    flt: Filter | None = None,
):
    """Embed `query` and return top-k similar items from Qdrant."""
    query_vector = model.encode(query).tolist()
    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=top_k,
        query_filter=flt,
    )
    return results


if __name__ == "__main__":
    # Safe default: don't ingest automatically.
    # Set INGEST=1 to run ingestion with your loaded dataset.
    if os.getenv("INGEST") == "1":
        raise RuntimeError(
            "Dataset loading is not implemented yet. "
            "Load `ds` (iterable of dicts with artist/song/text/link) then call main(ds)."
        )

    # Quick sanity check retrieval (assumes you've already ingested):
    try:
        hits = retrieve_similar("90s cyberpunk movie vibe", top_k=5)
        for hit in hits:
            print(hit.score, hit.payload)
    except Exception as e:
        print(
            "Qdrant not reachable or collection missing. "
            "Start docker compose and ingest first.\n"
            f"Error: {e}"
        )
