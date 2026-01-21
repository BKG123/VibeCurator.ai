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
    """Yield batches of specified size from iterable."""
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def main(ds, total: int | None = None):
    """Ingest dataset into Qdrant vector database."""
    from tqdm import tqdm

    # Get embedding dimension from model
    embedding_dim = model.get_sentence_embedding_dimension()

    # Create collection if it doesn't exist
    try:
        client.get_collection(COLLECTION_NAME)
        print(f"✓ Collection '{COLLECTION_NAME}' exists")
    except Exception:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE),
        )
        print(f"✓ Created collection '{COLLECTION_NAME}'")

    # Process and upsert with progress bar
    total_upserted = 0
    with tqdm(total=total, desc="Ingesting songs", unit="songs") as pbar:
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
                for embedding, payload in zip(embeddings, payloads)
            ]

            # Upsert points to collection
            client.upsert(collection_name=COLLECTION_NAME, points=points)
            total_upserted += len(batch)
            pbar.update(len(batch))

    print(f"✓ Ingested {total_upserted:,} songs")


def retrieve_similar(query: str, *, top_k: int = 10, flt: Filter | None = None):
    """Embed query and return top-k similar songs from vector database."""
    query_vector = model.encode(query).tolist()
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        query_filter=flt,
    )
    return results.points


def print_results(hits):
    """Pretty print search results."""
    for i, hit in enumerate(hits, 1):
        print(f"\n{i}. Score: {hit.score:.4f}")
        print(f"   Artist: {hit.payload['artist']}")
        print(f"   Song: {hit.payload['song']}")
        print(f"   Link: {hit.payload.get('link', 'N/A')}")
        print(f"   Preview: {hit.payload['text_preview']}")


if __name__ == "__main__":
    try:
        # Check collection status
        collection_info = client.get_collection(COLLECTION_NAME)
        print(
            f"📊 Collection '{COLLECTION_NAME}' has {collection_info.points_count:,} songs\n"
        )

        # Example queries
        queries = [
            "90s cyberpunk movie vibe",
            "heartbreak and lost love",
            "upbeat summer party anthem",
        ]

        for query in queries:
            print(f"🔍 Query: '{query}'")
            print("=" * 80)
            hits = retrieve_similar(query, top_k=5)
            print_results(hits)
            print("\n" + "=" * 80 + "\n")

    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure to:")
        print("   1. Start Qdrant: docker compose up -d")
        print("   2. Run ingestion: python ingest_spotify.py")
