"""Load and ingest Spotify Million Song Dataset into Qdrant."""

import argparse
import os
import uuid
from datasets import load_dataset
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Initialize model and client
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


def ingest_songs(ds, total: int | None = None):
    """Ingest dataset into Qdrant vector database."""
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


def load_spotify_dataset(limit: int | None = None, streaming: bool = True):
    """Load Spotify Million Song Dataset from Hugging Face."""
    print(f"📦 Loading dataset (streaming={streaming})...")

    ds = load_dataset(
        "vishnupriyavr/spotify-million-song-dataset", split="train", streaming=streaming
    )

    # Filter out entries with missing critical fields
    def is_valid(item):
        return item.get("artist") and item.get("song") and item.get("text")

    count = 0
    for item in ds:
        if not is_valid(item):
            continue

        yield item
        count += 1

        if limit and count >= limit:
            break

    if not limit:
        print(f"✓ Processed {count:,} total songs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest Spotify dataset into Qdrant")
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit songs (e.g., 5000 for testing)"
    )
    parser.add_argument(
        "--no-streaming", action="store_true", help="Load entire dataset into memory"
    )

    args = parser.parse_args()

    print(f"⚙️  Config: limit={args.limit or 'all'}, streaming={not args.no_streaming}")
    print("💡 Ensure Qdrant is running: docker compose up -d\n")

    dataset = load_spotify_dataset(limit=args.limit, streaming=not args.no_streaming)

    try:
        ingest_songs(dataset, total=args.limit)
        print("\n✅ Done! View at http://localhost:6333/dashboard")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("🔧 Troubleshoot: docker compose up -d && curl http://localhost:6333/health")
        raise
