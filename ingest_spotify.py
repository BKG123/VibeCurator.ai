"""Load and ingest Spotify Million Song Dataset into Qdrant."""
from datasets import load_dataset
from main import main as ingest_main
import argparse


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
        ingest_main(dataset, total=args.limit)
        print("\n✅ Done! View at http://localhost:6333/dashboard")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("🔧 Troubleshoot: docker compose up -d && curl http://localhost:6333/health")
        raise
