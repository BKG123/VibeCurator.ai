# VibeCurator.ai

Semantic music search using vector embeddings. Find songs by vibe and abstract concepts, not just keywords.

## Stack

- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2, 384-dim)
- **Vector DB**: Qdrant (cosine similarity)
- **Dataset**: Spotify Million Song Dataset (57K+ songs)

## Quick Start

**Prerequisites**: Python 3.11+, Docker

### 1. Start Qdrant

```bash
docker compose up -d
```

Dashboard: `http://localhost:6333/dashboard`

### 2. Install Dependencies

```bash
uv sync
```

### 3. Run Ingestion Pipeline

**Test with 5K songs (~5-10 min)**:
```bash
python ingest_spotify.py --limit 5000
```

**Full dataset (57K songs, ~2-3 hours)**:
```bash
python ingest_spotify.py
```

### 4. Test Retrieval

```bash
python main.py
```

## Usage

### Programmatic Search

```python
from main import retrieve_similar

results = retrieve_similar("upbeat workout music", top_k=10)
for hit in results:
    print(f"{hit.score:.3f} - {hit.payload['artist']}: {hit.payload['song']}")
```

### Re-ingest Data

```bash
python ingest_spotify.py --limit 5000  # Test with 5K songs
python ingest_spotify.py              # Full dataset
```

## Data Schema

**Vector**: 384-dim from `"Artist: {artist} Song: {song} Lyrics: {lyrics[:500]}"`
**Payload**: `artist`, `song`, `link`, `text_preview`

## Troubleshooting

**Qdrant not reachable**:
```bash
docker compose up -d
curl http://localhost:6333/health
```

**Reset collection**:
```python
from qdrant_client import QdrantClient
client = QdrantClient(host="localhost", port=6333)
client.delete_collection("songs")
```
