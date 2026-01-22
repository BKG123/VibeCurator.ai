# VibeCurator.ai

Semantic music search using vector embeddings. Find songs by vibe and abstract concepts, not just keywords. Create YouTube Music playlists instantly!

## Stack

- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2, 384-dim)
- **Vector DB**: Qdrant (cosine similarity)
- **Dataset**: Spotify Million Song Dataset (57K+ songs)
- **Playlist Creation**: YouTube Data API v3 integration

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

### 3. Configure Environment Variables

Copy `env.example` to `.env` and add your OpenAI API key:

```bash
cp env.example .env
```

### 4. Set Up YouTube API

Follow the [YouTube Setup Guide](YOUTUBE_SETUP.md) to:
1. Create a Google Cloud project
2. Enable YouTube Data API v3
3. Create OAuth credentials
4. Download `youtube_credentials.json` to project root

**Quick steps:**
- Go to [Google Cloud Console](https://console.cloud.google.com/)
- Create project → Enable YouTube Data API v3
- Create OAuth 2.0 credentials (Desktop app)
- Download JSON as `youtube_credentials.json`

### 5. Run Ingestion Pipeline

**Test with 5K songs (~5-10 min)**:
```bash
python ingest_spotify.py --limit 5000
```

**Full dataset (57K songs, ~2-3 hours)**:
```bash
python ingest_spotify.py
```

### 6. Run the App

```bash
python main.py
```

Or directly:
```bash
streamlit run app.py
```

## Usage

### Chat with the Assistant

Ask the agent to find songs and create playlists:
- "Find me 15 energetic workout songs and create a playlist"
- "I need sad breakup songs for a rainy day playlist"
- "Create a 90s cyberpunk vibe playlist"

The agent will search the database and create a YouTube Music playlist with a shareable link!

**First time:** A browser will open for Google authentication. Allow access to manage YouTube playlists.

### Programmatic Search

```python
from tools import search_songs, create_youtube_playlist

# Search for songs
results = search_songs("upbeat workout music", limit=10)
for song in results:
    print(f"{song['score']:.3f} - {song['artist']}: {song['song']}")

# Create playlist
songs = [{"artist": s["artist"], "song": s["song"]} for s in results]
playlist = create_youtube_playlist(songs, "My Workout Mix")
print(playlist["playlist_url"])
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
