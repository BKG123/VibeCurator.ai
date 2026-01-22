"""Tools for agent to interact with song embeddings."""

import os
from typing import Any
from agents import function_tool
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Filter

# Initialize model and client
model = SentenceTransformer("all-MiniLM-L6-v2")
client = QdrantClient(
    host=os.getenv("QDRANT_HOST", "localhost"),
    port=int(os.getenv("QDRANT_PORT", "6333")),
)

COLLECTION_NAME = "songs"


@function_tool
def search_songs(query: str, limit: int = 10) -> list[dict[str, Any]]:
    """Search for songs based on semantic similarity to the query.

    Use this tool to find songs that match a mood, vibe, theme, or description.

    Args:
        query: Natural language description of desired songs (e.g., "sad breakup songs",
               "energetic workout music", "90s cyberpunk vibes")
        limit: Maximum number of songs to return (default: 10, max: 50)

    Returns:
        List of songs with artist, title, link, and text preview

    Examples:
        - search_songs("upbeat summer party anthem", limit=5)
        - search_songs("melancholic rainy day songs", limit=10)
        - search_songs("aggressive heavy metal", limit=15)
    """
    # Validate limit
    limit = min(max(1, limit), 50)

    try:
        # Embed query
        query_vector = model.encode(query).tolist()

        # Search in Qdrant
        results = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=limit,
        )

        # Format results
        songs = []
        for point in results.points:
            songs.append(
                {
                    "artist": point.payload.get("artist"),
                    "song": point.payload.get("song"),
                    "link": point.payload.get("link"),
                    "preview": point.payload.get("text_preview"),
                    "score": round(point.score, 4),
                }
            )

        return songs

    except Exception as e:
        return [{"error": f"Failed to search songs: {str(e)}"}]


@function_tool
def search_songs_by_artist(artist_name: str, limit: int = 10) -> list[dict[str, Any]]:
    """Search for songs by a specific artist.

    Args:
        artist_name: Name of the artist
        limit: Maximum number of songs to return (default: 10, max: 50)

    Returns:
        List of songs by the artist
    """
    limit = min(max(1, limit), 50)

    try:
        # Use scroll to get songs by artist (no vector search needed)
        results = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=Filter(
                must=[{"key": "artist", "match": {"value": artist_name}}]
            ),
            limit=limit,
        )

        songs = []
        for point in results[0]:  # scroll returns (points, next_page_offset)
            songs.append(
                {
                    "artist": point.payload.get("artist"),
                    "song": point.payload.get("song"),
                    "link": point.payload.get("link"),
                    "preview": point.payload.get("text_preview"),
                }
            )

        return songs

    except Exception as e:
        return [{"error": f"Failed to search by artist: {str(e)}"}]


@function_tool
def get_collection_stats() -> dict[str, Any]:
    """Get statistics about the song collection.

    Returns:
        Dictionary with collection statistics
    """
    try:
        collection_info = client.get_collection(COLLECTION_NAME)
        return {
            "total_songs": collection_info.points_count,
            "collection_name": COLLECTION_NAME,
            "status": "active",
        }
    except Exception as e:
        return {"error": f"Failed to get stats: {str(e)}", "status": "unavailable"}
