"""Tools for agent to interact with song embeddings."""

import os
import json
from agents import function_tool
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Filter
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
import pickle
from pydantic import BaseModel

# Pydantic models for tool outputs
class Song(BaseModel):
    """Song information."""
    artist: str
    song: str
    link: str
    preview: str
    score: float | None = None

class ArtistStats(BaseModel):
    """Artist statistics."""
    artist: str
    song_count: int

class CollectionStats(BaseModel):
    """Collection statistics."""
    total_songs: int
    total_artists: int
    top_artists: list[ArtistStats]

class PlaylistResult(BaseModel):
    """YouTube playlist creation result."""
    success: bool
    playlist_id: str | None = None
    playlist_url: str | None = None
    message: str

# Initialize model and client
model = SentenceTransformer("all-MiniLM-L6-v2")
client = QdrantClient(
    host=os.getenv("QDRANT_HOST", "localhost"),
    port=int(os.getenv("QDRANT_PORT", "6333")),
)

COLLECTION_NAME = "songs"

# YouTube API configuration
SCOPES = ["https://www.googleapis.com/auth/youtube.force-ssl"]
TOKEN_PICKLE = "youtube_token.pickle"


def get_youtube_client():
    """Get authenticated YouTube client."""
    creds = None

    # Load saved credentials
    if os.path.exists(TOKEN_PICKLE):
        with open(TOKEN_PICKLE, "rb") as token:
            creds = pickle.load(token)

    # If no valid credentials, authenticate
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                "youtube_credentials.json", SCOPES
            )
            creds = flow.run_local_server(port=8080)

        # Save credentials for next time
        with open(TOKEN_PICKLE, "wb") as token:
            pickle.dump(creds, token)

    return build("youtube", "v3", credentials=creds)


@function_tool
def search_songs(query: str, limit: int = 10) -> list[Song]:
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
                Song(
                    artist=point.payload.get("artist", "Unknown"),
                    song=point.payload.get("song", "Unknown"),
                    link=point.payload.get("link", ""),
                    preview=point.payload.get("text_preview", ""),
                    score=round(point.score, 4),
                )
            )

        return songs

    except Exception:
        # Return empty list on error
        return []


@function_tool
def search_songs_by_artist(artist_name: str, limit: int = 10) -> list[Song]:
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
                Song(
                    artist=point.payload.get("artist", "Unknown"),
                    song=point.payload.get("song", "Unknown"),
                    link=point.payload.get("link", ""),
                    preview=point.payload.get("text_preview", ""),
                )
            )

        return songs

    except Exception:
        return []


@function_tool
def get_collection_stats() -> str:
    """Get statistics about the song collection.

    Returns:
        JSON string with collection statistics
    """
    try:
        collection_info = client.get_collection(COLLECTION_NAME)
        stats = {
            "total_songs": collection_info.points_count,
            "collection_name": COLLECTION_NAME,
            "status": "active",
        }
        return json.dumps(stats)
    except Exception as e:
        return json.dumps({"error": f"Failed to get stats: {str(e)}", "status": "unavailable"})


@function_tool
def create_youtube_playlist(
    songs_json: str,
    playlist_name: str,
    playlist_description: str = "Curated by VibeCurator AI",
) -> str:
    """Create a YouTube Music playlist with the given songs.

    Args:
        songs_json: JSON string of song list with 'artist' and 'song' keys
                    Example: '[{"artist": "The Beatles", "song": "Hey Jude"}]'
        playlist_name: Name for the playlist
        playlist_description: Description for the playlist (optional)

    Returns:
        JSON string with playlist URL and details, or error message

    Example:
        create_youtube_playlist(
            songs_json='[{"artist": "Radiohead", "song": "Creep"}]',
            playlist_name="My Vibe Playlist"
        )
    """
    try:
        # Parse songs JSON
        songs = json.loads(songs_json)

        youtube = get_youtube_client()

        # Create playlist
        playlist_response = (
            youtube.playlists()
            .insert(
                part="snippet,status",
                body={
                    "snippet": {
                        "title": playlist_name,
                        "description": playlist_description,
                    },
                    "status": {"privacyStatus": "unlisted"},
                },
            )
            .execute()
        )

        playlist_id = playlist_response["id"]
        playlist_url = f"https://www.youtube.com/playlist?list={playlist_id}"

        # Search for videos and add to playlist
        found_tracks = []
        not_found = []

        for song in songs:
            artist = song.get("artist", "")
            song_name = song.get("song", "")

            if not artist or not song_name:
                continue

            # Search for video
            search_query = f"{artist} {song_name} official audio"
            search_response = (
                youtube.search()
                .list(
                    q=search_query,
                    part="snippet",
                    type="video",
                    maxResults=1,
                    videoCategoryId="10",  # Music category
                )
                .execute()
            )

            if search_response["items"]:
                video = search_response["items"][0]
                video_id = video["id"]["videoId"]

                # Add video to playlist
                youtube.playlistItems().insert(
                    part="snippet",
                    body={
                        "snippet": {
                            "playlistId": playlist_id,
                            "resourceId": {
                                "kind": "youtube#video",
                                "videoId": video_id,
                            },
                        }
                    },
                ).execute()

                found_tracks.append(f"{artist} - {song_name}")
            else:
                not_found.append(f"{artist} - {song_name}")

        result = {
            "success": True,
            "playlist_url": playlist_url,
            "playlist_name": playlist_name,
            "tracks_added": len(found_tracks),
            "tracks_found": found_tracks,
            "tracks_not_found": not_found,
            "message": f"Created YouTube playlist with {len(found_tracks)} tracks! View it at: {playlist_url}",
        }
        return json.dumps(result)

    except Exception as e:
        result = {
            "success": False,
            "error": f"Failed to create playlist: {str(e)}",
            "message": "Make sure YouTube credentials are set up correctly. Run setup first.",
        }
        return json.dumps(result)
