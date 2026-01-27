"""Automated discovery pipeline for regular ingestion.

Run this script via cron job to discover and ingest new songs periodically.
"""

import os
import argparse
from datetime import datetime
from discovery import GeniusDiscovery, LastFmDiscovery, save_songs_to_json, Song
from ingest_spotify import ingest_songs, model, get_batches


def discover_multilingual_songs(
    english_artists: list[str] = None,
    hindi_artists: list[str] = None,
    bengali_artists: list[str] = None,
    songs_per_artist: int = 10,
) -> list[Song]:
    """Discover songs across English, Hindi, and Bengali."""

    genius = GeniusDiscovery()
    all_songs = []

    # Default artist lists (curated popular artists)
    english_artists = english_artists or [
        "The Beatles",
        "Radiohead",
        "Taylor Swift",
        "Ed Sheeran",
        "Arctic Monkeys",
        "Coldplay",
        "Billie Eilish",
    ]

    hindi_artists = hindi_artists or [
        "Arijit Singh",
        "Shreya Ghoshal",
        "Sonu Nigam",
        "Atif Aslam",
        "Pritam",
        "A.R. Rahman",
        "Badshah",
        "Divine",
    ]

    bengali_artists = bengali_artists or [
        "Rabindranath Tagore",
        "Nachiketa",
        "Rupankar Bagchi",
        "Arijit Singh",
        "Shreya Ghoshal",
        "Anupam Roy",
    ]

    print("🌍 Discovering Multilingual Songs\n")

    # English songs
    print("🇬🇧 English Artists:")
    for artist in english_artists:
        songs = genius.search_artist(artist, max_songs=songs_per_artist)
        all_songs.extend(songs)
        for song in songs:
            song.language = "english"
            genius.tracker.mark_processed(song)

    # Hindi songs
    print("\n🇮🇳 Hindi Artists:")
    for artist in hindi_artists:
        songs = genius.search_artist(artist, max_songs=songs_per_artist)
        all_songs.extend(songs)
        for song in songs:
            song.language = "hindi"
            genius.tracker.mark_processed(song)

    # Bengali songs
    print("\n🇧🇩 Bengali Artists:")
    for artist in bengali_artists:
        songs = genius.search_artist(artist, max_songs=songs_per_artist)
        all_songs.extend(songs)
        for song in songs:
            song.language = "bengali"
            genius.tracker.mark_processed(song)

    return all_songs


def discover_from_lastfm(tags: list[str] = None, limit_per_tag: int = 20) -> list[Song]:
    """Discover trending songs from Last.fm."""

    lastfm = LastFmDiscovery()
    genius = GeniusDiscovery()

    tags = tags or ["bollywood", "bengali", "indie", "rock", "pop"]

    all_songs = []

    print("📻 Discovering from Last.fm\n")

    for tag in tags:
        print(f"Tag: {tag}")
        trending = lastfm.get_top_tracks_by_tag(tag, limit=limit_per_tag)

        for track in trending[:10]:  # Limit to avoid rate limits
            song = genius.search_song(track["song"], track["artist"])
            if song:
                all_songs.append(song)
                genius.tracker.mark_processed(song)

    return all_songs


def ingest_discovered_songs(songs: list[Song], batch_size: int = 100):
    """Ingest discovered songs into Qdrant."""

    if not songs:
        print("⚠️  No songs to ingest")
        return

    print(f"\n🔄 Ingesting {len(songs)} songs into Qdrant")

    # Convert to format expected by ingest_songs
    def song_generator():
        for song in songs:
            yield {
                "artist": song.artist,
                "song": song.song,
                "text": song.lyrics,
                "link": song.link,
            }

    try:
        ingest_songs(song_generator(), total=len(songs))
        print("✅ Ingestion complete!")
    except Exception as e:
        print(f"❌ Ingestion error: {e}")


def run_discovery_pipeline(
    mode: str = "multilingual",
    dry_run: bool = False,
    save_json: bool = True,
    ingest: bool = True,
):
    """Run the full discovery and ingestion pipeline."""

    print(f"🚀 Starting Discovery Pipeline - Mode: {mode}")
    print(f"   Dry run: {dry_run}, Save JSON: {save_json}, Ingest: {ingest}\n")

    discovered_songs = []

    if mode == "multilingual":
        discovered_songs = discover_multilingual_songs(
            songs_per_artist=5  # Adjust based on your needs
        )

    elif mode == "lastfm":
        discovered_songs = discover_from_lastfm(
            tags=["bollywood", "bengali", "rock", "indie"], limit_per_tag=20
        )

    elif mode == "combined":
        # Combine both approaches
        songs1 = discover_multilingual_songs(songs_per_artist=3)
        songs2 = discover_from_lastfm(limit_per_tag=10)
        discovered_songs = songs1 + songs2

    else:
        print(f"❌ Unknown mode: {mode}")
        return

    print(f"\n📊 Discovery Summary:")
    print(f"   Total songs discovered: {len(discovered_songs)}")

    if not discovered_songs:
        print("⚠️  No new songs discovered")
        return

    # Save to JSON
    if save_json:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"discovered_songs_{timestamp}.json"
        save_songs_to_json(discovered_songs, output_path)

    # Ingest into Qdrant
    if ingest and not dry_run:
        ingest_discovered_songs(discovered_songs)
    elif dry_run:
        print("\n🔍 DRY RUN - Skipping ingestion")
        print("\nSample songs discovered:")
        for song in discovered_songs[:5]:
            print(f"  • {song.artist} - {song.song} ({len(song.lyrics)} chars)")

    print("\n✅ Pipeline complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Music discovery and ingestion pipeline"
    )

    parser.add_argument(
        "--mode",
        choices=["multilingual", "lastfm", "combined"],
        default="multilingual",
        help="Discovery mode",
    )

    parser.add_argument(
        "--dry-run", action="store_true", help="Run without ingesting to Qdrant"
    )

    parser.add_argument("--no-ingest", action="store_true", help="Skip ingestion step")

    parser.add_argument("--no-save", action="store_true", help="Skip saving to JSON")

    args = parser.parse_args()

    run_discovery_pipeline(
        mode=args.mode,
        dry_run=args.dry_run,
        save_json=not args.no_save,
        ingest=not args.no_ingest,
    )
