# -*- coding: utf-8 -*-
"""Taking a url instead of a file.

a youtube link, a playlist, or anything else yt-dlp can reach. only the audio is fetched —
the pipeline resamples to mono 16 kHz anyway, so pulling video would cost bandwidth and disk
for nothing.

**one item at a time.** the playlist is listed first, without downloading anything, and then
each item is fetched, transcribed and deleted before the next one starts. downloading the
whole playlist up front would mean a channel with two hundred uploads fills the disk before
producing a single transcript, and a crash halfway through would leave the lot behind. this
way peak disk is one episode, results appear as they are made, and an interrupted run
resumes without re-fetching what it already has.

resuming works off the video id, which yt-dlp writes into the filename as `[id]`. the audio
is gone by then, but the json is not, so an item whose transcript is already complete is
skipped before its download starts.

a note on what you download. a link is somebody's work under somebody's terms, and fetching
it may be against the terms of the site it sits on. that judgement is the caller's; this
module only does what it is told.
"""
import re
from pathlib import Path

from .logging import Progress, log

# a real url, or one of yt-dlp's own search prefixes (ytsearch5:query and friends)
URL_PATTERN = re.compile(r"^((https?|ftp)://|yt[a-z]*search\d*:)", re.I)

# the id in the name is what makes resuming possible: titles get sanitised and truncated,
# ids do not.
OUTTMPL = "%(title).120B [%(id)s].%(ext)s"

# audio only, at whatever quality is offered. the pipeline downmixes to mono 16 kHz, so a
# higher bitrate buys nothing and a smaller file downloads faster.
YTDLP_OPTIONS = {
    "format": "bestaudio/best",
    "outtmpl": OUTTMPL,
    "restrictfilenames": True,
    "windowsfilenames": True,
    "noplaylist": True,
    "ignoreerrors": True,
    "quiet": True,
    "no_warnings": True,
    "noprogress": True,
    "retries": 5,
}


def is_url(value: str) -> bool:
    return bool(URL_PATTERN.match(value.strip()))


def _ydl(extra=None):
    import yt_dlp
    options = dict(YTDLP_OPTIONS)
    options.update(extra or {})
    return yt_dlp.YoutubeDL(options)


def available() -> bool:
    try:
        import yt_dlp  # noqa: F401
    except ImportError:
        log("  ⚠ yt-dlp is not installed — a url cannot be fetched. pip install yt-dlp")
        return False
    return True


# ------------------------------------------------------------------------- listing
def enumerate_items(url: str, limit: int = 0) -> list:
    """What the url points at, without downloading. one entry per video.

    `extract_flat` asks yt-dlp for the playlist's index only — titles and ids, no formats —
    which turns listing a two-hundred-item channel into one request instead of two hundred.
    """
    if not available():
        return []
    extra = {"extract_flat": "in_playlist", "skip_download": True}
    if limit and limit > 0:
        extra["playlistend"] = limit
    try:
        with _ydl(extra) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception as exc:
        log(f"  ⚠ could not read {url}: {type(exc).__name__}: {exc}")
        return []
    if not info:
        return []

    raw = info.get("entries")
    playlist = raw is not None
    entries = [e for e in (raw if playlist else [info]) if e]

    items = []
    for entry in entries:
        video_id = entry.get("id")
        if not video_id:
            continue
        # webpage_url first, and the order is not cosmetic. for a flat playlist entry
        # `url` is the watch page, but for a single video — where extract_flat does not
        # apply — `url` is the direct media stream, and downloading that produces a file
        # named after the whole signed query string, which windows refuses to create.
        items.append({
            "id": video_id,
            "title": entry.get("title") or video_id,
            "url": entry.get("webpage_url") or entry.get("url") or video_id,
            "duration": entry.get("duration"),
        })
    if playlist:
        name = info.get("title") or "playlist"
        log(f"  {name}: {len(items)} item(s)"
            + (f", first {limit} taken" if limit and len(raw) >= limit else ""))
    return items


def existing_result(item_id: str, folder: Path):
    """The json already produced for this video id, if there is one.

    matched on the `[id]` yt-dlp puts in the filename rather than on the title, because the
    title is sanitised, truncated to 120 bytes and may change on the site between runs.
    """
    needle = f"[{item_id}]"
    for path in sorted(folder.glob("*.json")):
        if needle in path.name:
            return path
    return None


# ------------------------------------------------------------------------ fetching
def fetch_one(item: dict, target_dir: Path):
    """Download the audio of one item. returns the file, or None if it did not arrive."""
    if not available():
        return None
    target_dir.mkdir(parents=True, exist_ok=True)

    bar = Progress("downloading", total=None, unit="")

    def hook(status):
        if status.get("status") != "downloading":
            return
        total = status.get("total_bytes") or status.get("total_bytes_estimate")
        got = status.get("downloaded_bytes") or 0
        # percent rather than megabytes: a two-megabyte episode divided into whole MB
        # reads "1/1" and looks stuck, while a percentage is meaningful at any size.
        if total:
            bar.unit = "%"
            bar.set(int(got * 100 / total), total=100,
                    note=f"{got / 2 ** 20:.1f} of {total / 2 ** 20:.1f} MB")
        else:
            bar.unit = "MB"
            bar.set(int(got / 2 ** 20), note=item["title"][:40])

    extra = {"outtmpl": str(target_dir / OUTTMPL), "progress_hooks": [hook]}
    before = set(_audio_files(target_dir))
    try:
        with _ydl(extra) as ydl:
            ydl.download([item["url"]])
    except Exception as exc:
        bar.close("failed")
        log(f"  ⚠ could not fetch {item['title'][:60]}: {type(exc).__name__}: {exc}")
        return None

    arrived = sorted(set(_audio_files(target_dir)) - before)
    if not arrived:
        bar.close("nothing arrived")
        log(f"  ⚠ nothing downloaded for {item['title'][:60]}")
        return None
    path = arrived[0]
    bar.close(f"{path.stat().st_size / 2 ** 20:.0f} MB")
    return path


def _audio_files(folder: Path) -> list:
    from .config import MEDIA_EXTS
    return [f for f in folder.iterdir()
            if f.is_file() and f.suffix.lower() in MEDIA_EXTS]
