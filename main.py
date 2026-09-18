"""Download a video, extract its audio, and transcribe it with Whisper."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

import whisper


def run_command(command: list[str]) -> None:
    """Run a required external command and show its output in the terminal."""
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            f"Command failed with exit code {error.returncode}: {command[0]}"
        ) from error


def require_command(command: str) -> None:
    if shutil.which(command) is None:
        raise RuntimeError(
            f"{command} was not found on PATH. Install it and try again."
        )


def summarize_with_gemini(transcript: str, model: str) -> str:
    """Return a concise summary of a transcript using the Gemini API."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY must be set when using --summarize.")

    request_body = json.dumps(
        {
            "contents": [
                {
                    "parts": [
                        {
                            "text": (
                                # "Summarize the following video transcript concisely. "
                                # "Include the main points and any important conclusions.\n\n"
                                "reformat this text to be easily readable\n\n"
                                f"{transcript}"
                            )
                        }
                    ]
                }
            ]
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        data=request_body,
        headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Gemini API request failed: {detail}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"Could not reach the Gemini API: {error.reason}") from error

    try:
        summary = payload["candidates"][0]["content"]["parts"][0]["text"]
    except (IndexError, KeyError, TypeError) as error:
        raise RuntimeError("Gemini did not return a text summary.") from error
    if not isinstance(summary, str):
        raise RuntimeError("Gemini did not return a text summary.")
    return summary.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a video URL, extract audio with ffmpeg, and transcribe it with Whisper."
    )
    parser.add_argument("url", help="Video URL supported by yt-dlp")
    parser.add_argument(
        "-m", "--model", default="base", help="Whisper model name (default: base)"
    )
    parser.add_argument("-l", "--language", help="Spoken language code, for example en")
    parser.add_argument(
        "--summarize",
        action="store_true",
        help="Summarize the transcript with Gemini using GEMINI_API_KEY",
    )
    parser.add_argument(
        "--gemini-model",
        default="gemini-3.8-flash",
        help="Gemini model for --summarize (default: gemini-3.8-flash)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Transcript file to create (default: print to standard output)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    require_command("yt-dlp")
    require_command("ffmpeg")

    with tempfile.TemporaryDirectory(
        prefix="whisper-transcribe-"
    ) as temporary_directory:
        working_directory = Path(temporary_directory)
        video_template = working_directory / "video.%(ext)s"
        run_command(["yt-dlp", "--no-playlist", "-o", str(video_template), args.url])

        videos = [path for path in working_directory.iterdir() if path.name != ".part"]
        if len(videos) != 1:
            raise RuntimeError("yt-dlp did not produce exactly one video file.")

        audio_path = working_directory / "audio.wav"
        run_command(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(videos[0]),
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                str(audio_path),
            ]
        )

        model = whisper.load_model(args.model)
        result = model.transcribe(str(audio_path), language=args.language)

    transcript = result.get("text")
    if not isinstance(transcript, str):
        raise RuntimeError("Whisper did not return a text transcript.")

    print(args.url)
    text = transcript.strip()
    print(text)

    if args.summarize:
        text = summarize_with_gemini(text, args.gemini_model)
        print(text)

    if args.output:
        output_path = args.output.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")
        print(f"Output saved to {output_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
