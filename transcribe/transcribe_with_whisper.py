import os
import pandas as pd
from yt_dlp import YoutubeDL
from tqdm import tqdm
import datetime
import subprocess
import json


# ~/.conda/envs/bigdataproject/bin/ffmpeg

# Ensure required directories exist
os.makedirs("videos", exist_ok=True)
os.makedirs("audio", exist_ok=True)
os.makedirs("transcribe_data", exist_ok=True)
start_row = 90000
end_row = 92000
# Load dataset
INPUT_CSV = "/sise/home/mordeche/bigdata_youtube/data/youtube_childbirth_concated_videos.csv"  # Update this path
OUTPUT_PATH = "/sise/home/mordeche/bigdata_youtube/transcribe_data"
OUTPUT_CSV_PREFIX = "chunk"
CHUNK_SIZE = 100  # Number of videos per batch
OUTPUT_JSON = f"/sise/home/mordeche/bigdata_youtube/outputs/out_{start_row}_{end_row}.json"
# --- Load data ---
df = pd.read_csv(INPUT_CSV)

# --- Function: Download and extract audio ---
def download_audio(video_url, video_id):
    try:
        ydl_opts = {
            "format": "bestaudio/best",
            "--ffmpeg-location": "~/.conda/envs/bigdataproject/bin/",
            "outtmpl": f"videos/{video_id}.%(ext)s",
            "postprocessors": [
                {"key": "FFmpegExtractAudio", "preferredcodec": "wav", "preferredquality": "192"}
            ],
            "quiet": True,
        }
        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])
        audio_path = f"videos/{video_id}.wav"
        return audio_path if os.path.exists(audio_path) else None
    except Exception as e:
        print(f"Failed to download {video_url}: {e}")
        return None

# --- Function: Transcribe audio using insanely-fast-whisper CLI ---
def transcribe_audio(audio_path):
    try:
        result = subprocess.run(
            ["insanely-fast-whisper", "--file-name", audio_path, "--model-name", "openai/whisper-base",
             "--device-id", "0", "--transcript-path", OUTPUT_JSON],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            with open(OUTPUT_JSON, 'r') as f:
                transcription_data = json.load(f)
                transcription_text = " ".join(chunk["text"] for chunk in transcription_data.get("chunks", []))
                # print(transcription_text)
            return transcription_text
        else:
            print(f"Transcription failed for {audio_path} with error: {result.stderr}")
            return None
    except Exception as e:
        print(f"Failed to transcribe {audio_path}: {e}")
        return None

# --- Process DataFrame in Chunks ---
#num_rows = len(df) #TODO: change
num_rows = end_row
for start_idx in tqdm(range(start_row, num_rows, CHUNK_SIZE), desc="Processing Chunks"):
    end_idx = min(start_idx + CHUNK_SIZE, num_rows)
    chunk_df = df.iloc[start_idx:end_idx].copy()

    print(f"\n[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Processing chunk: rows {start_idx} to {end_idx - 1}")

    transcriptions = []

    for idx, row in tqdm(chunk_df.iterrows(), total=len(chunk_df), desc="Downloading & Transcribing", leave=False):
        video_id = row["video_id"]
        video_url = row["video_url"]

        audio_path = download_audio(video_url, video_id)
        transcription = transcribe_audio(audio_path) if audio_path else None
        audio_file = f"videos/{video_id}.wav"
        if os.path.exists(audio_file):
            os.remove(audio_file)
        transcriptions.append(transcription)

    chunk_df["transcription"] = transcriptions

    # --- Save to CSV for this chunk ---
    output_csv = f"{OUTPUT_PATH}/{OUTPUT_CSV_PREFIX}_{start_idx}_{end_idx}.csv"
    chunk_df.to_csv(output_csv, index=False)
    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Saved chunk to {output_csv}")

    # --- Cleanup audio files ---
    for row in chunk_df.itertuples():
        audio_file = f"videos/{row.video_id}.wav"
        if os.path.exists(audio_file):
            os.remove(audio_file)
    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Deleted audio files for rows {start_idx} to {end_idx - 1}.\n")

print("Finished processing all chunks.")
