import os
import glob
import subprocess
import mlx_whisper

TARGET_USERNAME = "premanand_maharaj"
OUTPUT_DIR = "transcripts"
DOWNLOAD_DIR = f"downloads_{TARGET_USERNAME}"
AUDIO_DIR = f"audio_{TARGET_USERNAME}"

def extract_audio(video_path: str, audio_path: str) -> bool:
    """Extracts lightweight 16kHz mono WAV audio from MP4 using FFmpeg."""
    try:
        command = [
            "ffmpeg",
            "-y",                   # Overwrite output file if exists
            "-i", video_path,       # Input video
            "-vn",                  # Disable video recording
            "-acodec", "pcm_s16le", # 16-bit PCM audio codec
            "-ar", "16000",         # 16kHz sample rate (Whisper native rate)
            "-ac", "1",             # Mono channel
            audio_path
        ]
        # Run ffmpeg quietly without stdout clutter
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except Exception as e:
        print(f"Error extracting audio from {video_path}: {e}")
        return False

def process_reels():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(AUDIO_DIR, exist_ok=True)

    video_files = sorted(glob.glob(os.path.join(DOWNLOAD_DIR, "*.mp4")))
    total_videos = len(video_files)

    if total_videos == 0:
        print(f"No video files found in '{DOWNLOAD_DIR}'. Run yt-dlp first.")
        return

    print(f"Found {total_videos} videos. Processing pipeline initialized...\n")

    master_transcript_path = os.path.join(OUTPUT_DIR, f"{TARGET_USERNAME}_all_254_transcripts.txt")

    with open(master_transcript_path, "a", encoding="utf-8") as master_file:
        for idx, video_path in enumerate(video_files, start=1):
            filename = os.path.basename(video_path)
            base_name = os.path.splitext(filename)[0]
            
            txt_filepath = os.path.join(OUTPUT_DIR, f"{base_name}.txt")
            audio_filepath = os.path.join(AUDIO_DIR, f"{base_name}.wav")

            # Skip if transcript already exists
            if os.path.exists(txt_filepath):
                print(f"[{idx}/{total_videos}] Skipping {filename} (Already transcribed).")
                continue

            print(f"[{idx}/{total_videos}] Processing {filename}...")

            # 1. Extract audio if not already extracted
            if not os.path.exists(audio_filepath):
                success = extract_audio(video_path, audio_filepath)
                if not success:
                    continue

            # 2. Transcribe lightweight audio file using Apple Silicon MLX
            try:
                result = mlx_whisper.transcribe(
                    audio_filepath,
                    path_or_hf_repo="mlx-community/whisper-small-mlx"
                )
                text = result["text"].strip()

                # Save individual text transcript
                with open(txt_filepath, "w", encoding="utf-8") as f:
                    f.write(text)

                # Append to master output file
                master_file.write(f"=== REEL: {filename} ===\n{text}\n\n")
                master_file.flush()

                # Clean up extracted audio file to save disk space
                if os.path.exists(audio_filepath):
                    os.remove(audio_filepath)

            except Exception as e:
                print(f"Error transcribing {filename}: {e}")

    # Cleanup temporary audio directory when complete
    if os.path.exists(AUDIO_DIR) and not os.listdir(AUDIO_DIR):
        os.rmdir(AUDIO_DIR)

    print(f"\nDone! All transcripts successfully saved in '{OUTPUT_DIR}/'.")

if __name__ == "__main__":
    process_reels()