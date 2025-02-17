import os, json, subprocess, ffmpeg, numpy as np
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip
from vosk import Model, KaldiRecognizer, SetLogLevel

def extract_audio_from_video(video_path):
    """Extracts raw audio data from video and saves it as a .wav file."""
    try:
        probe_output = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", video_path],
            capture_output=True, text=True
        ).stdout
        probe = json.loads(probe_output)
    except json.JSONDecodeError:
        raise ValueError("FFprobe output is not valid JSON. Check FFmpeg installation.")

    # Find the audio stream
    audio_stream = next((stream for stream in probe.get("streams", []) if stream.get("codec_type") == "audio"), None)
    if not audio_stream:
        raise ValueError("No audio stream found in the video.")

    sample_rate = int(audio_stream["sample_rate"])

    process = (
        ffmpeg
        .input(video_path)
        .output("pipe:", format="wav", acodec="pcm_s16le", ar=sample_rate, ac=1)
        .run(capture_stdout=True, capture_stderr=True)
    )

    audio_data = np.frombuffer(process[0], dtype=np.int16)
    
    return audio_data, sample_rate


def transcribe_audio(audio_data, sample_rate, model_path="vosk-model-small-en-us-0.15", debug=False):
    """Transcribes in-memory audio using Vosk and generates subtitles based on recognized text."""
    if not debug: SetLogLevel(-1)

    model = Model(model_path)
    recognizer = KaldiRecognizer(model, sample_rate)
    recognizer.SetWords(True)
    recognizer.SetPartialWords(True)

    subtitles = []
    chunk_size = 8000  # Size of audio chunks to process at a time
    segment_start = 0

    for i in range(0, len(audio_data), chunk_size):
        chunk = audio_data[i:i+chunk_size].tobytes()
        if recognizer.AcceptWaveform(chunk):
            result = json.loads(recognizer.Result())
            if debug: print(f"Recognized: {result}")  # Print result to check if speech is detected
            if result["text"].strip():
                # Extract word timestamps
                for word_info in result.get("result", []):
                    word = word_info["word"]
                    word_start = segment_start + word_info["start"]
                    word_end = segment_start + word_info["end"]
                    subtitles.append((word_start, word_end, word))

                segment_start += 3  # Adjust the start time for the next chunk

    if  debug: print("Subtitles:", subtitles)  # Print the final subtitles list
    return subtitles


def add_subtitles_to_video(video_path, subtitles, output_video=os.path.join(os.getcwd(), "output_with_subtitles.mp4")):
    """Overlays individual words as subtitles directly onto the video."""
    video_clip = VideoFileClip(video_path)
    subtitle_clips = []

    # Create individual TextClip for each word
    for start, end, text in subtitles:
        text_clip = TextClip(
            text,
            fontsize=70,
            font="Arial",
            color="yellow",
            bg_color="transparent"
        ).set_position(("center", "center")).set_duration(end - start).set_start(start)

        subtitle_clips.append(text_clip)

    # Overlay all subtitle clips on top of the video
    final_clip = CompositeVideoClip([video_clip] + subtitle_clips)
    final_clip.write_videofile(output_video, codec="libx264", audio_codec="aac", fps=24)

def main() -> None:
    video_file = os.path.join(os.getcwd(), "video.mp4")
    audio_data, sample_rate = extract_audio_from_video(video_file)
    subtitles = transcribe_audio(audio_data, sample_rate)
    add_subtitles_to_video(video_file, subtitles)
    print("Subtitle added! Saved as output_with_subtitles.mp4")

if __name__ == "__main__":
    main()
