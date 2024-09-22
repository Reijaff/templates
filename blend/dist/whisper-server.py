import base64
import time
import json
from flask import Flask, request, jsonify
import whisper_timestamped
import string


app = Flask(__name__)

# Load the Whisper model
whisper_model = whisper_timestamped.load_model("medium")

_punctuation = (
    "".join(c for c in string.punctuation if c not in ["-", "'"]) + "。，！？：”、…"
)


def split_long_segments(segments, max_length, use_space=True):
    new_segments = []
    for segment in segments:
        text = segment["text"]
        if len(text) <= max_length:
            new_segments.append(segment)
        else:
            meta_words = segment["words"]
            # Note: we do this in case punctuation were removed from words
            if use_space:
                # Split text around spaces and punctuations (keeping punctuations)
                words = text.split()
            else:
                words = [w["text"] for w in meta_words]
            if len(words) != len(meta_words):
                new_words = [w["text"] for w in meta_words]
                print(f"WARNING: {' '.join(words)} != {' '.join(new_words)}")
                words = new_words
            current_text = ""
            current_start = segment["start"]
            current_best_idx = None
            current_best_end = None
            current_best_next_start = None
            for i, (word, meta) in enumerate(zip(words, meta_words)):
                current_text_before = current_text
                if current_text and use_space:
                    current_text += " "
                current_text += word

                if len(current_text) > max_length and len(current_text_before):
                    start = current_start
                    if current_best_idx is not None:
                        text = current_text[:current_best_idx]
                        end = current_best_end
                        current_text = current_text[current_best_idx + 1 :]
                        current_start = current_best_next_start
                    else:
                        text = current_text_before
                        end = meta_words[i - 1]["end"]
                        current_text = word
                        current_start = meta["start"]

                    current_best_idx = None
                    current_best_end = None
                    current_best_next_start = None

                    new_segments.append({"text": text, "start": start, "end": end})

                # Try to cut after punctuation
                if current_text and current_text[-1] in _punctuation:
                    current_best_idx = len(current_text)
                    current_best_end = meta["end"]
                    current_best_next_start = (
                        meta_words[i + 1]["start"] if i + 1 < len(meta_words) else None
                    )

            if len(current_text):
                new_segments.append(
                    {
                        "text": current_text,
                        "start": current_start,
                        "end": segment["end"],
                    }
                )

    return new_segments


def format_timestamp(
    seconds: float, always_include_hours: bool = False, decimal_marker: str = "."
):
    assert seconds >= 0, "non-negative timestamp expected"
    milliseconds = round(seconds * 1000.0)

    hours = milliseconds // 3_600_000
    milliseconds -= hours * 3_600_000

    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000

    seconds = milliseconds // 1_000
    milliseconds -= seconds * 1_000

    hours_marker = f"{hours:02d}:" if always_include_hours or hours > 0 else ""
    return (
        f"{hours_marker}{minutes:02d}:{seconds:02d}{decimal_marker}{milliseconds:03d}"
    )


def write_vtt(result, file):
    print("WEBVTT\n", file=file)
    for segment in result:
        print(
            f"{format_timestamp(segment['start'])} --> {format_timestamp(segment['end'])}\n"
            f"{segment['text'].strip().replace('-->', '->')}\n",
            file=file,
            flush=True,
        )


def write_srt(result, file):
    for i, segment in enumerate(result, start=1):
        # # write srt lines

        tmp4 = " ".join(
            word.strip()
            for word in segment["text"]
            .strip()
            .replace("-->", "->")
            .lower()
            .translate(
                str.maketrans(_punctuation, " " * len(_punctuation))
            )  # Replace punctuation
            .split()
        )

        print(
            f"{i}\n"
            f"{format_timestamp(segment['start'], always_include_hours=True, decimal_marker=',')} --> "
            f"{format_timestamp(segment['end'], always_include_hours=True, decimal_marker=',')}\n"
            f"{tmp4}\n",
            file=file,
            flush=True,
        )


@app.route("/transcribe", methods=["POST"])
def transcribe_audio():
    data = request.get_json()

    if "audio_base64" not in data:
        return jsonify({"error": "No audio_base64 field provided"}), 400

    if "srt_file_path" not in data:
        return jsonify({"error": "No srt_file_path field provided"}), 400

    try:
        audio_data = base64.b64decode(data["audio_base64"])
    except Exception as e:
        return jsonify({"error": "Invalid base64 encoding"}), 400

    with open("temp_audio.wav", "wb") as f:
        f.write(audio_data)

    # Process the audio
    start_time = time.time()
    audio = whisper_timestamped.load_audio("temp_audio.wav")
    transcription_data = whisper_timestamped.transcribe(
        whisper_model,
        audio,
        language=None,
        # detect_disfluencies=True,
        remove_punctuation_from_words=True,
        compute_word_confidence=False,
        # #
        # beam_size=5,
        # best_of=5,
        # temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    )
    print("transcribed in ", time.time() - start_time)

    #
    # transcript = json.loads(transcription_data)
    segments = transcription_data["segments"]
    segments = split_long_segments(segments, 25, use_space=True)

    with open(data["srt_file_path"], "w") as f:
        write_srt(segments, file=f)
    #
    print(data["srt_file_path"])

    return jsonify(transcription_data)


if __name__ == "__main__":
    app.run(port=5302, debug=False)

