import os
import tempfile
import base64
import logging
import edge_tts
from flask import Flask, request, jsonify


import whisper_timestamped as whisper
from melo.api import TTS

# Use a production-grade WSGI server
from waitress import serve

# Robust error handling and user input validation
from werkzeug.exceptions import BadRequest, InternalServerError

# NLTK for natural language processing (if needed)
import nltk

nltk.download("averaged_perceptron_tagger_eng")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

app = Flask(__name__)

# Initialize TTS model once
tts_model = TTS(language="EN")
speaker_ids = tts_model.hps.data.spk2id

# Load Whisper model lazily
whisper_model = None

VOICE = "en-US-AndrewNeural"


@app.route("/api/btts", methods=["GET", "POST"])
def btts():
    """
    Handles text-to-speech (TTS) and optional transcription requests.
    """

    # Get text and transcription flag from request
    text = request.args.get("text") or request.form.get("text", "")
    is_transcription = request.args.get("transcription", "False").lower() == "true"

    # Get speech speed, handling potential errors
    try:
        speech_speed = float(request.args.get("speed", "1"))
    except ValueError:
        raise BadRequest("Invalid speech speed. Please provide a number.")

    # Log request details
    logging.info(
        f"Received request: text='{text}', is_transcription={is_transcription}, speech_speed={speech_speed}"
    )

    # Validate input text
    if not text:
        raise BadRequest("No text provided for TTS synthesis.")

    try:
        # Generate TTS audio using a temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            # tts_model.tts_to_file(
            #     text, speaker_ids["EN-Default"], tmp_file.name, speed=speech_speed
            # )

            communicate = edge_tts.Communicate(
                text=text, voice=VOICE, rate="+10%", pitch="-10Hz"
            )
            communicate.save_sync(tmp_file.name)

            logging.info("TTS synthesis completed.")

            # Perform transcription if requested
            transcription_data = None
            if is_transcription:
                logging.info("Starting transcription...")
                global whisper_model
                if whisper_model is None:
                    whisper_model = whisper.load_model("tiny", device="cpu")
                try:
                    audio = whisper.load_audio(tmp_file.name)
                    transcription_data = whisper.transcribe(
                        whisper_model, audio, language="en"
                    )
                    logging.info("Transcription completed.")
                except Exception as e:
                    logging.error(f"Error during transcription: {e}")
                    raise InternalServerError("An error occurred during transcription.")

            # Encode audio file to base64
            with open(tmp_file.name, "rb") as file:
                encoded_string = base64.b64encode(file.read()).decode("utf-8")

        # Clean up temporary file
        os.remove(tmp_file.name)

    # Handle unexpected errors
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        raise InternalServerError("An internal server error occurred.")

    # Log response and return JSON
    logging.info("Sending response.")
    return jsonify({"audio": encoded_string, "transcription": transcription_data})


if __name__ == "__main__":
    # Use a production WSGI server
    serve(app, host="0.0.0.0", port=5300)

