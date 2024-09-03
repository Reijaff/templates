import os
import tempfile
import base64
import logging
from flask import Flask, request, jsonify

import whisper_timestamped as whisper
from melo.api import TTS

# Use a production-grade WSGI server
from waitress import serve

# Robust error handling and user input validation
from werkzeug.exceptions import BadRequest, InternalServerError

# For asynchronous tasks if needed in the future
# import asyncio

# NLTK for natural language processing (if needed)
import nltk

nltk.download("averaged_perceptron_tagger_eng")

# Configure logging for production
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    # Consider logging to a file or centralized logging system
    # filename='app.log',
    # handlers=[logging.FileHandler('app.log')]
)

app = Flask(__name__)

# Initialize TTS model once (avoid doing it for every request)
tts_model = TTS(language="EN")
speaker_ids = tts_model.hps.data.spk2id

# Load Whisper model only if transcription is enabled (lazy loading)
whisper_model = None


@app.route("/api/btts", methods=["GET", "POST"])
def btts():
    text = request.args.get("text") or request.form.get("text", "")
    is_transcription = request.args.get("transcription", "False").lower() == "true"
    try:
        speech_speed = float(request.args.get("speed", "1"))
    except ValueError:
        raise BadRequest("Invalid speech speed. Please provide a number.")

    logging.info(
        f"Received request: text='{text}', is_transcription={is_transcription}, speech_speed={speech_speed}"
    )

    if not text:
        raise BadRequest("No text provided for TTS synthesis.")

    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tts_model.tts_to_file(
                text, speaker_ids["EN-Default"], tmp_file.name, speed=speech_speed
            )
            logging.info("TTS synthesis completed.")

            transcription_data = None
            if is_transcription:
                logging.info("Starting transcription...")
                global whisper_model  # Use the global variable
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

            with open(tmp_file.name, "rb") as file:
                encoded_string = base64.b64encode(file.read()).decode("utf-8")

        os.remove(tmp_file.name)

    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        raise InternalServerError("An internal server error occurred.")

    logging.info("Sending response.")
    return jsonify({"audio": encoded_string, "transcription": transcription_data})


if __name__ == "__main__":
    # Use a production WSGI server
    serve(app, host="0.0.0.0", port=5300)