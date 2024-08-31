
import whisper_timestamped as whisper
from melo.api import TTS
import tempfile
import base64
import time
import os

from flask import Flask, render_template, render_template_string, request, send_file


app = Flask(__name__)


tts_model = TTS(language='EN_NEWEST')
speaker_ids = tts_model.hps.data.spk2id


@app.route("/api/btts", methods=["GET", "POST"])
def btts():
    global tts_model, speaker_ids

    text = request.headers.get("text") or request.values.get("text", "")
    isTranscription = (
        request.headers.get("transcription")
        or request.values.get("transcription", "")
        or False
    )
    speech_speed = (
        request.headers.get("speed")
        or request.values.get("speed", "")
        or False
    )
    isEnhance = False 

    print("SPEED:", speech_speed)

    tmp_file_name = "/tmp/tmp.wav"
    # tmp_file = tempfile.TemporaryFile(suffix="wav")


    print(f" > Model input: {text}")

    speed = 1

    tts_model.tts_to_file(text, speaker_ids['EN-Newest'], tmp_file_name , speed=float(speech_speed))
    # out = io.BytesIO()
    # samples = tts.synthesize(text, speaker)

    # with wave.open(tmp_file_name, "w") as fp:
        # fp.setparams((1, 2, tts.get_sampling_rate(), len(samples), "NONE", "NONE"))
        # fp.writeframes(samples)

    transcription_data = None
    print(isTranscription)

    if isTranscription == "True":
        audio = whisper.load_audio(tmp_file_name)
        whisper_model = whisper.load_model("tiny", device="cpu")
        transcription_data = whisper.transcribe(whisper_model, audio, language="en")

    encoded_string = None
    with open(tmp_file_name, "rb") as file:
        encoded_string = base64.b64encode(file.read()).decode("utf-8")

    print(transcription_data)
    ret_data = {
        "audio": encoded_string,
        "transcription": transcription_data,
    }

    # tmp_file.close()

    return ret_data

app.run(port=5300)






