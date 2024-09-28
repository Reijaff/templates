from flask import Flask, request, jsonify
from waitress import serve


from scipy.ndimage import binary_dilation, binary_closing
import matplotlib.pyplot as plt
import json
import numpy as np
import datetime
import sys

import tempfile
import os
import subprocess

from pathlib import Path
from typing import Optional, Union, List
import webrtcvad
import librosa
import struct

from torch import nn
from time import perf_counter as timer
import torch


app = Flask(__name__)

## Mel-filterbank
mel_window_length = 25  # In milliseconds
mel_window_step = 10  # In milliseconds
mel_n_channels = 40


## Audio
sampling_rate = 16000
# Number of spectrogram frames in a partial utterance
partials_n_frames = 160  # 1600 ms


## Voice Activation Detection
# Window size of the VAD. Must be either 10, 20 or 30 milliseconds.
# This sets the granularity of the VAD. Should not need to be changed.
vad_window_length = 30  # In milliseconds
# Number of frames to average together when performing the moving average smoothing.
# The larger this value, the larger the VAD variations must be to not get smoothed out.
vad_moving_average_width = 8
# Maximum number of consecutive silent frames a segment can have.
vad_max_silence_length = 6


## Audio volume normalization
audio_norm_target_dBFS = -30


## Model parameters
model_hidden_size = 256
model_embedding_size = 256
model_num_layers = 3


int16_max = (2**15) - 1


def preprocess_wav(
    fpath_or_wav: Union[str, Path, np.ndarray], source_sr: Optional[int] = None
):
    """
    Applies preprocessing operations to a waveform either on disk or in memory such that
    The waveform will be resampled to match the data hyperparameters.

    :param fpath_or_wav: either a filepath to an audio file (many extensions are supported, not
    just .wav), either the waveform as a numpy array of floats.
    :param source_sr: if passing an audio waveform, the sampling rate of the waveform before
    preprocessing. After preprocessing, the waveform'speaker sampling rate will match the data
    hyperparameters. If passing a filepath, the sampling rate will be automatically detected and
    this argument will be ignored.
    """
    # Load the wav from disk if needed
    if isinstance(fpath_or_wav, str) or isinstance(fpath_or_wav, Path):
        wav, source_sr = librosa.load(str(fpath_or_wav), sr=None)
    else:
        wav = fpath_or_wav

    # Resample the wav
    if source_sr is not None:
        wav = librosa.resample(wav, orig_sr=source_sr, target_sr=sampling_rate)

    # Apply the preprocessing: normalize volume and shorten long silences
    wav = normalize_volume(wav, audio_norm_target_dBFS, increase_only=True)
    # wav = trim_long_silences(wav)

    return wav


def wav_to_mel_spectrogram(wav):
    """
    Derives a mel spectrogram ready to be used by the encoder from a preprocessed audio waveform.
    Note: this not a log-mel spectrogram.
    """
    frames = librosa.feature.melspectrogram(
        y=wav,
        sr=sampling_rate,
        n_fft=int(sampling_rate * mel_window_length / 1000),
        hop_length=int(sampling_rate * mel_window_step / 1000),
        n_mels=mel_n_channels,
    )
    return frames.astype(np.float32).T


def trim_long_silences(wav):
    """
    Ensures that segments without voice in the waveform remain no longer than a
    threshold determined by the VAD parameters in params.py.

    :param wav: the raw waveform as a numpy array of floats
    :return: the same waveform with silences trimmed away (length <= original wav length)
    """
    # Compute the voice detection window size
    samples_per_window = (vad_window_length * sampling_rate) // 1000

    # Trim the end of the audio to have a multiple of the window size
    wav = wav[: len(wav) - (len(wav) % samples_per_window)]

    # Convert the float waveform to 16-bit mono PCM
    pcm_wave = struct.pack(
        "%dh" % len(wav), *(np.round(wav * int16_max)).astype(np.int16)
    )

    # Perform voice activation detection
    voice_flags = []
    vad = webrtcvad.Vad(mode=3)
    for window_start in range(0, len(wav), samples_per_window):
        window_end = window_start + samples_per_window
        voice_flags.append(
            vad.is_speech(
                pcm_wave[window_start * 2 : window_end * 2], sample_rate=sampling_rate
            )
        )
    voice_flags = np.array(voice_flags)

    # Smooth the voice detection with a moving average
    def moving_average(array, width):
        array_padded = np.concatenate(
            (np.zeros((width - 1) // 2), array, np.zeros(width // 2))
        )
        ret = np.cumsum(array_padded, dtype=float)
        ret[width:] = ret[width:] - ret[:-width]
        return ret[width - 1 :] / width

    audio_mask = moving_average(voice_flags, vad_moving_average_width)
    audio_mask = np.round(audio_mask).astype(bool)

    # Dilate the voiced regions
    audio_mask = binary_dilation(audio_mask, np.ones(vad_max_silence_length + 1))
    audio_mask = np.repeat(audio_mask, samples_per_window)

    return wav[audio_mask == True]


def normalize_volume(wav, target_dBFS, increase_only=False, decrease_only=False):
    if increase_only and decrease_only:
        raise ValueError("Both increase only and decrease only are set")
    rms = np.sqrt(np.mean((wav * int16_max) ** 2))
    wave_dBFS = 20 * np.log10(rms / int16_max)
    dBFS_change = target_dBFS - wave_dBFS
    if dBFS_change < 0 and increase_only or dBFS_change > 0 and decrease_only:
        return wav
    return wav * (10 ** (dBFS_change / 20))


class VoiceEncoder(nn.Module):
    def __init__(
        self,
        device: Union[str, torch.device] = None,
        verbose=True,
        weights_fpath: Union[Path, str] = None,
    ):
        """
        If None, defaults to cuda if it is available on your machine, otherwise the model will
        run on cpu. Outputs are always returned on the cpu, as numpy arrays.
        :param weights_fpath: path to "<CUSTOM_MODEL>.pt" file path.
        If None, defaults to built-in "pretrained.pt" model
        """
        super().__init__()

        # Define the network
        self.lstm = nn.LSTM(
            mel_n_channels, model_hidden_size, model_num_layers, batch_first=True
        )
        self.linear = nn.Linear(model_hidden_size, model_embedding_size)
        self.relu = nn.ReLU()

        # Get the target device
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            device = torch.device(device)
        self.device = device

        # Load the pretrained model'speaker weights
        if weights_fpath is None:
            weights_fpath = Path(__file__).resolve().parent.joinpath("pretrained.pt")
        else:
            weights_fpath = Path(weights_fpath)

        if not weights_fpath.exists():
            raise Exception(
                "Couldn't find the voice encoder pretrained model at %s."
                % weights_fpath
            )
        start = timer()
        checkpoint = torch.load(weights_fpath, map_location="cpu", weights_only=True)
        self.load_state_dict(checkpoint["model_state"], strict=False)
        self.to(device)

        if verbose:
            print(
                "Loaded the voice encoder model on %s in %.2f seconds."
                % (device.type, timer() - start)
            )

    def forward(self, mels: torch.FloatTensor):
        """
        Computes the embeddings of a batch of utterance spectrograms.

        :param mels: a batch of mel spectrograms of same duration as a float32 tensor of shape
        (batch_size, n_frames, n_channels)
        :return: the embeddings as a float 32 tensor of shape (batch_size, embedding_size).
        Embeddings are positive and L2-normed, thus they lay in the range [0, 1].
        """
        # Pass the input through the LSTM layers and retrieve the final hidden state of the last
        # layer. Apply a cutoff to 0 for negative values and L2 normalize the embeddings.
        _, (hidden, _) = self.lstm(mels)
        embeds_raw = self.relu(self.linear(hidden[-1]))
        return embeds_raw / torch.norm(embeds_raw, dim=1, keepdim=True)

    @staticmethod
    def compute_partial_slices(n_samples: int, rate, min_coverage):
        """
        Computes where to split an utterance waveform and its corresponding mel spectrogram to
        obtain partial utterances of <partials_n_frames> each. Both the waveform and the
        mel spectrogram slices are returned, so as to make each partial utterance waveform
        correspond to its spectrogram.

        The returned ranges may be indexing further than the length of the waveform. It is
        recommended that you pad the waveform with zeros up to wav_slices[-1].stop.

        :param n_samples: the number of samples in the waveform
        :param rate: how many partial utterances should occur per second. Partial utterances must
        cover the span of the entire utterance, thus the rate should not be lower than the inverse
        of the duration of a partial utterance. By default, partial utterances are 1.6s long and
        the minimum rate is thus 0.625.
        :param min_coverage: when reaching the last partial utterance, it may or may not have
        enough frames. If at least <min_pad_coverage> of <partials_n_frames> are present,
        then the last partial utterance will be considered by zero-padding the audio. Otherwise,
        it will be discarded. If there aren't enough frames for one partial utterance,
        this parameter is ignored so that the function always returns at least one slice.
        :return: the waveform slices and mel spectrogram slices as lists of array slices. Index
        respectively the waveform and the mel spectrogram with these slices to obtain the partial
        utterances.
        """
        assert 0 < min_coverage <= 1

        # Compute how many frames separate two partial utterances
        samples_per_frame = int((sampling_rate * mel_window_step / 1000))
        n_frames = int(np.ceil((n_samples + 1) / samples_per_frame))
        frame_step = int(np.round((sampling_rate / rate) / samples_per_frame))
        assert 0 < frame_step, "The rate is too high"
        assert (
            frame_step <= partials_n_frames
        ), "The rate is too low, it should be %f at least" % (
            sampling_rate / (samples_per_frame * partials_n_frames)
        )

        # Compute the slices
        wav_slices, mel_slices = [], []
        steps = max(1, n_frames - partials_n_frames + frame_step + 1)
        for i in range(0, steps, frame_step):
            mel_range = np.array([i, i + partials_n_frames])
            wav_range = mel_range * samples_per_frame
            mel_slices.append(slice(*mel_range))
            wav_slices.append(slice(*wav_range))

        # Evaluate whether extra padding is warranted or not
        last_wav_range = wav_slices[-1]
        coverage = (n_samples - last_wav_range.start) / (
            last_wav_range.stop - last_wav_range.start
        )
        if coverage < min_coverage and len(mel_slices) > 1:
            mel_slices = mel_slices[:-1]
            wav_slices = wav_slices[:-1]

        return wav_slices, mel_slices

    def embed_utterance(
        self, wav: np.ndarray, return_partials=False, rate=1.3, min_coverage=0.75
    ):
        """
        Computes an embedding for a single utterance. The utterance is divided in partial
        utterances and an embedding is computed for each. The complete utterance embedding is the
        L2-normed average embedding of the partial utterances.

        TODO: independent batched version of this function

        :param wav: a preprocessed utterance waveform as a numpy array of float32
        :param return_partials: if True, the partial embeddings will also be returned along with
        the wav slices corresponding to each partial utterance.
        :param rate: how many partial utterances should occur per second. Partial utterances must
        cover the span of the entire utterance, thus the rate should not be lower than the inverse
        of the duration of a partial utterance. By default, partial utterances are 1.6s long and
        the minimum rate is thus 0.625.
        :param min_coverage: when reaching the last partial utterance, it may or may not have
        enough frames. If at least <min_pad_coverage> of <partials_n_frames> are present,
        then the last partial utterance will be considered by zero-padding the audio. Otherwise,
        it will be discarded. If there aren't enough frames for one partial utterance,
        this parameter is ignored so that the function always returns at least one slice.
        :return: the embedding as a numpy array of float32 of shape (model_embedding_size,). If
        <return_partials> is True, the partial utterances as a numpy array of float32 of shape
        (n_partials, model_embedding_size) and the wav partials as a list of slices will also be
        returned.
        """
        # Compute where to split the utterance into partials and pad the waveform with zeros if
        # the partial utterances cover a larger range.
        wav_slices, mel_slices = self.compute_partial_slices(
            len(wav), rate, min_coverage
        )
        max_wave_length = wav_slices[-1].stop
        if max_wave_length >= len(wav):
            wav = np.pad(wav, (0, max_wave_length - len(wav)), "constant")

        # Split the utterance into partials and forward them through the model
        mel = wav_to_mel_spectrogram(wav)
        mels = np.array([mel[s] for s in mel_slices])
        with torch.no_grad():
            mels = torch.from_numpy(mels).to(self.device)
            partial_embeds = self(mels).cpu().numpy()

        # Compute the utterance embedding from the partial embeddings
        raw_embed = np.mean(partial_embeds, axis=0)
        embed = raw_embed / np.linalg.norm(raw_embed, 2)

        if return_partials:
            return embed, partial_embeds, wav_slices
        return embed

    def embed_speaker(self, wavs: List[np.ndarray], **kwargs):
        """
        Compute the embedding of a collection of wavs (presumably from the same speaker) by
        averaging their embedding and L2-normalizing it.

        :param wavs: list of wavs a numpy arrays of float32.
        :param kwargs: extra arguments to embed_utterance()
        :return: the embedding as a numpy array of float32 of shape (model_embedding_size,).
        """
        raw_embed = np.mean(
            [
                self.embed_utterance(wav, return_partials=False, **kwargs)
                for wav in wavs
            ],
            axis=0,
        )
        return raw_embed / np.linalg.norm(raw_embed, 2)


def seconds_to_time_str(seconds, include_hours=True):
    """Converts seconds to a formatted time string (e.g., 120 -> '2:00' or '0:02:00')"""
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if include_hours:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    else:
        return f"{minutes:02d}:{seconds:02d}"


def process_audio_chunk(
    encoder, speaker_embed, chunk_filename, chunk_start_time, temp_dir
):
    """Processes a single audio chunk, performs speaker diarization, and returns the data to be dumped into JSON."""

    full_path = os.path.join(temp_dir, chunk_filename)
    print(f"Processing: {full_path}")

    try:
        wav = preprocess_wav(full_path)
    except Exception as e:
        print(f"Error preprocessing {full_path}: {e}")
        return  # Skip this chunk if there's an error

    # Get embeddings and similarity
    _, cont_embeds, wav_splits = encoder.embed_utterance(
        wav, return_partials=True, rate=16
    )
    similarity_dict = {"Speaker 1": cont_embeds @ speaker_embed}

    data_to_dump = {k: list(map(float, v)) for k, v in similarity_dict.items()}
    data_to_dump["wav_splits"] = [
        {"start": int(s.start), "stop": int(s.stop)} for s in wav_splits
    ]  # Include wav_splits in the JSON
    data_to_dump["chunk_start_time"] = chunk_start_time

    print("Chunk processed.")
    return data_to_dump


def process_audio_and_generate_json(main_path, sample_path, chunk_duration, t_dir):
    """Processes the main audio file, generates chunks, and saves similarity data to JSON files."""

    with tempfile.TemporaryDirectory(prefix="audio_chunks_", dir=t_dir) as temp_dir:
        # Chunking using ffmpeg
        ffmpeg_cmd = f"ffmpeg -i {main_path} -f segment -segment_time {chunk_duration} -c copy {temp_dir}/chunk_%d.mp3"
        subprocess.call(ffmpeg_cmd, shell=True)

        # Load reference audio and initialize encoder
        speaker_wav = preprocess_wav(sample_path)
        encoder = VoiceEncoder("cpu")
        speaker_embed = encoder.embed_utterance(speaker_wav)

        # Get sorted chunk filenames
        chunk_filenames = sorted(
            os.listdir(temp_dir), key=lambda x: int(x.split("_")[1].split(".")[0])
        )

        segment_cache_dir = tempfile.TemporaryDirectory(
            prefix="segment_cache_", dir=t_dir, delete=False
        )

        # Process each chunk and save data to JSON
        for i, chunk_filename in enumerate(chunk_filenames):
            chunk_start_time = i * chunk_duration
            data_to_dump = process_audio_chunk(
                encoder, speaker_embed, chunk_filename, chunk_start_time, temp_dir
            )

            # Write similarity data to JSON (append mode)
            with open(
                segment_cache_dir.name + f"/segmentation_{i}.json",
                "a",
                encoding="utf-8",
            ) as f:
                json.dump(data_to_dump, f, ensure_ascii=False, indent=4)
                f.write("\n")  # Add newline for better readability

            # if i > 1:  # For demonstration, process only first few chunks
            #     break

    return segment_cache_dir.name  # Return the directory containing JSON files


def tolerate_threshold(similarities, duration, threshold=0.7, min_gap=4, max_gap=6):
    points_per_second = int(len(similarities) / duration)

    # Convert similarities to a NumPy array
    similarities_array = np.array(similarities)

    # Thresholding
    binary_similarities = similarities_array > threshold

    # Closing operation to fill gaps
    structuring_element = np.ones(max_gap * points_per_second)  # Adjust size as needed
    smoothed_similarities = binary_closing(
        binary_similarities, structure=structuring_element
    )

    # Ensure long gaps remain zero
    gap_starts = np.where(np.diff(smoothed_similarities.astype(int)) == -1)[0] + 1
    gap_ends = np.where(np.diff(smoothed_similarities.astype(int)) == 1)[0]
    for start, end in zip(gap_starts, gap_ends):
        if (end - start) > min_gap * points_per_second:
            smoothed_similarities[start : end + 1] = 0

    return {"Speaker 1": smoothed_similarities.astype(int)}


def draw_graph_from_json(json_dir, t_dir):
    """Reads JSON files from the specified directory and draws a graph."""

    segment_filenames = sorted(
        os.listdir(json_dir), key=lambda x: int(x.split("_")[1].split(".")[0])
    )

    # Plotting setup
    fig, axes = plt.subplots(
        nrows=len(segment_filenames),
        figsize=(15, len(segment_filenames) * 2),
        sharex=True,
    )
    plt.subplots_adjust(hspace=0.5)  # Adjust vertical spacing between subplots

    smoothed_data = {}

    # Process each JSON file and plot
    for i, (ax, segment_filename) in enumerate(zip(axes, segment_filenames)):

        with open(os.path.join(json_dir, segment_filename), "r", encoding="utf-8") as f:
            data = json.load(f)

        similarity_dict = {
            k: v for k, v in data.items() if k not in ["wav_splits", "chunk_start_time"]
        }

        wav_splits = data["wav_splits"]

        chunk_start_time = int(data["chunk_start_time"])

        duration = int(wav_splits[-1]["stop"] / sampling_rate)

        similarity_dict = tolerate_threshold(similarity_dict["Speaker 1"], duration)

        smoothed_data[i] = {}
        smoothed_data[i]["wav_splits"] = wav_splits
        smoothed_data[i]["start_time"] = chunk_start_time
        smoothed_data[i]["duration"] = duration
        smoothed_data[i]["segments"] = list(map(int, similarity_dict["Speaker 1"]))
        smoothed_data[i]["sampling_rate"] = sampling_rate

        print(f"Chunk {i} processed.")
        if i == len(segment_filenames) - 1:
            print("This is the last iteration, graph breaks for some reason, break")
            break        

        lines = [ax.plot([], [], label=name)[0] for name in similarity_dict.keys()]
        times = [((s["start"] + s["stop"]) / 2) / sampling_rate for s in wav_splits]

        for line, (name, similarities) in zip(lines, similarity_dict.items()):
            line.set_data(times, similarities)
            ax.fill_between(times, similarities, color=line.get_color(), alpha=0.4)

        # Plot adjustments (same as in the original `process_audio_chunk` function)
        ax.set_ylim(0.4, 1)
        ax.set_xlim(0, duration)
        # ax.set_ylabel("Similarity")

        ax.set_xticks(range(0, duration, 60))
        ax.set_xticklabels(
            [
                seconds_to_time_str(x, include_hours=False)
                for x in range(0, duration, 60)
            ]
        )

        ax.tick_params(axis="x", which="both", labelbottom=True)
        ax.set_title(seconds_to_time_str(chunk_start_time))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    json_output_path = t_dir / "segmentation.json"
    with open(
        json_output_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(smoothed_data, f, ensure_ascii=False, indent=4)
        f.write("\n")  # Add newline for better readability

    # Overall plot adjustments and save
    fig.suptitle("Diarization Results for Each Chunk", fontsize=16)
    # plt.tight_layout(rect=[0, 0, 1, 0.98])

    current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = t_dir / f"diarization_{current_datetime}.png"
    plt.savefig(filename.name)
    print(f"Plot saved as {filename.name}")

    return json_output_path


@app.route("/api/segment", methods=["GET", "POST"])
def api_segment():

    config_dir = os.getenv("XDG_CONFIG_HOME", default=Path.home() / ".config")
    ss_path = Path(config_dir) / "speech_segment"
    ss_path.mkdir(parents=True, exist_ok=True)

    chunk_duration = 10 * 60  # in seconds

    # Get text and transcription flag from request
    main_path = request.args.get("main_path") or request.form.get("main_path", "")
    sample_path = request.args.get("sample_path") or request.form.get("sample_path", "")
    # is_transcription = request.args.get("transcription", "False").lower() == "true"

    json_dir = process_audio_and_generate_json(
        main_path, sample_path, chunk_duration, ss_path
    )
    # json_dir = ss_path / "segment_cache_y3ayunl8"
    seg_data_path = draw_graph_from_json(json_dir, ss_path)

    return jsonify({"segmentation_data": str(seg_data_path)})


# Main execution
if __name__ == "__main__":
    # main_path = "../test1/2024.09.21.mp3"
    # chunk_duration = 10 * 60  # in seconds

    # json_dir = process_audio_and_generate_json(main_path, chunk_duration)
    # draw_graph_from_json(json_dir)

    serve(app, host="0.0.0.0", port=5303)
    # draw_graph_from_json("../test1/segment_cache_chzkpkjz")