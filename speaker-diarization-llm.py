## Speaker diarization implementation using WhisperX and Pyannote
## Author: Lauren Hendley

## Install dependencies:
##   pip install whisperx datasets pyannote.metrics ffmpeg-python soundfile
##   !apt install -y ffmpeg  (Linux/Colab)
##   pip install diarizationlm

import whisperx
import torch
import numpy as np
import getpass
from datasets import load_dataset, Audio

from pyannote.metrics.diarization import DiarizationErrorRate
from pyannote.core import Annotation
from pyannote.database.util import load_rttm

## LLM for diarization
import diarizationlm


# -----------------------------
# INPUT FUNCTIONS
# -----------------------------


## Getting the file
def get_file():
    ds = load_dataset("diarizers-community/voxconverse", 
                      split="test", 
                      streaming=True)
    
    ds = ds.cast_column("audio", 
                        Audio(sampling_rate=16000))  

    return next(iter(ds))

## Getting the user's token
try:
    from google.colab import userdata
    def get_token():
        try:
            return userdata.get('HF_TOKEN')
        except:
            return getpass.getpass("Enter HuggingFace token: ")
except Exception:
    def get_token():
        return getpass.getpass("Enter HuggingFace token: ")


# -----------------------------
# TRANSCRIBE AND ALIGN
# -----------------------------


## Load whisper model, transcribe and align words
def load_transcribe_align(audio, device):
    compute_type = "float16" if device == "cuda" else "int8"

    # Load model
    model = whisperx.load_model("small", # Can be upgraded to medium or large to improve 
                                device, 
                                compute_type = compute_type)

    # Transcribe
    res = model.transcribe(audio, 
                           batch_size = 16)

    # Align model
    model_a, metadata = whisperx.load_align_model(
        language_code = "en",
        device = device
    )

    res = whisperx.align(res["segments"], 
                         model_a, 
                         metadata, 
                         audio, 
                         device)

    return res


# -----------------------------
# DIARIZATION
# -----------------------------


def diarization(token, res, audio, device):
    diarize_model = whisperx.DiarizationPipeline(
        use_auth_token = token,
        device = device
    )

    diarize = diarize_model(audio)

    final = whisperx.assign_word_speakers(diarize, res)

    return final, diarize



# -----------------------------
# LLM IMPLEMENTATION
# -----------------------------


def llm(final_segs):
    parts = []

    for seg in final_segs:
        speaker = seg.get("speaker", "UNKNOWN")
        text = seg.get("text", "").strip()
        parts.append(f"<spk:{speaker}> {text}")

    raw = " ".join(parts)

    prompt = diarizationlm.create_diarized_transcript_prompt(raw)
    completer = diarizationlm.LLMCompleter()
    response = completer.complete(prompt)

    res = diarizationlm.extract_diarized_transcript(response)

    return res


# -----------------------------
# MAIN
# -----------------------------


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    fn = get_file()
    audio = fn["audio"]["array"].astype(np.float32)
    token = get_token()

    res = load_transcribe_align(audio, device)

    final, diarize = diarization(token, res, audio, device)

    for seg in final["segments"]:
        speaker = seg.get("speaker", "UNKNOWN")

        print(f"[{speaker}] : {seg['text']}")


    ref_res = llm(final["segments"])
    print("\n--- LLM-Refined Transcript ---")
    print(ref_res)

    ref = load_rttm(fn["rttm"])
    ref_ann = list(ref.values())[0]

    metric = DiarizationErrorRate()
    der = metric(ref_ann, diarize)
    print(f"\nDiarization Error Rate: {der:.3f}")
