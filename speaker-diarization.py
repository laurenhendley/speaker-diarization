## Speaker diarization implementation using WhisperX and Pyannote
## Author: Lauren Hendley

## Install dependencies:
##   pip install whisperx datasets pyannote.metrics ffmpeg-python soundfile
##   !apt install -y ffmpeg  (Linux/Colab)

import whisperx
import torch
import subprocess
import numpy as np
import getpass
import os
from datasets import load_dataset, Audio

from pyannote.metrics.diarization import DiarizationErrorRate
from pyannote.database.util import load_rttm

from huggingface_hub import login


# -----------------------------
# INPUT FUNCTIONS
# -----------------------------


def get_file():
    """ 
    Getting the file 

    Returns: 
        dict: sample of VoxConverse test
    """
    print("Getting the file...")

    ds = load_dataset("diarizers-community/voxconverse", # VoxConverse dataset
                      split = "test", # 16-sample test split
                      streaming = True) # Avoids downloading full dataset
    
    ds = ds.cast_column("audio", Audio(sampling_rate = 16000)) # Resample to 16kHz (required for WhisperX)

    return next(iter(ds))


# Tries locally in colab first, then requests if fails (either not in colab or token isn't stored in colab)
try:
    from google.colab import userdata
    def get_token():
        """ 
        Getting the user's token

        Returns:
            String: user's token
        """
        try:
            return userdata.get('HF_TOKEN')
        except:
            return getpass.getpass("Enter HuggingFace token: ")
except Exception:
    def get_token():
        """ 
        Getting the user's token

        Returns:
            String: user's token
        """
        return getpass.getpass("Enter HuggingFace token: ")


def clone_voxconverse_rttms(dest="voxconverse"):
    """
    Clone the VoxConverse GitHub repo to access its RTTM annotation files.
    Skips cloning if the repo already exists locally.
 
    Args:
        dest (str): local directory name to clone into
 
    Returns:
        str: path to the test/ RTTM directory inside the cloned repo
    """
    if not os.path.exists(dest):
        print("Cloning VoxConverse RTTM annotations...")
        subprocess.run(
            ["git", "clone", "--depth", "1",
             "https://github.com/joonson/voxconverse.git", dest],
            check = True
        )
    else:
        print(f"VoxConverse repo already exists at '{dest}', skipping clone.")
 
    return os.path.join(dest, "test")


def get_rttm_annotations(rttm_dir, file_id):
    """
    Load the reference RTTMs.
 
    Args:
        rttm_dir (str): path to the cloned repo's test/ directory
        file_id (str): the sample identifier (e.g. 'abjxc')
 
    Returns:
        Annotation: pyannote reference annotation for DER evaluation
    """
    rttm_path = os.path.join(rttm_dir, f"{file_id}.rttm")
 
    if not os.path.exists(rttm_path):
        raise FileNotFoundError(f"RTTM file not found: {rttm_path}")
 
    # load_rttm returns a dict of {uri: Annotation}
    ref_annotations = load_rttm(rttm_path)
 
    # There is one entry per file; return the single annotation
    return list(ref_annotations.values())[0]


# -----------------------------
# TRANSCRIBE AND ALIGN
# -----------------------------


def load_transcribe_align(audio, device):
    """ 
    Load whisper model, transcribe and align timestamps

    Args:
        audio (array): audio dervied from file
        device (String): 'cuda' or 'cpu

    Returns:
        dict: segments with timestamps
    """
    print("Starting transcribe...")

    compute_type = "float16" if device == "cuda" else "int8" # Changing the compute type if using GPU
    # float16 used on GPU (faster inference)
    # int8 used on CPU (reduce memory usage, improve speed)


    ## Load model (WhisperX)
    model = whisperx.load_model("small", # Can be upgraded to medium or large to improve accuracy
                                device,
                                compute_type = compute_type)

    ## Transcribe (into segments with timestamps)
    res = model.transcribe(audio, batch_size = 16) 


    ## Align words to audio (using alignment model)
    model_a, metadata = whisperx.load_align_model( 
        language_code = "en", # Strictly English
        device = device
    )

    res = whisperx.align(res["segments"], ## Transcribed audio
                         model_a, 
                         metadata, 
                         audio, 
                         device)

    return res


# -----------------------------
# DIARIZATION
# -----------------------------


def diarization(token, res, audio, device):
    """ Perform diarization on the audio
    Args:
        token (string): user's token
        res (dict): aligned transcription segments
        audio (np.ndarray): audio array 
        device (string): device
    
    Returns:
        dict: segments with speaker labels
        Annotation: Pyannote annotation 
    """
    print("Starting diarization...")

    login(token = token)

    diarize_model = whisperx.diarize.DiarizationPipeline(
        device = device
    )

    ## Diarization pipeline (limits to 2-6 speakers)
    diarization_res = diarize_model(audio, min_speakers = 2, max_speakers = 6)

    ## Mapping speakers onto word segments
    final = whisperx.assign_word_speakers(diarization_res, res)

    return final, diarization_res


# -----------------------------
# MAIN
# -----------------------------


if __name__ == "__main__":
    torch.manual_seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu" # Choosing device based on GPU use or not

    rttm_dir = clone_voxconverse_rttms()

    ## Get the file, token, and audio
    fn = get_file()
    audio = fn["audio"]["array"].astype(np.float32) # Extracts the audio as an NP array (float32) - required by WhisperX
    file_id = fn["audio"]["path"].split("/")[-1].replace(".wav", "")
    token = get_token()

    ## Transcribing and aligning
    res = load_transcribe_align(audio, device)

    ## Diarization process + assigning speakers
    final, diarization_res = diarization(token, res, audio, device)

    ## Outputs the results (speaker-labelled)
    print("Printing segments...")
    for seg in final["segments"]:
        speaker = seg.get("speaker", "UNKNOWN")

        print(f"[{speaker}] : {seg['text']}")

    ref_ann = get_rttm_annotations(rttm_dir, file_id)

    ## Calculating the error (against annotation)
    metric = DiarizationErrorRate()
    metric(ref_ann, diarization_res)
    der = abs(metric)
    print(f"\nDiarization Error Rate: {der:.3f}")