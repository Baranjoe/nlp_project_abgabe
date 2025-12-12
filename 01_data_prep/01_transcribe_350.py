import io
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI

import pandas as pd
import numpy as np
import csv

import time
from datetime import datetime

start_idx=0
end_idx=25000+1  # end_idx ist exklusiv, also wenn 10, dann gehts bis Zeile 9 Anzahl 152252 bei SDS200



BASE_PATH = Path(__file__).parent.resolve()

INPUT_FILENAME = "train_all.tsv"  ; AUDIO_FOLDERNAME = "clips__train_valid"
INPUT_FILENAME = "test.tsv" ; AUDIO_FOLDERNAME = "clips__test"
INPUT_FILENAME = "valid.tsv"  ; AUDIO_FOLDERNAME = "clips__train_valid"
#INPUT_FILENAME = "train_balanced.tsv"  ; AUDIO_FOLDERNAME = "clips__train_valid"
INPUT_PATH = os.path.join(BASE_PATH, "datasets", "STT4SG-350", "Data_300", "Data_300", INPUT_FILENAME)  

OUTPUT_PATH = os.path.join(BASE_PATH, "transcripts_tsv", "tsv_350", f"transcripted_{INPUT_FILENAME}")
AUDIO_PATH = os.path.join(BASE_PATH, "datasets", "STT4SG-350", "Data_300", "Data_300", AUDIO_FOLDERNAME)
TRANSCRIPT_PATH = os.path.join(BASE_PATH, "transcripts_json", "json_350")

load_dotenv()
API_KEY = os.getenv("ALPINEAI_API_KEY")
BASE_URL = "https://whirlpoolllm.stage.pre.alpineai.ch/sst/v1"
MODEL = "Systran/faster-whisper-large-v3"

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

##############################################################################################

def _compute_confidence_from_segments(segments: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Return overall and per-segment confidences derived from avg_logprob."""
    total_tokens = 0
    cumulative_logprob = 0.0
    enriched_segments: List[Dict[str, Any]] = []

    for segment in segments or []:
        tokens = segment.get("tokens") or []
        avg_logprob = segment.get("avg_logprob")
        confidence = None
        if avg_logprob is not None:
            confidence = math.exp(avg_logprob)
            if tokens:
                cumulative_logprob += avg_logprob * len(tokens)
                total_tokens += len(tokens)

        enriched_segment = dict(segment)
        enriched_segment["confidence"] = confidence
        enriched_segments.append(enriched_segment)

    overall_confidence = None
    if total_tokens:
        overall_confidence = math.exp(cumulative_logprob / total_tokens)

    return {
        "overall": overall_confidence,
        "segments": enriched_segments,
    }


def _extract_token_confidences(logprobs: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    tokens_with_confidence: List[Dict[str, Any]] = []
    if not logprobs:
        return tokens_with_confidence

    for entry in logprobs:
        token = entry.get("token")
        logprob = entry.get("logprob")
        if token is None or logprob is None:
            continue
        tokens_with_confidence.append(
            {
                "token": token,
                "confidence": math.exp(logprob),
            }
        )

    # surface the least confident tokens first so the caller can double-check them
    tokens_with_confidence.sort(key=lambda item: item["confidence"])
    return tokens_with_confidence[:50]


def _prepare_audio_bytes(path: Path) -> bytes:
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")
    return path.read_bytes()


def _call_transcription_api(audio_bytes: bytes, *, temperature: float = 0.0, include_logprobs: bool = True):
    request_args: Dict[str, Any] = {
        "model": MODEL,
        "response_format": "verbose_json",
        "timestamp_granularities": ["segment", "word"],
        "temperature": temperature,
        "language": "de",
    }

    if include_logprobs:
        request_args["include"] = ["logprobs"]

    # Each API call needs a fresh file-like object
    buffer = io.BytesIO(audio_bytes)
    return client.audio.transcriptions.create(file=buffer, **request_args)


def _format_transcription_response(resp: Any) -> Dict[str, Any]:
    data = resp.model_dump()
    segments_info = _compute_confidence_from_segments(data.get("segments") or [])

    detailed_result: Dict[str, Any] = {
        "text": data.get("text"),
        "language": data.get("language"),
        "duration": data.get("duration"),
        "confidence": segments_info["overall"],
        "segments": segments_info["segments"],
        "words": data.get("words"),
    }

    logprobs = data.get("logprobs")
    if logprobs:
        detailed_result["tokens"] = _extract_token_confidences(logprobs)

    usage = data.get("usage")
    if usage:
        detailed_result["usage"] = usage

    return detailed_result


def transcribe(path: str, *, num_alternatives: int = 1, temperature: float = 0.0) -> Dict[str, Any]:
    if not API_KEY:
        raise RuntimeError("Missing ALPINEAI_API_KEY environment variable")

    audio_path = Path(path)
    audio_bytes = _prepare_audio_bytes(audio_path)

    alternatives: List[Dict[str, Any]] = []

    for idx in range(max(1, num_alternatives)):
        # For alternatives we slightly increase the temperature across attempts to encourage diversity
        if num_alternatives == 1:
            temp = temperature
        else:
            base_temp = temperature or 0.4
            temp = min(1.0, base_temp + idx * 0.1)
        resp = _call_transcription_api(audio_bytes, temperature=temp, include_logprobs=True)
        formatted = _format_transcription_response(resp)
        formatted["metadata"] = {
            "temperature": temp,
            "attempt": idx + 1,
        }
        alternatives.append(formatted)

    if num_alternatives <= 1:
        return alternatives[0]

    return {
        "primary": alternatives[0],
        "alternatives": alternatives[1:],
    }

###############################################################################################

def transcribe_range(
    input_path: str,
    output_path: str,
    audio_path: str,
    transcript_path: str,
    start_idx: int = 0,
    end_idx: Optional[int] = None,
):
    # Lese die bereits verarbeiteten clip_ids aus dem Output-File
    processed_clips = set()
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH, "r", newline='', encoding="utf-8") as outfile:
            reader = csv.DictReader(outfile, delimiter="\t")
            for row in reader:
                processed_clips.add(row["clip_path"])

    # Öffne Input- und Output-File
    with open(INPUT_PATH, "r", newline='', encoding="utf-8") as infile, \
        open(OUTPUT_PATH, "a", newline='', encoding="utf-8") as outfile:
        reader = csv.DictReader(infile, delimiter="\t")
        #fieldnames = reader.fieldnames + ["transcript", "json_path", "duration", "confidence"] # wenn alle columns gewünscht sind
        fieldnames = ["clip_path", "duration", "sentence", "transcript", "json_path", "confidence"]

        # counters
        counter_json_exists = 0
        counter_already_listed = 0
        counter_transcribed = 0

        # Schreibe den Header nur, wenn die Datei neu ist
        if outfile.tell() == 0:
            writer = csv.DictWriter(outfile, fieldnames=fieldnames, delimiter="\t")
            writer.writeheader()

        for i, row in enumerate(reader):
            if i < start_idx:
                continue  # Überspringe Zeilen vor start_idx
            if end_idx is not None and i >= end_idx:
                break  # Stoppe nach end_idx

            path = row["path"]
            if path in processed_clips:
                counter_already_listed += 1
                print(f"path: {path} is already listed in output and skipped.")
                continue

            audiopath = os.path.join(AUDIO_PATH, row["path"])
            json_filename = f"{os.path.splitext(row['path'])[0].replace('/', '_')}.json" 
            jsonpath = os.path.join(TRANSCRIPT_PATH, json_filename)

            if os.path.exists(jsonpath):
                counter_json_exists += 1
                print(f"json for path: {row['path']} already exists and is not transcribed again.")
                with open(jsonpath, "r", encoding="utf-8") as jsonfile:
                    result = json.load(jsonfile)
            else:
                row_start_time = time.time()
                counter_transcribed +=1
                result = transcribe(audiopath, num_alternatives=1,)
                with open(jsonpath, "w", encoding="utf-8") as jsonfile:
                     json.dump(result, jsonfile, ensure_ascii=False, indent=2)
                
                row_end_time = time.time()
                row_duration = row_end_time - row_start_time
                print(f"clip_id: {row['path']} transcribed and stored in json. Duration: {row_duration}")


            # Schreibe die Zeile in die Output-Datei
            writer = csv.DictWriter(outfile, fieldnames=fieldnames, delimiter="\t")
            writer.writerow({
                "clip_path": row.get("path", ""),
                "duration": result.get("duration", np.nan),
                "sentence": row.get("sentence", ""),
                "transcript": result.get("text", ""),
                "json_path": json_filename,
                "confidence": result.get("confidence", np.nan),
            })
    counters = [counter_already_listed, counter_json_exists, counter_transcribed]
    return counters

#####################################################################################
print(f"Script started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")



print(f"rows {start_idx} to {end_idx} will be processed")

counters = transcribe_range(
                            input_path=INPUT_PATH,
                            output_path=OUTPUT_PATH,
                            audio_path=AUDIO_PATH,
                            transcript_path=TRANSCRIPT_PATH,
                            start_idx=start_idx,
                            end_idx=end_idx,  # end_idx ist exklusiv, also wenn 10, dann gehts bis Zeile 9 Anzahl 152252 bei SDS200
                            )

print(f"rows {start_idx} to {end_idx-1} are processed")
print(f"not processed because already listed: {counters[0]}, only updated rows {counters[1]}, transcribed: {counters[2]}")
print(f"Script finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")