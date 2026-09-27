#!/usr/bin/env python
"""
Live demo: BC-ResNet-8 against zero-shot Parakeet-TDT-0.6B-v3 on the same
2 s window, with each model's size and CPU latency.

Usage:
    python app.py                                   # BC-ResNet-8 deploy model + Parakeet
    python app.py --no-asr                          # BC-ResNet only, starts in seconds
    python app.py --checkpoint runs/bcresnet-8/seed0/fold1_F01.pt   # score F01's clips held out

The Hugging Face Space runs this file from the bundle scripts/build_hf_space.py
writes, with the deploy model at weights/bcresnet-8.pt.
"""

import argparse
import logging
from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple

import gradio as gr
import numpy as np
import torch

from src.baselines.asr.transcribers import build_transcriber, resolve_device
from src.config import Config
from src.demo.audio_input import AudioInputError
from src.demo.handler import AsrAnswer, DemoResult, recognize
from src.demo.kws import KeywordSpotter
from src.demo.results_table import DemoResults, ModelCost, load_results

APP_DIR = Path(__file__).resolve().parent
SPACE_CHECKPOINT = APP_DIR / "weights" / "bcresnet-8.pt"
LOCAL_CHECKPOINT = APP_DIR / "runs" / "bcresnet-8" / "seed0" / "deploy.pt"
ASR_NAME = "parakeet-tdt-0.6b-v3"
GITHUB_URL = "https://github.com/FYC23/dysarthric-voice-command-classifier"
TOP_K = 5
# 10 s of 48 kHz stereo float32 is under 4 MB; this stops a huge upload being decoded
MAX_UPLOAD_SIZE = "20mb"

EMPTY_NOTICE = "Record or upload one word."
NO_SPEECH_NOTICE = "No speech detected: the models heard the loudest 2 seconds of the clip."
TABLE_CAPTION = (
    "\nAccuracy on 8 TORGO speakers with dysarthria, each held out of training in turn and "
    "averaged per speaker (array microphone; 95% bootstrap interval over speakers). "
    "A recording here is one example, not a benchmark."
)

logger = logging.getLogger(__name__)


def default_checkpoint() -> Path:
    """The Space bundle's copy of the deploy model if present, else the local training run."""
    for path in (SPACE_CHECKPOINT, LOCAL_CHECKPOINT):
        if path.is_file():
            return path
    raise FileNotFoundError(f"no checkpoint at {SPACE_CHECKPOINT} or {LOCAL_CHECKPOINT}; "
                            f"pass --checkpoint")


def resolve_checkpoint(arg: Optional[str]) -> Path:
    return Path(arg) if arg else default_checkpoint()


def header_markdown(results: DemoResults) -> str:
    return (
        "# Dysarthric voice commands: a tiny keyword spotter against zero-shot ASR\n\n"
        f"**BC-ResNet-8** ({results.kws_cost.params} parameters) is fine-tuned to recognise 20 "
        "commands from speakers with dysarthria. **Parakeet-TDT-0.6B-v3** "
        f"({results.asr_cost.params} parameters) is an off-the-shelf speech recogniser; its "
        "transcript is mapped to the nearest command. Both hear the same 2 s clip.\n\n"
        f"**Say one word:** {', '.join(Config.DIGITS)}, {', '.join(Config.COMMANDS)}.\n\n"
        "BC-ResNet always picks one of the 20 (it has no \"not a command\" answer), so an "
        "off-list word still comes out as a command.\n\n"
        f"[Code, method and full results]({GITHUB_URL})"
    )


def caption(cost: ModelCost, latency_ms: float, device_label: str) -> str:
    return f"{cost.params} params · {cost.macs} MACs · {latency_ms:.0f} ms ({device_label})"


def asr_markdown(answer: AsrAnswer, cost: ModelCost, device_label: str) -> str:
    if answer.error:
        return f"**Not available.** {answer.error}"
    line = caption(cost, answer.latency_ms, device_label)
    if answer.command is None:  # empty, or nothing a command can be read from (e.g. ".")
        return f"Heard: *(nothing recognised)*\n\n{line}"
    return f"Heard: “{answer.transcript}”\n\n### → {answer.command}\n\n{line}"


def render(result: DemoResult, results: DemoResults, device_label: str) -> tuple:
    """(heard audio, notice, BC-ResNet label, BC-ResNet caption, Parakeet panel)."""
    # int16, so Gradio plays the window at its real level instead of peak-normalising it
    heard = (np.clip(result.heard, -1.0, 1.0) * 32767).astype(np.int16)
    return ((Config.SAMPLE_RATE, heard),
            "" if result.speech_found else NO_SPEECH_NOTICE,
            result.probabilities,
            caption(results.kws_cost, result.kws_latency_ms, device_label),
            asr_markdown(result.asr, results.asr_cost, device_label))


def make_handler(kws, asr, asr_error: Optional[str], results: DemoResults,
                 device_label: str) -> Callable:
    def handle(audio: Optional[Tuple[int, np.ndarray]]) -> tuple:
        if audio is None:
            return None, EMPTY_NOTICE, None, "", ""
        sample_rate, samples = audio
        try:
            result = recognize(sample_rate, samples, kws, asr, asr_error)
        except AudioInputError as err:
            gr.Warning(str(err))
            return None, str(err), None, "", ""
        return render(result, results, device_label)
    return handle


def create_app(handle: Callable, results: DemoResults) -> gr.Blocks:
    with gr.Blocks(title="Dysarthric voice commands") as demo:
        gr.Markdown(header_markdown(results))
        audio = gr.Audio(sources=["microphone", "upload"], type="numpy",
                         label="Say one command, or upload a clip",
                         editable=False)  # a trim would not be re-scored
        heard = gr.Audio(label="What the models heard (the 2 s window)", interactive=False)
        notice = gr.Markdown()
        with gr.Row():
            with gr.Column():
                gr.Markdown("### BC-ResNet-8")
                label = gr.Label(num_top_classes=TOP_K, label=f"Top {TOP_K}")
                kws_md = gr.Markdown()
            with gr.Column():
                gr.Markdown("### Parakeet-TDT-0.6B-v3 (zero-shot)")
                asr_md = gr.Markdown()
        with gr.Accordion("How these models score on TORGO", open=False):
            gr.Markdown(results.table_markdown + TABLE_CAPTION)
        outputs = [heard, notice, label, kws_md, asr_md]
        audio.stop_recording(handle, audio, outputs, api_name="recognize_recording")
        audio.upload(handle, audio, outputs, api_name="recognize_upload")
        audio.clear(handle, audio, outputs, api_name=False)  # clears the panels
    return demo


def load_asr(device: torch.device):
    """(transcriber, None) or (None, the reason it could not be loaded)."""
    try:
        asr = build_transcriber(ASR_NAME, device, torch.float32)
        asr.transcribe([np.zeros(Config.MAX_AUDIO_SAMPLES, dtype=np.float32)])  # warm-up
        return asr, None
    except Exception as err:  # the BC-ResNet half of the demo still works
        logger.exception("Parakeet failed to load")
        return None, f"Parakeet could not be loaded: {err}"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BC-ResNet-8 vs zero-shot Parakeet demo")
    parser.add_argument("--checkpoint", help="BC-ResNet fine-tuning checkpoint (default: "
                        "weights/bcresnet-8.pt in the Space bundle, else "
                        "runs/bcresnet-8/seed0/deploy.pt)")
    parser.add_argument("--no-asr", action="store_true", help="skip loading Parakeet")
    parser.add_argument("--device", default="cpu", help="cpu (default), cuda or mps")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--share", action="store_true", help="create a public Gradio link")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args(argv)
    device = resolve_device(args.device)
    checkpoint = resolve_checkpoint(args.checkpoint)
    logger.info("BC-ResNet checkpoint: %s", checkpoint)
    kws = KeywordSpotter.from_checkpoint(checkpoint, device)
    kws.warm_up()
    asr, asr_error = (None, None) if args.no_asr else load_asr(device)
    results = load_results()
    handle = make_handler(kws, asr, asr_error, results, device.type.upper())
    create_app(handle, results).launch(server_port=args.port, share=args.share,
                                         max_file_size=MAX_UPLOAD_SIZE)


if __name__ == "__main__":
    main()
