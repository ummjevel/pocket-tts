"""CV3-Eval Korean zero-shot generation and quality scoring.

The official CV3-Eval scorer routes Korean through Whisper large-v3 with the
language fixed to Korean, removes punctuation, and scores character edits.  We
keep that protocol and add prompt-speaker similarity plus UTMOS. Generation,
ASR, and quality JSONL files make every stage of a long 500-item evaluation
resumable after interruption.

Usage:
    python -m training.eval.cv3_ko runs/korean_lsd_scratch_4k \
        --checkpoint runs/korean_lsd_scratch_4k/checkpoint_00001000.pt
"""

import argparse
import gc
import io
import json
import logging
import re
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import huggingface_hub
import jiwer
import pyarrow.parquet as pq
import soundfile
import sphn
import torch

from pocket_tts.modules.stateful_module import init_states
from training.eval.librispeech import load_run

logger = logging.getLogger("eval_cv3_ko")

DATASET_REPO = "yuekai/CV3-Eval"
DATASET_REVISION = "6ea9d3650fffcbed7c6279e6d1546d01ef1d2796"
DATASET_FILE = "data/zero_shot_ko-00000-of-00001.parquet"
DATASET_SPLIT = "zero_shot_ko"
DEFAULT_LOCAL_PARQUET = Path("data/cv3_eval") / DATASET_FILE

ASR_REPO = "openai/whisper-large-v3"
ASR_REVISION = "06f233fe06e710322aca913c1bc4249a0d71fce1"
ASR_LANGUAGE = "ko"
SPEAKER_REPO = "microsoft/wavlm-base-plus-sv"
SPEAKER_REVISION = "feb593a6c23c1cc3d9510425c29b0a14d2b07b1e"
MIN_FRAMES = 8


@dataclass(frozen=True)
class EvalItem:
    index: int
    id: str
    prompt_text: str
    prompt_audio: bytes
    prompt_audio_name: str
    target_text: str


@dataclass(frozen=True)
class EvalConfig:
    dataset_repo: str
    dataset_revision: str
    dataset_file: str
    dataset_split: str
    asr_repo: str
    asr_revision: str
    asr_language: str
    checkpoint: str
    step: int
    use_ema: bool
    num_items: int
    seed: int
    temp: float
    cfg: float
    n_steps: int
    eos_threshold: float
    max_sec: float
    voice_sec: float


def resolve_parquet(path: str | None = None) -> Path:
    """Use an explicit/local copy, otherwise fetch only the pinned Korean shard."""
    if path:
        result = Path(path)
        if not result.is_file():
            raise FileNotFoundError(result)
        return result
    if DEFAULT_LOCAL_PARQUET.is_file():
        return DEFAULT_LOCAL_PARQUET
    return Path(
        huggingface_hub.hf_hub_download(
            repo_id=DATASET_REPO,
            filename=DATASET_FILE,
            repo_type="dataset",
            revision=DATASET_REVISION,
        )
    )


def load_items(path: str | Path, limit: int | None = None) -> list[EvalItem]:
    table = pq.read_table(
        path, columns=["id", "prompt_text", "prompt_audio", "target_text"]
    )
    rows = table.to_pylist()
    if limit is not None:
        if limit < 1:
            raise ValueError("num_items must be positive")
        rows = rows[:limit]
    items = []
    seen = set()
    for index, row in enumerate(rows):
        item_id = str(row["id"])
        audio = row["prompt_audio"] or {}
        audio_bytes = audio.get("bytes")
        target = str(row["target_text"] or "").strip()
        if not item_id or item_id in seen:
            raise ValueError(f"missing or duplicate CV3 id at row {index}: {item_id!r}")
        if not audio_bytes or not target:
            raise ValueError(f"missing prompt audio or target text for {item_id}")
        seen.add(item_id)
        items.append(
            EvalItem(
                index=index,
                id=item_id,
                prompt_text=str(row["prompt_text"] or "").strip(),
                prompt_audio=audio_bytes,
                prompt_audio_name=str(audio.get("path") or f"{item_id}.wav"),
                target_text=target,
            )
        )
    if not items:
        raise ValueError(f"no CV3 items in {path}")
    return items


def normalize_korean_for_cer(text: str) -> str:
    """CV3-compatible Korean normalization: NFC, no punctuation, stable spaces.

    Spaces remain characters, matching the official CV3 implementation.  A
    second no-space CER is reported as a useful Korean-specific diagnostic.
    """
    text = unicodedata.normalize("NFC", text)
    text = "".join(ch for ch in text if not unicodedata.category(ch).startswith("P"))
    return re.sub(r"\s+", " ", text).strip()


def _read_prompt(item: EvalItem, sample_rate: int, voice_sec: float) -> torch.Tensor:
    wav, source_rate = soundfile.read(
        io.BytesIO(item.prompt_audio), dtype="float32", always_2d=True
    )
    wav = wav.mean(axis=1)
    if source_rate != sample_rate:
        wav = sphn.resample(wav, src_sample_rate=source_rate, dst_sample_rate=sample_rate)
    if voice_sec > 0:
        wav = wav[: int(voice_sec * sample_rate)]
    if len(wav) == 0:
        raise ValueError(f"empty prompt audio for {item.id}")
    return torch.from_numpy(wav.copy()).float()


def _jsonl_map(path: Path) -> dict[str, dict[str, Any]]:
    records = {}
    if not path.exists():
        return records
    with path.open() as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            item_id = record["id"]
            if item_id in records:
                raise ValueError(f"duplicate id {item_id!r} in {path}:{line_number}")
            records[item_id] = record
    return records


def _append_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        return
    with path.open("a") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
        output.flush()


def _safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)[:80]


@torch.no_grad()
def generate_items(
    model,
    mimi,
    items: list[EvalItem],
    output_dir: Path,
    device: torch.device,
    *,
    batch_size: int,
    seed: int,
    temp: float,
    cfg: float,
    n_steps: int,
    eos_threshold: float,
    max_sec: float,
    voice_sec: float,
) -> dict[str, dict[str, Any]]:
    """Generate missing CV3 items and return all resumable generation records."""
    record_path = output_dir / "generation.jsonl"
    records = _jsonl_map(record_path)
    missing = [item for item in items if item.id not in records]
    if not missing:
        logger.info("generation already complete (%d items)", len(records))
        return records

    wav_dir = output_dir / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = model.flow_lm.conditioner.tokenizer.sp.encode
    max_frames = int(max_sec * mimi.frame_rate)
    ratio = round(mimi.encoder_frame_rate / mimi.frame_rate)
    model.eval()

    for start in range(0, len(missing), batch_size):
        chunk = missing[start : start + batch_size]
        # Batch-local seed makes an interrupted/resumed run bitwise-equivalent:
        # completed batches are skipped without shifting the RNG stream.
        torch.manual_seed(seed + chunk[0].index)
        tokens = [torch.tensor(tokenizer(item.target_text), dtype=torch.long) for item in chunk]
        voices = [
            mimi.encode_to_latent(
                _read_prompt(item, mimi.sample_rate, voice_sec)[None, None].to(device)
            )[0]
            for item in chunk
        ]
        latents = model.generate(
            tokens,
            voices,
            max_frames=max_frames,
            temp=temp,
            n_steps=n_steps,
            cfg_coef=cfg,
            eos_threshold=eos_threshold,
        )
        new_records = []
        for item, latent in zip(chunk, latents):
            no_eos = int(latent.shape[0] >= max_frames)
            record = {
                "index": item.index,
                "id": item.id,
                "target_text": item.target_text,
                "prompt_text": item.prompt_text,
                "prompt_audio_name": item.prompt_audio_name,
                "num_frames": int(latent.shape[0]),
                "silent": int(latent.shape[0] < MIN_FRAMES),
                "no_eos": no_eos,
                "audio": None,
            }
            if not record["silent"]:
                state = init_states(mimi, 1, (latent.shape[0] + MIN_FRAMES) * ratio)
                audio = mimi.decode_from_latent(latent[None].to(device), state)[0, 0]
                name = f"{item.index:04d}_{_safe_id(item.id)}.wav"
                audio_path = wav_dir / name
                soundfile.write(audio_path, audio.float().cpu().numpy(), mimi.sample_rate)
                record["audio"] = str(audio_path.relative_to(output_dir))
            new_records.append(record)
            records[item.id] = record
        _append_jsonl(record_path, new_records)
        logger.info("generated %d/%d", min(start + len(chunk), len(missing)), len(missing))
    return records


def _load_asr(device: torch.device):
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    dtype = torch.float16 if device.type == "cuda" else torch.float32
    processor = AutoProcessor.from_pretrained(ASR_REPO, revision=ASR_REVISION)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        ASR_REPO, revision=ASR_REVISION, dtype=dtype
    ).to(device)
    return processor, model.eval()


def _load_16k(path: Path) -> Any:
    wav, sample_rate = soundfile.read(path, dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)
    if sample_rate != 16000:
        wav = sphn.resample(wav, src_sample_rate=sample_rate, dst_sample_rate=16000)
    return wav


def _cast_floating_inputs(inputs: dict[str, torch.Tensor], dtype: torch.dtype) -> None:
    """Match floating processor outputs to the ASR model without casting masks."""
    for name, value in inputs.items():
        if torch.is_floating_point(value):
            inputs[name] = value.to(dtype=dtype)


@torch.no_grad()
def transcribe_items(
    items: list[EvalItem],
    generation: dict[str, dict[str, Any]],
    output_dir: Path,
    device: torch.device,
    batch_size: int,
) -> dict[str, dict[str, Any]]:
    record_path = output_dir / "asr.jsonl"
    records = _jsonl_map(record_path)
    missing = [item for item in items if item.id not in records]
    if not missing:
        logger.info("ASR already complete (%d items)", len(records))
        return records

    audible = [item for item in missing if not generation[item.id]["silent"]]
    processor = model = None
    if audible:
        processor, model = _load_asr(device)

    for start in range(0, len(missing), batch_size):
        chunk = missing[start : start + batch_size]
        speaking = [item for item in chunk if not generation[item.id]["silent"]]
        hypotheses: dict[str, str] = {}
        if speaking:
            wavs = [
                _load_16k(output_dir / generation[item.id]["audio"]) for item in speaking
            ]
            inputs = processor(
                wavs,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True,
                return_attention_mask=True,
            ).to(device)
            _cast_floating_inputs(inputs, next(model.parameters()).dtype)
            output_ids = model.generate(
                **inputs,
                language=ASR_LANGUAGE,
                task="transcribe",
                do_sample=False,
                num_beams=1,
                max_new_tokens=256,
            )
            decoded = processor.batch_decode(output_ids, skip_special_tokens=True)
            hypotheses = dict(zip((item.id for item in speaking), decoded))

        new_records = []
        for item in chunk:
            hypothesis = hypotheses.get(item.id, "")
            reference_normalized = normalize_korean_for_cer(item.target_text)
            hypothesis_normalized = normalize_korean_for_cer(hypothesis)
            detail = jiwer.process_characters(reference_normalized, hypothesis_normalized)
            record = {
                "index": item.index,
                "id": item.id,
                "reference": item.target_text,
                "hypothesis": hypothesis.strip(),
                "reference_normalized": reference_normalized,
                "hypothesis_normalized": hypothesis_normalized,
                "cer": detail.cer,
                "hits": detail.hits,
                "substitutions": detail.substitutions,
                "deletions": detail.deletions,
                "insertions": detail.insertions,
                "silent": generation[item.id]["silent"],
                "no_eos": generation[item.id]["no_eos"],
            }
            new_records.append(record)
            records[item.id] = record
        _append_jsonl(record_path, new_records)
        logger.info("transcribed %d/%d", min(start + len(chunk), len(missing)), len(missing))
    return records


@torch.no_grad()
def score_quality_items(
    items: list[EvalItem],
    generation: dict[str, dict[str, Any]],
    output_dir: Path,
    device: torch.device,
    batch_size: int,
    voice_sec: float,
) -> dict[str, dict[str, Any]]:
    """Score prompt-speaker similarity and UTMOS, resuming via quality.jsonl."""
    from transformers import AutoFeatureExtractor, WavLMForXVector
    from utmos_pytorch import UTMOSScoreTorch

    record_path = output_dir / "quality.jsonl"
    records = _jsonl_map(record_path)
    missing = [item for item in items if item.id not in records]
    if not missing:
        logger.info("quality scoring already complete (%d items)", len(records))
        return records

    audible = [item for item in missing if not generation[item.id]["silent"]]
    feature_extractor = speaker_model = utmos = None
    if audible:
        feature_extractor = AutoFeatureExtractor.from_pretrained(
            SPEAKER_REPO, revision=SPEAKER_REVISION
        )
        speaker_model = (
            WavLMForXVector.from_pretrained(SPEAKER_REPO, revision=SPEAKER_REVISION)
            .to(device)
            .eval()
        )
        utmos = UTMOSScoreTorch(device=str(device))

    def speaker_embeddings(wavs: list[Any]) -> torch.Tensor:
        inputs = feature_extractor(
            [wav.cpu().numpy() if torch.is_tensor(wav) else wav for wav in wavs],
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
        ).to(device)
        return speaker_model(**inputs).embeddings

    for start in range(0, len(missing), batch_size):
        chunk = missing[start : start + batch_size]
        speaking = [item for item in chunk if not generation[item.id]["silent"]]
        scores: dict[str, tuple[float, float]] = {}
        if speaking:
            generated = [
                _load_16k(output_dir / generation[item.id]["audio"]) for item in speaking
            ]
            prompts = [_read_prompt(item, 16000, voice_sec) for item in speaking]
            generated_embeddings = speaker_embeddings(generated)
            prompt_embeddings = speaker_embeddings(prompts)
            similarities = torch.nn.functional.cosine_similarity(
                generated_embeddings, prompt_embeddings, dim=-1
            ).tolist()
            # UTMOS is intentionally unbatched: padding changes its score, so
            # every generated clip is evaluated at its true duration.
            utmos_scores = [
                float(utmos.score(torch.as_tensor(wav)[None, None].to(device)))
                for wav in generated
            ]
            scores = dict(
                zip((item.id for item in speaking), zip(similarities, utmos_scores))
            )

        new_records = []
        for item in chunk:
            score = scores.get(item.id)
            record = {
                "index": item.index,
                "id": item.id,
                "speaker_similarity": score[0] if score else None,
                "utmos": score[1] if score else None,
                "silent": generation[item.id]["silent"],
            }
            new_records.append(record)
            records[item.id] = record
        _append_jsonl(record_path, new_records)
        logger.info("quality-scored %d/%d", min(start + len(chunk), len(missing)), len(missing))
    return records


def summarize(
    items: list[EvalItem],
    records: dict[str, dict[str, Any]],
    config: EvalConfig,
    quality_records: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    ordered = [records[item.id] for item in items]
    references = [record["reference_normalized"] for record in ordered]
    hypotheses = [record["hypothesis_normalized"] for record in ordered]
    detail = jiwer.process_characters(references, hypotheses)
    references_no_space = [re.sub(r"\s+", "", text) for text in references]
    hypotheses_no_space = [re.sub(r"\s+", "", text) for text in hypotheses]
    quality = quality_records or {}
    similarities = [
        quality[item.id]["speaker_similarity"]
        for item in items
        if item.id in quality and quality[item.id]["speaker_similarity"] is not None
    ]
    utmos_scores = [
        quality[item.id]["utmos"]
        for item in items
        if item.id in quality and quality[item.id]["utmos"] is not None
    ]
    return {
        **asdict(config),
        "cer": detail.cer,
        "cer_no_space": jiwer.cer(references_no_space, hypotheses_no_space),
        "hits": detail.hits,
        "substitutions": detail.substitutions,
        "deletions": detail.deletions,
        "insertions": detail.insertions,
        "silent": sum(record["silent"] for record in ordered),
        "no_eos": sum(record["no_eos"] for record in ordered),
        "speaker_similarity": (
            sum(similarities) / len(similarities) if similarities else None
        ),
        "utmos": sum(utmos_scores) / len(utmos_scores) if utmos_scores else None,
        "quality_items": len(utmos_scores),
    }


def _eval_dir_name(args, step: int) -> str:
    name = f"cv3_ko_eval_step{step}_t{args.temp}_cfg{args.cfg}"
    if not args.use_ema:
        name += "_raw"
    if args.num_items:
        name += f"_n{args.num_items}"
    if args.seed:
        name += f"_seed{args.seed}"
    return name


def _write_config(path: Path, config: EvalConfig) -> None:
    serialized = json.dumps(asdict(config), ensure_ascii=False, indent=2) + "\n"
    if path.exists() and path.read_text() != serialized:
        raise ValueError(f"evaluation configuration changed for existing directory: {path.parent}")
    path.write_text(serialized)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
        datefmt="%d-%m %H:%M:%S",
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--parquet", default=None)
    parser.add_argument("--num-items", type=int, default=None, help="default: all 500 items")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=8, help="TTS generation batch size")
    parser.add_argument("--asr-batch-size", type=int, default=16)
    parser.add_argument("--quality-batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temp", type=float, default=0.3)
    parser.add_argument("--cfg", type=float, default=2.0)
    parser.add_argument("--n-steps", type=int, default=1)
    parser.add_argument("--eos-threshold", type=float, default=-1.0)
    parser.add_argument("--max-sec", type=float, default=30.0)
    parser.add_argument(
        "--voice-sec", type=float, default=0.0, help="0 uses each complete CV3 prompt"
    )
    parser.add_argument("--use-ema", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1 or args.asr_batch_size < 1 or args.quality_batch_size < 1:
        parser.error("batch sizes must be positive")

    device = torch.device(args.device)
    items = load_items(resolve_parquet(args.parquet), args.num_items)
    model, mimi, step = load_run(
        args.run_dir, device, use_ema=args.use_ema, checkpoint=args.checkpoint
    )
    checkpoint = str(Path(args.checkpoint).resolve()) if args.checkpoint else "latest"
    config = EvalConfig(
        dataset_repo=DATASET_REPO,
        dataset_revision=DATASET_REVISION,
        dataset_file=DATASET_FILE,
        dataset_split=DATASET_SPLIT,
        asr_repo=ASR_REPO,
        asr_revision=ASR_REVISION,
        asr_language=ASR_LANGUAGE,
        checkpoint=checkpoint,
        step=step,
        use_ema=args.use_ema,
        num_items=len(items),
        seed=args.seed,
        temp=args.temp,
        cfg=args.cfg,
        n_steps=args.n_steps,
        eos_threshold=args.eos_threshold,
        max_sec=args.max_sec,
        voice_sec=args.voice_sec,
    )
    output_dir = Path(args.run_dir) / _eval_dir_name(args, step)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_config(output_dir / "config.json", config)

    generation = generate_items(
        model,
        mimi,
        items,
        output_dir,
        device,
        batch_size=args.batch_size,
        seed=args.seed,
        temp=args.temp,
        cfg=args.cfg,
        n_steps=args.n_steps,
        eos_threshold=args.eos_threshold,
        max_sec=args.max_sec,
        voice_sec=args.voice_sec,
    )
    del model, mimi
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    asr_records = transcribe_items(
        items, generation, output_dir, device, args.asr_batch_size
    )
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    quality_records = score_quality_items(
        items,
        generation,
        output_dir,
        device,
        args.quality_batch_size,
        args.voice_sec,
    )
    results = summarize(items, asr_records, config, quality_records)
    temporary = output_dir / "results.json.tmp"
    temporary.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(output_dir / "results.json")
    logger.info("FINAL %s", json.dumps(results, ensure_ascii=False))


if __name__ == "__main__":
    main()
