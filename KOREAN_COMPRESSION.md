# Korean Pocket TTS compression

This branch separates Korean compression research from the unrelated acoustic
model project. The upstream `pocket_tts/` and `training/` algorithms are retained.

## Layout

- `pocket_tts/`, `training/`: official upstream implementation.
- `research/korean_compression/`: Korean backbone/head/decoder orchestration and CV3 scorer.
- `configs/korean_compression/`: portable training templates and a settings example.
- `.local/`: **ignored** machine settings, assets, dataset paths, transfer manifests.
- `runs/`: **ignored** checkpoints, optimizer state, generated configurations and audio.

No dataset, weights, scheduler jobs, GPU UUIDs, machine names or absolute paths
belong in a Git commit. Use `.local/settings.json` or `POCKET_LOCAL_CONFIG` for
environment values. The checked-in example is a low-memory starting point, not
a measured compatibility or speed guarantee. Global batch remains64; splitting
microbatches can change numerical results and does not guarantee identical training.

## Setup on another machine

Use Linux, or a Linux environment on Windows. Install a CUDA-compatible torch
build first, then the repository's training dependencies (see training/README.md).
Avoid silently replacing it with a CPU-only torch build.

1. Copy `configs/korean_compression/settings.example.json` to `.local/settings.json`.
2. Transfer the separate ignored artifacts and `runs/` pack. Git clone does not include weights.
3. Obtain the Korean filtered dataset separately and configure aligned train/valid manifests.
   Rebase each manifest's `path` field to the local audio root; do not edit transcript/word timing.
4. Put Whisper-large-v3, UTMOSv1 and WavLM scorer assets under the configured metrics directory.
5. Check paths and start with a fresh three-update smoke, not the full pipeline.

```bash
python research/korean_compression/pocket_small_b6k.py smoke width768
python research/korean_compression/pocket_small_b6k.py pipeline
# Only after successful backbone completion:
python research/korean_compression/pocket_small_followup.py pipeline
```

The copied backbone resume state is step18131 with its optimizer. Native EMA,
optimizer and LR schedule are restored; the native data loader does not promise
bit-exact uninterrupted replay. These commands do not submit scheduler jobs.
The server's private profile requires a real scheduler allocation for CUDA.
On another machine configure the scheduler requirement according to its policy.

## Experiment scope and caveats

Teacher: Korean B6000 EMA, already guidance-baked; use CFG1, not a second CFG2 bake.
Backbone: official PCA/attention/FFN shrinking, width768 then512. Followup:
native small drifting4x192 head distillation, width224/GQA/text-rank reduction,
then another head and same-latent full/small-decoder comparisons.

The adaptation retains six layers and z1024. It is **not an exact reproduction**
of the released eight-layer/z64 small English models. The small decoder is
reused from the official release, not retrained by an unpublished recipe.
The original prompt encoder is retained exactly; shape compatibility alone
does not establish audio compatibility. Decoder acceptance requires identical
latent hashes plus CER/UTMOS/speaker/silence/EOS checks. Failed candidates are
retained for analysis, never automatically deployed.

CV3 Korean500 is reused validation, not an untouched final test. Metrics are
UTMOSv1, Whisper-large-v3 CER and WavLM speaker similarity. No inference RTF
measurement has been established by these quality evaluations.

## Attribution and artifact licenses

Official source: kyutai-labs/pocket-tts, pinned source commit
77c3d76b63d6cfe08c674b4f728db6774cf01df6; code under the repository MIT license.
Korean tokenizer, dataset preparation and frozen CV3 scorer derive from
seastar105/pocket-tts-korean-training; the scorer's MIT notice is retained at
`research/korean_compression/LICENSE.seastar105`. Model weights and datasets
have their own licenses; code licensing does not grant permission to redistribute
training audio or voice prompts. Do not publish local artifacts without checking
their provenance, attribution and voice/data permissions.
