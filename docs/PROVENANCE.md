# Provenance and release notes

Packaged on 23 September 2026 from the local ALTeGraD Molecular Graph Captioning coursework directories. Relative source paths and SHA-256 values are recorded in [`source_manifest.json`](source_manifest.json). Absolute personal computer paths are deliberately omitted.

## Sources and scope

The local archive `molecular-graph-captioning_JY_without_checkpoint.zip` is a traceable coursework snapshot, not a verified final-submission receipt. Eleven of twelve compared text files in the extracted `molecular-graph-captioning_copy/data_baseline` directory were byte-identical to their archive versions; its GPT2 trainer differs only by an extra tokenizer import and line endings. Shared retrieval/data utilities are identical across the two source directories. The manifest identifies the exact selected source bytes.

The official ALTeGraD 2025–2026 assignment describes molecular graph captioning as a multimodal graph-to-text task. That third-party assignment PDF, course data and example submissions are not republished here. The course starter is the source of baseline structure and data utilities; the feature-aware GINE/contrastive and graph-conditioned generation work is presented in that context.

The source collection did not include a final project report, a complete team contribution table or a verified leaderboard result. The repository is maintained for Ying Jin's portfolio without claiming sole authorship of all scaffold or experiment code.

## Maintained-copy changes

Original files were left unchanged. This release:

1. Renames the GINE retrieval trainer and inference scripts so filenames describe their active role; updates imports and messages accordingly.
2. Selects the actual frozen-GNN/DistilGPT2 training script. The unrelated original `train_gpt2.py` entry point only ran a limited checkpoint evaluation and is omitted to avoid presenting it as training code.
3. Fixes `GPTCollate(tokenizer_name=...)` to use its existing `tokenizer` parameter.
4. Aligns the GNN checkpoint default with retrieval training's `model_checkpoint.pt`, and the generation default with `checkpoints_gen/best_model.pth`.
5. Requires compatible pretrained GNN weights rather than silently freezing a random encoder after a missing/failed load. Wrapped `model_state_dict` and `module.` prefixes remain supported.
6. Masks padded text positions to `-100` in GraphGPT loss labels. This changes the training objective relative to that historical source; past results cannot be attributed to this corrected copy without rerunning.
7. Replaces a missing-data message pointing to a nonexistent preparation script with the actual authorized-data requirement.
8. Removes notebook outputs, execution counts and session metadata; normalizes filenames. Notebook model/training configurations remain unchanged.

Data, weights, embedding/prediction CSVs, course handouts, redundant versions and debug scripts are excluded. No dataset or checkpoint was deserialized during packaging. Dependencies were derived from imports, not resolved into a validated lockfile.

## Verification

Six Python modules and 34 notebook code cells passed static syntax parsing. Local module references and the renamed entry points were checked. Selected files were scanned for common hardcoded credential patterns; none were found. No file exceeds 1 MB.

The packaging environment does not include PyTorch, PyTorch Geometric or Transformers. Consequently no model imports, end-to-end training, numerical benchmark or CUDA compatibility test was performed. Static checks do not establish runtime or score reproduction. Validation metrics in the retrieval code and the competition's captioning evaluation are distinct.
