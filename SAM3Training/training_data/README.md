# Training-data directory

`manifests/train.jsonl` and `manifests/val.jsonl` are portable source
descriptors. They intentionally reference project-relative movies and labels;
they do not duplicate multi-gigabyte image data. Procedural training scenes are
generated from deterministic per-epoch/per-sample seeds. The fixed procedural
validation recipes live in `val.jsonl`. Procedural backgrounds are dynamic:
weak drift is always present, with independently sampled transient slow blobs
and crowded, gently wiggling microtubule-like curves. Positive scenes include
an exact 5% merge variant and shallow as well as regular branch variants.

Rebuild and re-audit after changing labels:

```bash
python scripts/build_manifest.py --config configs/campaign.yaml
```

`manifests/leakage_audit.json` must report `status: passed` before training.
