# Reproducibility

## What is recorded

Every `Embeddings` object carries a provenance record, saved as JSON next to the CSV by
`emb.save()` and by `simcitynet embed`:

```python
emb.provenance
```

```json
{
  "simcitynet_version": "0.1.0",
  "created_utc": "2026-11-19T09:12:44+00:00",
  "python": "3.12.7",
  "platform": "Linux-6.8.0-x86_64-with-glibc2.39",
  "device": "cuda",
  "blocks": ["vegetation", "color", "pedestrians", "vehicles", "surfaces"],
  "config": {"vegetation_model": "facebook/mask2former-swin-large-ade-semantic", "detection_conf": 0.25, "...": "..."},
  "libraries": {"torch": "2.5.1", "transformers": "4.46.3", "ultralytics": "8.3.40", "...": "..."},
  "paper": "Caliskan et al. (2025), SimCityNet, SSRN 5369073, doi:10.2139/ssrn.5369073"
}
```

Report at least the `simcitynet` version, the model IDs and the normalisation you used.

## Determinism

- K-means uses a fixed seed (`Config.random_state = 0`) and a fixed pixel sample, so the colour
  block is identical across runs on the same machine.
- The neural models are deterministic for a given input on the same hardware and library
  versions. GPU and CPU results can differ in the last decimals, which can occasionally change a
  detection near the threshold or a pixel at a class boundary.

## Pinning versions

For a study, pin everything that affects results:

```bash
pip install "simcitynet[all]==0.1.0" "torch==2.5.*" "transformers==4.46.*" "ultralytics==8.3.*"
pip freeze > requirements-lock.txt
```

To pin model weights too, download each model once and point the configuration at the local copy, for example
`Config(vegetation_model="/models/mask2former-ade")`.

## Regression tests

The package's slow test suite (`pytest -m slow`) runs the real models on six fixed example
photos and compares them with stored reference embeddings, so an upgrade of PyTorch,
Transformers or Ultralytics that changes results is caught before release.
