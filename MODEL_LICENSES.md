# Model and dependency licences

The `simcitynet` code is MIT-licensed. It does **not** ship any model weights: they are
downloaded from their publishers the first time a block needs them, and each keeps its own
licence. Review these terms before redistributing results or using the package commercially.

| Component | Used for | Where it comes from | Licence (as published) |
| --- | --- | --- | --- |
| `facebook/mask2former-swin-large-ade-semantic` | Vegetation masks | Hugging Face Hub | "other" on the model card; Mask2Former code is MIT. Trained on ADE20K, whose images have their own terms of use. |
| `facebook/mask2former-swin-large-cityscapes-semantic` | Road, sidewalk, building | Hugging Face Hub | "other" on the model card. Trained on Cityscapes, whose dataset licence restricts use to non-commercial purposes. |
| `Intel/dpt-large` | Depth weighting of vegetation | Hugging Face Hub | Apache-2.0 |
| `yolov9e.pt` via `ultralytics` | People and vehicle counts | Ultralytics (downloaded by the library) | AGPL-3.0 (the `ultralytics` package and its weights); Ultralytics also sells an enterprise licence. |
| PyTorch, Transformers | Model runtime (`[models]` extra) | PyPI | BSD-3-Clause, Apache-2.0 |
| NumPy, pandas, SciPy, scikit-learn, Matplotlib, Pillow, OpenCV, Typer | Core | PyPI | BSD / HPND / Apache-2.0 / MIT-style permissive licences |

## Why YOLO is an optional extra

Ultralytics is licensed under AGPL-3.0. To keep the core package MIT-licensed, it is installed
only with `pip install "simcitynet[yolo]"` (or `[all]`). If AGPL obligations are a problem for
you, pass any object implementing the `simcitynet.backends.Detector` protocol as
`SimCityNet(detector=...)` — for example a detector with a permissive licence.

## Example images

`simcitynet.datasets.fetch_examples()` downloads photos from Wikimedia Commons at run time and
writes their author and licence to `CREDITS.csv`. Many are CC BY-SA: figures you make from them
must credit the author and be shared under the same licence.

*This file is a summary, not legal advice. Check the linked model cards and dataset terms for
the authoritative wording.*
