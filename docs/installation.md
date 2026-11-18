# Installation

`simcitynet` needs Python 3.10 or newer.

=== "Full pipeline"

    ```bash
    pip install "simcitynet[all]"
    ```

    Installs PyTorch, Hugging Face Transformers and Ultralytics YOLO, so all five blocks work.

=== "Core only"

    ```bash
    pip install simcitynet
    ```

    No deep-learning libraries. The colour block, similarity measures, plots and loading of
    saved embeddings work; the other blocks raise a clear error asking for the extras.

=== "Pick extras"

    ```bash
    pip install "simcitynet[models]"   # vegetation, depth and surfaces (PyTorch + Transformers)
    pip install "simcitynet[yolo]"     # pedestrians and vehicles (Ultralytics, AGPL-3.0)
    ```

!!! note "Before the first PyPI release"

    Install straight from GitHub:

    ```bash
    pip install "simcitynet[all] @ git+https://github.com/hpa-code/simcitynet.git"
    ```

## Check your setup

```bash
simcitynet doctor
```

reports the Python version, which extras are installed and whether a CUDA GPU is available.

## GPU or CPU?

All models run on CPU, which is fine for tens of images. For hundreds or thousands, use a GPU
(`device="cuda"`) or Google Colab. The device is chosen automatically (`device="auto"` picks
CUDA, then Apple MPS, then CPU).

With all five blocks, three large neural networks run per image, so a CPU is many times slower
than a GPU; time a few images before launching a large batch. The colour block alone needs no
neural network and is fast on any machine.

## Model downloads

The first time a block runs, its model is downloaded and cached (about 3 GB in total):

| Block | Model | Cache |
| --- | --- | --- |
| Vegetation | `facebook/mask2former-swin-large-ade-semantic` | Hugging Face cache (`~/.cache/huggingface`) |
| Vegetation depth | `Intel/dpt-large` | Hugging Face cache |
| Surfaces | `facebook/mask2former-swin-large-cityscapes-semantic` | Hugging Face cache |
| Pedestrians, vehicles | `yolov9e.pt` | current folder (Ultralytics default) |

To work offline later, run once with internet, then set `HF_HUB_OFFLINE=1`. Read
[Model licences](model_licenses.md) before commercial use.

## Development install

```bash
git clone https://github.com/hpa-code/simcitynet.git
cd simcitynet
pip install -e ".[dev]"
pytest
```
