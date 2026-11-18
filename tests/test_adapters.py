"""The real model adapters, run with tiny randomly initialised models (no downloads).

These check that our code works with the installed PyTorch / Transformers / Ultralytics
versions: loading from a model folder, preprocessing, post-processing, output shapes and the
label maps. Values are meaningless because the weights are random; the `slow` tests check
real values. Skipped unless the `[all]` extra is installed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("ultralytics")

from simcitynet import Config, SimCityNet  # noqa: E402
from simcitynet.backends import DepthEstimator, Detector, Segmenter  # noqa: E402
from simcitynet.backends.huggingface import DPTDepthEstimator, Mask2FormerSegmenter  # noqa: E402
from simcitynet.backends.yolo import YOLODetector  # noqa: E402

from .conftest import street_scene  # noqa: E402

ADE = {
    0: "wall",
    1: "building",
    2: "sky",
    4: "tree",
    6: "road",
    9: "grass",
    11: "sidewalk",
    17: "plant",
}
CITYSCAPES = {0: "road", 1: "sidewalk", 2: "building", 8: "vegetation", 10: "sky"}


def _dense_labels(names: dict[int, str]) -> dict[int, str]:
    n = max(names) + 1
    return {i: names.get(i, f"class_{i}") for i in range(n)}


def _tiny_mask2former(folder: Path, labels: dict[int, str]) -> Path:
    torch.manual_seed(0)
    id2label = _dense_labels(labels)
    backbone = transformers.SwinConfig(
        image_size=64,
        embed_dim=32,
        depths=[1, 1, 1, 1],
        num_heads=[1, 1, 2, 2],
        window_size=4,
        out_features=["stage1", "stage2", "stage3", "stage4"],
    )
    config = transformers.Mask2FormerConfig(
        backbone_config=backbone,
        feature_size=32,  # GroupNorm in the pixel decoder uses 32 groups
        mask_feature_size=32,
        hidden_dim=32,
        encoder_feedforward_dim=64,
        dim_feedforward=64,
        encoder_layers=1,
        decoder_layers=2,
        num_attention_heads=2,
        num_queries=6,
        id2label=id2label,
        label2id={v: k for k, v in id2label.items()},
    )
    transformers.Mask2FormerForUniversalSegmentation(config).save_pretrained(folder)
    transformers.Mask2FormerImageProcessor(
        size={"shortest_edge": 64, "longest_edge": 128}, size_divisor=32, ignore_index=255
    ).save_pretrained(folder)
    return folder


def _tiny_dpt(folder: Path) -> Path:
    torch.manual_seed(0)
    config = transformers.DPTConfig(
        image_size=64,
        patch_size=16,
        hidden_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=37,
        backbone_out_indices=[0, 1, 2, 3],
        neck_hidden_sizes=[16, 16, 32, 32],
        fusion_hidden_size=16,
        head_in_index=-1,
    )
    transformers.DPTForDepthEstimation(config).save_pretrained(folder)
    transformers.DPTImageProcessor(size={"height": 64, "width": 64}).save_pretrained(folder)
    return folder


@pytest.fixture(scope="module")
def models(tmp_path_factory: pytest.TempPathFactory) -> dict[str, str]:
    root = tmp_path_factory.mktemp("models")
    return {
        "ade": str(_tiny_mask2former(root / "ade", ADE)),
        "city": str(_tiny_mask2former(root / "city", CITYSCAPES)),
        "dpt": str(_tiny_dpt(root / "dpt")),
    }


@pytest.fixture(scope="module")
def image() -> np.ndarray:
    return street_scene(96, 128)


def test_mask2former_adapter(models, image):
    seg = Mask2FormerSegmenter(models["ade"], device="cpu")
    assert isinstance(seg, Segmenter)
    assert seg.labels[4] == "tree" and seg.labels[17] == "plant"
    out = seg.segment(image)
    assert out.shape == image.shape[:2]
    assert np.issubdtype(out.dtype, np.integer)
    assert set(np.unique(out)) <= set(seg.labels)


def test_dpt_adapter(models, image):
    depth = DPTDepthEstimator(models["dpt"], device="cpu")
    assert isinstance(depth, DepthEstimator)
    out = depth.estimate(image)
    assert out.shape == image.shape[:2]
    assert np.isfinite(out).all()


def test_yolo_adapter_builds_from_yaml(image):
    # A model architecture file needs no download; weights are random, so we only check types.
    det = YOLODetector("yolov8n.yaml", device="cpu")
    assert isinstance(det, Detector)
    out = det.detect(image)
    assert isinstance(out, list)
    for d in out:
        assert isinstance(d.label, str)
        assert 0 <= d.confidence <= 1
        assert len(d.box) == 4


def test_full_pipeline_with_real_adapters(models, image):
    cfg = Config(
        device="cpu",
        vegetation_model=models["ade"],
        surface_model=models["city"],
        depth_model=models["dpt"],
        detector_weights="yolov8n.yaml",
    )
    emb = SimCityNet(cfg).embed([image, image[:, ::-1].copy()], names=["a", "b"])
    assert emb.raw.shape == (2, 33)
    assert not emb.raw.isna().any().any()
    assert emb.raw["vegetation"].between(0, 1).all()
    assert (
        emb.raw[["road_share", "sidewalk_share", "building_share"]].sum(axis=1).le(1 + 1e-9).all()
    )
    scene = emb.scene("a")
    assert scene.surface_map is not None and scene.surface_map.shape == scene.preview.shape[:2]
    assert scene.depth is not None and scene.depth.shape == scene.preview.shape[:2]
    assert emb.provenance["libraries"]["torch"] is not None
