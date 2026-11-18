"""End-to-end pipeline with fake models: inputs, embeddings, I/O, video, plots and CLI."""

from __future__ import annotations

import json
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure
from PIL import Image
from typer.testing import CliRunner

import simcitynet
from simcitynet import Config, Embeddings, SimCityNet
from simcitynet.cli import app
from simcitynet.io import expand_paths, iter_inputs, load_image, sample_video_frames
from simcitynet.preprocess import cap_long_side, crop_black_borders, prepare, to_rgb_array

from .conftest import (
    FakeADESegmenter,
    FakeCityscapesSegmenter,
    FakeDepth,
    FakeDetector,
    street_scene,
)

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------- embedding
def test_embed_returns_complete_33_column_table(fake_net, scenes):
    emb = fake_net.embed(scenes, names=["a", "b", "c"])
    assert emb.raw.shape == (3, 33)
    assert not emb.raw.isna().any().any()
    assert emb.names == ["a", "b", "c"]
    assert emb.raw.loc["a", "person"] == 2
    assert emb.raw.loc["a", "car"] == 1
    assert emb.raw.loc["a", "motorcycle"] == 1
    assert emb.raw.loc["b", "vegetation"] == 0.0
    assert emb.raw.loc["a", "vegetation"] > 0
    assert emb.raw.loc["c", "sidewalk_share"] == 0.0
    assert "vegetation_tree" in emb.meta.columns
    assert emb.provenance["simcitynet_version"] == simcitynet.__version__
    assert emb.provenance["device"] == "cpu"
    assert repr(emb) == "<Embeddings: 3 images × 33 features>"


def test_similarity_distance_and_nearest(fake_net, scenes):
    emb = fake_net.embed(scenes, names=["a", "b", "c"])
    sim = emb.similarity()
    assert sim.shape == (3, 3)
    np.testing.assert_allclose(np.diag(sim), 1.0)
    np.testing.assert_allclose(sim.to_numpy(), sim.to_numpy().T)
    dist = emb.distance()
    assert (dist.to_numpy() >= 0).all()
    np.testing.assert_allclose(emb.distance("cosine").to_numpy(), 1 - sim.to_numpy())
    emd = emb.distance("color_emd")
    assert emd.loc["a", "a"] == pytest.approx(0, abs=1e-9)
    assert list(emb.nearest("a", k=2).index) == list(
        sim["a"].drop("a").sort_values(ascending=False).index
    )
    sub = emb.similarity(blocks=["surfaces"])
    assert sub.loc["a", "b"] == pytest.approx(1.0)  # identical surfaces
    with pytest.raises(ValueError):
        emb.distance("manhattan")  # type: ignore[arg-type]


def test_normalisation_methods_and_blocks(fake_net, scenes):
    emb = fake_net.embed(scenes)
    assert emb.normalized("batch").max().max() <= 1.0
    assert emb.normalized("fixed").max().max() <= 1.0
    assert list(emb.block("surfaces").columns) == ["road_share", "sidewalk_share", "building_share"]
    with pytest.raises(KeyError):
        emb.block("sky")
    summary = emb.summary()
    assert list(summary.columns) == [
        "greenery",
        "pedestrians",
        "vehicles",
        "road",
        "sidewalk",
        "building",
        "brightness",
        "colourfulness",
    ]
    assert summary["vehicles"].iloc[0] == 2


def test_colour_only_mode_needs_no_models(scenes):
    emb = SimCityNet(blocks=["color"]).embed(scenes)
    assert emb.raw[list(simcitynet.BLOCKS["color"])].notna().all().all()
    assert emb.raw["vegetation"].isna().all()
    with pytest.warns(UserWarning, match="missing values"):
        sim = emb.similarity()
    assert sim.shape == (3, 3)
    assert emb.provenance["device"] == "cpu"


def test_partial_dynamic_blocks(scene):
    net = SimCityNet(blocks=["pedestrians"], detector=FakeDetector())
    emb = net.embed(scene)
    assert emb.raw.loc["image_0", "person"] == 2
    assert np.isnan(emb.raw.loc["image_0", "car"])
    assert all(d.label == "person" for d in emb.scene(0).detections)


def test_depth_weighting_can_be_disabled(scene):
    net = SimCityNet(
        Config(depth_weighting=False),
        blocks=["vegetation"],
        vegetation_segmenter=FakeADESegmenter(),
    )
    emb = net.embed(scene)
    assert emb.raw["vegetation"].iloc[0] > 0
    assert emb.scene(0).depth is not None  # uniform weights


def test_constructor_validation():
    with pytest.raises(ValueError):
        SimCityNet(blocks=["sky"])
    with pytest.raises(ValueError):
        SimCityNet(blocks=[])
    net = SimCityNet(device="cpu", detection_conf=0.5)
    assert net.config.detection_conf == 0.5
    assert net.device == "cpu"
    assert net.blocks == simcitynet.ALL_BLOCKS


def test_n_colors_must_be_five_for_embedding(scene):
    with pytest.raises(ValueError, match="n_colors=5"):
        SimCityNet(blocks=["color"], n_colors=4).embed(scene)


def test_embeddings_validation_and_indexing(fake_net, scenes):
    emb = fake_net.embed(scenes, names=["a", "b", "c"])
    assert emb.scene("b").name == "b"
    assert emb.scene(-1).name == "c"
    with pytest.raises(IndexError):
        emb.scene(5)
    with pytest.raises(KeyError):
        emb.scene("zzz")
    with pytest.raises(ValueError):
        Embeddings(emb.raw.drop(columns=["car"]))
    with pytest.raises(ValueError):
        Embeddings(pd.concat([emb.raw, emb.raw]))
    bare = Embeddings.from_frame(emb.raw)
    with pytest.raises(ValueError, match="keep_scenes"):
        bare.scene(0)
    assert fake_net.embed(scenes, keep_scenes=False).scenes is None


def test_save_and_load_roundtrip(fake_net, scenes, tmp_path):
    emb = fake_net.embed(scenes, names=["a", "b", "c"])
    csv = emb.save(tmp_path / "out" / "emb.csv")
    assert csv.exists() and csv.with_suffix(".json").exists()
    back = Embeddings.load(csv)
    pd.testing.assert_frame_equal(back.raw, emb.raw, check_names=False)
    assert back.provenance["simcitynet_version"] == simcitynet.__version__
    pd.testing.assert_frame_equal(back.similarity(), emb.similarity(), check_names=False)
    assert emb.to_frame("batch").max().max() <= 1


# ------------------------------------------------------------------------------- inputs
def test_iter_inputs_paths_folders_globs_and_names(tmp_path):
    for name in ("b", "a"):
        Image.fromarray(street_scene()).save(tmp_path / f"{name}.png")
    (tmp_path / "notes.txt").write_text("x")
    assert [p.name for p in expand_paths([tmp_path])] == ["a.png", "b.png"]
    assert [p.name for p in expand_paths([str(tmp_path / "*.png")])] == ["a.png", "b.png"]
    items = list(iter_inputs([tmp_path, tmp_path / "a.png"]))
    assert [n for n, _ in items] == ["a", "b", "a_1"]
    arrays = list(iter_inputs([street_scene(), Image.fromarray(street_scene())]))
    assert [n for n, _ in arrays] == ["image_0", "image_1"]
    assert [n for n, _ in iter_inputs(street_scene(), names=["x"])] == ["x"]
    with pytest.raises(ValueError):
        list(iter_inputs([street_scene()], names=["x", "y"]))


def test_load_image_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_image(tmp_path / "missing.jpg")
    bad = tmp_path / "bad.jpg"
    bad.write_text("not an image")
    with pytest.raises(ValueError):
        load_image(bad)


def test_embed_with_no_images_fails(fake_net, tmp_path):
    with pytest.raises(ValueError, match="No images"):
        fake_net.embed(tmp_path)


def test_preprocess_conversions_cropping_and_resizing():
    gray = np.full((10, 12), 128, dtype=np.uint8)
    assert to_rgb_array(gray).shape == (10, 12, 3)
    rgba = np.zeros((4, 4, 4), dtype=np.uint8)
    assert to_rgb_array(rgba).shape == (4, 4, 3)
    floats = np.full((2, 2, 3), 0.5)
    assert to_rgb_array(floats)[0, 0, 0] == 127
    with pytest.raises(ValueError):
        to_rgb_array(np.zeros((2, 2, 5)))
    framed = np.zeros((40, 60, 3), dtype=np.uint8)
    framed[5:35, 10:50] = 200
    assert crop_black_borders(framed).shape == (30, 40, 3)
    black = np.zeros((5, 5, 3), dtype=np.uint8)
    assert crop_black_borders(black).shape == black.shape
    big = np.zeros((100, 400, 3), dtype=np.uint8)
    assert cap_long_side(big, 200).shape == (50, 200, 3)
    assert cap_long_side(big, 1000).shape == big.shape
    pano = prepare(street_scene(), Config(panorama=True, panorama_size=(64, 32)))
    assert pano.shape == (32, 64, 3)


# -------------------------------------------------------------------------------- video
@pytest.fixture
def video(tmp_path):
    path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (160, 120))
    assert writer.isOpened()
    for _ in range(25):
        writer.write(cv2.cvtColor(street_scene(), cv2.COLOR_RGB2BGR))
    writer.release()
    return path


def test_sample_video_frames_and_embed_video(video, fake_net):
    frames = list(sample_video_frames(video, every=1.0))
    assert [round(t, 1) for t, _ in frames] == [0.0, 1.0, 2.0]
    assert len(list(sample_video_frames(video, every=0.5, max_frames=2))) == 2
    emb = fake_net.embed_video(video, every=1.0)
    assert emb.names == ["clip@0.0s", "clip@1.0s", "clip@2.0s"]
    assert emb.meta["timestamp"].tolist() == [0.0, 1.0, 2.0]
    with pytest.raises(ValueError):
        list(sample_video_frames(video, every=0))
    with pytest.raises(FileNotFoundError):
        list(sample_video_frames(video.with_name("nope.avi")))


def test_unreadable_video(tmp_path, fake_net):
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"not a video")
    with pytest.raises(ValueError):
        fake_net.embed_video(bad)


# -------------------------------------------------------------------------------- plots
def test_all_plots_render(fake_net, scenes, tmp_path):
    emb = fake_net.embed(scenes, names=["a", "b", "c"])
    figs = [
        emb.plot.scene_card("a"),
        emb.plot.palette(1),
        emb.plot.radar(),
        emb.plot.radar(["a", "b"], normalize="fixed"),
        emb.plot.heatmap(),
        emb.plot.heatmap(thumbnails=False, annotate=False),
        emb.plot.map(),
        emb.plot.map(thumbnails=False),
        emb.plot.neighbors("a", k=2),
    ]
    for i, fig in enumerate(figs):
        assert isinstance(fig, Figure)
        fig.savefig(tmp_path / f"fig{i}.png")
    many = fake_net.embed([*scenes, *scenes], keep_scenes=True)
    many.plot.radar().savefig(tmp_path / "small_multiples.png")


def test_scene_card_without_model_blocks(scene, tmp_path):
    emb = SimCityNet(blocks=["color"]).embed(scene)
    emb.plot.scene_card(0).savefig(tmp_path / "card.png")


# ---------------------------------------------------------------------------------- CLI
def _install_fakes(monkeypatch):
    original = SimCityNet.__init__

    def patched(self, config=None, **kwargs):
        kwargs.setdefault("vegetation_segmenter", FakeADESegmenter())
        kwargs.setdefault("depth_estimator", FakeDepth())
        kwargs.setdefault("surface_segmenter", FakeCityscapesSegmenter())
        kwargs.setdefault("detector", FakeDetector())
        original(self, config, **kwargs)

    monkeypatch.setattr(SimCityNet, "__init__", patched)


def test_cli_end_to_end(tmp_path, monkeypatch, video):
    _install_fakes(monkeypatch)
    for name in ("x", "y"):
        Image.fromarray(street_scene()).save(tmp_path / f"{name}.png")
    runner = CliRunner()
    out = tmp_path / "emb.csv"
    res = runner.invoke(
        app,
        [
            "embed",
            str(tmp_path / "x.png"),
            str(tmp_path / "y.png"),
            str(video),
            "-o",
            str(out),
            "--device",
            "cpu",
        ],
    )
    assert res.exit_code == 0, res.output
    assert "Wrote 5 embeddings" in res.output
    for metric in ("cosine", "euclidean", "color_emd"):
        sim = tmp_path / f"{metric}.csv"
        res = runner.invoke(app, ["similarity", str(out), "-o", str(sim), "--metric", metric])
        assert res.exit_code == 0, res.output
        assert pd.read_csv(sim, index_col=0).shape == (5, 5)
    res = runner.invoke(app, ["similarity", str(out), "--metric", "nope"])
    assert res.exit_code != 0
    res = runner.invoke(app, ["card", str(tmp_path / "x.png"), "-o", str(tmp_path / "card.png")])
    assert res.exit_code == 0 and (tmp_path / "card.png").exists()
    res = runner.invoke(
        app, ["embed", str(tmp_path / "x.png"), "-o", str(tmp_path / "c.csv"), "--blocks", "color"]
    )
    assert res.exit_code == 0, res.output


def test_cli_info_commands():
    runner = CliRunner()
    res = runner.invoke(app, ["--version"])
    assert res.exit_code == 0 and simcitynet.__version__ in res.output
    res = runner.invoke(app, ["features"])
    assert res.exit_code == 0 and len(res.output.strip().splitlines()) == 33
    res = runner.invoke(app, ["doctor"])
    assert res.exit_code == 0 and "simcitynet" in res.output


# ------------------------------------------------------------------------- consistency
def test_citation_and_changelog_match_version():
    version = simcitynet.__version__
    cff = (ROOT / "CITATION.cff").read_text()
    assert re.search(rf"^version:\s*\"?{re.escape(version)}\"?\s*$", cff, re.M), (
        "CITATION.cff version"
    )
    assert f"[{version}]" in (ROOT / "CHANGELOG.md").read_text(), "CHANGELOG entry"


def test_provenance_is_json_serialisable(fake_net, scene):
    emb = fake_net.embed(scene)
    json.dumps(emb.provenance)
