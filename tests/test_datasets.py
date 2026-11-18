"""Example downloader, with the network replaced by fakes."""

from __future__ import annotations

import pytest

from simcitynet import datasets


@pytest.fixture
def fake_network(monkeypatch):
    calls: list[str] = []

    def fake_get_json(url: str) -> dict:
        calls.append(url)
        if "Missing" in url:
            return {"query": {"pages": {"-1": {"missing": ""}}}}
        return {
            "query": {
                "pages": {
                    "1": {
                        "imageinfo": [
                            {
                                "url": "https://upload.example/full.jpg",
                                "thumburl": "https://upload.example/thumb.jpg",
                                "descriptionurl": "https://commons.wikimedia.org/wiki/File:X.jpg",
                                "extmetadata": {
                                    "Artist": {"value": '<a href="u">Jane &amp; Doe</a>'},
                                    "LicenseShortName": {"value": "CC BY-SA 4.0"},
                                    "LicenseUrl": {
                                        "value": "https://creativecommons.org/licenses/by-sa/4.0"
                                    },
                                },
                            }
                        ]
                    }
                }
            }
        }

    def fake_download(url: str, path) -> None:
        calls.append(url)
        path.write_bytes(b"jpeg")

    monkeypatch.setattr(datasets, "_get_json", fake_get_json)
    monkeypatch.setattr(datasets, "_download", fake_download)
    return calls


def test_fetch_examples_downloads_and_writes_credits(tmp_path, fake_network):
    paths = datasets.fetch_examples(tmp_path, names=["dublin", "doha"])
    assert [p.name for p in paths] == ["dublin.jpg", "doha.jpg"]
    assert all(p.read_bytes() == b"jpeg" for p in paths)
    credits = datasets.credits_table(tmp_path)
    assert credits[0].author == "Jane & Doe"
    assert credits[0].license == "CC BY-SA 4.0"
    assert credits[1].region == "Middle East"
    n_calls = len(fake_network)
    datasets.fetch_examples(tmp_path, names=["dublin"])  # cached: no second download
    assert len(fake_network) == n_calls + 1


def test_fetch_examples_rejects_unknown_names(tmp_path, fake_network):
    with pytest.raises(ValueError, match="Unknown example"):
        datasets.fetch_examples(tmp_path, names=["atlantis"])


def test_missing_commons_file(monkeypatch, fake_network):
    with pytest.raises(FileNotFoundError):
        datasets._lookup("Missing.jpg", 100)


def test_examples_cover_both_regions():
    regions = {e.region for e in datasets.EXAMPLES}
    assert regions == {"Europe", "Middle East"}
    assert len({e.name for e in datasets.EXAMPLES}) == len(datasets.EXAMPLES)
