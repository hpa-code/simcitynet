# Command line

Installing the package adds a `simcitynet` command.

```text
simcitynet [--version] COMMAND [OPTIONS]
```

| Command | What it does |
| --- | --- |
| `embed` | Compute embeddings for images, folders and videos; write a CSV and a JSON provenance file |
| `similarity` | Compute a cosine similarity, Euclidean distance or colour EMD matrix from that CSV |
| `card` | Save a scene card PNG for one image |
| `features` | List the 33 dimensions |
| `doctor` | Check Python, optional extras and GPU |

## `simcitynet embed`

```bash
simcitynet embed INPUTS... [-o embeddings.csv] [--device auto|cpu|cuda|mps]
                 [--blocks color,surfaces,...] [--panorama] [--every 1.0]
```

- `INPUTS` — image files, folders (non-recursive) and videos (`.mp4`, `.mov`, `.avi`, `.mkv`,
  `.webm`). Video frames are sampled every `--every` seconds and named `<clip>@<t>s`.
- `--blocks` — compute only some blocks (others are left empty), e.g. `--blocks color` runs
  without any deep-learning model.
- `--panorama` — resize inputs to 4000 × 2000 as in the paper.

## `simcitynet similarity`

```bash
simcitynet similarity embeddings.csv [-o similarity.csv]
                      [--metric cosine|euclidean|color_emd] [--normalize batch|fixed|none]
```

## `simcitynet card`

```bash
simcitynet card street.jpg -o street_card.png [--device cuda]
```
