# Quick start

## 1. Get some street images

Use your own photos, panoramas or video frames, or download six openly licensed examples:

```python
from simcitynet.datasets import fetch_examples

paths = fetch_examples("examples")   # amsterdam, dublin, zurich, doha, dubai, tel_aviv
```

The authors and licences are written to `examples/CREDITS.csv`.

## 2. Embed them

```python
from simcitynet import SimCityNet

scn = SimCityNet()            # or SimCityNet(device="cuda")
emb = scn.embed(paths)        # also accepts a folder, a glob such as "photos/*.jpg", PIL images or arrays
```

`emb.raw` is a pandas DataFrame with one row per image and the
[33 named columns](concepts.md):

```python
emb.raw[["vegetation", "person", "car", "road_share", "building_share"]]
```

`emb.summary()` condenses them into eight readable numbers per image (greenery, pedestrians,
vehicles, road, sidewalk, building, brightness, colourfulness).

## 3. Compare

```python
emb.similarity()                  # cosine similarity, paper's batch normalisation
emb.distance("euclidean")         # equal-weight Euclidean distance
emb.distance("color_emd")         # Earth Mover's distance between colour palettes (RGB units)
emb.nearest("zurich", k=3)        # most similar images
emb.similarity(blocks=["vegetation", "surfaces"])   # compare on some blocks only
```

## 4. Look

```python
emb.plot.scene_card("dubai")      # photo, vegetation, surfaces, detections, depth, palette
emb.plot.heatmap()                # similarity matrix with thumbnails
emb.plot.map()                    # images placed by similarity
emb.plot.radar(["zurich", "doha", "dubai"])
emb.plot.neighbors("amsterdam")
emb.plot.palette("doha")
```

Every plot returns a Matplotlib `Figure`: save it with `fig.savefig("card.png", dpi=200)`.

## 5. Save

```python
emb.save("results/cities")        # results/cities.csv + results/cities.json (provenance)

from simcitynet import Embeddings
emb = Embeddings.load("results/cities.csv")
```

## Videos

```python
clip = scn.embed_video("sora_clip.mp4", every=1.0)   # one frame per second
clip.raw["person"].plot()                             # pedestrians over time
```

## Command line

```bash
simcitynet embed examples/ -o cities.csv
simcitynet similarity cities.csv -o cities_similarity.csv --metric cosine
simcitynet card examples/zurich.jpg -o zurich.png
```

See [Command line](cli.md) for every option.
