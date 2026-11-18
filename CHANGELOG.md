# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.0] - 2026-11-19

### Added

- First public release, implementing the 33-dimensional SimCityNet embedding of
  Caliskan et al. (2025): vegetation (depth-weighted Green View Index), colour palette
  (5 K-means colours and shares), pedestrian count, 8 vehicle counts and road / sidewalk /
  building shares.
- `SimCityNet.embed()` for images, folders and globs; `embed_video()` for frame sampling.
- Batch and fixed normalisation; cosine similarity, Euclidean distance and colour
  Earth Mover's distance.
- Plots: scene card, colour palette, radar, similarity heatmap with thumbnails,
  similarity map, nearest neighbours.
- Command-line tool: `simcitynet embed | similarity | card | features | doctor`.
- Provenance record (versions, settings, device) saved with every embedding file.
