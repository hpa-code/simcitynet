# Python API

The main entry points are [`SimCityNet`](#simcitynet.SimCityNet) (compute embeddings) and
[`Embeddings`](#simcitynet.Embeddings) (compare and plot them).

## Pipeline

::: simcitynet.SimCityNet

::: simcitynet.Config

## Results

::: simcitynet.Embeddings

::: simcitynet.SceneResult

## Plots

::: simcitynet.viz.Plotter

## Features

::: simcitynet.features
    options:
      show_root_heading: false
      members: [FEATURE_NAMES, BLOCKS, FEATURES, FeatureInfo]

## Measurement blocks

::: simcitynet.blocks.vegetation.compute_vegetation

::: simcitynet.blocks.color.dominant_colors

::: simcitynet.blocks.dynamics.count_objects

::: simcitynet.blocks.surfaces.surface_shares

## Normalisation and similarity

::: simcitynet.normalize.normalize

::: simcitynet.similarity.cosine_similarity_matrix

::: simcitynet.similarity.euclidean_distance_matrix

::: simcitynet.similarity.color_emd

## Model interfaces

Pass your own objects implementing these protocols to `SimCityNet(...)`.

::: simcitynet.backends.Segmenter

::: simcitynet.backends.DepthEstimator

::: simcitynet.backends.Detector

::: simcitynet.backends.Detection

## Example data

::: simcitynet.datasets.fetch_examples
