# Faithfulness to the paper

`simcitynet` follows Section 4 of Caliskan et al. (2025). Where the paper is explicit, the
package does the same. Where it leaves a detail open, the package uses the default below and
exposes it in [`Config`](api.md#simcitynet.Config) so you can change it.

## Matches the paper

| Aspect | Paper | Package |
| --- | --- | --- |
| Embedding layout | 1 + 20 + 1 + 8 + 3 = 33 dimensions | Same, in the same order (`FEATURE_NAMES`) |
| Vegetation segmenter | Mask2Former, ADE20K classes 4, 9, 17 (tree, grass, plant) | `facebook/mask2former-swin-large-ade-semantic`; classes found by name, indices 4/9/17 as fallback |
| Vegetation refinement | HSV thresholds, trees hue–saturation [20, 40]–[95, 255] | Same range for trees |
| Depth weighting | Dense Prediction Transformer, min–max normalised | `Intel/dpt-large`, min–max normalised relative inverse depth |
| Vegetation index | Depth-weighted GVI over the union of masks | Same formula |
| Colour | K-means, \(k = 5\), centroids + shares | Same |
| Detector | YOLOv9e, fixed confidence threshold \(\tau\) | `yolov9e.pt` via Ultralytics |
| Counted classes | person + bicycle, car, motorcycle, airplane, bus, train, truck, boat | Same |
| Surfaces | Pixel fractions of road, sidewalk, building | Same |
| Normalisation | \(n_k = 1/\max_I \Phi_k(I)\) "or a chosen upper bound" | `normalize="batch"` (default) or `"fixed"` |
| Similarity | Cosine and equal-weight Euclidean | `emb.similarity()`, `emb.distance("euclidean")` |
| Colour distance | 1-Wasserstein / EMD over 5 centroids, RGB ground cost | `emb.distance("color_emd")` |

## Choices the paper leaves open

| Aspect | Paper says | Package default | Reason |
| --- | --- | --- | --- |
| Vegetation output size | Per-class indices "concatenated to form the single component" | One number: the combined index; per-class values stored in `emb.meta` | Only a single value makes the total 33 |
| HSV thresholds for grass and plants | Not given | Same range as trees | No better information; configurable via `Config.vegetation_hsv` |
| Mask2Former and DPT checkpoints | Model families only | Swin-Large ADE20K semantic; DPT-Large | Largest public checkpoints of those families on the Hugging Face Hub |
| Surface segmenter | "e.g. a DeepLabV3+ variant fine-tuned on Cityscapes" | Mask2Former Swin-Large trained on Cityscapes | Also a Cityscapes model; keeps one model library and installs from the Hugging Face Hub |
| Detection threshold \(\tau\) | Fixed, value not given | 0.25 | Ultralytics default; consistent with the high recall reported in the paper |
| Order of K-means colours | Not specified | Sorted by share, ties by brightness | Makes the 20 colour coordinates comparable across images |
| K-means details | Not specified | 50,000 random pixels, 4 initialisations, seed 0 | Speed and determinism |
| Normalisation constants | Corpus maxima, not published | Batch maxima (`"batch"`) or caps 50 / 30 / 10 (`"fixed"`) | Paper's own rule; caps for cross-run comparisons |
| Preprocessing | Crop black borders, resize panoramas to 4000 × 2000 | Crop always; resize to 4000 × 2000 only with `panorama=True`; otherwise cap the long side at 2048 px | Forcing 2:1 would distort ordinary photos |

## Not included

The paper's statistical analysis (per-block Mahalanobis distances against Street View,
cluster-aware Mann–Whitney tests, mixed models, variance-ratio tests) uses the embeddings but
is not part of the package. `emb.raw`, `emb.block(...)` and `emb.distance("color_emd")` provide
everything needed to reproduce it with SciPy or statsmodels.
