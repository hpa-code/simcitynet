# The 33 dimensions

SimCityNet maps an image \(I\) to a vector of five interpretable blocks (paper, Section 4):

\[
\Phi(I) = \big[\, f_{\text{veg}}(I),\; f_{\text{color}}(I),\; c_{\text{ped}}(I),\; d_{\neg\text{ped}}(I),\; u(I) \,\big] \in \mathbb{R}^{33}
\]

| # | Feature | Block | Unit | Meaning |
| --- | --- | --- | --- | --- |
| 1 | `vegetation` | <span class="block-chip" style="background:var(--scn-veg)"></span>Vegetation | 0–1 | Depth-weighted Green View Index (trees, grass, plants) |
| 2–16 | `color1_r` … `color5_b` | <span class="block-chip" style="background:var(--scn-color)"></span>Colour | 0–255 | RGB of the five dominant colours, largest share first |
| 17–21 | `color1_share` … `color5_share` | <span class="block-chip" style="background:var(--scn-color)"></span>Colour | 0–1 | Share of pixels in each colour (sum to 1) |
| 22 | `person` | <span class="block-chip" style="background:var(--scn-ped)"></span>Pedestrians | count | People detected |
| 23–30 | `bicycle`, `car`, `motorcycle`, `airplane`, `bus`, `train`, `truck`, `boat` | <span class="block-chip" style="background:var(--scn-veh)"></span>Vehicles | count | Other moving objects detected |
| 31–33 | `road_share`, `sidewalk_share`, `building_share` | <span class="block-chip" style="background:var(--scn-surf)"></span>Surfaces | 0–1 | Share of pixels labelled road, sidewalk, building |

The same list is available in code as `simcitynet.FEATURES` and on the command line with
`simcitynet features`.

## Vegetation: depth-weighted Green View Index

1. **Segment** the image with Mask2Former trained on ADE20K and keep the classes *tree*,
   *grass* and *plant*.
2. **Refine** each mask in HSV colour space, keeping only pixels whose hue and saturation are in
   the green range (hue 20–95, saturation 40–255 in OpenCV units). This removes false positives
   such as green-painted walls.
3. **Weight by depth.** A monocular depth model (DPT-Large) predicts relative inverse depth,
   min–max normalised to \(D(i,j) \in [0,1]\), so that nearby foliage counts more than distant
   foliage.
4. **Index** over the union \(M\) of the refined masks:

\[
f_{\text{veg}}(I) = \frac{\sum_{i,j} M(i,j)\, D(i,j)}{\sum_{i,j} D(i,j)}
\]

Per-class values (tree, grass, plant) are also computed and stored in `emb.meta`
(`vegetation_tree`, …) but are not part of the 33 dimensions.

## Colour palette

The pixels are clustered with K-means (\(k = 5\)). Each cluster gives a centroid
\(\mu_k \in [0,255]^3\) and the share of pixels \(P_k\) assigned to it:

\[
f_{\text{color}}(I) = [\mu_1, \dots, \mu_5, P_1, \dots, P_5] \in \mathbb{R}^{20}
\]

K-means returns clusters in arbitrary order, so `simcitynet` **sorts them by share**
(largest first, ties broken by brightness). Without this, the same palette could appear in
different coordinates for different images.

## Pedestrians and vehicles

A YOLOv9e detector finds objects; detections with confidence at least \(\tau\) (default 0.25)
are counted. People form the pedestrian block; bicycles, cars, motorcycles, airplanes, buses,
trains, trucks and boats form the vehicle block.

## Surfaces

A street-scene segmenter trained on Cityscapes labels every pixel; the block is the fraction of
pixels labelled road, sidewalk and building:

\[
u_c(I) = \frac{1}{|\Omega_I|} \sum_{p \in \Omega_I} \mathbb{1}[L_I(p) = c], \qquad c \in \{\text{road}, \text{sidewalk}, \text{building}\}
\]

## Normalisation

Counts and colour values have very different ranges, so coordinates are scaled before
comparison with a diagonal matrix \(N = \operatorname{diag}(n_1, \dots, n_{33})\):

| `normalize=` | Rule | Use it when |
| --- | --- | --- |
| `"batch"` (default) | \(n_k = 1 / \max_I \Phi_k(I)\) over the images being compared (the paper's rule) | You compare a fixed set of images |
| `"fixed"` | Natural bounds: colours ÷ 255, shares and vegetation as is, counts ÷ a cap (people 50, cars 30, others 10), clipped to [0, 1] | You want results comparable across separate runs |
| `"none"` | Raw values | You do your own scaling |

## Similarity and distance

On the normalised vectors \(\tilde\Phi\), with all coordinates weighted equally:

\[
\operatorname{sim}(I,J) = \frac{\tilde\Phi(I) \cdot \tilde\Phi(J)}{\lVert\tilde\Phi(I)\rVert_2\, \lVert\tilde\Phi(J)\rVert_2},
\qquad
d(I,J) = \lVert \tilde\Phi(I) - \tilde\Phi(J) \rVert_2
\]

For the colour block, the paper also uses the first-order Wasserstein (Earth Mover's)
distance between the two palettes, with ground cost \(c_{km} = \lVert \mu_k - \nu_m \rVert_2\)
in RGB space (`emb.distance("color_emd")`). Moving pixel mass between similar colours costs
less than between distant colours.

You can restrict any comparison to some blocks, e.g.
`emb.similarity(blocks=["vegetation", "surfaces"])` for static scene structure only.
