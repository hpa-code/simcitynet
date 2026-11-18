"""SimCityNet: interpretable 33-dimensional embeddings of street-level images.

Implements the embedding of Caliskan, Iskakov, Li, McCollum and Ren (2025),
*SimCityNet: Quantifying Geo-Cultural Bias in AI-Generated Urban Videos through Interpretable
Scene Embeddings*, SSRN 5369073.

Quick start::

    from simcitynet import SimCityNet

    scn = SimCityNet()
    emb = scn.embed("my_street_photos/")
    emb.raw                 # (n, 33) table of interpretable features
    emb.similarity()        # pairwise cosine similarity
    emb.plot.scene_card(0)  # visual summary of the first image
"""

from ._version import __version__
from .config import Config
from .embedder import ALL_BLOCKS, SimCityNet
from .embeddings import Embeddings
from .features import BLOCKS, EMBEDDING_DIM, FEATURE_NAMES, FEATURES
from .scene import SceneResult

__all__ = [
    "ALL_BLOCKS",
    "BLOCKS",
    "EMBEDDING_DIM",
    "FEATURES",
    "FEATURE_NAMES",
    "Config",
    "Embeddings",
    "SceneResult",
    "SimCityNet",
    "__version__",
]
