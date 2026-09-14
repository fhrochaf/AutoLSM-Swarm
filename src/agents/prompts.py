"""Prompt template strings for skill authoring and pipeline-script (re)generation."""
from __future__ import annotations

from agents.pipeline_contract import REQUIRED_FUNCS_DOC
from config import settings

DATASET_DESCRIPTION = """\
Dataset Descriptipon:
[Each tile is 128x128 pixels with 14 input channels that include:
- Multispectral data from Sentinel-2: B1, B2, B3, B4, B5, B6, B7, B8, B9, B10, B11, B12.
- Slope data from ALOS PALSAR: B13.
- Digital elevation model (DEM) from ALOS PALSAR: B14.
The label is a binary landslide/no-landslide mask, 128x128, values in {0,1}.
X arrays are shaped (N,128,128,14) float32; y arrays are shaped (N,128,128) uint8.]
"""

# The corpus-retrieval query for round 0: derived from the dataset itself rather than an
# assigned niche, so every agent's search is grounded in "how do I map this kind of
# target given this kind of dataset" rather than a predetermined method family.
RETRIEVAL_QUERY = f"""\
Methods for landslide mapping/detection in this kind of dataset:
{DATASET_DESCRIPTION}"""

SKILL_SYSTEM_PROMPT = """\
You are a research agent whose job is to develop a landslide detection/mapping \
pipeline for satellite imagery, grounded in published methodology. You will write \
a `skill.md` describing YOUR strategy: which data sources/channels you will use, \
what preprocessing/feature engineering you will apply, what model family you will \
use, and why -- citing the specific retrieved papers that motivate each choice. \
This skill.md is your own accumulated strategy; you will revise it in later rounds, \
so write it as durable guidance to your future self, not as a one-off report."""

PIPELINE_SYSTEM_PROMPT = f"""\
You write a single Python module, pipeline.py, implementing a fixed contract so it can \
be executed and evaluated by code you do not control. You MUST implement exactly these \
functions:

{REQUIRED_FUNCS_DOC}

Rules:
- You may import any published, pip-installable library that fits your strategy --
including deep learning frameworks (PyTorch, TensorFlow/Keras, etc.), geospatial \
libraries (rasterio, GDAL, geopandas, ...), or anything else. Only real, \
correctly-named PyPI packages install successfully, so double-check the import name \
matches the actual package (e.g. `import cv2` needs the PyPI package `opencv-python`, \
not `cv2`; when they differ, prefer a library whose import name matches its package \
name, or you'll fail on a bad install rather than a bad model).
- `predict` must return binary {{0,1}} masks -- threshold internally if your model \
produces probabilities.
- Reproducibility is mandatory: at the very top of the module, before any other code, \
deterministically seed every source of randomness any of your imports could draw from \
-- Python's `random`, `numpy`, and the equivalent seeding call or constructor argument \
for every other library you import that exposes one (deep learning frameworks, \
gradient-boosting/classical-ML libraries, etc.). Use the fixed seed {settings.seed} \
everywhere you seed something. A Dice change between rounds must come from your \
strategy, not from unseeded weight init, shuffling, or bootstrapping -- never rely on a \
library's default RNG behavior.
- Output ONLY the Python code for pipeline.py, in a single ```python code block. No \
prose before or after."""
