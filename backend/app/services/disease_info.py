"""
Loads the structured disease information dataset (backend/app/data/disease_info.json)
and exposes lookups by class name.
"""

import json
from functools import lru_cache
from typing import Dict

from app.config import settings


class DiseaseNotFoundError(KeyError):
    pass


@lru_cache(maxsize=1)
def _load_data() -> dict:
    with open(settings.DISEASE_INFO_PATH) as f:
        return json.load(f)


def get_disclaimer() -> str:
    return _load_data()["disclaimer"]


def get_disease_info(class_name: str) -> Dict:
    """Look up structured disease info for a predicted class name (e.g.
    'Tomato_Early_Blight'). Raises DiseaseNotFoundError if the class is not
    documented, so a missing entry is surfaced loudly rather than silently
    returning empty/fake data.
    """
    diseases = _load_data()["diseases"]
    if class_name not in diseases:
        raise DiseaseNotFoundError(
            f"No disease information documented for class '{class_name}'. "
            "Update backend/app/data/disease_info.json."
        )
    return diseases[class_name]
