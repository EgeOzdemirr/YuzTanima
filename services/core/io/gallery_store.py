import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class PersonEntry:
    person_id: str
    photo_path: str
    face3d_path: Optional[str]
    index: int


def write_gallery_json(path: str, persons: List[PersonEntry], embedding_dim: int) -> None:
    payload = {
        "version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "embedding_dim": embedding_dim,
        "persons": [
            {
                "personId": p.person_id,
                "photoPath": p.photo_path,
                "face3dPath": p.face3d_path,
                "index": p.index,
            }
            for p in persons
        ],
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def write_names(names: List[str], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(names, f, indent=2)


def read_names(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)
