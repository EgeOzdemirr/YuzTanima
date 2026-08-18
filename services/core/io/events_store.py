import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


class EventsStore:
    def __init__(self, events_path: str) -> None:
        self.events_path = Path(events_path)
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.events_path.exists():
            self.write_default()

    def write_default(self) -> None:
        now = datetime.now(timezone.utc).isoformat()
        payload = {
            "version": 1,
            "timestamp": now,
            "status": "CLEAR",
            "captureId": None,
            "cameraFacePath": None,
            "matchedPersonId": None,
            "galleryPhotoPath": None,
            "galleryFace3dPath": None,
            "similarity": 0.0,
            "threshold": 0.45,
            "bbox": [],
            "detectorScore": 0.0,
            "frameSize": [0, 0],
        }
        self.save_event(payload)

    def save_event(self, payload: Dict[str, Any]) -> None:
        Path(self.events_path).parent.mkdir(parents=True, exist_ok=True)
        payload["timestamp"] = datetime.now(timezone.utc).isoformat()
        with open(self.events_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def load_latest(self) -> Dict[str, Any]:
        if not self.events_path.exists():
            self.write_default()
        with open(self.events_path, "r", encoding="utf-8-sig") as f:
            return json.load(f)
