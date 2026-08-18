import sys
from collections import deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.camera_runtime.pipeline import CameraPipeline, _TrackMatchState, can_embed_face


def _pipeline(**overrides):
    p = object.__new__(CameraPipeline)
    p.match_threshold = 0.55
    p.match_margin = 0.05
    p.confirmation_frames = 2
    p.relock_after = 0
    for key, value in overrides.items():
        setattr(p, key, value)
    return p


def test_can_embed_face_gate():
    # coasting kare (kps yok) -> reddedilir
    assert not can_embed_face(has_kps=False, bbox_side=100, detector_frames_only=True, min_face_px=60)
    # detector_frames_only kapaliysa kps'siz kare gecer
    assert can_embed_face(has_kps=False, bbox_side=100, detector_frames_only=False, min_face_px=60)
    # kucuk yuz -> reddedilir
    assert not can_embed_face(has_kps=True, bbox_side=40, detector_frames_only=True, min_face_px=60)
    # min_face_px=0 -> boyut kontrolu kapali
    assert can_embed_face(has_kps=True, bbox_side=40, detector_frames_only=True, min_face_px=0)
    assert can_embed_face(has_kps=True, bbox_side=100, detector_frames_only=True, min_face_px=60)


def test_embedding_average_math():
    window = deque(maxlen=5)
    rng = np.random.default_rng(0)
    first = None
    for i in range(6):
        v = rng.normal(size=8).astype(np.float32)
        v /= np.linalg.norm(v)
        if i == 0:
            first = v
        window.append(v)
    # 6. embedding 1.'yi dusurdu
    assert len(window) == 5
    assert not any(np.allclose(first, w) for w in window)
    avg = np.mean(np.stack(window, axis=0), axis=0)
    avg /= np.linalg.norm(avg) + 1e-12
    assert abs(float(np.linalg.norm(avg)) - 1.0) < 1e-6


def test_lock_holds_without_relock():
    p = _pipeline()
    s = _TrackMatchState(emb_window=deque(maxlen=5))
    assert p._update_confirmed_match(s, "Ahmet", 0.70, True, raw_person_id="Ahmet (2)") is None
    assert p._update_confirmed_match(s, "Ahmet", 0.72, True, raw_person_id="Ahmet (2)") == "Ahmet"
    assert s.confirmed_raw_id == "Ahmet (2)"
    # celiskiler kilidi bozamaz (relock_after=0)
    for _ in range(20):
        assert p._update_confirmed_match(s, "Mehmet", 0.90, True) == "Ahmet"
    # eslesme kaybi da bozamaz
    for _ in range(20):
        assert p._update_confirmed_match(s, None, 0.0, True) == "Ahmet"


def test_relock_after_strong_contradictions():
    p = _pipeline(relock_after=3)
    s = _TrackMatchState(emb_window=deque(maxlen=5))
    p._update_confirmed_match(s, "Ahmet", 0.70, True)
    p._update_confirmed_match(s, "Ahmet", 0.70, True)
    assert s.confirmed_person_id == "Ahmet"
    # esik+marj altindaki celiski sayilmaz
    assert p._update_confirmed_match(s, "Mehmet", 0.56, True) == "Ahmet"
    assert s.contradiction_hits == 0
    # 3 ardisik guclu celiski kilidi cozer, yeni aday konfirmasyona baslar
    p._update_confirmed_match(s, "Mehmet", 0.70, True)
    p._update_confirmed_match(s, "Mehmet", 0.70, True)
    assert s.confirmed_person_id == "Ahmet"  # henuz 2 celiski
    result = p._update_confirmed_match(s, "Mehmet", 0.70, True)
    assert result is None  # kilit cozuldu
    assert s.candidate_person_id == "Mehmet" and s.candidate_hits == 1
    # bir eslesme daha -> yeni kimlik onaylanir
    assert p._update_confirmed_match(s, "Mehmet", 0.70, True) == "Mehmet"


def test_same_person_match_resets_contradictions():
    p = _pipeline(relock_after=3)
    s = _TrackMatchState(emb_window=deque(maxlen=5))
    p._update_confirmed_match(s, "Ahmet", 0.70, True)
    p._update_confirmed_match(s, "Ahmet", 0.70, True)
    p._update_confirmed_match(s, "Mehmet", 0.70, True)
    p._update_confirmed_match(s, "Mehmet", 0.70, True)
    assert s.contradiction_hits == 2
    # dogru kisi tekrar eslesince sayac sifirlanir
    p._update_confirmed_match(s, "Ahmet", 0.75, True)
    assert s.contradiction_hits == 0
    assert s.confirmed_person_id == "Ahmet"
    assert s.confirmed_similarity == 0.75
