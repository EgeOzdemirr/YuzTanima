import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.core.utils.identity import canonical_person_id, dedupe_embeddings, resolve_match


def test_canonical_person_id():
    assert canonical_person_id("Abidin AYGÜN (2)") == "Abidin AYGÜN"
    assert canonical_person_id("Abidin AYGÜN(2)") == "Abidin AYGÜN"
    assert canonical_person_id("Ad (3) SOYAD") == "Ad (3) SOYAD"  # ortadaki sonek dokunulmaz
    assert canonical_person_id("Ad SOYAD") == "Ad SOYAD"
    assert canonical_person_id("  Ad SOYAD (10)  ") == "Ad SOYAD"
    assert canonical_person_id("") == ""
    assert canonical_person_id(None) == ""


def test_resolve_match_groups_same_person_rows():
    # Ayni kisinin coklu satirlari tek adaya gruplanir; marj testi
    # ayni kisiye degil FARKLI kisiye karsi yapilir.
    results = [("Ahmet K (2)", 0.72, 0), ("Ahmet K", 0.70, 1), ("Mehmet Y", 0.60, 2)]
    cid, score, raw = resolve_match(results, threshold=0.55, margin=0.05)
    assert cid == "Ahmet K"
    assert score == 0.72
    assert raw == "Ahmet K (2)"


def test_resolve_match_margin_rejects_two_close_people():
    results = [("Ahmet K", 0.60, 0), ("Mehmet Y", 0.58, 1)]
    cid, score, raw = resolve_match(results, threshold=0.55, margin=0.05)
    assert cid is None
    assert raw is None
    assert score == 0.60


def test_resolve_match_margin_passes_when_runner_up_is_same_person():
    results = [("Ahmet K", 0.60, 0), ("Ahmet K (2)", 0.59, 1), ("Mehmet Y", 0.40, 2)]
    cid, _, _ = resolve_match(results, threshold=0.55, margin=0.05)
    assert cid == "Ahmet K"


def test_resolve_match_below_threshold():
    results = [("Ahmet K", 0.50, 0)]
    cid, score, raw = resolve_match(results, threshold=0.55, margin=0.05)
    assert cid is None and raw is None
    assert score == 0.50


def test_resolve_match_empty():
    assert resolve_match([], threshold=0.55, margin=0.05) == (None, 0.0, None)


def test_resolve_match_margin_disabled():
    results = [("Ahmet K", 0.60, 0), ("Mehmet Y", 0.59, 1)]
    cid, _, _ = resolve_match(results, threshold=0.55, margin=0.0)
    assert cid == "Ahmet K"


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def test_dedupe_embeddings():
    rows = np.stack(
        [
            _unit([1.0, 0.0, 0.0]),   # Celal KAYA
            _unit([1.0, 0.001, 0.0]), # Celal KAYA - birebir kopyaya yakin -> atilir
            _unit([0.0, 1.0, 0.0]),   # Celal KAYA - farkli poz -> kalir
            _unit([1.0, 0.0, 0.0]),   # Veli DEMIR - ayni vektor ama farkli kisi -> kalir
        ]
    )
    names = ["Celal KAYA", "Celal KAYA", "Celal KAYA (2)", "Veli DEMIR"]
    deduped, kept_names, dropped = dedupe_embeddings(rows, names, threshold=0.999)
    assert kept_names == ["Celal KAYA", "Celal KAYA (2)", "Veli DEMIR"]
    assert deduped.shape == (3, 3)
    assert dropped == [(1, 0)]
    # embedding-isim hizasi korunur
    assert np.allclose(deduped[0], rows[0])
    assert np.allclose(deduped[1], rows[2])
    assert np.allclose(deduped[2], rows[3])


def test_dedupe_disabled():
    rows = np.eye(3, dtype=np.float32)
    names = ["a", "a", "b"]
    deduped, kept_names, dropped = dedupe_embeddings(rows, names, threshold=0.0)
    assert kept_names == names and dropped == [] and deduped.shape == (3, 3)
