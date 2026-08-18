"""Identity resolution policy: canonical person ids and top-k match disambiguation.

Pure functions (no models) so they can be unit-tested in isolation.
"""

import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# "Ahmet YILMAZ (2)" -> "Ahmet YILMAZ": duplicate enrollments of the same person
# live in suffixed directories and must not compete as separate identities.
_SUFFIX_RE = re.compile(r"\s*\(\d+\)\s*$")


def canonical_person_id(person_id: str) -> str:
    return _SUFFIX_RE.sub("", (person_id or "").strip())


def resolve_match(
    results: Sequence[Tuple[str, float, int]],
    threshold: float,
    margin: float,
) -> Tuple[Optional[str], float, Optional[str]]:
    """Pick the winning person from raw FAISS rows.

    ``results`` is FaissMatcher.search output: (raw_person_id, score, row_idx),
    highest score first. Rows are grouped by canonical person (best score per
    person). The winner must clear ``threshold`` and beat the best OTHER person
    by ``margin`` (0 disables the margin test).

    Returns (canonical_id or None, top1_score, raw_person_id or None).
    """
    best: Dict[str, Tuple[float, str]] = {}
    for raw_id, score, _idx in results:
        cid = canonical_person_id(raw_id)
        prev = best.get(cid)
        if prev is None or score > prev[0]:
            best[cid] = (float(score), raw_id)
    if not best:
        return None, 0.0, None
    ranked = sorted(best.items(), key=lambda kv: -kv[1][0])
    top_cid, (top_score, top_raw) = ranked[0]
    if top_score < threshold:
        return None, top_score, None
    if margin > 0 and len(ranked) >= 2 and (top_score - ranked[1][1][0]) < margin:
        # Ambiguous between two different people; safer to report no match.
        return None, top_score, None
    return top_cid, top_score, top_raw


def dedupe_embeddings(
    embeddings: np.ndarray,
    names: List[str],
    threshold: float,
) -> Tuple[np.ndarray, List[str], List[Tuple[int, int]]]:
    """Drop rows whose cosine to an already-kept row of the same canonical
    person is >= ``threshold``. Returns (embeddings, names, dropped) where
    dropped lists (dropped_row, kept_row) pairs from the original indexing.
    Row order of survivors is preserved.
    """
    if threshold <= 0 or len(names) == 0:
        return embeddings, names, []
    normed = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12)
    kept_idx: List[int] = []
    kept_by_person: Dict[str, List[int]] = {}
    dropped: List[Tuple[int, int]] = []
    for i, name in enumerate(names):
        cid = canonical_person_id(name)
        dup_of = None
        for j in kept_by_person.get(cid, []):
            if float(np.dot(normed[i], normed[j])) >= threshold:
                dup_of = j
                break
        if dup_of is not None:
            dropped.append((i, dup_of))
            continue
        kept_by_person.setdefault(cid, []).append(i)
        kept_idx.append(i)
    return embeddings[kept_idx], [names[i] for i in kept_idx], dropped
