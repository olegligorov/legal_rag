"""Schema validation for evaluation datasets.

Catches regressions like typos in expected_articles, duplicate ids, or
references to articles that don't exist in the source corpus — bugs that
otherwise surface only as silent recall=0 in the head-to-head numbers.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from citations import normalize_article, normalize_law_id

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DOCS_DIR = REPO_ROOT / "documentation"
DATASETS_DIR = Path(__file__).resolve().parent.parent / "evaluation"


def _load_corpus() -> dict[tuple[str, str], dict]:
    """Map (normalized_law_id, normalized_article) → article dict."""
    out: dict[tuple[str, str], dict] = {}
    for fp in sorted(DOCS_DIR.glob("*.json")):
        with open(fp, encoding="utf-8") as f:
            for art in json.load(f):
                key = (
                    normalize_law_id(art.get("law_id", "")),
                    normalize_article(art.get("article", "")),
                )
                out[key] = art
    return out


CORPUS = _load_corpus()
DATASETS = sorted(DATASETS_DIR.glob("dataset*.json"))


@pytest.mark.parametrize("dataset_path", DATASETS, ids=lambda p: p.name)
class TestDatasetSchema:
    """Run every test against every dataset file in evaluation/."""

    def test_loadable_json(self, dataset_path: Path) -> None:
        with open(dataset_path, encoding="utf-8") as f:
            data = json.load(f)
        assert isinstance(data, list)
        assert data, f"{dataset_path.name} is empty"

    def test_required_fields_per_question(self, dataset_path: Path) -> None:
        data = json.loads(dataset_path.read_text(encoding="utf-8"))
        for tc in data:
            assert "id" in tc, f"{tc} missing id"
            assert "question" in tc, f"{tc.get('id')} missing question"
            assert "ground_truth" in tc, f"{tc.get('id')} missing ground_truth"
            assert "expected_articles" in tc, f"{tc.get('id')} missing expected_articles"
            assert isinstance(tc["expected_articles"], list)
            assert tc["expected_articles"], f"{tc['id']} has empty expected_articles"

    def test_expected_article_shape(self, dataset_path: Path) -> None:
        data = json.loads(dataset_path.read_text(encoding="utf-8"))
        for tc in data:
            for ea in tc["expected_articles"]:
                assert "law_id" in ea, f"{tc['id']}: expected_article missing law_id"
                assert "article" in ea, f"{tc['id']}: expected_article missing article"
                assert ea["law_id"], f"{tc['id']}: empty law_id"
                assert ea["article"], f"{tc['id']}: empty article"

    def test_unique_ids(self, dataset_path: Path) -> None:
        data = json.loads(dataset_path.read_text(encoding="utf-8"))
        ids = [tc["id"] for tc in data]
        assert len(ids) == len(set(ids)), f"duplicate ids in {dataset_path.name}"

    def test_expected_articles_resolve_in_corpus(self, dataset_path: Path) -> None:
        """Every expected (law_id, article) must exist in the source JSONs.

        Catches typos like ``Чл. 105.`` when the law actually has ``Чл. 105а`` —
        without this, the eval silently scores the system 0 on that question
        because the expected article can never appear in retrieval.
        """
        data = json.loads(dataset_path.read_text(encoding="utf-8"))
        missing: list[tuple[str, str, str]] = []
        for tc in data:
            for ea in tc["expected_articles"]:
                key = (normalize_law_id(ea["law_id"]), normalize_article(ea["article"]))
                if key not in CORPUS:
                    missing.append((tc["id"], ea["law_id"], ea["article"]))
        assert not missing, f"Expected articles not found in corpus: {missing}"


def test_corpus_loaded() -> None:
    """Sanity: the test setup actually found articles."""
    assert len(CORPUS) > 0, f"No articles loaded from {DOCS_DIR}"
