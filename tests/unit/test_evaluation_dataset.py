import json
from pathlib import Path

import pytest

from agentic_rag.evaluation.runner import load_dataset

DATASET = Path(__file__).resolve().parents[2] / "benchmarks/dense_v1/dataset.json"


def test_frozen_development_set_has_complete_labels() -> None:
    dataset, documents, labels = load_dataset(DATASET)
    assert len(documents) == 10
    assert len(dataset.questions) == 24
    assert {q.language for q in dataset.questions} == {"en", "ru"}
    chunks = {str(chunk.id) for doc in documents.values() for chunk in doc.chunks}
    assert all(set(qrels) <= chunks and qrels for qrels in labels.values())


@pytest.mark.parametrize("damage", ["checksum", "evidence", "duplicate", "unknown"])
def test_invalid_labels_fail_before_indexing(tmp_path: Path, damage: str) -> None:
    data = json.loads(DATASET.read_text())
    for source in data["documents"]:
        source["path"] = str(DATASET.parent / source["path"])
    if damage == "checksum":
        data["documents"][0]["sha256"] = "0" * 64
    elif damage == "evidence":
        data["questions"][0]["evidence"][0]["text"] = "Absent evidence span."
    elif damage == "duplicate":
        data["questions"].append(data["questions"][0])
    else:
        data["questions"][0]["evidence"][0]["document"] = "unknown"
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        load_dataset(path)
