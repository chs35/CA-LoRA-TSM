"""Selections must come from CA measured on base-model self-generated images.

CA-LoRA (Sec. 3.2) measures CA on x0 the pretrained model generated from c; the
measurement is made once in Improving-CA-LoRA and imported. These tests pin the
two gates: `select` refuses measurements that are not `data_source=selfgen`
(or that mix sources), and `foster` refuses a selection that does not say it
came from such measurements.
"""

from __future__ import annotations

import json

import pytest

from calora.ca.select import select_topk
from calora.ca.stats import save_measurement
from calora.models.projections import ProjectionTable
from catsm.select import load_measurements, load_selection
from conftest import make_adapter
from test_diagnose import fake_result

SELFGEN = {"data_source": "selfgen", "selfgen_manifest_sha256": "ab" * 32, "precision": "tf32"}


@pytest.fixture(scope="module")
def table():
    return ProjectionTable.from_model(make_adapter().unet, granularity="head")


def _write(directory, table, provenances):
    directory.mkdir(parents=True, exist_ok=True)
    for t, provenance in provenances.items():
        result = fake_result(table, t, seed=t)
        result.provenance = dict(provenance)
        save_measurement(result, directory / f"t{t:04d}.safetensors")
    return directory


def test_selfgen_measurements_load(tmp_path, table):
    directory = _write(tmp_path / "m", table, {1: SELFGEN, 81: SELFGEN})
    results = load_measurements(directory)
    assert sorted(results) == [1, 81]
    assert results[81].provenance == SELFGEN


def test_legacy_real_image_measurements_are_refused_unless_asked(tmp_path, table):
    directory = _write(tmp_path / "m", table, {1: {}, 81: {}})  # no header = pre-2026-11-07
    with pytest.raises(ValueError, match="data_source='dataset'"):
        load_measurements(directory)
    assert sorted(load_measurements(directory, require_source=None)) == [1, 81]


def test_mixed_sources_or_precisions_are_refused_even_when_allowed(tmp_path, table):
    mixed = _write(tmp_path / "a", table, {1: SELFGEN, 81: {}})
    with pytest.raises(ValueError, match="data_source"):
        load_measurements(mixed, require_source=None)
    other = _write(tmp_path / "b", table, {1: SELFGEN, 81: {**SELFGEN, "precision": "fp32"}})
    with pytest.raises(ValueError, match="precision"):
        load_measurements(other)


def _selection_file(path, table, meta):
    import torch

    selection = select_topk(torch.arange(len(table), dtype=torch.float64), table, fraction=0.05,
                            budget_basis="unit_count")
    payload = selection.to_dict()
    payload["meta"] = meta
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_foster_refuses_a_selection_without_selfgen_provenance(tmp_path, table):
    legacy = _selection_file(tmp_path / "legacy.json", table, {"timesteps": [81]})
    with pytest.raises(ValueError, match="data_source='dataset'"):
        load_selection(legacy, require_source="selfgen")
    assert len(load_selection(legacy)) > 0  # historical runs still readable

    fresh = _selection_file(tmp_path / "fresh.json", table, {"timesteps": [81], **SELFGEN})
    assert load_selection(fresh, require_source="selfgen").meta["data_source"] == "selfgen"
