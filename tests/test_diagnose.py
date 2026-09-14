"""scripts/diagnose_ca.py on synthetic head measurements (hermetic, CPU)."""

from __future__ import annotations

import importlib.util
import json
import sys

import pytest
import torch

from calora.ca.stats import AxisStats, MeasurementResult, save_measurement
from conftest import ROOT, make_adapter
from calora.models.projections import ProjectionTable


@pytest.fixture(scope="module")
def diagnose_ca():
    spec = importlib.util.spec_from_file_location("diagnose_ca", ROOT / "scripts" / "diagnose_ca.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fake_result(table: ProjectionTable, t: int, seed: int, n_items: int = 4) -> MeasurementResult:
    gen = torch.Generator().manual_seed(seed)
    n = len(table)
    per_axis = {}
    for axis in ("style", "viewpoint"):
        ca = torch.rand(n, generator=gen, dtype=torch.float32) + 1e-3
        per_axis[axis] = AxisStats(ca_mean=ca, ca_std=torch.zeros(n), gc_norm_mean=ca.clone(), direction_mean=None)
    return MeasurementResult(
        timestep=t, num_samples=n_items, axes=("style", "viewpoint"), names=tuple(table.names),
        per_axis=per_axis, gd_norm_mean=torch.ones(n), valid_count=torch.full((n,), n_items),
        cos_of_means={}, mean_of_cos={}, unit_granularity="head", num_items=n_items,
        table_records=table.to_records(),
    )


@pytest.fixture(scope="module")
def measurements_dir(tmp_path_factory):
    table = ProjectionTable.from_model(make_adapter().unet, granularity="head")
    directory = tmp_path_factory.mktemp("measurements")
    for t, seed in ((1, 7), (81, 7), (201, 3), (750, 5)):  # 1 and 81 share a ranking
        save_measurement(fake_result(table, t, seed), directory / f"t{t:04d}.safetensors")
        if t != 750:  # 750 has no halves -> split_half_jaccard must be None
            for half in (0, 1):
                save_measurement(fake_result(table, t, seed * 10 + half), directory / f"t{t:04d}_half{half}.safetensors")
    return directory


def test_diagnosis_outputs(diagnose_ca, measurements_dir, tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)
    monkeypatch.setattr(sys, "argv", [
        "diagnose_ca.py", "--config", str(ROOT / "configs" / "tiny_cpu.yaml"),
        "--measurements", str(measurements_dir), "--set", f"output_dir={tmp_path}",
    ])
    diagnose_ca.main()
    out = tmp_path / "tiny_cpu" / "diagnosis"
    diag = json.loads((out / "diagnosis.json").read_text())
    assert diag["timesteps"] == [1, 81, 201, 750] and "divergence" not in diag["per_timestep"]["81"]
    for t, e in diag["per_timestep"].items():
        assert len(e["ca_normalised"]["style"]) == diag["num_units_total"] == 80
        assert abs(sum(e["ca_normalised"]["viewpoint"]) - 1.0) < 1e-6
        assert e["num_units"] == {"style": 8, "viewpoint": 8}
        assert 0 <= e["cross_axis_jaccard"]["style|viewpoint"] <= 1
        assert (e["split_half_jaccard"] is None) == (t == "750")
    m = diag["jaccard_matrix"]["values"]
    assert all(m[i][i] == 1.0 for i in range(4)) and all(m[i][j] == m[j][i] for i in range(4) for j in range(4))
    assert diag["per_timestep"]["81"]["jaccard_vs_81"] == 1.0
    assert diag["per_timestep"]["1"]["jaccard_vs_81"] == 1.0  # same seed -> same selection
    assert diag["per_timestep"]["201"]["jaccard_vs_81"] < 1.0
    bk = diag["per_timestep"]["81"]["block_kind"]["style"]
    assert set(diag["blocks"]) == set(bk) and abs(sum(sum(r.values()) for r in bk.values()) - 1.0) < 1e-6
    lk = diag["per_timestep"]["81"]["layer_kind"]["style"]
    assert len(diag["layers"]) == 10 and set(lk) == set(diag["layers"])  # toy: 5 blocks x attn1/attn2
    assert abs(sum(sum(r.values()) for r in lk.values()) - 1.0) < 1e-6
    for name in ("ca_distribution.png", "jaccard_matrix.png", "ca_heatmap_style_layers.png",
                 "ca_heatmap_viewpoint_layers_shared.png", "ca_heatmap_style_blocks.png"):
        assert (out / name).exists(), name
    assert not (out / "kld_by_t.png").exists()
    template = (out / "DECISION_TEMPLATE.md").read_text()
    assert "n2_i1" in template and "[500, 999]" in template and "t=750" in template
    assert "n1_i0" in template and template.count("- decision:") == 3  # n1 (1) + n2 (2)


def test_timestep_subset_and_missing(diagnose_ca, measurements_dir, tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)
    base = ["diagnose_ca.py", "--config", str(ROOT / "configs" / "tiny_cpu.yaml"),
            "--measurements", str(measurements_dir), "--set", f"output_dir={tmp_path}", "--out", str(tmp_path / "sub")]
    monkeypatch.setattr(sys, "argv", base + ["--timesteps", "81", "750"])
    diagnose_ca.main()
    diag = json.loads((tmp_path / "sub" / "diagnosis.json").read_text())
    assert diag["timesteps"] == [81, 750] and len(diag["jaccard_matrix"]["values"]) == 2
    monkeypatch.setattr(sys, "argv", base + ["--timesteps", "81", "999"])
    with pytest.raises(FileNotFoundError, match="999"):
        diagnose_ca.main()
