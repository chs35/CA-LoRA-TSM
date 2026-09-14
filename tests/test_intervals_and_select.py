from __future__ import annotations

import pytest
import torch
from torch import nn

from calora.ca.stats import AxisStats, MeasurementResult
from calora.config import SelectConfig
from catsm.config import IntervalsConfig, load_config
from catsm.intervals import (
    Interval,
    IntervalScheme,
    build_scheme,
    halving_intervals,
    intervals_from_starts,
    logsnr,
    parse_expert_name,
    scheme_record,
    validate_partition,
)
from catsm.select import select_for_interval, select_scheme, selection_sha
from conftest import ROOT, make_adapter

T = 1000


def spans(intervals):
    return [iv.as_list() for iv in intervals]


def test_halving_intervals_match_the_draft():
    assert spans(halving_intervals(1, T)) == [[0, 999]]
    assert spans(halving_intervals(3, T)) == [[0, 249], [250, 499], [500, 999]]
    assert spans(halving_intervals(4, T)) == [[0, 124], [125, 249], [250, 499], [500, 999]]
    n5 = halving_intervals(5, T)
    assert spans(n5) == [[0, 61], [62, 124], [125, 249], [250, 499], [500, 999]]
    assert [iv.length for iv in n5] == [62, 63, 125, 250, 500]
    # n=5 is the scheme that separates the two known candidates.
    scheme = IntervalScheme("n5", n5)
    assert scheme.owner(41) != scheme.owner(81)
    assert scheme.owner(81) == scheme.owner(124)


def test_partition_validation_catches_gaps_and_overlaps():
    validate_partition(halving_intervals(4, T), T)
    with pytest.raises(ValueError):
        validate_partition((Interval(0, 100), Interval(102, 999)), T)
    with pytest.raises(ValueError):
        validate_partition((Interval(0, 100), Interval(100, 999)), T)
    with pytest.raises(ValueError):
        validate_partition((Interval(0, 998),), T)
    with pytest.raises(ValueError):
        Interval(5, 4)


def test_boundary_override_is_validated():
    assert spans(intervals_from_starts([125, 250, 500], T)) == spans(halving_intervals(4, T))
    cfg = IntervalsConfig(schemes=[4], boundaries={"n4": [100, 300, 600]})
    assert spans(build_scheme("n4", T, cfg).intervals) == [[0, 99], [100, 299], [300, 599], [600, 999]]
    with pytest.raises(ValueError):
        intervals_from_starts([500, 250], T)
    with pytest.raises(ValueError):
        intervals_from_starts([0, 250], T)
    with pytest.raises(ValueError):
        IntervalsConfig(schemes=[4], boundaries={"n4": [125, 250]})


def test_expert_names_are_moduledict_safe_and_parse_back():
    scheme = build_scheme("n4", T)
    assert scheme.expert_names() == ["n4_i0", "n4_i1", "n4_i2", "n4_i3"]
    assert parse_expert_name("n4_i2") == ("n4", 2)
    assert scheme.interval_of("n4_i0") == Interval(0, 124)
    nn.ModuleDict({name: nn.Identity() for name in scheme.expert_names()})
    for bad in ("n4.i0", "e0", "n4_i4", "n0_i0"):
        with pytest.raises(ValueError):
            parse_expert_name(bad)
    lookup = scheme.lookup(T)
    assert lookup.tolist()[:2] == [0, 0] and int(lookup[124]) == 0 and int(lookup[125]) == 1
    assert int(lookup[999]) == 3


def test_logsnr_decreases_with_t_and_is_recorded():
    adapter = make_adapter()
    a = adapter.alphas_cumprod
    values = [logsnr(a, t) for t in (0, 81, 500, 999)]
    assert values == sorted(values, reverse=True)
    record = scheme_record(build_scheme("n2", T), a)
    assert record["intervals"][0]["logsnr_lo"] == pytest.approx(values[0])
    assert record["intervals"][1]["t_lo"] == 500


# --------------------------------------------------------------------------------------
# Selection modes on synthetic measurements
# --------------------------------------------------------------------------------------


def fake_result(table, timestep: int, seed: int) -> MeasurementResult:
    """A measurement whose style-CA ranking depends on ``seed`` only."""
    gen = torch.Generator().manual_seed(seed)
    n = len(table)
    ca = torch.rand(n, generator=gen, dtype=torch.float32) + 1e-3
    stats = AxisStats(ca_mean=ca, ca_std=torch.zeros(n), gc_norm_mean=ca, direction_mean=None)
    return MeasurementResult(
        timestep=timestep,
        num_samples=4,
        axes=("style",),
        names=tuple(table.names),
        per_axis={"style": stats},
        gd_norm_mean=torch.ones(n),
        valid_count=torch.full((n,), 4),
        cos_of_means={},
        mean_of_cos={},
        unit_granularity="head",
        table_records=table.to_records(),
    )


@pytest.fixture(scope="module")
def measurements(head_table):
    # Same seed for 1 and 81 (same ranking), different for the rest.
    return {t: fake_result(head_table, t, seed) for t, seed in
            [(1, 7), (81, 7), (201, 3), (481, 4), (750, 5)]}


SELECT = SelectConfig(granularity="head", fraction=0.1, budget_basis="unit_count", axis="style")


def test_representative_mode_uses_the_named_timestep(measurements, head_table):
    low = Interval(0, 124)
    sel = select_for_interval(measurements, head_table, low, mode="representative",
                              cfg=SELECT, representative_t=81)
    assert sel.meta["timesteps"] == [81] and sel.granularity == "head"
    assert sel.meta["selection_sha"] == selection_sha(sel.names)
    with pytest.raises(ValueError):  # t outside the interval
        select_for_interval(measurements, head_table, low, mode="representative",
                            cfg=SELECT, representative_t=201)


def test_aggregate_mode_pools_only_measured_t_inside_the_interval(measurements, head_table):
    sel = select_for_interval(measurements, head_table, Interval(0, 249), mode="aggregate", cfg=SELECT)
    assert sel.meta["timesteps"] == [1, 81, 201]
    single = select_for_interval(measurements, head_table, Interval(500, 999), mode="aggregate", cfg=SELECT)
    assert single.meta["timesteps"] == [750]
    with pytest.raises(ValueError):
        select_for_interval(measurements, head_table, Interval(900, 999), mode="aggregate", cfg=SELECT)


def test_fixed_mode_gives_every_interval_the_same_selection(measurements, head_table):
    scheme = build_scheme("n4", T)
    fixed = select_scheme(measurements, head_table, scheme, mode="fixed", cfg=SELECT, fixed_t=81)
    shas = {s.meta["selection_sha"] for s in fixed.values()}
    assert len(shas) == 1 and set(fixed) == set(scheme.expert_names())
    per_interval = select_scheme(measurements, head_table, scheme, mode="aggregate", cfg=SELECT)
    assert len({s.meta["selection_sha"] for s in per_interval.values()}) > 1
    assert per_interval["n4_i0"].meta["expert"] == "n4_i0"


def test_tiny_config_loads_and_unknown_keys_are_rejected(tmp_path):
    cfg = load_config(ROOT / "configs" / "tiny_cpu.yaml")
    assert cfg.assemble.core in cfg.assemble.sets and cfg.interval_select.mode == "aggregate"
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: x\nassemble:\n  sets: [n1, n4]\n  core: n4\n  bogus: 1\n")
    with pytest.raises(ValueError):
        load_config(bad)
    bad.write_text("name: x\nassemble:\n  sets: [n1]\n  core: n4\n")
    with pytest.raises(ValueError):
        load_config(bad)
