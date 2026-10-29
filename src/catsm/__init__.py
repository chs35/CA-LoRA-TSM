"""CA-LoRA with TimeStep Master (docs/design.md).

Per-interval concept-aware head selection, one sliced LoRA expert per timestep
interval (Fostering, trained only on that interval's t), and a router that
combines the frozen experts (Assembling). Everything CA-LoRA-specific — CA
measurement, top-k head selection, the sliced type A/B LoRA — comes from the
vendored ``calora`` package and is reused, not reimplemented.
"""

__version__ = "0.1.0"
