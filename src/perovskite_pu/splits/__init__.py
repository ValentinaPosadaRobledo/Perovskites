"""Esquemas de partición con control de fuga (reglas 2.1.3–2.1.5 de CLAUDE.md).

Esquemas: random, group_system (solo diagnóstico), loeo_A, loeo_B, family, cluster y
temporal. La implementación actual es PROVISIONAL (F0): en F1.7 se decide si se
generan con MatFold o con esta implementación propia, y se registra en la bitácora.
"""

from perovskite_pu.splits.schemes import (
    SCHEMES,
    Fold,
    LeakageError,
    assert_no_leakage,
    assign_b_family,
    group_size_report,
    make_folds,
)

__all__ = [
    "SCHEMES",
    "Fold",
    "LeakageError",
    "assert_no_leakage",
    "assign_b_family",
    "group_size_report",
    "make_folds",
]
