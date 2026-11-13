"""perovskite_pu: evaluación de PU learning para la sintetizabilidad de perovskitas ABO3.

Subpaquetes:
    data        descarga, identificación por conectividad, deduplicación y etiquetado (F1)
    features    MAGPIE, t, μ, τ y estados de oxidación (F2)
    splits      esquemas de partición con control de fuga (F1.7–F1.8)
    training    bagging PU, modelos y CV anidada (F3). NO puede leer el conjunto reservado
    evaluation  métricas PU, prior de clase, agregación, tests estadísticos (F3–F4)
    pourbaix    extensión de tesis: estabilidad acuosa (F5, OE4)
    utils       semillas, metadatos de corrida y hashing canónico
"""

__version__ = "0.1.0"
