# ATENCIÓN: ningún módulo de este subpaquete puede leer el conjunto de prueba reservado
# ni importar su función de carga. Lo verifica tests/test_leakage.py con análisis
# estático (ast) y con un audit hook sobre el evento `open`.
# (Esta advertencia va en comentarios y no en el docstring porque el test estático
# revisa también las cadenas de texto del código.)
"""Entrenamiento: bagging PU (Mordelet y Vert, 2014), modelos y CV anidada (Fase 3).

Todo el entrenamiento parte de ``data/processed/train_pool.parquet``. Módulos: ``cv``
(ajuste por fold), ``baselines`` (líneas base de F2.4), ``pu_bagging`` (F3.1) y ``runner``
(ejecutor de experimentos con CV anidada y corridas reanudables).
"""
