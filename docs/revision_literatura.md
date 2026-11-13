# Revisión de literatura (F0.5)

> **Estado: parcial (2026-11-13).** Las 22 referencias tienen metadatos verificados. Solo 5
> se leyeron a texto completo. Faltan lecturas completas que no pude hacer porque la
> editorial RSC bloquea el acceso automatizado (sección 8). La brecha de la sección 6 queda
> **condicionada** a la lectura de Chung et al. (2025).
>
> Bibliografía en BibTeX: `paper/references.bib`, generada a partir de los registros de
> Crossref de cada DOI. Cada entrada indica su nivel de verificación.

## 1. Método

**Fuentes consultadas:**

| Fuente | Uso |
|---|---|
| Crossref | Metadatos por DOI: autores, revista, volumen, páginas, año |
| arXiv | Metadatos y texto completo HTML de los preprints |
| nature.com | Texto completo de los artículos de acceso abierto de *npj Comput. Mater.* |
| Semantic Scholar, OpenAlex | Resúmenes y versiones de acceso abierto |
| Consensus | Búsqueda semántica: 1 búsqueda dirigida (10 resultados) y 1 amplia (50 resultados, leídos en su totalidad) |
| PyPI | README oficial de MatFold v1.3.3 |

**Niveles de verificación del contenido**, usados en todo el documento:

- **[TC]** texto completo leído; las cifras se citan textualmente.
- **[R]** solo resumen.
- **[M]** solo metadatos.

**Limitaciones:**

- `pubs.rsc.org` devuelve 403 a cualquier acceso automatizado, incluidos los PDF de acceso
  abierto. Por eso MatFold, Meredig et al., Chung et al. y la versión publicada de SynCoTrain
  no se leyeron completos.
- No es una revisión sistemática (PRISMA). La afirmación de que "ningún trabajo hace X" vale
  para el conjunto revisado, no para toda la literatura.

## 2. Correcciones a las referencias del prompt

| Referencia | En el prompt | Verificado |
|---|---|---|
| Frey et al. | 2020 | **2019**, *ACS Nano* 13(3):3031–3041 |
| SynCoTrain | arXiv 2411.12011 | Publicado: *Digital Discovery* 4(6):1437–1448 (2025) |
| Desai et al. | arXiv 2510.06166 | Publicado: *Mach. Learn.: Sci. Technol.* 6:045061 (2025); misma obra, mismos autores |
| MatFold | *Digital Discovery*, 2025 | Correcto: 4(3):625–635. Preprint en ChemRxiv (2024) |
| Desai et al. (alcance) | "PU para sintetizabilidad" | Solo perovskitas **haluro y calcogenuro**: no cubre óxidos |
| Gu et al. (datos) | "943 sintetizadas, 11 964 virtuales" | Son **cristales**, no composiciones; provienen de MP + OQMD + AFLOW (octubre de 2020) |

## 3. Referencias revisadas

| Clave BibTeX | Trabajo | Nivel | Papel en el proyecto |
|---|---|---|---|
| `gu2022perovskite` | Gu et al., *npj Comput. Mater.* 8:71 (2022) | TC | Modelo que se reevalúa (H3, F3.5) |
| `amariamir2025syncotrain` | SynCoTrain, *Digital Discovery* (2025) | TC (arXiv v1) | PU para óxidos; "generalización" sin particiones por grupos |
| `desai2025bridging` | Desai et al., *MLST* (2025) | TC (arXiv) | PU para perovskitas haluro/calcogenuro |
| `antoniuk2023synthnn` | SynthNN, *npj Comput. Mater.* 9:155 (2023) | TC | Composicional; precedente de partición temporal |
| `chung2025solidstate` | Chung, Walsh y Payne, *Digital Discovery* 4:2439 (2025) | M + R | **El más cercano: PU para óxidos ternarios.** Sin leer |
| `meredig2018` | Meredig et al., *MSDE* 3:819 (2018) | R | LOCO-CV (esquema `cluster`) |
| `witman2025matfold` | Witman y Schindler, *Digital Discovery* 4:625 (2025) | R + README | Herramienta de particiones (F1.7) |
| `bartel2019tolerance` | Bartel et al., *Sci. Adv.* 5:eaav0693 (2019) | R | τ: línea base y comparador de H2 |
| `jang2020structure` | Jang et al., *JACS* 142:18836 (2020) | R | PU con grafos; precedente temporal |
| `jang2024stoichiometry` | Jang et al., *Matter* 7:2294 (2024) | R | PU por estequiometría (óxidos) |
| `frey2019mxenes` | Frey et al., *ACS Nano* 13:3031 (2019) | R | PU en materiales (MXenes) |
| `sun2025cpul` | Sun y Yuan, *Comput. Phys. Commun.* 308:109465 (2025) | R | Precedente: reserva todos los compuestos con Fe |
| `li2025ood` | Li et al., *Commun. Mater.* 6:9 (2025) | R | Las pruebas OOD suelen ser interpolación |
| `davariashtiyani2024bias` | Davariashtiyani et al., *MLST* 5:040501 (2024) | R | Sesgo por la fuente de los datos (apoya el control de H3) |
| `zhu2023synthesizability` | Zhu et al., *ACS Omega* 8:8210 (2023) | R | Partición temporal por fecha de MP |
| `schlesinger2026thermodynamic` | Schlesinger et al., *Mater. Horiz.* 13:8188 (2026) | R | Los modelos de sintetizabilidad sobrepredicen |
| `aydin2026tree` | Aydın et al., *Adv. Theory Simul.* 9:e70395 (2026) | R | Benchmark PU con árboles |
| `mordelet2014bagging` | Mordelet y Vert, *Pattern Recognit. Lett.* 37:201 (2014) | M | Bagging PU (F3.1) |
| `elkan2008learning` | Elkan y Noto, KDD 2008 | M | Prior de clase (F3.2) |
| `bekker2018tice` | Bekker y Davis, AAAI 2018 | M | TIcE, prior de clase (F3.2) |
| `nadeau2003inference` | Nadeau y Bengio, *Mach. Learn.* 52:239 (2003) | M | Test corregido (F4.2) |
| `singh2017electrochemical` | Singh et al., *Chem. Mater.* 29:10159 (2017) | M | ΔG_pbx (F5.1) |
| — | Lee y Liu, ICML 2003 | **No verificada** | Métrica de Lee–Liu (sin DOI) |
| — | Ramaswamy et al., ICML 2016 (KM2) | **No verificada** | Prior de clase (sin DOI) |

## 4. Fichas de los trabajos núcleo

### 4.1 Gu et al. (2022): perovskitas con grafos [TC]

- **Datos.** Preentrenamiento con MP (octubre de 2020): "46,546 crystals with inorganic
  crystal structure databases (ICSD) id and 79,789 crystals without ICSD id". Perovskitas de
  MP, OQMD y AFLOW, identificadas con "the StructureMatcher function in pymatgen and
  perovskite prototype structures in the AFLOW database". Resultado: "943 synthesized and
  11,964 virtual perovskite crystals". Todas las clases de perovskita, no solo óxidos.
- **Evaluación.** "10% of randomly sampled synthesized crystals are set aside". Es una
  única partición aleatoria, a nivel de **cristal**. El ensamble de 100 modelos varía solo los
  virtuales: "the training and validation set for the virtual crystals change, whereas those
  for the synthesized crystals remain fixed".
- **Resultados.** TPR = 0,957 (GCNN + PUL + DSL + TL). En óxidos, la TPR pasa "from 0.837 to
  0.930" con transferencia de dominio. "84% of the perovskite crystals are predicted
  unsynthesizable".
- **Comparación con τ.** "Only 310 crystals out of 943 perovskites were within their
  selection of elements … we calculated the TPR of 0.806". La comparación usa **subconjuntos
  distintos y umbrales distintos**, no la misma P(ŷ = 1).
- **Datos y código:** Zenodo 10.5281/zenodo.6348980 (citado en el artículo).
- **Para este proyecto:**
  - **Posible fuga entre polimorfos.** Al particionar por cristal, polimorfos o variantes
    distorsionadas de una misma composición podrían quedar a ambos lados. Es una inferencia
    nuestra que hay que verificar con sus datos en F1.2/F3.5.
  - Ninguna partición por grupos.
  - La comparación con τ no se hace a igual P(ŷ = 1), que es lo que corrige H2.

### 4.2 SynCoTrain (Amariamir, George y Benner, 2025) [TC, arXiv v1]

- **Datos.** Óxidos de ICSD vía MP: "10206 experimental and 31245 unlabeled data points". La
  experimentalidad se toma del atributo `theoretical`, como en nuestra definición.
- **Evaluación.** "a leave-out test set with 5% of the positive data … For the dynamic test
  set, 10% of the positive data is chosen at each run". Ambos son aleatorios. La
  "generalización" se aborda con co-entrenamiento (ALIGNN + SchNet), no con particiones por
  grupos.
- **Hallazgo útil.** Validan la *recall* contra una "Ground Truth Recall", construyendo una
  tarea PU artificial a partir de la estabilidad. Es una idea aplicable a F3.2.
- **Para este proyecto:** óxidos en general, sin foco en perovskitas y sin extrapolación
  química explícita.

### 4.3 Desai et al. (2025) [TC, arXiv]

- **Datos.** Perovskitas haluro y calcogenuro: "a total positive label set of 239 compounds"
  (206 extraídos de la literatura con LLM y ChemDataExtractor, más 33 de ICSD) y 832 no
  etiquetados.
- **Evaluación.** K-fold aleatorio repetido (R = 10, K = 5, T = 500), con "TPR of 86% for the
  DT model on the cumulative random test set". También reportan ROC-AUC.
- **Para este proyecto:** otra clase química y sin particiones por grupos. No compite
  directamente.

### 4.4 SynthNN (Antoniuk et al., 2023) [TC]

- **Datos.** Composiciones de ICSD más fórmulas no sintetizadas generadas artificialmente.
  Las métricas se calculan "by treating the artificially generated unsynthesized materials as
  negative examples", lo que en un problema PU equivale a cotas inferiores.
- **Evaluación.** "a 90:5:5 split" aleatoria, más una prueba **temporal por décadas** (Fig. 4).
- **Para este proyecto:** precedente del esquema `temporal` y de modelos solo composicionales
  (nuestra familia `pre_dft`).

### 4.5 Otros precedentes de evaluación fuera de distribución [R]

- **Jang et al. (2020):** TPR del 87,4 % en test aleatorio y del 86,2 % en materiales
  reportados en 2015–2019, con un modelo entrenado hasta 2014. Precedente temporal.
- **Zhu et al. (2023):** partición temporal por **fecha de carga en MP**: entrena antes de 2015
  y obtiene TPR del 88,60 % con precisión del 9,81 % en el test posterior a 2019. Ojo: la
  fecha de carga en MP no es el año de síntesis.
- **Sun y Yuan (CPUL):** "using all Fe-containing materials from the MP database as the test
  set, obtaining a true positive rate of 88.89%". Es un único elemento reservado, sin análisis
  sistemático.
- **Li et al. (2025):** "most OOD tests reflect interpolation, not true extrapolation", porque
  los datos de test quedan en regiones del espacio de representación bien cubiertas por el
  entrenamiento.
- **Davariashtiyani et al. (2024):** los conjuntos que mezclan fuentes experimentales y
  computacionales inducen sesgos que el modelo aprende. Apoya el control por procedencia de
  la estructura en H3.
- **Schlesinger et al. (2026, grupo de Bartel):** los modelos de sintetizabilidad publicados
  "generally overpredict the likelihood of synthesis". Apoya reportar siempre P(ŷ = 1 | U).

### 4.6 Herramientas de partición

- **Meredig et al. (2018) [R]:** introduce LOCO-CV y una línea base de vecino más cercano.
  "Traditional machine learning (ML) metrics overestimate model performance for materials
  discovery."
- **MatFold (2025) [R + README v1.3.3]:**
  - Esquemas disponibles: `random`, `structureid`, `composition`, `chemsys`, `sgnum`,
    `pointgroup`, `crystalsys`, `elements`, `periodictablerows` y `periodictablegroups`, con
    CV anidada (`n_inner_splits`, `n_outer_splits`) y LOO de un solo grupo
    (`create_loo_split('elements', 'Fe')`).
  - Exige una estructura de pymatgen por entrada (`bulk_dict`). Licencia MIT; depende de
    numpy, pandas y pymatgen ≥ 2024.

### 4.7 Chung, Walsh y Payne (2025): **lectura prioritaria** [M + R]

- Resumen gráfico: "a human curated dataset on solid-state reactions of ternary metal
  oxides. Applications … include outlier detection of text-mined datasets, synthesis data
  analysis, and synthesizability prediction."
- Según Desai et al., "predicted the synthesizability of ternary oxides using PU learning,
  with labels applied specifically for oxides that can be synthesized via solid-state
  methods", y usa un conjunto de test fijo.
- Aron Walsh es coautor de Gu et al. (2022).
- **Se desconoce cómo particionan.** Es el trabajo más cercano en química (óxidos ternarios
  con PU) y debe leerse antes de redactar la introducción.

## 5. Matriz comparativa

| Trabajo | Materiales | Representación | Evaluación | Extrapolación química sistemática | Perovskitas óxido | τ a igual P(ŷ=1) |
|---|---|---|---|---|---|---|
| Gu 2022 | Perovskitas (todas) | Grafo | 10 % aleatorio, 1 partición, por cristal | No | Subconjunto | No |
| SynCoTrain 2025 | Óxidos | Grafo | 5 % fijo + 10 % dinámico, aleatorios | No | No específico | — |
| Desai 2025 | Perovskitas haluro/calcogenuro | Descriptores | K-fold aleatorio repetido | No | No | — |
| SynthNN 2023 | Inorgánicos | Composición | 90:5:5 aleatorio + temporal | Temporal | No específico | — |
| Jang 2020 | Inorgánicos | Grafo | Aleatorio + temporal | Temporal | No específico | — |
| Zhu 2023 | Inorgánicos | FTCP | Aleatorio + temporal (fecha MP) | Temporal | No específico | — |
| CPUL 2025 | Inorgánicos | Grafo | Aleatorio + todos los compuestos con Fe | 1 elemento | No específico | — |
| Chung 2025 | Óxidos ternarios | ? | ? (test fijo, según Desai) | ? | ? | ? |
| **Este proyecto** | **Perovskitas óxido ABO₃** | Composición y grafo | **Aleatoria, por A, por B, por familia, por clúster y temporal; holdout por grupos** | **Sí** | **Sí** | **Sí (H2)** |

## 6. Delimitación de la brecha (actualizada)

En el conjunto revisado **ya existen**:

- herramientas genéricas de partición fuera de distribución (LOCO-CV, MatFold);
- particiones temporales en sintetizabilidad (Jang 2020, SynthNN, Zhu 2023);
- un caso de un único elemento reservado (CPUL, Fe);
- evidencia de que muchas pruebas OOD interpolan (Li 2025) y de que la fuente de los datos
  sesga los modelos de sintetizabilidad (Davariashtiyani 2024).

**Ningún trabajo revisado**:

1. evalúa la sintetizabilidad PU de **perovskitas óxido** con particiones sistemáticas por
   elemento A, elemento B, familia química y clúster químico;
2. etiqueta **a nivel de composición**, con identificación de la perovskita por conectividad
   y separación de polimorfos no perovskita (Gu et al. etiqueta cristales por coincidencia
   con prototipos);
3. compara con τ **a igual P(ŷ = 1)** y con un margen de no inferioridad preregistrado;
4. controla la **procedencia de la estructura** con una línea base de solo simetría;
5. está **preregistrado**.

**Condición:** si Chung et al. (2025) evalúa óxidos ternarios con particiones por grupos, la
brecha (1) se estrecha a "perovskitas ABO₃ y comparación con τ" y debe citarse como trabajo
más cercano. La contribución sigue siendo una **evaluación**, no un algoritmo nuevo, como
establece el preregistro.

## 7. Implicaciones para el proyecto

> Las propuestas 1 y 3 fueron **aprobadas el 2026-11-13** y quedan preregistradas como
> análisis secundarios S2 y S1 (`docs/preregistro.md` §10b).

1. **Reevaluación de Gu (F3.5).** Además de las particiones por grupos, deduplicar sus
   cristales a nivel de composición y medir cuánto cambia su TPR solo por eso. Esto separa la
   fuga entre polimorfos de la debida a la extrapolación química.
2. **MatFold (F1.7), resultado preliminar: compatibilidad parcial.** Sirve para `random`,
   `chemsys` (= `group_system`) y como verificación cruzada, y aporta `sgnum` para
   diagnósticos de la línea base de simetría. **No** cubre `loeo_A`/`loeo_B` por sitio
   (`elements` reserva el elemento en cualquier sitio, y el O está en todas las entradas),
   `family`, `cluster` ni `temporal`. Propuesta: esquema híbrido. La decisión final queda
   en F1.7.
3. **Análisis descriptivo de "extrapolación real" (por Li et al., 2025).** Para cada grupo
   reservado, reportar la distancia en el espacio MAGPIE a la composición de entrenamiento más
   cercana y relacionarla con la TPR del grupo. Si se aprueba, se añade al preregistro como
   análisis secundario antes de F4.0.
4. **`temporal`.** Si se usa una fecha, debe ser la de la primera entrada en ICSD (año de
   publicación), no la de carga en MP (Zhu 2023). Se verifica en F1.1 si MP la expone.
5. **Introducción del artículo.** Citar a Schlesinger 2026 y Davariashtiyani 2024 para
   motivar P(ŷ = 1 | U) y el control de procedencia, y a Li 2025 para justificar particiones
   más exigentes que las heurísticas.

## 8. Pendientes de la autora

1. **Leer Chung et al. (2025)** a texto completo (prioridad 1) y anotar su protocolo de
   evaluación.
2. Leer a texto completo MatFold, Meredig et al., Bartel et al. (2019) y la versión publicada
   de SynCoTrain (puede diferir de arXiv v1).
3. Verificar Lee y Liu (2003) y Ramaswamy et al. (2016, KM2): no tienen DOI y OpenAlex limitó
   las consultas.
4. Decidir sobre las propuestas 1 y 3 de la sección 7.

*Para que yo pueda leerlos, basta con subir los PDF a la sesión. Desde aquí no puedo
descargarlos de la RSC.*
