# F5.3 (borrador para la tesis): lixiviación del sitio A y precipitación con fosfatos y carbonatos

> Borrador de discusión, exploratorio (OE4, solo tesis). Datos: `results/f5_3_lixiviacion/`
> (informe, `summary.csv`, `front.csv`), a partir de F5.1 (`results/f5_1_pourbaix/`) y F5.2
> (`results/f5_2_pareto/`). Todas las cifras salen de esos archivos; las referencias están al
> final. Pendiente de revisión por la autora.

## 1. Qué mide el análisis

El diagrama de Pourbaix de cada sistema A–B–O–H (GGA/GGA+U de Materials Project, con
`MaterialsProjectAqueousCompatibility`) da las fases que son estables en equilibrio a cada pH y
potencial [Patel et al., 2019]. ΔG_pbx es la energía libre de la perovskita por encima de esa
combinación estable [Singh et al., 2017]. Aquí se leen además **qué productos** forman esa
combinación en la ventana fisiológica (−0,2 a +0,4 V frente a SHE): si el metal de un sitio
aparece en una especie disuelta, el sitio "se lixivia" en sentido termodinámico.

Es una lectura de equilibrio, no una cinética. Una ΔG_pbx positiva no implica que el material se
disuelva a una velocidad apreciable: Singh et al. (2017) muestran que materiales con ΔG_pbx de
hasta ~0,5 eV/átomo pueden persistir, a menudo porque forman una capa pasivante.

## 2. Resultados

**La lixiviación del sitio A domina.** A pH 7,4 y 10⁻⁶ M, el 90 % de las perovskitas conocidas
(P) liberan el catión A: el 55 % solo A y el 35 % A y B. Solo el 2 % liberan únicamente B, y el
9 % no liberan ninguno. Las no etiquetadas (U) siguen el mismo patrón, con más casos de
liberación conjunta (44 %). A pH 5 (lisosomal), la liberación conjunta pasa a ser mayoritaria:
65 % en P y 66 % en U.

| Grupo (pH 7,4; 10⁻⁶ M) | n | Solo A | Solo B | A y B | Ninguno |
|---|---|---|---|---|---|
| P | 192 | 55 % | 2 % | 35 % | 9 % |
| U | 539 | 38 % | 8 % | 44 % | 10 % |
| U elegibles en F5.2 | 390 | 41 % | 6 % | 41 % | 12 % |
| Frente de F5.2 | 5 | 3 | 0 | 1 | 1 |

Es lo esperable para óxidos ABO₃: el sitio B suele ser un catión de carga alta (Ti⁴⁺, Hf⁴⁺,
Al³⁺, Ga³⁺) cuyo óxido o hidróxido es poco soluble a pH neutro, mientras que el sitio A es un
catión grande y de baja carga (alcalinotérreos, alcalinos, lantánidos) con una forma acuosa
estable. SrTiO₃ es el caso de referencia: se descompone en TiO₂ + Sr²⁺ (ΔG_pbx = 0,16
eV/átomo), y la liberación de Sr²⁺ desde SrTiO₃ sinterizado en tampón fosfato a 37 °C está
documentada experimentalmente [Sahoo et al., 2019].

**Frente de Pareto de F5.2** (pH 7,4; 10⁻⁶ M):

| Candidata | ΔG_pbx (eV/átomo) | Se libera | Productos sólidos |
|---|---|---|---|
| CaHfO₃ | 0,20 | Ca²⁺ | HfO₂ |
| ErAlO₃ | 0,04 | Er³⁺ | Er₃Al₅O₁₂ |
| EuVO₃ | 0,21 | Eu³⁺ y VO₃⁻ | V₃O₅, VO₂ |
| LuAlO₃ | 0,02 | ninguno | Lu₂O₃, Lu₃Al₅O₁₂ |
| TbGaO₃ | 0,07 | Tb³⁺ | Tb₃Ga₅O₁₂ |

Cuatro de las cinco liberan un catión del sitio A. En ErAlO₃ y TbGaO₃ el producto sólido es un
granate Ln₃B₅O₁₂, de modo que la descomposición sería parcial: un sólido rico en B más el
lantánido en disolución. LuAlO₃ es la única que no libera nada: se descompone por completo en
sólidos (Lu₂O₃ + Lu₃Al₅O₁₂), con ΔG_pbx de solo 0,02 eV/átomo.

## 3. Precipitación con fosfatos y carbonatos

El diagrama solo contiene los elementos A, B, O y H. En los fluidos fisiológicos hay además
fosfato (~1 mM en plasma) y carbonato/bicarbonato (~25 mM), que pueden **retirar de la
disolución** los cationes liberados y cambiar el balance de la descomposición. Esto no está en
ΔG_pbx y es la principal limitación química del análisis. Por grupos:

- **Lantánidos (Eu³⁺, Er³⁺, Tb³⁺; 3 de las 5 candidatas).** Los fosfatos de lantánido (LnPO₄)
  son muy poco solubles. En óxidos de tierras raras, la liberación de iones en el lisosoma
  ácido lleva a la complejación con fosfato biológico, a la transformación de la partícula y a
  daño lisosomal, con efectos pro-fibróticos [Li et al., 2014]; en nanopartículas de
  conversión ascendente se observa depósito de REPO₄ en la superficie en fluido lisosomal
  simulado [Li et al., 2015]. En medios con fosfato, el La³⁺ puede quedar sobresaturado durante
  días, pero acaba precipitando como LaPO₄ [Kastenhofer et al., 2024]. Para las candidatas con
  lantánidos, la lixiviación del sitio A tendería a acoplarse con una **precipitación de LnPO₄**,
  lo que puede reducir el ion libre pero también es el mecanismo de toxicidad descrito. El
  pretratamiento con fosfato a pH neutro evita esa biotransformación [Li et al., 2014], una
  posible estrategia de diseño.
- **Alcalinotérreos (Ca²⁺, Sr²⁺; CaHfO₃ y SrTiO₃).** El Sr²⁺ liberado desde vidrios bioactivos
  sustituye al Ca²⁺ y participa en la formación de una capa de apatita en fluido corporal
  simulado [Lao et al., 2009; Fredholm et al., 2011]. En este caso la lixiviación del sitio A
  no es necesariamente un defecto: el Sr²⁺ estimula la formación de hueso [Fredholm et al.,
  2011], y SrTiO₃ se ha propuesto como biomaterial por eso [Sahoo et al., 2019]. La
  precipitación de carbonatos (CaCO₃, SrCO₃) es plausible a las concentraciones de bicarbonato
  del plasma, pero no se ha cuantificado aquí *(falta una referencia específica)*.
- **Vanadato (EuVO₃).** EuVO₃ es la única del frente que libera también el sitio B, como VO₃⁻
  (V⁵⁺). El vanadato es un análogo estructural del fosfato y compite con él en quinasas y
  fosfatasas [Sharfalddin et al., 2022]; los compuestos inorgánicos de vanadio son citotóxicos,
  genotóxicos e inmunotóxicos in vitro [Zwolak, 2013], aunque sus efectos dependen mucho de la
  especie, la dosis y el medio [Treviño et al., 2018]. El vanadio no estaba en el filtro de
  toxicidad preregistrado (Pb, Ba²⁺, Cr⁶⁺, Co, Ni, Cd, Tl); para un uso biomédico, EuVO₃
  debería marcarse o, al menos, discutirse aparte.

## 4. Limitaciones

1. **Equilibrio, no cinética**: no hay velocidades de disolución ni capas pasivantes
   [Singh et al., 2017].
2. **Circularidad con el hull**: ΔG_pbx sigue a la E_hull de la propia entrada (Spearman 0,83),
   y la diferencia entre P y U arrastra la circularidad del hull de MP (bitácora del
   2026-11-20).
3. **Sin fosfato ni carbonato** en los diagramas (sección 3). La opción "hull extendido con
   fosfatos y carbonatos" del plan (F5.3) cuantificaría este efecto, pero requiere otra
   descarga de MP.
4. **Datos de iones** de MPContribs (`ion_ref_data`), con energías experimentales de distintas
   fuentes; **una sola funcional** (GGA/GGA+U), aunque r2SCAN mejora las energías de los
   diagramas de Pourbaix [Wang et al., 2020].
5. **Concentraciones fijas** (10⁻⁶ y 10⁻⁸ M) y ventana de potencial aproximada; el medio
   biológico real (proteínas, complejantes) no se modela.

## Referencias

- Fredholm, Y. C. et al. (2011). [Influence of strontium for calcium substitution in bioactive glasses on degradation, ion release and apatite formation](https://consensus.app/papers/details/w:9ee235a2-78f8-4fa7-a268-644b6cb20f18/?utm_source=claude_desktop). *J. R. Soc. Interface*.
- Kastenhofer, J. et al. (2024). [Roles of pH and phosphate in rare earth element biosorption with living acidophilic microalgae](https://consensus.app/papers/details/w:0becc755-a6ad-4d89-80cd-87175687f8ea/?utm_source=claude_desktop). *Appl. Microbiol. Biotechnol.*
- Lao, J. et al. (2009). [New strontium-based bioactive glasses: physicochemical reactivity and delivering capability of biologically active dissolution products](https://consensus.app/papers/details/w:c85d1a79-c656-4801-a5d2-f00eb0c7c3bd/?utm_source=claude_desktop). *J. Mater. Chem.*
- Li, R. et al. (2014). [Surface Interactions with Compartmentalized Cellular Phosphates Explain Rare Earth Oxide Nanoparticle Hazard and Provide Opportunities for Safer Design](https://consensus.app/papers/details/w:85a238bd-139a-4f40-94ab-a0d75b7d19bf/?utm_source=claude_desktop). *ACS Nano*.
- Li, R. et al. (2015). [Enhancing the Imaging and Biosafety of Upconversion Nanoparticles through Phosphonate Coating](https://consensus.app/papers/details/w:0bbcc293-a6ff-432a-bd1a-412e1dedef5f/?utm_source=claude_desktop). *ACS Nano*.
- Patel, A. M. et al. (2019). [Efficient Pourbaix diagrams of many-element compounds](https://consensus.app/papers/details/w:894ea555-b3eb-4ca1-9e76-2e3f96635615/?utm_source=claude_desktop). *Phys. Chem. Chem. Phys.*
- Sahoo, S. et al. (2019). [Synthesis, characterization and in vitro biocompatibility study of strontium titanate ceramic: A potential biomaterial](https://consensus.app/papers/details/w:d8b640ba-f437-4916-a2ee-bc4771be73a8/?utm_source=claude_desktop). *J. Mech. Behav. Biomed. Mater.*
- Sharfalddin, A. A. et al. (2022). [Therapeutic Properties of Vanadium Complexes](https://consensus.app/papers/details/w:659122ca-61bb-482c-b7f4-004025ec8b84/?utm_source=claude_desktop). *Inorganics*.
- Singh, A. K. et al. (2017). [Electrochemical Stability of Metastable Materials](https://consensus.app/papers/details/w:ccb3d78c-8740-410f-87ab-1eec4a68a2e6/?utm_source=claude_desktop). *Chem. Mater.*
- Treviño, S. et al. (2018). [Vanadium in Biological Action: Chemical, Pharmacological Aspects, and Metabolic Implications in Diabetes Mellitus](https://consensus.app/papers/details/w:83f03edb-0eb9-4ef1-b825-ea1da2d4c8f3/?utm_source=claude_desktop). *Biol. Trace Elem. Res.*
- Wang, Z. et al. (2020). [Predicting aqueous stability of solid with computed Pourbaix diagram using SCAN functional](https://consensus.app/papers/details/w:ecbb82ae-509e-4d79-8423-b2f84aeddfab/?utm_source=claude_desktop). *npj Comput. Mater.*
- Zwolak, I. (2013). [Vanadium carcinogenic, immunotoxic and neurotoxic effects: a review of in vitro studies](https://consensus.app/papers/details/w:19530bfd-a532-424d-831f-81d616e33f52/?utm_source=claude_desktop). *Toxicol. Mech. Methods*.
