# Source attribution and distribution notes

| Component | Provider/source | Package treatment |
|---|---|---|
| TerraClass Amazon land-use vectors, 2020/2024 | INPE and Embrapa; https://www.terraclass.gov.br/download-de-dados | URLs/hashes and derived chip labels supplied; large state archives omitted. Cite product providers and Almeida et al. (2016). |
| Dated validated clearing alerts | MapBiomas Alerta RAD 2024; source archive URL in manifest | Alert identifiers, areas and derived references supplied; national raw archive omitted. Cite MapBiomas Alerta. |
| Sentinel-1 and Sentinel-2 | Copernicus/ESA; Google Earth Engine collections | Cached derived features and displayed composites; collection identifiers and acquisition hashes retained. |
| TerraFM | Danish et al., arXiv:2506.06281; https://github.com/mbzuai-oryx/TerraFM (MIT) | Backbone used to compute the cached features. The fine-tuned checkpoint and inference code are released separately as forest-to-agriculture-inference. |
| BigEarthNet / BigEarthNet-MM | Sumbul et al. (2019, 2021) | Classification nomenclature, preprocessing and adaptation provenance identified; original dataset omitted. |
| IBGE Biomas 1:250,000 (2019) | https://geoftp.ibge.gov.br/informacoes_ambientais/estudos_ambientais/biomas/vetores/Biomas_250mil.zip | Source vector archive supplied for the post hoc biome-boundary check. |
| Benchmark code, references, predictions and fitted models | AI4DIS project (the four authors) | Code under MIT (`LICENSE`); derived data and fitted models under CC BY-SA 4.0 (`LICENSE-DATA`). |

The official INPE TerraClass project page
https://www.gov.br/inpe/pt-br/area-conhecimento/unidade-amazonia/projetos-e-pesquisas/terraclass
identifies Creative Commons Attribution-ShareAlike 3.0 Unported for the project;
MapBiomas Alerta's FAQ and platform terms also identify attribution/share-alike
conditions. The reference labels, label grids and every table derived from them
are therefore released under CC BY-SA 4.0 (a compatible later version). Sentinel
features and composites contain modified Copernicus Sentinel data (2020, 2024).

An upstream data licence does not license project code, and the project licences do
not replace the terms of the original sources. Users redistributing the raw source
archives should obtain them from their providers and follow those terms. This
document records attributions; it does not claim ownership of source datasets.
