# ML-Guided Interpretable Code-Format GMPE for Shallow Active Regions of Pakistan

Reproduction package for the manuscript

> *Machine-Learning-Guided Interpretable Code-Format Ground-Motion Prediction Equation with
> Magnitude-Dependent Site Response for Shallow Active Regions of Pakistan*

## Final equation (structure E5c)

```
ln(PGA[g]) = -5.3068 + 0.9024 f(Mw) + 1.0024 f(R) + S(Mw) f(Vs30)

S(Mw)     = max[0, 0.9084 - 0.9628 (Mw - 5.5)]
f(Mw)     = -4.572 + 7.641 / (1 + exp(-(Mw - 5.00)/1.090))
f(R)      = 5.243 - 1.169 ln(Rhyp),   Rhyp = sqrt(Repi^2 + 10^2)
f(Vs30)   = -0.414 - 0.882 ln(Vs30/760)

sigma = 0.846 (natural-log units)
Valid for 4.1 <= Mw <= 7.8, 8 <= Repi <= 470 km, 310 <= Vs30 <= 900 m/s
```

## Contents

| Folder / file | Description |
|---|---|
| `Manuscript_ML_guided_GMPE_Pakistan.docx` | Full manuscript: A4, template styles, native Word (OMML) equations, EMF figures |
| `figures/svg/` | 21 vector figures, numbered as in the manuscript |
| `figures/emf/` | The same figures as Enhanced Metafiles (as embedded in the DOCX) |
| `references.bib` | BibTeX database of the 57 cited references, in citation order |
| `ML_guided_GMPE_Pakistan.ipynb` | Annotated notebook (markdown + LaTeX), executed, with outputs |
| `code/` | Python source (see below) |
| `data/` | Digitised databank (147 records) and engineered features |
| `results/` | `results.json` and CSV tables behind every number in the paper |

## Code

| Script | Role |
|---|---|
| `build_dataset.py` | Digitised Table A1 of Waseem et al. (2022) |
| `gmpe_ml.py` | Core library: features, 7 learners, PINN, exact SHAP & interaction indices, function families, LH/LLH/EDR, SH12, final-equation predictor |
| `run_analysis.py` | Staged, cached pipeline (`tune`, `pinn`, `gep`, `cv`, `final`) |
| `equation_search.py` | Interaction analysis, 12 candidate structures, 1-SE + VIF + admissibility selection |
| `conventional_gmm.py` | Five conventional regression-based GMMs (OLS and mixed effects) vs E5c and PINN |
| `nested_e5c.py` | Fully nested validation of the ML-guided pipeline (PINN -> SHAP -> fit -> E5c in every fold) |
| `sota_benchmark.py` | Supplementary: LightGBM, CatBoost, Gaussian process, stacking, PINN ensemble |
| `make_figures.py` | All figures (SVG + PNG previews), opaque colours for EMF compatibility |
| `omml.py`, `docx_helpers.py`, `refs.py`, `build_manuscript.py` | Manuscript builder |
| `make_notebook.py` | Regenerates the notebook |

## Reproduce

```bash
pip install -r code/requirements.txt
cd code
python run_analysis.py tune && python run_analysis.py pinn && python run_analysis.py gep
python run_analysis.py cv && python run_analysis.py final
python equation_search.py
python conventional_gmm.py
python nested_e5c.py
python make_figures.py
python build_manuscript.py
```

The notebook has a `FAST` switch (default `True`) that uses reduced grids so it runs in a few minutes;
its numbers are therefore close to, but not identical with, the paper. `FAST = False` or the scripts
above reproduce the paper exactly (seed 42).
