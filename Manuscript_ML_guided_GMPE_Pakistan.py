import json
import os
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, train_test_split

import gmpe_ml as G
from build_dataset import build

warnings.filterwarnings("ignore")
os.makedirs("cache", exist_ok=True)
os.makedirs("out", exist_ok=True)
os.makedirs("data", exist_ok=True)

CLASSIC = ["SVM", "KNN", "ANN", "RF", "XGBoost"]
MODELS = CLASSIC + ["PINN", "GEP"]

PINN_CFG = dict(hidden=(10, 10), w=(10.0, 10.0, 5.0, 1.0), n_coll=250, maxiter=600)
PINN_CV_CFG = dict(hidden=(10, 10), w=(10.0, 10.0, 5.0, 1.0), n_coll=200, maxiter=300)
GEP_CFG = dict(population_size=2000, generations=30, tournament_size=20,
               function_set=("add", "sub", "mul", "div", "sqrt", "log", "neg"),
               metric="rmse", parsimony_coefficient=0.0015, p_crossover=0.70,
               p_subtree_mutation=0.10, p_hoist_mutation=0.05, p_point_mutation=0.10,
               const_range=(-5.0, 5.0), init_depth=(2, 5), random_state=G.SEED,
               n_jobs=1, verbose=0)
GEP_CV_CFG = dict(GEP_CFG, population_size=800, generations=15)


def log(*a):
    print(*a, flush=True)


# --------------------------------------------------------------------------- #
#  data (identical in every stage)
# --------------------------------------------------------------------------- #
def load():
    df = G.make_features(build())
    df.to_csv("data/pakistan_databank_features.csv", index=False)
    X = df[G.FEATURES].values
    y = df["lnPGA"].values
    idx_tr, idx_te = train_test_split(np.arange(len(y)), test_size=0.20,
                                      random_state=G.SEED, shuffle=True)
    return df, X, y, idx_tr, idx_te


# --------------------------------------------------------------------------- #
#  stage: tune classical learners
# --------------------------------------------------------------------------- #
GRIDS = {
    "SVM": (G.build_svr(), {"m__C": [1, 5, 10, 30, 100, 300],
                            "m__gamma": [0.05, 0.1, 0.25, 0.5, 1.0],
                            "m__epsilon": [0.02, 0.05, 0.1, 0.2]}),
    "KNN": (G.build_knn(), {"m__n_neighbors": [3, 4, 5, 6, 7, 8, 10, 12],
                            "m__weights": ["uniform", "distance"],
                            "m__p": [1, 2]}),
    "ANN": (G.build_ann(), {"m__hidden_layer_sizes": [(6,), (8,), (10,), (12,), (8, 6), (12, 8)],
                            "m__alpha": [1e-2, 1e-1, 0.5, 1.0, 3.0, 10.0]}),
    "RF": (G.build_rf(), {"n_estimators": [100, 300],
                          "max_depth": [3, 4, 6, 8, None],
                          "min_samples_leaf": [1, 2, 4],
                          "max_features": [1.0, 0.67]}),
    "XGBoost": (G.build_xgb(), {"n_estimators": [100, 300, 600],
                                "max_depth": [2, 3, 4],
                                "learning_rate": [0.03, 0.05, 0.1],
                                "subsample": [0.8, 1.0]}),
}


def stage_tune(names=CLASSIC):
    df, X, y, itr, ite = load()
    store = joblib.load("cache/classic.joblib") if os.path.exists("cache/classic.joblib") else {}
    for n in names:
        log("tuning", n)
        est, grid = GRIDS[n]
        gs = GridSearchCV(est, grid, cv=10, scoring="r2", n_jobs=1).fit(X[itr], y[itr])
        params = {k.replace("m__", ""): (list(v) if isinstance(v, tuple) else v)
                  for k, v in gs.best_params_.items()}
        store[n] = dict(model=gs.best_estimator_, params=params,
                        cv_best=float(gs.best_score_))
        joblib.dump(store, "cache/classic.joblib")
        log("   best", params, "inner-CV R2 = %.3f" % gs.best_score_)


def factory(name, params):
    if name == "SVM":
        return lambda: G.build_svr(**params)
    if name == "KNN":
        return lambda: G.build_knn(k=params["n_neighbors"], weights=params["weights"],
                                   p=params["p"])
    if name == "ANN":
        return lambda: G.build_ann(hidden=tuple(params["hidden_layer_sizes"]),
                                   alpha=params["alpha"])
    if name == "RF":
        return lambda: G.build_rf(**params)
    if name == "XGBoost":
        return lambda: G.build_xgb(**params)
    if name == "PINN":
        return lambda: G.PINN(**PINN_CV_CFG)
    if name == "GEP":
        from gplearn.genetic import SymbolicRegressor
        return lambda: SymbolicRegressor(**GEP_CV_CFG)
    raise KeyError(name)


# --------------------------------------------------------------------------- #
#  stage: PINN / GEP
# --------------------------------------------------------------------------- #
def stage_pinn():
    df, X, y, itr, ite = load()
    log("training PINN", PINN_CFG)
    m = G.PINN(seed=G.SEED, **PINN_CFG).fit(X[itr], y[itr])
    joblib.dump(dict(model=m, params=dict(PINN_CFG, hidden=list(PINN_CFG["hidden"]),
                                          w=list(PINN_CFG["w"]))), "cache/pinn.joblib")
    log("   done; final loss %.4f" % m.history_["loss"][-1])


def stage_gep():
    from gplearn.genetic import SymbolicRegressor
    df, X, y, itr, ite = load()
    log("evolving GEP")
    m = SymbolicRegressor(**GEP_CFG).fit(X[itr], y[itr])
    joblib.dump(dict(model=m, params=dict(population_size=GEP_CFG["population_size"],
                                          generations=GEP_CFG["generations"],
                                          tournament_size=GEP_CFG["tournament_size"],
                                          parsimony_coefficient=GEP_CFG["parsimony_coefficient"],
                                          functions=["+", "-", "x", "/", "sqrt", "ln", "neg"])),
                "cache/gep.joblib")
    log("   program:", m._program)


def get_models():
    store = joblib.load("cache/classic.joblib")
    out = {n: store[n] for n in CLASSIC}
    out["PINN"] = joblib.load("cache/pinn.joblib")
    out["GEP"] = joblib.load("cache/gep.joblib")
    return out


# --------------------------------------------------------------------------- #
#  stage: cross-validation of one model
# --------------------------------------------------------------------------- #
def stage_cv(name):
    df, X, y, itr, ite = load()
    ms = get_models()
    k = 5 if name in ("PINN", "GEP") else 10
    log(f"cross-validating {name} ({k}-fold)")
    cv = G.cv_scores(factory(name, ms[name]["params"]), X, y, k=k)
    cv["k"] = k
    json.dump(cv, open(f"cache/cv_{name}.json", "w"))
    log("  ", cv)


# --------------------------------------------------------------------------- #
#  stage: SHAP -> ML-guided equation -> goodness of fit
# --------------------------------------------------------------------------- #
def stage_final():
    df, X, y, itr, ite = load()
    ms = get_models()
    R = {}
    R["data"] = dict(
        n=int(len(df)), n_events=int(df.groupby(["Date", "Mw"]).ngroups),
        desc={c: dict(mean=float(df[c].mean()), min=float(df[c].min()),
                      max=float(df[c].max()), std=float(df[c].std()),
                      skew=float(df[c].skew()), kurt=float(df[c].kurt()))
              for c in ["Mw", "Repi", "Vs30", "PGA_g", "lnPGA"]})
    R["corr"] = df[["Mw", "lnR", "lnVs", "lnPGA"]].corr().round(3).to_dict()
    R["split"] = dict(n_train=int(len(itr)), n_test=int(len(ite)))

    # ---- scores ---------------------------------------------------------- #
    scores, tuned, pred = {}, {}, {}
    for n in MODELS:
        m = ms[n]["model"]
        scores[n] = G.all_metrics(m, X[itr], y[itr], X[ite], y[ite])
        cvf = f"cache/cv_{n}.json"
        if os.path.exists(cvf):
            scores[n].update(json.load(open(cvf)))
        pred[n] = m.predict(X)
        scores[n]["sigma"] = float(np.std(y - pred[n], ddof=1))
        tuned[n] = ms[n]["params"]
    R["model_scores"], R["tuned"] = scores, tuned
    np.save("out/predictions.npy", np.vstack([pred[k] for k in MODELS]))
    json.dump(MODELS, open("out/model_order.json", "w"))

    gep = ms["GEP"]["model"]
    R["gep_program"] = str(gep._program)
    # closed form of the evolved program (verified against gplearn to 1e-15):
    #   ln PGA = ln Mw - c1 lnR / Mw - c2 sqrt((lnR + lnVs) / Mw)
    Mw_, L_, V_ = X[:, 0], X[:, 1], X[:, 2]
    A_ = np.column_stack([L_ / Mw_, np.sqrt(np.abs(L_ + V_) / Mw_)])
    c_gep = -np.linalg.lstsq(A_, gep.predict(X) - np.log(Mw_), rcond=None)[0]
    R["gep_closed_form"] = dict(c1=float(c_gep[0]), c2=float(c_gep[1]),
                                max_err=float(np.max(np.abs(np.log(Mw_) - A_ @ c_gep
                                                            - gep.predict(X)))))
    R["gep_length"] = int(gep._program.length_)
    R["gep_depth"] = int(gep._program.depth_)
    R["gep_evolution"] = [float(v) for v in np.ravel(gep.run_details_["best_fitness"])]
    R["gep_avg_len"] = [float(v) for v in np.ravel(gep.run_details_["average_length"])]
    R["pinn_history"] = dict(
        loss=[float(v) for v in ms["PINN"]["model"].history_["loss"][::50]],
        data=[float(v) for v in ms["PINN"]["model"].history_["data"][::50]],
        phys=[float(v) for v in ms["PINN"]["model"].history_["phys"][::50]])
    g = ms["PINN"]["model"].gradients(X)
    R["pinn_physics_check"] = dict(
        frac_dM_pos=float(np.mean(g[:, 0] >= 0)),
        frac_dR_neg=float(np.mean(g[:, 1] <= 0)),
        frac_dV_neg=float(np.mean(g[:, 2] <= 0)))
    ann = ms["ANN"]["model"]
    ga = np.column_stack([(ann.predict(X + e) - ann.predict(X - e)) / 0.02
                          for e in np.eye(3) * 0.01])
    R["ann_physics_check"] = dict(
        frac_dM_pos=float(np.mean(ga[:, 0] >= 0)),
        frac_dR_neg=float(np.mean(ga[:, 1] <= 0)),
        frac_dV_neg=float(np.mean(ga[:, 2] <= 0)))

    # tree-model feature importances (for discussion)
    R["rf_importance"] = dict(zip(G.FEATURES, [float(v) for v in
                                               ms["RF"]["model"].feature_importances_]))
    R["xgb_importance"] = dict(zip(G.FEATURES, [float(v) for v in
                                                ms["XGBoost"]["model"].feature_importances_]))

    # ---- exact SHAP of the best model ----------------------------------- #
    best = max(scores, key=lambda k: scores[k]["R2_Test"])
    R["best_model"] = best
    log("best model:", best)
    phi, base = G.exact_shap(ms[best]["model"].predict, X, X, n_bg=120)
    R["shap_base"] = float(base)
    R["shap_mean_abs"] = {f: float(np.mean(np.abs(phi[:, i]))) for i, f in enumerate(G.FEATURES)}
    R["shap_additivity_error"] = float(np.max(np.abs(phi.sum(1) + base - pred[best])))
    np.save("out/shap_values.npy", phi)
    pd.DataFrame(phi, columns=[f"SHAP_{f}" for f in G.FEATURES]).assign(
        Mw=df.Mw.values, Repi=df.Repi.values, Vs30=df.Vs30.values).to_csv(
        "out/shap_values.csv", index=False)
    # SHAP of every model (mean |phi|) for the cross-model comparison
    R["shap_all_models"] = {}
    for n in MODELS:
        ph, _ = G.exact_shap(ms[n]["model"].predict, X, X, n_bg=60)
        R["shap_all_models"][n] = {f: float(np.mean(np.abs(ph[:, i])))
                                   for i, f in enumerate(G.FEATURES)}

    # ---- nonlinear SHAP-function fitting --------------------------------- #
    # unconstrained fit (reported for transparency: parameters are non-identifiable)
    _pu, r2_mu = G.fit_family(G.f_logistic, df.Mw.values, phi[:, 0], p0=[3.0, -3.0, 5.8, 0.6])
    # identifiable fit: inflection constrained to the moderate-magnitude range 5-7
    p_m, r2_m = G.fit_family(G.f_logistic, df.Mw.values, phi[:, 0], p0=[3.0, -3.0, 5.8, 0.8],
                             bounds=([-10, -10, 5.0, 0.3], [10, 10, 7.0, 2.5]))
    p_r, r2_r = G.fit_family(G.f_atten, df.Repi.values, phi[:, 1], p0=[1.0, 0.002, 4.0],
                             bounds=([0.0, 0.0, -50.0], [5.0, 0.05, 50.0]))
    p_v, r2_v = G.fit_family(G.f_site, df.Vs30.values, phi[:, 2], p0=[0.3, 0.0])
    R["shap_fits"] = {
        "Mw": dict(family="logistic (saturation)", params=[float(v) for v in p_m], R2=float(r2_m)),
        "R": dict(family="attenuation (geometric + anelastic)", params=[float(v) for v in p_r], R2=float(r2_r)),
        "Vs30": dict(family="logarithmic site", params=[float(v) for v in p_v], R2=float(r2_v))}
    # alternative families (justification of the choice)
    alt = {}
    for key, fn, x, col, p0 in [
        ("Mw_linear", lambda x, a, b: a * x + b, df.Mw.values, 0, [1, 0]),
        ("Mw_quadratic", lambda x, a, b, c: a * x ** 2 + b * x + c, df.Mw.values, 0, [0, 1, 0]),
        ("R_expdecay", G.f_expdecay, df.Repi.values, 1, [3.0, 80.0, -2.0]),
        ("R_loglinear", lambda x, a, b: a - b * np.log(x), df.Repi.values, 1, [3, 1]),
        ("Vs30_linear", lambda x, a, b: a + b * x, df.Vs30.values, 2, [0, 0]),
    ]:
        try:
            _, rr = G.fit_family(fn, x, phi[:, col], p0=p0)
            alt[key] = float(rr)
        except Exception:
            pass
    alt["Mw_logistic_unconstrained"] = float(r2_mu)
    alt["Mw_logistic_unconstrained_params"] = [float(v) for v in _pu]
    R["shap_fit_alternatives"] = alt

    # ---- ML-guided code-format equation ---------------------------------- #
    F = np.column_stack([G.f_logistic(df.Mw.values, *p_m),
                         G.f_atten(df.Repi.values, *p_r),
                         G.f_site(df.Vs30.values, *p_v)])
    mlr = LinearRegression().fit(F, y)
    k0 = float(mlr.intercept_)
    k1, k2, k3 = [float(v) for v in mlr.coef_]
    ml_pred = mlr.predict(F)
    mlr_tr = LinearRegression().fit(F[itr], y[itr])
    R["mlguided"] = dict(
        k0=k0, k1=k1, k2=k2, k3=k3,
        f_Mw=[float(v) for v in p_m], f_R=[float(v) for v in p_r], f_Vs=[float(v) for v in p_v],
        metrics_ln=G.metrics(y, ml_pred), metrics_g=G.metrics(np.exp(y), np.exp(ml_pred)),
        sigma=float(np.std(y - ml_pred, ddof=1)),
        mean_res=float(np.mean(y - ml_pred)),
        holdout=dict(train=G.metrics(y[itr], mlr_tr.predict(F[itr])),
                     test=G.metrics(y[ite], mlr_tr.predict(F[ite])),
                     k=[float(mlr_tr.intercept_)] + [float(v) for v in mlr_tr.coef_]))

    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import SplineTransformer
    sp = make_pipeline(SplineTransformer(n_knots=6, degree=3), LinearRegression()).fit(X, y)
    R["additive_ceiling_R2"] = float(G.metrics(y, sp.predict(X))["R2"])

    # ---- SH12 benchmark --------------------------------------------------- #
    ln_sh = G.sh12(df.Mw.values, df.Repi.values)
    R["sh12"] = dict(metrics_ln=G.metrics(y, ln_sh), metrics_g=G.metrics(np.exp(y), np.exp(ln_sh)),
                     sigma=G.SH12_SIGMA, mean_res=float(np.mean(y - ln_sh)),
                     sigma_empirical=float(np.std(y - ln_sh, ddof=1)))
    np.save("out/mlguided_pred.npy", ml_pred)
    np.save("out/sh12_pred.npy", ln_sh)

    # ---- LH / LLH / EDR --------------------------------------------------- #
    entries = dict(pred)
    entries["ML-guided Eq."] = ml_pred
    entries["SH12"] = ln_sh
    sig = {n: scores[n]["sigma"] for n in MODELS}
    sig["ML-guided Eq."] = R["mlguided"]["sigma"]
    sig["SH12"] = G.SH12_SIGMA
    gof = {}
    for n, p in entries.items():
        res = y - p
        d = G.lh_method(res, sig[n])
        d["LLH"] = G.llh_method(res, sig[n])
        d.update(G.edr_method(y, p, sig[n]))
        d["sigma"] = float(sig[n])
        d["mean_res"] = float(np.mean(res))
        d.update(G.metrics(y, p))
        gof[n] = d
    for key in ("LLH", "EDR"):
        for i, n in enumerate(sorted(gof, key=lambda k: gof[k][key])):
            gof[n][key + "_rank"] = i + 1
    R["gof"] = gof

    R["waseem_ranking"] = {
        "AK14": dict(MEDLH=0.76, LLH=1.63, EDR=0.75, grade="A"),
        "AB10": dict(MEDLH=0.06, LLH=4.79, EDR=0.96, grade="D"),
        "BA14": dict(MEDLH=0.34, LLH=1.88, EDR=1.31, grade="D"),
        "BI14": dict(MEDLH=0.13, LLH=10.96, EDR=0.95, grade="D"),
        "CY14": dict(MEDLH=0.05, LLH=4.39, EDR=2.25, grade="D"),
        "CB14": dict(MEDLH=0.41, LLH=1.67, EDR=1.12, grade="D"),
        "CZ15": dict(MEDLH=0.57, LLH=1.42, EDR=0.57, grade="B"),
        "GK15": dict(MEDLH=0.55, LLH=2.59, EDR=1.12, grade="C"),
        "ID14": dict(MEDLH=0.36, LLH=1.78, EDR=1.34, grade="C"),
        "KAN06": dict(MEDLH=0.16, LLH=13.19, EDR=0.94, grade="D"),
        "RK14": dict(MEDLH=0.56, LLH=4.05, EDR=1.35, grade="C"),
        "SH12": dict(MEDLH=0.74, LLH=1.68, EDR=1.77, grade="A"),
        "ZF18": dict(MEDLH=0.11, LLH=14.56, EDR=1.23, grade="D")}

    # ---- parametric study -------------------------------------------------- #
    def mlg(Mw, Repi, Vs):
        return (k0 + k1 * G.f_logistic(Mw, *p_m) + k2 * G.f_atten(Repi, *p_r)
                + k3 * G.f_site(Vs, *p_v))
    base_case = dict(Mw=5.5, Repi=50.0, Vs30=560.0)
    R["parametric"] = dict(base=base_case)
    pts = {}
    for M in [4.5, 5.5, 6.5, 7.5]:
        pts[str(M)] = dict(R10=float(np.exp(mlg(M, 10.0, 760.0))),
                           R50=float(np.exp(mlg(M, 50.0, 760.0))),
                           R100=float(np.exp(mlg(M, 100.0, 760.0))),
                           R200=float(np.exp(mlg(M, 200.0, 760.0))),
                           sh12_R50=float(np.exp(G.sh12(M, 50.0))))
    R["scenario_table"] = pts
    R["site_amp_310_760"] = float(np.exp(mlg(5.5, 50, 310) - mlg(5.5, 50, 760)))

    # ---- by-magnitude-bin residual statistics ----------------------------- #
    bins = [(4.0, 5.0), (5.0, 5.5), (5.5, 6.5), (6.5, 8.0)]
    rb = {}
    for lo, hi in bins:
        m = (df.Mw >= lo) & (df.Mw < hi)
        rb[f"{lo}-{hi}"] = dict(n=int(m.sum()),
                                ml=float(np.mean((y - ml_pred)[m])),
                                sh=float(np.mean((y - ln_sh)[m])))
    R["res_by_mag"] = rb

    with open("out/results.json", "w") as fh:
        json.dump(R, fh, indent=1, default=float)
    pd.DataFrame(scores).T.to_csv("out/model_scores.csv")
    pd.DataFrame(gof).T.to_csv("out/gof.csv")

    log(pd.DataFrame(scores).T[["R2_Train", "R2_Test", "RMSE_Test"] +
                               (["R2_CV"] if "R2_CV" in scores[MODELS[0]] else [])].round(3))
    log("SHAP mean|phi|:", {k: round(v, 3) for k, v in R["shap_mean_abs"].items()})
    log("fits R2:", {k: round(v["R2"], 3) for k, v in R["shap_fits"].items()})
    log("k:", [round(v, 3) for v in (k0, k1, k2, k3)])
    log("ML-guided:", {k: round(v, 3) for k, v in R["mlguided"]["metrics_ln"].items()},
        "sigma %.3f" % R["mlguided"]["sigma"])
    log("SH12:", {k: round(v, 3) for k, v in R["sh12"]["metrics_ln"].items()})
    log(pd.DataFrame(gof).T[["MEDLH", "Grade", "LLH", "EDR", "R2"]].round(3))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd == "tune":
        stage_tune(sys.argv[2:] or CLASSIC)
    elif cmd == "pinn":
        stage_pinn()
    elif cmd == "gep":
        stage_gep()
    elif cmd == "cv":
        for n in sys.argv[2:] or MODELS:
            stage_cv(n)
    elif cmd == "final":
        stage_final()
    elif cmd == "all":
        stage_tune(); stage_pinn(); stage_gep()
        for n in MODELS:
            stage_cv(n)
        stage_final()
