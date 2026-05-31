"""
Monitoring: data drift + performance degradation detection.
Outputs:
  - reports/drift_report.json
  - reports/performance_report.html
  - artifacts/metrics/monitoring_summary.json
  - retrain_needed.txt  (true/false — read by GitHub Actions)
"""
import os
import sys
import json
import logging
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import yaml
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf

# ── Logging ───────────────────────────────────────────────────────────────────
os.makedirs("logs", exist_ok=True)
os.makedirs("reports", exist_ok=True)
os.makedirs("artifacts/metrics", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/monitoring.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger("monitor")

# ── Load params ───────────────────────────────────────────────────────────────
with open("params.yaml") as f:
    params = yaml.safe_load(f)

PERF_THRESHOLD  = params["monitor"]["performance_threshold"]  # 0.15
DRIFT_THRESHOLD = params["monitor"]["drift_threshold"]        # 0.05


def load_artifacts():
    """Load model, scaler, feature columns, and reference data."""
    required = [
        "models/model.keras",
        "artifacts/data/X_train_cnn.npy",
        "artifacts/data/y_train.npy",
        "artifacts/data/X_test_cnn.npy",
        "artifacts/data/y_test.npy",
        "artifacts/preprocessing/freq_maps.json",
        "artifacts/preprocessing/feature_columns.json",
    ]
    missing = [p for p in required if not os.path.exists(p)]
    if missing:
        logger.error(f"Missing files: {missing}")
        logger.error("Run model.py first.")
        return None

    def r2_metric(y_true, y_pred):
        SS_res = tf.reduce_sum(tf.square(y_true - y_pred))
        SS_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
        return 1 - SS_res / (SS_tot + tf.keras.backend.epsilon())

    model = tf.keras.models.load_model("models/model.keras",
                                       custom_objects={"r2_metric": r2_metric})

    with open("artifacts/preprocessing/freq_maps.json") as f:
        freq_maps = json.load(f)
    with open("artifacts/preprocessing/feature_columns.json") as f:
        feature_columns = json.load(f)

    data = {
        "model":           model,
        "X_train":         np.load("artifacts/data/X_train_cnn.npy"),
        "y_train":         np.load("artifacts/data/y_train.npy"),
        "X_test":          np.load("artifacts/data/X_test_cnn.npy"),
        "y_test":          np.load("artifacts/data/y_test.npy"),
        "freq_maps":       freq_maps,
        "feature_columns": feature_columns,
    }
    logger.info(f"Loaded model and artifacts. Train: {data['X_train'].shape}")
    return data


def load_new_data(artifacts):
    """Load and preprocess new_data.csv if it exists and has a y column."""
    new_data_path = "data/new_data.csv"
    if not os.path.exists(new_data_path):
        logger.warning("No new_data.csv found — using test split as reference for drift.")
        return None, None

    df = pd.read_csv(new_data_path)
    if "y" not in df.columns:
        logger.warning("new_data.csv has no 'y' column — cannot assess performance on new data.")
        return None, None

    logger.info(f"Loaded new_data.csv: {df.shape}")
    y_new = df["y"].values
    df = df.drop("y", axis=1)
    if "ID" in df.columns:
        df = df.drop("ID", axis=1)

    # Frequency-encode categorical columns
    freq_maps = artifacts["freq_maps"]
    cat_cols  = [c for c in df.columns if df[c].dtype == "O"]
    for col in cat_cols:
        if col in freq_maps:
            df[f"{col}_freq"] = df[col].map(freq_maps[col]).fillna(0)
        else:
            df[f"{col}_freq"] = 0
    df = df.drop(cat_cols, axis=1)

    # Align columns
    feature_columns = artifacts["feature_columns"]
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0.0
    df = df[feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0)

    # Scale
    if os.path.exists("artifacts/preprocessing/scaler.pkl"):
        import joblib
        scaler = joblib.load("artifacts/preprocessing/scaler.pkl")
        X_new  = scaler.transform(df.values)
    else:
        X_new = df.values

    X_new_cnn = X_new.reshape(X_new.shape[0], X_new.shape[1], 1)
    return X_new_cnn, y_new


def detect_drift(X_ref, X_new, feature_names, threshold):
    """
    KS test on each feature between reference (training) and new data.
    X_ref / X_new shape: (n_samples, n_features, 1)
    Returns dict of drift results per feature + overall summary.
    """
    # Flatten from (n, features, 1) to (n, features)
    ref_flat = X_ref[:, :, 0]
    new_flat = X_new[:, :, 0]

    n_features = ref_flat.shape[1]
    drifted    = []
    results    = {}

    for i in range(n_features):
        ks_stat, p_val = stats.ks_2samp(ref_flat[:, i], new_flat[:, i])
        drifted_flag   = bool(p_val < threshold)
        feat_name      = feature_names[i] if i < len(feature_names) else f"feature_{i}"
        results[feat_name] = {
            "ks_statistic": float(ks_stat),
            "p_value":      float(p_val),
            "drift_detected": drifted_flag
        }
        if drifted_flag:
            drifted.append(feat_name)

    drift_ratio = len(drifted) / n_features if n_features > 0 else 0
    return {
        "features":         results,
        "drifted_features": drifted,
        "n_drifted":        len(drifted),
        "n_total":          n_features,
        "drift_ratio":      float(drift_ratio),
        "overall_drift":    drift_ratio > 0.2   # >20% features drifted
    }


def check_performance(model, X_ref, y_ref, X_new, y_new, threshold):
    """Compare R² on reference vs new data. Flag if degradation > threshold."""
    y_ref_pred = model.predict(X_ref, verbose=0).flatten()
    y_new_pred = model.predict(X_new, verbose=0).flatten()

    ref_r2 = float(r2_score(y_ref, y_ref_pred))
    new_r2 = float(r2_score(y_new, y_new_pred))

    perf_change = (new_r2 - ref_r2) / max(abs(ref_r2), 1e-8)
    degraded    = perf_change < -threshold

    return {
        "reference_r2":       ref_r2,
        "reference_mse":      float(mean_squared_error(y_ref, y_ref_pred)),
        "reference_mae":      float(mean_absolute_error(y_ref, y_ref_pred)),
        "new_data_r2":        new_r2,
        "new_data_mse":       float(mean_squared_error(y_new, y_new_pred)),
        "new_data_mae":       float(mean_absolute_error(y_new, y_new_pred)),
        "performance_change": float(perf_change),
        "threshold":          threshold,
        "degraded":           degraded,
    }


def generate_html_report(drift_result, perf_result, retrain_needed, timestamp):
    """Write a simple HTML performance report."""
    drift_color = "red" if drift_result.get("overall_drift") else "green"
    perf_color  = "red" if perf_result and perf_result.get("degraded") else "green"

    html = f"""<!DOCTYPE html>
<html>
<head><title>Monitoring Report — {timestamp}</title>
<style>
  body {{ font-family: Arial, sans-serif; margin: 30px; }}
  h1   {{ color: #333; }}
  .ok  {{ color: green; font-weight: bold; }}
  .bad {{ color: red;   font-weight: bold; }}
  table {{ border-collapse: collapse; width: 60%; }}
  th, td {{ border: 1px solid #ccc; padding: 8px 12px; text-align: left; }}
  th {{ background: #f0f0f0; }}
</style></head>
<body>
<h1>MLOps Monitoring Report</h1>
<p><b>Generated:</b> {timestamp}</p>
<hr>
<h2>Summary</h2>
<p>Retrain needed:
  <span class="{'bad' if retrain_needed else 'ok'}">
    {'YES' if retrain_needed else 'NO'}
  </span>
</p>

<h2>Data Drift (KS Test)</h2>
<p>Features checked: {drift_result['n_total']}</p>
<p>Features drifted: <span class="{'bad' if drift_result['n_drifted'] > 0 else 'ok'}">{drift_result['n_drifted']}</span></p>
<p>Drift ratio: {drift_result['drift_ratio']:.1%}</p>
<p>Overall drift: <span class="{'bad' if drift_result['overall_drift'] else 'ok'}">{drift_result['overall_drift']}</span></p>
"""

    if drift_result["drifted_features"]:
        html += "<h3>Top Drifted Features</h3><ul>"
        for feat in drift_result["drifted_features"][:20]:
            d = drift_result["features"][feat]
            html += f"<li>{feat} — KS={d['ks_statistic']:.4f}, p={d['p_value']:.4f}</li>"
        html += "</ul>"

    if perf_result:
        html += f"""
<h2>Performance</h2>
<table>
<tr><th>Metric</th><th>Reference (test split)</th><th>New Data</th></tr>
<tr><td>R²</td>
    <td>{perf_result['reference_r2']:.4f}</td>
    <td class="{'bad' if perf_result['degraded'] else 'ok'}">{perf_result['new_data_r2']:.4f}</td></tr>
<tr><td>MSE</td><td>{perf_result['reference_mse']:.4f}</td><td>{perf_result['new_data_mse']:.4f}</td></tr>
<tr><td>MAE</td><td>{perf_result['reference_mae']:.4f}</td><td>{perf_result['new_data_mae']:.4f}</td></tr>
</table>
<p>Performance change: {perf_result['performance_change']:+.1%}</p>
"""
    html += "</body></html>"
    return html


def main():
    logger.info("=" * 70)
    logger.info("STARTING MONITORING")
    logger.info("=" * 70)

    artifacts = load_artifacts()
    if artifacts is None:
        return False

    model           = artifacts["model"]
    X_train         = artifacts["X_train"]
    y_train         = artifacts["y_train"]
    X_test          = artifacts["X_test"]
    y_test          = artifacts["y_test"]
    feature_columns = artifacts["feature_columns"]

    # ── Load new data ─────────────────────────────────────────────────────────
    X_new, y_new = load_new_data(artifacts)
    has_new_data = X_new is not None

    # ── Drift detection ───────────────────────────────────────────────────────
    # Compare training distribution vs new data (or test set if no new data)
    X_compare = X_new if has_new_data else X_test
    logger.info(f"Running KS drift test: {X_train.shape[0]} ref vs {X_compare.shape[0]} new")
    drift_result = detect_drift(X_train, X_compare, feature_columns, DRIFT_THRESHOLD)

    logger.info(f"Drift: {drift_result['n_drifted']}/{drift_result['n_total']} features "
                f"(ratio={drift_result['drift_ratio']:.1%}, "
                f"overall={drift_result['overall_drift']})")

    # ── Performance check ─────────────────────────────────────────────────────
    perf_result = None
    if has_new_data:
        logger.info("Checking performance on new labeled data...")
        perf_result = check_performance(model, X_test, y_test, X_new, y_new, PERF_THRESHOLD)
        logger.info(f"Ref R²={perf_result['reference_r2']:.4f}  "
                    f"New R²={perf_result['new_data_r2']:.4f}  "
                    f"Change={perf_result['performance_change']:+.1%}  "
                    f"Degraded={perf_result['degraded']}")
    else:
        logger.info("No new labeled data — skipping performance degradation check.")

    # ── Decision ──────────────────────────────────────────────────────────────
    retrain_needed = (
        drift_result["overall_drift"] or
        (perf_result is not None and perf_result["degraded"])
    )

    if retrain_needed:
        reasons = []
        if drift_result["overall_drift"]:
            reasons.append(f"data drift ({drift_result['drift_ratio']:.0%} features)")
        if perf_result and perf_result["degraded"]:
            reasons.append(f"performance drop ({perf_result['performance_change']:+.1%})")
        logger.warning(f"RETRAIN NEEDED: {', '.join(reasons)}")
    else:
        logger.info("No retraining needed — model is healthy.")

    # ── Save reports ──────────────────────────────────────────────────────────
    timestamp = datetime.now().isoformat()

    drift_report = {
        "timestamp":  timestamp,
        "threshold":  DRIFT_THRESHOLD,
        **drift_result
    }
    with open("reports/drift_report.json", "w") as f:
        json.dump(drift_report, f, indent=2)
    logger.info("Saved: reports/drift_report.json")

    html = generate_html_report(drift_result, perf_result, retrain_needed, timestamp)
    with open("reports/performance_report.html", "w") as f:
        f.write(html)
    logger.info("Saved: reports/performance_report.html")

    summary = {
        "timestamp":       timestamp,
        "retrain_needed":  retrain_needed,
        "drift":           drift_result,
        "performance":     perf_result,
    }
    with open("artifacts/metrics/monitoring_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Flag file for GitHub Actions
    with open("retrain_needed.txt", "w") as f:
        f.write("true" if retrain_needed else "false")

    logger.info("=" * 70)
    logger.info(f"MONITORING COMPLETE — retrain_needed={retrain_needed}")
    logger.info("=" * 70)
    return retrain_needed


if __name__ == "__main__":
    try:
        result = main()
        sys.exit(0)
    except Exception as e:
        logger.error(f"Monitoring failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(0)
