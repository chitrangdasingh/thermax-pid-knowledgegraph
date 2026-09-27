"""Measured-data analytics; gates prevent training a model on the DXF alone."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from platform_core import DB, connect

MODEL_DIR = Path(__file__).resolve().parent / "runtime" / "models"


def series(sensor_tag, path=DB):
    with connect(path) as db:
        data=pd.read_sql_query("""SELECT timestamp,value,unit,asset_id FROM observations
            WHERE sensor_tag=? AND quality_code='GOOD' ORDER BY timestamp""",db,params=(sensor_tag,))
    if not data.empty:
        data["timestamp"]=pd.to_datetime(data.timestamp,utc=True)
    return data


def trend(sensor_tag,path=DB):
    df=series(sensor_tag,path)
    if len(df)<2:return {"status":"INSUFFICIENT_DATA","count":len(df)}
    y=df.value.to_numpy(dtype=float)
    return {"status":"DESCRIPTIVE","count":len(y),"mean":float(y.mean()),"min":float(y.min()),
            "max":float(y.max()),"latest":float(y[-1]),"change_from_previous":float(y[-1]-y[-2]),
            "last_timestamp":df.timestamp.iloc[-1].isoformat(),"unit":df.unit.iloc[-1]}


def train_forecaster(sensor_tag,path=DB,window=8,minimum=80):
    """One-step tabular forecast. Chronological holdout versus persistence."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.metrics import mean_absolute_error
    import joblib

    df=series(sensor_tag,path)
    if len(df)<minimum:
        return {"status":"INSUFFICIENT_DATA","count":len(df),"required":minimum,"model":"not trained"}
    y=df.value.to_numpy(float)
    if len(set(np.round(y,7)))<5:
        return {"status":"LOW_VARIATION","count":len(y),"model":"not trained"}
    x=np.array([y[i-window:i] for i in range(window,len(y))])
    targets=y[window:]
    split=int(len(targets)*.8)
    if split<30 or len(targets)-split<10:return {"status":"INSUFFICIENT_SPLIT","model":"not trained"}
    model=HistGradientBoostingRegressor(max_iter=80,max_leaf_nodes=8,random_state=42)
    model.fit(x[:split],targets[:split])
    estimate=model.predict(x[split:]); mae=float(mean_absolute_error(targets[split:],estimate))
    baseline=float(mean_absolute_error(targets[split:],x[split:,-1]))
    info={"status":"VALIDATED" if mae<baseline else "BELOW_BASELINE", "model":"lagged_gradient_boosting",
          "sensor_tag":sensor_tag,"n_train":split,"n_test":len(targets)-split,"window":window,
          "test_mae":mae,"persistence_mae":baseline,"test_start":df.timestamp.iloc[window+split].isoformat(),
          "forecast_next":float(model.predict(y[-window:].reshape(1,-1))[0]),"unit":df.unit.iloc[-1],
          "caution":"A single-tag model; validate sampling frequency, drift and process regime before deployment."}
    MODEL_DIR.mkdir(parents=True,exist_ok=True)
    safe="".join(c for c in sensor_tag if c.isalnum() or c in "_- ").strip().replace(" ","_")
    joblib.dump(model,MODEL_DIR/f"{safe}.joblib")
    (MODEL_DIR/f"{safe}.json").write_text(json.dumps(info,indent=2))
    return info


def heat_duty(mass_flow_kg_s, cp_kj_kg_k, inlet_c, outlet_c):
    """Steady sensible-heat calculation: kW = (kg/s)(kJ/kg/K)(K)."""
    mass,cp=float(mass_flow_kg_s),float(cp_kj_kg_k)
    if mass<0 or cp<=0:raise ValueError("Mass flow must be nonnegative and Cp positive")
    return {"sensible_heat_kw":mass*cp*(float(outlet_c)-float(inlet_c)),
            "equation":"Q = m_dot * Cp * (T_out - T_in)",
            "assumptions":"Steady single phase; no heat loss, reaction, pressure effect or phase change. Inputs are user supplied."}


def advise(alert, asset=None):
    """Advisory rules only. Does not issue control commands or setpoints."""
    if alert["rule"]=="HIGH":
        recommendation="Verify the sensor and operating context; inspect the approved high-limit procedure with an operator."
    elif alert["rule"]=="LOW":
        recommendation="Verify the sensor and operating context; inspect the approved low-limit procedure with an operator."
    else:recommendation="Review the alert with the relevant procedure."
    return {"asset_id":alert["asset_id"],"recommendation":recommendation,
            "basis":f"{alert['rule']} threshold {alert['threshold']} versus reading {alert['measured']}",
            "approval":"Operator and engineering review required; advisory only."}
