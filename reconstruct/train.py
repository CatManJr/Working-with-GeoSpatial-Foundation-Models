import os
import sys
import pandas as pd
import lightgbm as lgb
import numpy as np

# Import paths
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import LGBM_DIR

BASE = LGBM_DIR
train = pd.read_parquet(f"{BASE}/train.parquet")
val   = pd.read_parquet(f"{BASE}/val.parquet")

feat_cols = [c for c in train.columns if c.startswith("f")]
targ_cols = [f"t{i}" for i in range(6)]
FILL = -9999.0

params = dict(
    objective="regression_l2",
    metric=["l2","l1"],
    learning_rate=0.05,
    num_leaves=64,
    feature_fraction=0.9,
    bagging_fraction=0.8,
    bagging_freq=1,
    min_data_in_leaf=50,
    seed=42,
    verbosity=-1,
    missing=FILL,  # tell LightGBM to treat -9999 as missing
)

models = []
for i, tcol in enumerate(targ_cols):
    callbacks = [
        lgb.early_stopping(stopping_rounds=100, verbose=True),
        lgb.log_evaluation(period=50),
    ]

    # Optional: convert -9999 to NaN so any LightGBM version treats them as missing
    def to_nan(a, fill=-9999.0):
        a = a.copy()
        a[a == fill] = np.nan
        return a

    Xtr = to_nan(train[feat_cols].values)
    ytr = train[tcol].values
    Xva = to_nan(val[feat_cols].values)
    yva = val[tcol].values

    dtrain = lgb.Dataset(Xtr, label=ytr)
    dval   = lgb.Dataset(Xva, label=yva, reference=dtrain)

    model = lgb.train(
        params,                 # keep your params dict
        dtrain,
        num_boost_round=10000,
        valid_sets=[dtrain, dval],
        valid_names=["train","val"],
        callbacks=callbacks,
    )
    model.save_model(f"{BASE}/model_t{i}.txt", num_iteration=model.best_iteration)
    models.append(model)
# save models
for i, model in enumerate(models):
    model.save_model(f"{BASE}/model_t{i}.txt", num_iteration=model.best_iteration)

print("Models saved")
