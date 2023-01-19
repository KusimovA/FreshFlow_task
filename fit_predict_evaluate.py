# I put everything at the same file for the speed and simplicity. It's better to have modular architecture as always.
# This file is definitely not what I would write in the production environment.

import json
import pandas as pd
from etna.datasets.tsdataset import TSDataset
from etna.models import NaiveModel
from etna.metrics import SMAPE

# Read the data
pd_df = pd.read_csv("data.csv", index_col=0)
pd_df = pd_df.drop_duplicates()

original_df = pd_df.copy()
original_df["timestamp"] = pd.to_datetime(original_df["day"])
original_df["target"] = original_df["sales_quantity"]
original_df["segment"] = original_df["item_number"]
original_df.drop(columns=["day", "sales_quantity", "item_number", 'item_name'], inplace=True)
original_df = original_df.reset_index(drop=True)

df = TSDataset.to_dataset(original_df)
ts = TSDataset(df, freq="D")

# Let's predict for 2 weeks now, even in real setting I've read that you need to predict only for 1 day ahead.
HORIZON = 14  # all hyperparams like this should be stored in the config file
train_ts, test_ts = ts.train_test_split(test_size=HORIZON)

# Fit the model
# This is just SeasonalMovingAverageModel, so there is nothing to save at this point
model = NaiveModel(lag=7)
model.fit(train_ts)

# Make the forecast
HORIZON_TO_PREDICT = 14
future_ts = train_ts.make_future(future_steps=HORIZON_TO_PREDICT, tail_steps=model.context_size)
forecast_ts = model.forecast(future_ts, prediction_size=HORIZON_TO_PREDICT)

# Save forecast
forecast_ts.to_pandas().to_csv('forecast.csv', header=True)

# Evaluate the forecast
smape = SMAPE()
evaluating_results = smape(y_true=test_ts, y_pred=forecast_ts)

# Save evaluation results
json = json.dumps(evaluating_results)
f = open("evaluating_results.json", "w")
f.write(json)
f.close()