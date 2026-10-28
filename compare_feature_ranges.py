import pandas as pd

train = pd.read_csv("data/ddos_cleaned.csv")
live = pd.read_csv("outputs/live_model_features_provisional.csv")

cols = [c for c in live.columns if c in train.columns]

result = pd.DataFrame({
    "Train Min": train[cols].min(),
    "Train Max": train[cols].max(),
    "Live Min": live[cols].min(),
    "Live Max": live[cols].max()
})

print("\nTRAINING VS LIVE FEATURE RANGES\n")
print(result.to_string())
print("\nTraining rows:", len(train))
print("Live intervals:", len(live))
print("Compared features:", len(cols))
