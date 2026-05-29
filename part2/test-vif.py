import sys
from pathlib import Path

# Them thu muc goc vao Python Path de import part1 va part2
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from part1.ols_implementation import vif 
from part2.data_pipeline import DataPipeline, load_data, train_test_split

print("1. Loading and cleaning data...")
data_path = PROJECT_ROOT / "part2" / "data" / "melb_data.csv"
df = load_data(str(data_path))
df_train, _ = train_test_split(df, test_size=0.3, random_state=42)

# Khoi tao Pipeline
pipeline = DataPipeline()
X_train, y_train = pipeline.fit_transform(df_train)

print(f"-> X_train shape: {X_train.shape}")
print("2. Measuring VIF")

# Khong bo cot nao vi Pipeline khong tu dong them Intercept
vif_table = vif(X_train) 

# Gan ten cot vao bang VIF
vif_table.insert(0, "Feature_Name", pipeline.feature_names)

print("\n--- VIF SCORES ---")
print(vif_table.to_string(index=False))
