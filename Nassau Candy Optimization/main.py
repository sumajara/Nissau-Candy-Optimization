import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# --- Step 1: Load Data & Assign Current Factories ---
excel_file = 'Nassau_Candy_Project_Workbook.xlsx'
df = pd.read_excel(excel_file, sheet_name='Cleaned Data')

product_factory_map = {
    "Wonka Bar - Nutty Crunch Surprise": "Lot's O' Nuts",
    "Wonka Bar - Fudge Mallows": "Lot's O' Nuts",
    "Wonka Bar -Scrumdiddlyumptious": "Lot's O' Nuts",
    "Wonka Bar - Milk Chocolate": "Wicked Choccy's",
    "Wonka Bar - Triple Dazzle Caramel": "Wicked Choccy's",
    'Laffy Taffy': 'Sugar Shack',
    'SweeTARTS': 'Sugar Shack',
    'Nerds': 'Sugar Shack',
    'Fun Dip': 'Sugar Shack',
    'Fizzy Lifting Drinks': 'Sugar Shack',
    'Everlasting Gobstopper': 'Secret Factory',
    'Hair Toffee': 'The Other Factory',
    'Lickable Wallpaper': 'Secret Factory',
    'Wonka Gum': 'Secret Factory',
    'Kazookles': 'The Other Factory',
}
df['Current Factory'] = df['Product Name'].map(product_factory_map)

# --- Step 2: Predictive Modeling ---
features = [
    'Ship Mode',
    'Region',
    'Division',
    'Current Factory',
    'Units',
    'Sales',
    'Cost',
    'Gross Profit',
]
X = pd.get_dummies(df[features], drop_first=True)
y = df['Adjusted Lead Time (days)']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(
        n_estimators=100, random_state=42
    ),
}

print('=== Step 2: Predictive Model Evaluation ===')
best_model = None
for name, model in models.items():
  model.fit(X_train, y_train)
  preds = model.predict(X_test)
  rmse = np.sqrt(mean_squared_error(y_test, preds))
  mae = mean_absolute_error(y_test, preds)
  r2 = r2_score(y_test, preds)
  print(f'{name} -> RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}')
  if name == 'Gradient Boosting':
    best_model = model

# --- Step 3: Reallocation & Recommendation Engine Simulation ---
print('\n=== Step 3: Top Reallocation Recommendations Sample ===')
all_factories = [
    "Lot's O' Nuts",
    "Wicked Choccy's",
    'Sugar Shack',
    'Secret Factory',
    'The Other Factory',
]

recommendations = []
for idx, row in df.head(5).iterrows():
  current_fac = row['Current Factory']
  best_fac = current_fac
  min_lead_time = row['Adjusted Lead Time (days)']

  for fac in all_factories:
    temp_row = pd.DataFrame([row[features]])
    temp_row['Current Factory'] = fac
    temp_encoded = pd.get_dummies(temp_row)
    temp_encoded = temp_encoded.reindex(columns=X.columns, fill_value=0)

    pred_lead = best_model.predict(temp_encoded)[0]
    if pred_lead < min_lead_time:
      min_lead_time = pred_lead
      best_fac = fac

  recommendations.append({
      'Product Name': row['Product Name'],
      'Current Factory': current_fac,
      'Recommended Factory': best_fac,
      'Original Lead Time': row['Adjusted Lead Time (days)'],
      'Predicted Lead Time': round(min_lead_time, 2),
  })

rec_df = pd.DataFrame(recommendations)
print(rec_df.to_string(index=False))