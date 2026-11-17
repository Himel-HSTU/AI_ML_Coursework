
# ==========================================================
# PROJECT: UNDERFITTING vs GOOD FIT vs OVERFITTING
# Example: House Size vs House Price Prediction
# Algorithm: Random Forest Regression
# ==========================================================

# 1. Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


# 2. Generate a sample house-price dataset
np.random.seed(42)

number_of_houses = 180

house_size = np.random.uniform(500, 3000, number_of_houses)

# Create a realistic nonlinear relationship
actual_price = (
    18
    + 0.025 * house_size
    + 0.000004 * (house_size - 1700) ** 2
)

# Add random noise to simulate real-world variation
house_price = actual_price + np.random.normal(0, 8, number_of_houses)

# Create a dataset
data = pd.DataFrame({
    'House_Size_sqft': house_size,
    'Price_Lakh_BDT': house_price
})

print("Complete House Price Dataset:")
print(data.to_string(index=False))


# 3. Separate input and output
X = data[['House_Size_sqft']]
y = data['Price_Lakh_BDT']


# 4. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)


# 5. Create three Random Forest models
models = {
    'Underfitting': RandomForestRegressor(
        n_estimators=200, max_depth=1, random_state=42
    ),

    'Good Fit': RandomForestRegressor(
        n_estimators=200, max_depth=6,
        min_samples_leaf=4, random_state=42
    ),

    'Overfitting': RandomForestRegressor(
        n_estimators=200, max_depth=None,
        min_samples_leaf=1, random_state=42
    )
}


# 6. Train the models and calculate errors
results = {}

for name, model in models.items():

    model.fit(X_train, y_train)

    train_prediction = model.predict(X_train)
    test_prediction = model.predict(X_test)

    train_rmse = np.sqrt(mean_squared_error(y_train, train_prediction))
    test_rmse = np.sqrt(mean_squared_error(y_test, test_prediction))

    results[name] = {
        'model': model,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_r2': r2_score(y_train, train_prediction),
        'test_r2': r2_score(y_test, test_prediction)
    }


# 7. Display model performance
print("\nMODEL PERFORMANCE COMPARISON")
print("-" * 65)

for name, result in results.items():
    print(f"\n{name}")
    print(f"Training RMSE: {result['train_rmse']:.2f}")
    print(f"Testing RMSE : {result['test_rmse']:.2f}")
    print(f"Training R2  : {result['train_r2']:.3f}")
    print(f"Testing R2   : {result['test_r2']:.3f}")


# 8. Plot three model predictions
X_range = pd.DataFrame({
    'House_Size_sqft': np.linspace(500, 3000, 300)
})

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, (name, result) in zip(axes, results.items()):

    model = result['model']
    prediction = model.predict(X_range)

    ax.scatter(X_train['House_Size_sqft'], y_train,
               color='lightgray', label='Training Data', alpha=0.6)

    ax.scatter(X_test['House_Size_sqft'], y_test,
               color='black', label='Testing Data', alpha=0.7)

    ax.plot(X_range['House_Size_sqft'], prediction,
            color='red', linewidth=2, label='Model Prediction')

    ax.set_title(name)
    ax.set_xlabel('House Size (sq ft)')
    ax.set_ylabel('House Price (Lakh BDT)')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

plt.suptitle('Random Forest: Underfitting vs Good Fit vs Overfitting',
             fontsize=15)
plt.tight_layout()
plt.savefig('three_model_comparison.png', dpi=200)
plt.show()


# 9. Compare Training and Testing Errors
names = list(results.keys())
training_errors = [results[n]['train_rmse'] for n in names]
testing_errors = [results[n]['test_rmse'] for n in names]

x = np.arange(len(names))
width = 0.35

plt.figure(figsize=(9, 5))

plt.bar(x - width/2, training_errors, width, label='Training Error')
plt.bar(x + width/2, testing_errors, width, label='Testing Error')

plt.xticks(x, names)
plt.ylabel('RMSE (Lower is Better)')
plt.title('Training Error vs Testing Error')
plt.legend()
plt.grid(axis='y', alpha=0.2)

plt.tight_layout()
plt.savefig('error_comparison.png', dpi=200)
plt.show()


# 10. Save dataset and performance results
data.to_csv('house_price_dataset.csv', index=False)

performance = pd.DataFrame([
    {
        'Model': name,
        'Training_RMSE': result['train_rmse'],
        'Testing_RMSE': result['test_rmse'],
        'Training_R2': result['train_r2'],
        'Testing_R2': result['test_r2']
    }
    for name, result in results.items()
])

performance.to_csv('model_performance.csv', index=False)

print("\nAll experiments completed successfully!")
print("Dataset and graphs have been saved.")
