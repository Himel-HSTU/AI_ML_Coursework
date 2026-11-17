
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


df = pd.read_csv("HousePrice_Cleaned.csv")

X = df[["Bedrooms", "Bathrooms", "Floor_area"]]


y = df["Price_in_taka"] / 100000

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

models = {
    "Underfitting": RandomForestRegressor(
        n_estimators=200,
        max_depth=1,
        random_state=42
    ),

    "Balanced Fit": RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=5,
        random_state=42
    ),

    "Overfitting": RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=1,
        random_state=42
    )
}

results = []

for name, model in models.items():

    model.fit(X_train, y_train)

    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)

    train_rmse = mean_squared_error(
        y_train, train_predictions
    ) ** 0.5

    test_rmse = mean_squared_error(
        y_test, test_predictions
    ) ** 0.5

    train_r2 = r2_score(y_train, train_predictions)
    test_r2 = r2_score(y_test, test_predictions)

    results.append({
        "Model": name,
        "Training RMSE (Lakh)": train_rmse,
        "Testing RMSE (Lakh)": test_rmse,
        "Training R2": train_r2,
        "Testing R2": test_r2
    })

results_df = pd.DataFrame(results)

print("\nMODEL PERFORMANCE COMPARISON")
print(results_df.round(3).to_string(index=False))

results_df.to_csv("Model_Comparison.csv", index=False)


median_bedrooms = X["Bedrooms"].median()
median_bathrooms = X["Bathrooms"].median()

x_min = X["Floor_area"].quantile(0.05)
x_max = X["Floor_area"].quantile(0.95)

floor_sizes = pd.DataFrame({
    "Bedrooms": median_bedrooms,
    "Bathrooms": median_bathrooms,
    "Floor_area": [
        x_min + (x_max - x_min) * i / 199
        for i in range(200)
    ]
})

x_low = X_test["Floor_area"].quantile(0.05)
x_high = X_test["Floor_area"].quantile(0.95)

y_low = y_test.quantile(0.05)
y_high = y_test.quantile(0.95)

visible = (
    (X_test["Floor_area"] >= x_low) &
    (X_test["Floor_area"] <= x_high) &
    (y_test >= y_low) &
    (y_test <= y_high)
)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, (name, model) in zip(axes, models.items()):

    predicted_prices = model.predict(floor_sizes)

    ax.scatter(
        X_test.loc[visible, "Floor_area"],
        y_test.loc[visible],
        color="blue",
        alpha=0.5,
        s=25,
        label="Actual Data"
    )


    ax.plot(
        floor_sizes["Floor_area"],
        predicted_prices,
        color="red",
        linewidth=2.5,
        label="Fitting Curve"
    )

    ax.set_title(name, fontsize=14)
    ax.set_xlabel("Floor Area (Sq Ft)")
    ax.set_ylabel("Price (Lakh Taka)")

    ax.set_xlim(x_min, x_max)

    ax.set_ylim(
        min(y_low, predicted_prices.min()) * 0.90,
        max(y_high, predicted_prices.max()) * 1.10
    )

    ax.grid(alpha=0.25)
    ax.legend()

plt.tight_layout()
plt.savefig("Model_Fitting_Graphs.png", dpi=300)
plt.show()


results_df.plot(
    x="Model",
    y=["Training RMSE (Lakh)", "Testing RMSE (Lakh)"],
    kind="bar",
    figsize=(9, 5)
)

plt.title("Training vs Testing Error")
plt.ylabel("RMSE (Lakh Taka)")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig("RMSE_Comparison.png", dpi=300)
plt.show()

print("\nAll results and graphs saved successfully!")
