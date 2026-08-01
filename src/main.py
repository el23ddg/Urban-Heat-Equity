"""
main.py
-------
Urban Heat Island & Green Equity Spatial-Data Analyzer

Pipeline:
  1. Load neighborhood polygons (data/neighborhoods.geojson)
  2. Clean / standardize CRS, compute area metrics
  3. Engineer the Green Equity Index (Green Cover Area / Built-Up Area)
  4. Exploratory Data Analysis (summary stats + correlation heatmap)
  5. Train two regression models (Linear Regression, Random Forest) to
     predict surface temperature anomaly from Green Equity Index,
     population density, and building height/density
  6. Evaluate models (R^2, MAE) and report feature importance
  7. Build two visual outputs:
       - outputs/heat_equity_map.html   (interactive Folium map)
       - outputs/green_vs_temp.png      (static Seaborn scatter/regression)
       - outputs/correlation_heatmap.png
       - outputs/feature_importance.png

Run:  python src/main.py
"""

import json
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless backend, safe for scripts/CI
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.features import GeoJsonTooltip

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

warnings.filterwarnings("ignore")

DATA_PATH = "data/neighborhoods.geojson"
OUT_DIR = "outputs"

sns.set_theme(style="whitegrid", context="talk")


# --------------------------------------------------------------------------
# Step 1-2: Load + clean
# --------------------------------------------------------------------------
def load_and_clean(path: str) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path)

    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")

    # Reproject to a metric CRS for accurate area calculations.
    # EPSG:3857 (Web Mercator) is fine for a quick demo; for production use
    # the correct local UTM zone for your city.
    gdf_metric = gdf.to_crs("EPSG:3857")
    gdf["area_km2"] = gdf_metric.geometry.area / 1e6

    # Land-cover component areas derived from the % columns
    gdf["green_area_km2"] = gdf["area_km2"] * gdf["green_cover_pct"] / 100
    gdf["built_area_km2"] = gdf["area_km2"] * gdf["building_pct"] / 100

    return gdf


# --------------------------------------------------------------------------
# Step 3: Green Equity Index
# --------------------------------------------------------------------------
def compute_green_equity_index(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf = gdf.copy()
    gdf["green_equity_index"] = (
        gdf["green_area_km2"] / gdf["built_area_km2"].clip(lower=1e-6)
    ).round(3)
    return gdf


# --------------------------------------------------------------------------
# Step 4: EDA
# --------------------------------------------------------------------------
def run_eda(gdf: gpd.GeoDataFrame) -> None:
    cols = [
        "green_cover_pct",
        "building_pct",
        "population_density_km2",
        "avg_building_height_m",
        "green_equity_index",
        "temp_anomaly_c",
    ]
    print("\n=== Summary statistics ===")
    print(gdf[cols].describe().round(2).to_string())

    print("\n=== Correlation with temperature anomaly ===")
    corr = gdf[cols].corr()["temp_anomaly_c"].sort_values()
    print(corr.round(3).to_string())

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        gdf[cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0,
        square=True, cbar_kws={"label": "correlation"},
    )
    plt.title("Correlation Matrix: Land Cover, Density & Heat")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/correlation_heatmap.png", dpi=150)
    plt.close()
    print(f"Saved {OUT_DIR}/correlation_heatmap.png")


# --------------------------------------------------------------------------
# Step 5-6: Predictive model
# --------------------------------------------------------------------------
def train_models(gdf: gpd.GeoDataFrame):
    features = ["green_equity_index", "population_density_km2", "avg_building_height_m"]
    target = "temp_anomaly_c"

    X = gdf[features]
    y = gdf[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    results = {}

    lin = LinearRegression().fit(X_train, y_train)
    lin_pred = lin.predict(X_test)
    results["LinearRegression"] = {
        "r2": r2_score(y_test, lin_pred),
        "mae": mean_absolute_error(y_test, lin_pred),
        "coefficients": dict(zip(features, lin.coef_.round(3))),
    }

    rf = RandomForestRegressor(n_estimators=300, max_depth=4, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    results["RandomForest"] = {
        "r2": r2_score(y_test, rf_pred),
        "mae": mean_absolute_error(y_test, rf_pred),
        "feature_importance": dict(
            zip(features, np.round(rf.feature_importances_, 3))
        ),
    }

    print("\n=== Model performance (held-out test set) ===")
    for name, res in results.items():
        print(f"{name}: R^2 = {res['r2']:.3f}, MAE = {res['mae']:.3f} degC")

    print("\nLinear Regression coefficients (standardized effect direction):")
    for feat, coef in results["LinearRegression"]["coefficients"].items():
        print(f"  {feat}: {coef}")

    print("\nRandom Forest feature importance:")
    for feat, imp in results["RandomForest"]["feature_importance"].items():
        print(f"  {feat}: {imp}")

    # Predict on the FULL dataset (for mapping) using the random forest,
    # generally the stronger of the two on nonlinear interactions.
    gdf = gdf.copy()
    gdf["predicted_temp_anomaly_c"] = rf.predict(X).round(2)

    # Feature importance bar chart
    plt.figure(figsize=(7, 5))
    fi = pd.Series(results["RandomForest"]["feature_importance"]).sort_values()
    fi.plot(kind="barh", color="#d1495b")
    plt.xlabel("Importance")
    plt.title("What Drives Heat Vulnerability?\n(Random Forest Feature Importance)")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/feature_importance.png", dpi=150)
    plt.close()
    print(f"Saved {OUT_DIR}/feature_importance.png")

    return gdf, results


# --------------------------------------------------------------------------
# Step 7a: Static seaborn plot
# --------------------------------------------------------------------------
def plot_green_vs_temp(gdf: gpd.GeoDataFrame) -> None:
    plt.figure(figsize=(9, 7))
    ax = sns.regplot(
        data=gdf,
        x="green_cover_pct",
        y="temp_anomaly_c",
        scatter_kws={"s": 90, "alpha": 0.8, "edgecolor": "white"},
        line_kws={"color": "#d1495b", "linewidth": 3},
    )
    corr = gdf["green_cover_pct"].corr(gdf["temp_anomaly_c"])
    ax.set_title("Green Cover vs. Surface Temperature Anomaly", fontsize=18, pad=15)
    ax.set_xlabel("Green Cover (%)")
    ax.set_ylabel("Temperature Anomaly (°C above city baseline)")
    ax.annotate(
        f"Pearson r = {corr:.2f}\nMore green cover -> cooler neighborhoods",
        xy=(0.97, 0.95), xycoords="axes fraction",
        ha="right", va="top", fontsize=13,
        bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.9),
    )
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/green_vs_temp.png", dpi=150)
    plt.close()
    print(f"Saved {OUT_DIR}/green_vs_temp.png")


# --------------------------------------------------------------------------
# Step 7b: Interactive Folium map
# --------------------------------------------------------------------------
def build_interactive_map(gdf: gpd.GeoDataFrame) -> None:
    gdf_wgs84 = gdf.to_crs("EPSG:4326")
    centroid = gdf_wgs84.geometry.union_all().centroid

    m = folium.Map(location=[centroid.y, centroid.x], zoom_start=13, tiles="cartodbpositron")

    folium.Choropleth(
        geo_data=gdf_wgs84,
        data=gdf_wgs84,
        columns=["neighborhood_id", "predicted_temp_anomaly_c"],
        key_on="feature.properties.neighborhood_id",
        fill_color="RdYlGn_r",
        fill_opacity=0.75,
        line_opacity=0.4,
        legend_name="Predicted Temperature Anomaly (°C)",
        name="Predicted Temperature Anomaly",
    ).add_to(m)

    tooltip = GeoJsonTooltip(
        fields=[
            "name",
            "postal_code",
            "green_cover_pct",
            "green_equity_index",
            "predicted_temp_anomaly_c",
        ],
        aliases=[
            "Neighborhood:",
            "Postal code:",
            "Green cover (%):",
            "Green Equity Index:",
            "Predicted temp anomaly (°C):",
        ],
        localize=True,
        sticky=True,
    )

    folium.GeoJson(
        gdf_wgs84,
        style_function=lambda x: {"fillOpacity": 0, "color": "#444", "weight": 0.6},
        tooltip=tooltip,
        name="Neighborhood details",
    ).add_to(m)

    folium.LayerControl().add_to(m)

    out_path = f"{OUT_DIR}/heat_equity_map.html"
    m.save(out_path)
    print(f"Saved {out_path}")


# --------------------------------------------------------------------------
def main():
    print("Loading and cleaning geospatial data...")
    gdf = load_and_clean(DATA_PATH)

    print("Computing Green Equity Index...")
    gdf = compute_green_equity_index(gdf)

    run_eda(gdf)

    print("\nTraining regression models...")
    gdf, results = train_models(gdf)

    plot_green_vs_temp(gdf)
    build_interactive_map(gdf)

    # Save the enriched dataset (with predictions) for inspection / reuse
    gdf.drop(columns="geometry").to_csv(f"{OUT_DIR}/neighborhood_metrics.csv", index=False)
    print(f"Saved {OUT_DIR}/neighborhood_metrics.csv")

    with open(f"{OUT_DIR}/model_results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"Saved {OUT_DIR}/model_results.json")

    print("\nDone. Open outputs/heat_equity_map.html in a browser to explore the map.")


if __name__ == "__main__":
    main()
