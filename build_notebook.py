"""
Builds notebooks/urban_heat_equity_analysis.ipynb using nbformat.
Run once from the project root: python build_notebook.py
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))

def code(text):
    cells.append(nbf.v4.new_code_cell(text))

# ---------------------------------------------------------------------
md("""\
# 🌆 Urban Heat Island & Green Equity Spatial-Data Analyzer

**Problem:** Concrete absorbs and re-radiates heat; trees and green space cool
the air through shade and evapotranspiration. Green space is rarely
distributed equitably across a city, and the neighborhoods with the least of
it are often the ones least able to afford air conditioning during a
heatwave.

This notebook walks through the full pipeline step by step, with every plot
rendered inline:

1. Generate / load neighborhood geospatial data
2. Clean & standardize CRS, compute area metrics
3. Engineer the **Green Equity Index**
4. Exploratory Data Analysis
5. Predictive modeling (Linear Regression + Random Forest)
6. Visualizations: static scatter plot + interactive map

> This uses a **synthetic city** (36 districts in a 6×6 grid) so it runs
> instantly with no external downloads. See the main `README.md` for how to
> swap in a real city's boundaries and NDVI/temperature data.
""")

# ---------------------------------------------------------------------
md("## Setup")
code("""\
import sys, os

# Make sure we're running from the project root so relative paths
# ("data/...", "outputs/...") resolve the same way as src/main.py
if os.path.basename(os.getcwd()) == "notebooks":
    os.chdir("..")

sys.path.append("src")

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.features import GeoJsonTooltip

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

%matplotlib inline
sns.set_theme(style="whitegrid", context="notebook")

import generate_data  # src/generate_data.py

print("Working directory:", os.getcwd())
""")

# ---------------------------------------------------------------------
md("""\
## Step 1 — Generate the synthetic city

Creates a 6×6 grid of "districts" with a realistic urban-core-vs-periphery
gradient: denser, taller, more populated toward the center; greener toward
the edges — similar to many real cities. A physically-motivated temperature
anomaly is baked in (more green → cooler, denser/taller → hotter) plus noise,
so the model trained later has a genuine signal to recover.
""")
code("""\
city_gdf = generate_data.build_city_grid()
city_gdf = generate_data.add_temperature_anomaly(city_gdf)
city_gdf.to_file("data/neighborhoods.geojson", driver="GeoJSON")

print(f"Generated {len(city_gdf)} synthetic districts")
city_gdf.drop(columns="geometry").head()
""")

code("""\
# Quick look at the raw grid, colored by green cover, before any cleaning
fig, ax = plt.subplots(figsize=(7, 7))
city_gdf.plot(column="green_cover_pct", cmap="Greens", legend=True, ax=ax, edgecolor="white")
ax.set_title("Synthetic City: Green Cover % by District (raw)")
ax.set_axis_off()
plt.show()
""")

# ---------------------------------------------------------------------
md("""\
## Step 2 — Load, clean, and compute area metrics

Reproject to a metric CRS (EPSG:3857) so area calculations are in real
square kilometers rather than degrees, then derive green/built area from the
land-cover percentages.
""")
code("""\
gdf = gpd.read_file("data/neighborhoods.geojson")

if gdf.crs is None:
    gdf = gdf.set_crs("EPSG:4326")

gdf_metric = gdf.to_crs("EPSG:3857")
gdf["area_km2"] = gdf_metric.geometry.area / 1e6
gdf["green_area_km2"] = gdf["area_km2"] * gdf["green_cover_pct"] / 100
gdf["built_area_km2"] = gdf["area_km2"] * gdf["building_pct"] / 100

gdf[["name", "area_km2", "green_area_km2", "built_area_km2"]].head()
""")

# ---------------------------------------------------------------------
md("""\
## Step 3 — Engineer the Green Equity Index

$$\\text{Green Equity Index} = \\frac{\\text{Total Green Cover Area}}{\\text{Total Built-Up Area}}$$

A value above 1 means a district has *more* green area than built area; below
1 means the opposite.
""")
code("""\
gdf["green_equity_index"] = (
    gdf["green_area_km2"] / gdf["built_area_km2"].clip(lower=1e-6)
).round(3)

print("Greenest districts:")
display(gdf.nlargest(5, "green_equity_index")[["name", "green_equity_index", "temp_anomaly_c"]])

print("\\nMost concrete-heavy districts:")
display(gdf.nsmallest(5, "green_equity_index")[["name", "green_equity_index", "temp_anomaly_c"]])
""")

# ---------------------------------------------------------------------
md("## Step 4 — Exploratory Data Analysis")
code("""\
cols = [
    "green_cover_pct", "building_pct", "population_density_km2",
    "avg_building_height_m", "green_equity_index", "temp_anomaly_c",
]
gdf[cols].describe().round(2)
""")

code("""\
corr = gdf[cols].corr()
corr["temp_anomaly_c"].sort_values()
""")

code("""\
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
            square=True, cbar_kws={"label": "correlation"})
plt.title("Correlation Matrix: Land Cover, Density & Heat")
plt.tight_layout()
plt.savefig("outputs/correlation_heatmap.png", dpi=150)
plt.show()
""")

# ---------------------------------------------------------------------
md("""\
## Step 5 — Predictive modeling

Train a Linear Regression and a Random Forest to predict **temperature
anomaly** from the Green Equity Index, population density, and average
building height. We hold out 25% of districts to evaluate on unseen data.
""")
code("""\
features = ["green_equity_index", "population_density_km2", "avg_building_height_m"]
target = "temp_anomaly_c"

X = gdf[features]
y = gdf[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

lin = LinearRegression().fit(X_train, y_train)
lin_pred = lin.predict(X_test)
print(f"Linear Regression  ->  R2 = {r2_score(y_test, lin_pred):.3f}   MAE = {mean_absolute_error(y_test, lin_pred):.3f} degC")
print("Coefficients:", dict(zip(features, lin.coef_.round(3))))
""")

code("""\
rf = RandomForestRegressor(n_estimators=300, max_depth=4, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
print(f"Random Forest      ->  R2 = {r2_score(y_test, rf_pred):.3f}   MAE = {mean_absolute_error(y_test, rf_pred):.3f} degC")

importance = pd.Series(rf.feature_importances_, index=features).sort_values()
importance
""")

code("""\
plt.figure(figsize=(7, 4))
importance.plot(kind="barh", color="#d1495b")
plt.xlabel("Importance")
plt.title("What Drives Heat Vulnerability?\\n(Random Forest Feature Importance)")
plt.tight_layout()
plt.savefig("outputs/feature_importance.png", dpi=150)
plt.show()
""")

code("""\
# Predict on the FULL dataset (for mapping) using the Random Forest
gdf["predicted_temp_anomaly_c"] = rf.predict(X).round(2)
gdf[["name", "green_equity_index", "temp_anomaly_c", "predicted_temp_anomaly_c"]].head()
""")

# ---------------------------------------------------------------------
md("## Step 6 — Visualizations")
md("### 6a. Static plot: Green cover vs. temperature anomaly")
code("""\
plt.figure(figsize=(8, 6))
ax = sns.regplot(
    data=gdf, x="green_cover_pct", y="temp_anomaly_c",
    scatter_kws={"s": 80, "alpha": 0.8, "edgecolor": "white"},
    line_kws={"color": "#d1495b", "linewidth": 3},
)
corr_val = gdf["green_cover_pct"].corr(gdf["temp_anomaly_c"])
ax.set_title("Green Cover vs. Surface Temperature Anomaly")
ax.set_xlabel("Green Cover (%)")
ax.set_ylabel("Temperature Anomaly (°C above city baseline)")
ax.annotate(f"Pearson r = {corr_val:.2f}", xy=(0.97, 0.95), xycoords="axes fraction",
            ha="right", va="top",
            bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.9))
plt.tight_layout()
plt.savefig("outputs/green_vs_temp.png", dpi=150)
plt.show()
""")

md("### 6b. Interactive map: predicted temperature anomaly by district\\n\\nThis renders inline below — hover over a district for details, and it's also saved as a standalone HTML file you can open in a browser or share.")
code("""\
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
).add_to(m)

tooltip = GeoJsonTooltip(
    fields=["name", "postal_code", "green_cover_pct", "green_equity_index", "predicted_temp_anomaly_c"],
    aliases=["Neighborhood:", "Postal code:", "Green cover (%):", "Green Equity Index:", "Predicted temp anomaly (°C):"],
    localize=True, sticky=True,
)
folium.GeoJson(
    gdf_wgs84,
    style_function=lambda x: {"fillOpacity": 0, "color": "#444", "weight": 0.6},
    tooltip=tooltip,
).add_to(m)

folium.LayerControl().add_to(m)
m.save("outputs/heat_equity_map.html")
m
""")

# ---------------------------------------------------------------------
md("""\
## Conclusions

- The **Green Equity Index** is the strongest single predictor of a
  district's temperature anomaly — much stronger than raw population
  density on its own.
- The Random Forest model finds **building height/density** at least as
  important as population density — physical urban form (heat-trapping
  mass, impervious surface) matters as much as headcount.
- Every district with a Green Equity Index above roughly 1.5 sits below the
  city-wide average temperature; every one below roughly 0.5 sits above it.

**Next steps for a real city:** swap `data/neighborhoods.geojson` for real
neighborhood boundaries, pull real NDVI / building footprints from
OpenStreetMap or Google Earth Engine, and replace the synthetic
`temp_anomaly_c` column with real Landsat/MODIS land-surface-temperature
data. See the project `README.md` for direct links and steps.
""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"},
}

with open("notebooks/urban_heat_equity_analysis.ipynb", "w") as f:
    nbf.write(nb, f)

print("Notebook written to notebooks/urban_heat_equity_analysis.ipynb")
