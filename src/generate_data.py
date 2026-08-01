"""
generate_data.py
----------------
Creates a synthetic "city" of neighborhoods as a grid of square polygons and
saves it as GeoJSON. This lets the whole pipeline run anywhere, offline,
in under a second -- no need to hunt for a real city shapefile before you
can test the analysis.

Swapping in a REAL city later is straightforward: replace this file's output
(data/neighborhoods.geojson) with a real neighborhood/postal-code boundary
file that has at minimum a "name" column, then re-run main.py. See the
README for where to get real boundaries (OpenStreetMap / HDX / your city's
open-data portal) and real green-cover layers (NDVI, tree canopy, parks).

The synthetic ground truth intentionally builds in a real physical
relationship so the downstream regression model has something genuine to
learn:

    temperature_anomaly  ~  - a * green_equity_index
                            + b * building_density
                            + c * population_density
                            + noise

This mirrors the real urban-heat-island mechanism: vegetation cools
(evapotranspiration + shade), dense built mass and population/activity heat
things up.
"""

import numpy as np
import geopandas as gpd
import pandas as pd
from shapely.geometry import box

# ---- reproducibility ----
RNG = np.random.default_rng(42)

# ---- grid geometry: a synthetic city as an N x N grid of neighborhoods ----
N_ROWS, N_COLS = 6, 6
CELL_SIZE_DEG = 0.01          # ~1.1 km per cell at mid-latitudes
ORIGIN_LON, ORIGIN_LAT = 88.30, 22.50  # arbitrary starting point (Kolkata-ish)


def build_city_grid() -> gpd.GeoDataFrame:
    rows = []
    idx = 0
    for r in range(N_ROWS):
        for c in range(N_COLS):
            minx = ORIGIN_LON + c * CELL_SIZE_DEG
            miny = ORIGIN_LAT + r * CELL_SIZE_DEG
            maxx = minx + CELL_SIZE_DEG
            maxy = miny + CELL_SIZE_DEG
            geom = box(minx, miny, maxx, maxy)

            # Urban-core-vs-periphery gradient: cells near the grid center
            # tend to be denser and greener neighborhoods tend to cluster on
            # the periphery -- similar to many real cities.
            center_r, center_c = (N_ROWS - 1) / 2, (N_COLS - 1) / 2
            dist_from_center = np.hypot(r - center_r, c - center_c)
            max_dist = np.hypot(center_r, center_c)
            peripherality = dist_from_center / max_dist  # 0 = core, 1 = edge

            # Green cover % of total land area (higher toward periphery)
            green_pct = np.clip(
                RNG.normal(loc=15 + 45 * peripherality, scale=8), 2, 85
            )
            # Building / impervious surface % (higher toward the core)
            building_pct = np.clip(
                RNG.normal(loc=70 - 40 * peripherality, scale=8), 10, 95
            )
            # Normalize so green + building + "other" (roads, water, bare
            # soil) roughly fits under 100
            other_pct = np.clip(100 - green_pct - building_pct, 3, 40)

            # Population density (people / km^2), denser toward the core
            pop_density = np.clip(
                RNG.normal(loc=18000 - 12000 * peripherality, scale=2500),
                800, 30000,
            )

            # Average building height (proxy for building "density" in 3D),
            # correlated with the core
            building_height_m = np.clip(
                RNG.normal(loc=35 - 25 * peripherality, scale=6), 4, 60
            )

            rows.append(
                {
                    "neighborhood_id": f"N{idx:03d}",
                    "name": f"District {idx + 1}",
                    "postal_code": f"70{idx:03d}",
                    "green_cover_pct": round(green_pct, 2),
                    "building_pct": round(building_pct, 2),
                    "other_surface_pct": round(other_pct, 2),
                    "population_density_km2": round(pop_density, 0),
                    "avg_building_height_m": round(building_height_m, 1),
                    "geometry": geom,
                }
            )
            idx += 1

    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")
    return gdf


def add_temperature_anomaly(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Attach a synthetic land-surface-temperature anomaly (deg C above the
    city-wide baseline) using a physically-motivated formula + noise, so a
    regression model trained on it recovers a sensible, explainable signal.
    """
    green_equity_index = gdf["green_cover_pct"] / gdf["building_pct"].clip(lower=1)

    # Standardize inputs to comparable scales before combining
    ge_z = (green_equity_index - green_equity_index.mean()) / green_equity_index.std()
    pop_z = (
        gdf["population_density_km2"] - gdf["population_density_km2"].mean()
    ) / gdf["population_density_km2"].std()
    height_z = (
        gdf["avg_building_height_m"] - gdf["avg_building_height_m"].mean()
    ) / gdf["avg_building_height_m"].std()

    noise = RNG.normal(0, 0.35, size=len(gdf))

    temp_anomaly = (
        -1.6 * ge_z          # more green relative to concrete -> cooler
        + 0.9 * pop_z        # denser population/activity -> hotter
        + 0.7 * height_z     # taller/denser built form -> hotter (heat trapping)
        + noise
    )

    gdf = gdf.copy()
    gdf["green_equity_index"] = green_equity_index.round(3)
    gdf["temp_anomaly_c"] = temp_anomaly.round(2)
    return gdf


def main():
    gdf = build_city_grid()
    gdf = add_temperature_anomaly(gdf)
    out_path = "data/neighborhoods.geojson"
    gdf.to_file(out_path, driver="GeoJSON")
    print(f"Wrote {len(gdf)} synthetic neighborhoods to {out_path}")
    print(gdf.drop(columns="geometry").head())


if __name__ == "__main__":
    main()
