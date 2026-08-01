"""
fetch_real_data.py
-------------------
Replaces the synthetic city with REAL data for an actual city, using Google
Earth Engine's free public datasets. You provide neighborhood boundaries;
this script does zonal statistics (per-polygon averages) for everything
else, and writes a data/neighborhoods.geojson with the exact same schema
that src/main.py and the notebook already expect -- so nothing downstream
needs to change.

WHAT THIS PULLS, AND FROM WHERE
--------------------------------
  green_cover_pct, building_pct
      Dynamic World V1 (10m land cover, near-real-time), Google/DeepMind/WRI.
      Per pixel, it gives a probability for each of 9 classes (trees, grass,
      crops, built, water, etc). We average "trees + grass + crops +
      flooded_vegetation" as the green fraction and "built" as the building
      fraction, per polygon.

  temp_anomaly_c
      MODIS MOD11A2 (8-day Land Surface Temperature, 1km). We take a recent
      summer composite (edit SUMMER_START/END below for your hemisphere and
      city), convert Kelvin -> Celsius, then subtract the city-wide mean so
      it's an ANOMALY (relative to that city, not an absolute temperature).

  population_density_km2
      WorldPop Global Project Population Data (100m). Already people/pixel;
      we sum within each polygon and divide by polygon area.

  avg_building_height_m
      JRC Global Human Settlement Layer, GHS-BUILT-H (average building
      height, ~100m resolution).

BEFORE YOU RUN THIS
--------------------
1. pip install earthengine-api geemap
2. One-time setup:
     earthengine authenticate
   This opens a browser to log into your Google account and generates a
   local credentials file. You'll also need a GEE "Cloud Project" -- when
   you sign up at https://code.earthengine.google.com/ for the first time,
   Google walks you through creating one (any project name is fine, e.g.
   "urban-heat-equity"). Put that project ID below in EE_PROJECT.
3. Get real neighborhood boundaries and save them as
   data/real_neighborhoods_raw.geojson before running this script. Options:
     - Overpass Turbo (https://overpass-turbo.eu/): run a query like
         [out:json][timeout:60];
         area["name"="YOUR_CITY"]->.searchArea;
         relation["boundary"="administrative"]["admin_level"="10"](area.searchArea);
         out geom;
       then Export -> GeoJSON. (admin_level varies by country/city -- try
       8-10 and see what returns actual neighborhoods vs. the whole city.)
     - Your city's open-data portal (search "<your city> open data GIS
       neighborhoods" or "<your city> ward boundaries geojson").
     - GADM (https://gadm.org/) for administrative boundaries if your city
       doesn't have finer neighborhood-level data.
   The file needs AT MINIMUM a name/id field per polygon. Update
   NAME_FIELD below to match whatever that field is actually called in
   your file (open it and check -- common names: "name", "NAME",
   "neighborhood", "ward_name").

RUN
----
    python src/fetch_real_data.py
"""

import json

import ee
import geopandas as gpd
import pandas as pd

# --------------------------------------------------------------------------
# CONFIG -- edit these for your city
# --------------------------------------------------------------------------
EE_PROJECT = "my-data-project-1010-382120"          # <-- put your GEE cloud project ID here
RAW_BOUNDARIES_PATH = "data/real_neighborhoods_raw.geojson"  # <-- your downloaded boundaries
NAME_FIELD = "name"                          # <-- the column with each neighborhood's name/id in your file
OUTPUT_PATH = "data/neighborhoods.geojson"

# Pick a recent, cloud-free-ish summer window for LST (your hottest season).
# Northern hemisphere cities: e.g. "2025-06-01" to "2025-08-31"
# Southern hemisphere cities: e.g. "2025-12-01" to "2026-02-28"
SUMMER_START = "2025-06-01"
SUMMER_END = "2025-08-31"

GREEN_CLASSES = ["trees", "grass", "crops", "flooded_vegetation"]
BUILT_CLASS = "built"


# --------------------------------------------------------------------------
def init_ee():
    try:
        ee.Initialize(project=EE_PROJECT)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=EE_PROJECT)


def load_boundaries():
    gdf = gpd.read_file(RAW_BOUNDARIES_PATH)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    gdf = gdf.to_crs("EPSG:4326")

    if NAME_FIELD not in gdf.columns:
        raise ValueError(
            f"Column '{NAME_FIELD}' not found in {RAW_BOUNDARIES_PATH}. "
            f"Available columns: {list(gdf.columns)}. "
            f"Update NAME_FIELD in this script to match."
        )

    gdf = gdf[[NAME_FIELD, "geometry"]].reset_index(drop=True)
    gdf["neighborhood_id"] = [f"N{idx:03d}" for idx in range(len(gdf))]
    gdf = gdf.rename(columns={NAME_FIELD: "name"})
    return gdf


def gdf_to_ee_fc(gdf: gpd.GeoDataFrame) -> ee.FeatureCollection:
    features = []
    for _, row in gdf.iterrows():
        geom = ee.Geometry(json.loads(gpd.GeoSeries([row.geometry]).to_json())["features"][0]["geometry"])
        features.append(ee.Feature(geom, {"neighborhood_id": row["neighborhood_id"], "name": row["name"]}))
    return ee.FeatureCollection(features)


# --------------------------------------------------------------------------
def compute_dynamic_world(fc: ee.FeatureCollection, start: str, end: str) -> ee.FeatureCollection:
    """Mean class probabilities -> green_cover_pct and building_pct per polygon."""
    dw = (
        ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
        .filterDate(start, end)
        .filterBounds(fc)
        .select(GREEN_CLASSES + [BUILT_CLASS])
        .mean()
    )
    green_img = dw.select(GREEN_CLASSES).reduce(ee.Reducer.sum()).rename("green_frac")
    built_img = dw.select(BUILT_CLASS).rename("built_frac")
    combined = green_img.addBands(built_img)

    return combined.reduceRegions(
        collection=fc, reducer=ee.Reducer.mean(), scale=10
    )


def compute_lst(fc: ee.FeatureCollection, start: str, end: str) -> ee.FeatureCollection:
    """MODIS LST (Kelvin*0.02) -> Celsius, mean per polygon."""
    lst = (
        ee.ImageCollection("MODIS/061/MOD11A2")
        .filterDate(start, end)
        .filterBounds(fc)
        .select("LST_Day_1km")
        .mean()
        .multiply(0.02)
        .subtract(273.15)
        .rename("lst_c")
    )
    return lst.reduceRegions(collection=fc, reducer=ee.Reducer.mean(), scale=1000)


def compute_population(fc: ee.FeatureCollection, year: int = 2020) -> ee.FeatureCollection:
    """WorldPop population count -> density per km^2 per polygon."""
    pop = (
        ee.ImageCollection("WorldPop/GP/100m/pop")
        .filter(ee.Filter.eq("year", year))
        .mosaic()
        .rename("pop_count")
    )
    summed = pop.reduceRegions(collection=fc, reducer=ee.Reducer.sum(), scale=100)
    return summed


def compute_building_height(fc: ee.FeatureCollection) -> ee.FeatureCollection:
    """GHSL average building height per polygon."""
    ghsl_h = ee.Image("JRC/GHSL/P2023A/GHS_BUILT_H/2018").select("built_height").rename("bldg_height_m")
    return ghsl_h.reduceRegions(collection=fc, reducer=ee.Reducer.mean(), scale=100)


def fc_to_df(fc: ee.FeatureCollection, value_field: str) -> pd.DataFrame:
    info = fc.getInfo()
    rows = []
    for f in info["features"]:
        props = f["properties"]
        rows.append({"neighborhood_id": props["neighborhood_id"], value_field: props.get("mean", props.get("sum"))})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
def main():
    print("Initializing Earth Engine...")
    init_ee()

    print(f"Loading boundaries from {RAW_BOUNDARIES_PATH} ...")
    gdf = load_boundaries()
    print(f"  {len(gdf)} neighborhoods loaded")

    gdf_metric = gdf.to_crs("EPSG:3857")
    gdf["area_km2"] = gdf_metric.geometry.area / 1e6

    fc = gdf_to_ee_fc(gdf)

    print("Querying Dynamic World (green cover / built-up fraction)...")
    dw_fc = compute_dynamic_world(fc, SUMMER_START, SUMMER_END)
    dw_info = dw_fc.getInfo()
    dw_rows = [
        {
            "neighborhood_id": f["properties"]["neighborhood_id"],
            "green_frac": f["properties"].get("green_frac"),
            "built_frac": f["properties"].get("built_frac"),
        }
        for f in dw_info["features"]
    ]
    dw_df = pd.DataFrame(dw_rows)

    print("Querying MODIS Land Surface Temperature...")
    lst_fc = compute_lst(fc, SUMMER_START, SUMMER_END)
    lst_df = fc_to_df(lst_fc, "lst_c")

    print("Querying WorldPop population...")
    pop_fc = compute_population(fc)
    pop_df = fc_to_df(pop_fc, "pop_count")

    print("Querying GHSL building height...")
    height_fc = compute_building_height(fc)
    height_df = fc_to_df(height_fc, "bldg_height_m")

    # ---- merge everything back onto the boundaries ----
    merged = gdf.merge(dw_df, on="neighborhood_id").merge(lst_df, on="neighborhood_id") \
                .merge(pop_df, on="neighborhood_id").merge(height_df, on="neighborhood_id")

    merged["green_cover_pct"] = (merged["green_frac"] * 100).round(2)
    merged["building_pct"] = (merged["built_frac"] * 100).round(2)
    merged["population_density_km2"] = (merged["pop_count"] / merged["area_km2"]).round(0)
    merged["avg_building_height_m"] = merged["bldg_height_m"].round(1)

    city_mean_lst = merged["lst_c"].mean()
    merged["temp_anomaly_c"] = (merged["lst_c"] - city_mean_lst).round(2)

    merged["postal_code"] = merged["neighborhood_id"]  # placeholder if you don't have real postal codes

    final_cols = [
        "neighborhood_id", "name", "postal_code",
        "green_cover_pct", "building_pct", "population_density_km2",
        "avg_building_height_m", "temp_anomaly_c", "geometry",
    ]
    final_gdf = gpd.GeoDataFrame(merged[final_cols], geometry="geometry", crs="EPSG:4326")

    final_gdf.to_file(OUTPUT_PATH, driver="GeoJSON")
    print(f"\nWrote real data for {len(final_gdf)} neighborhoods to {OUTPUT_PATH}")
    print(final_gdf.drop(columns="geometry").head())
    print("\nNow just re-run: python src/main.py  (or re-run the notebook)")


if __name__ == "__main__":
    main()
