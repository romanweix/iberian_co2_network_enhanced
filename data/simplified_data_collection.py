import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import os
import rasterio
from matplotlib.lines import Line2D
from pathlib import Path
from shapely.geometry import Point, MultiPoint, LineString, box
from shapely.ops import nearest_points
from sklearn.cluster import KMeans
from scipy.spatial import Delaunay
from geopy.distance import geodesic
from pyproj import Transformer


"""

0. User‑controlled parameters

"""

# Bounding box (lon_min, lat_min, lon_max, lat_max)
BOUNDING_BOX = (-10.0, 36.0, 4.5, 44)      # lon_min, lat_min, lon_max, lat_max
PORTUGUESE_BBOX = (-11, 36.0, -6.0, 42.3)  # lon_min, lat_min, lon_max, lat_max (approximate)
BALEARIC_BBOX = (0.5, 38,  4.5, 40.2)     # lon_min, lat_min, lon_max, lat_max (approximate)

N_SOURCE_CLUSTERS = 12                  # total → includes manual island clusters      PREVIOUS VALUE: 12
N_SINK_CLUSTERS = 6                     # confirmed sink clusters                      PREVIOUS VALUE: 6 
EMISSION_THRESHOLD = 10_000_000         # kg CO2 
UNCERTAIN_SINK_THRESHOLD = 1_000        # Mt CO2 (uncertain sinks) (set to 1000)
SINK_THRESHOLD = 15                     # Mt CO2 (confirmed sinks) (set to 15)

# Spain and Portugal GeoJSON files loading
spain_prov = gpd.read_file("data/spain-provinces.geojson") # only for altitudes approach
spain = gpd.read_file("sdata/pain.json")
portugal = gpd.read_file("data/portugal.json")
france = gpd.read_file("data/france.json")
andorra = gpd.read_file("data/andorra.json")
#morocco = gpd.read_file("data/morocco.json")
#algeria = gpd.read_file("data/algeria.json")

################
# Joining Spain and Portugal in a single GeoDataFrame
peninsula = gpd.GeoDataFrame(pd.concat([spain, portugal], ignore_index=True))
################

# Joining all of the neighbouring countries in a single different GeoDataFrame
#neighbours = gpd.GeoDataFrame(pd.concat([france, andorra, morocco, algeria], ignore_index=True))
neighbours = gpd.GeoDataFrame(pd.concat([france, andorra], ignore_index=True))

# Creating the figure
fig, ax = plt.subplots(figsize=(8, 6))

# Drawing all of the countries
spain.plot(ax=ax, color="#D6EAF8", edgecolor="black") #D6EAF8 is a clear blue
portugal.plot(ax=ax, color="#E8D4EF", edgecolor="black") #E6C8EB is a clear purple
france.plot(ax=ax, color="#FFF9C4", edgecolor="black", alpha=0.5) # neighbours are plotted in a lighter color
andorra.plot(ax=ax, color="#F5D6C6", edgecolor="black", alpha=0.5) # neighbours are plotted in a lighter color
#morocco.plot(ax=ax, color="#FFEB99", edgecolor="black", alpha=0.5) # neighbours are plotted in a lighter color
#algeria.plot(ax=ax, color="#E6EE9C", edgecolor="black", alpha=0.5) # neighbours are plotted in a lighter color


"""

1. CO2 sources identification & clustering

"""

# --------------------------------------------------------------------------- #
# 1. Data loading and initial processing
# --------------------------------------------------------------------------- #

excel_database = 'Spanish and Portuguese emissions and sinks database.xlsx'
emissions_df = pd.read_excel(excel_database,
                             sheet_name="2020 Spain + 2017 Portugal em.")
emissions_df["orig_idx"] = emissions_df.index
emissions_df['geometry'] = emissions_df.apply(
    lambda row: Point(row['Longitude'], row['Latitude']), axis=1)

emissions_gdf = gpd.GeoDataFrame(emissions_df, geometry='geometry')
emissions_gdf.set_crs('EPSG:4326', inplace=True)
emissions_gdf = emissions_gdf[emissions_gdf.geometry.notnull()
                              & ~emissions_gdf.geometry.is_empty]

# --------------------------------------------------------------------------- #
# 2. Emissions filtering: relevant vs non-relevant (threshold-based)
# --------------------------------------------------------------------------- #

emission_threshold = 10_000_000           # kg CO2
relevant_emissions_gdf = emissions_gdf[
    emissions_gdf["CO2 emissions"] >= emission_threshold].copy()
non_relevant_emissions_gdf = emissions_gdf[
    emissions_gdf["CO2 emissions"] < emission_threshold].copy()

# --------------------------------------------------------------------------- #
# 3. Manual clustering (only if the centroid falls within the Peninsula + Balearic Islands)
# --------------------------------------------------------------------------- #

manual_clusters = [[422],
    [418, 419, 423, 425, 670, 680],
    [420, 421, 424],
    [442],
    [321],
    [0],
    [32],
    [157, 902],
    [679],
    [821],
    [709, 948, 158, 151, 155, 159],
    [156, 160, 947, 154],
    [716, 152, 951],
    [965, 691, 153]]

relevant_emissions_gdf["Source cluster"] = np.nan

def centroid_inside_bbox(points, bbox):
    """Return True if the centroid of *points* lies within *bbox*."""
    centroid = MultiPoint(points).centroid
    lon_min, lat_min, lon_max, lat_max = bbox
    return (lon_min <= centroid.x <= lon_max) and (lat_min <= centroid.y <= lat_max)

rows_to_drop = []
manual_cluster_id = 0                        # consecutive IDs only for clusters kept

for group in manual_clusters:
    subset = relevant_emissions_gdf[relevant_emissions_gdf["orig_idx"].isin(group)]
    if subset.empty:
        continue
    # bbox check -------------------------------------------------------------
    if centroid_inside_bbox(subset.geometry.tolist(), BOUNDING_BOX):
        # keep the cluster
        relevant_emissions_gdf.loc[subset.index, "Source cluster"] = manual_cluster_id
        manual_cluster_id += 1
    else:
        # drop every point of this cluster from further analysis
        rows_to_drop.extend(subset.index.tolist())

# Remove the discarded island/Africa points
relevant_emissions_gdf.drop(index=rows_to_drop, inplace=True)

# --------------------------------------------------------------------------- #
# 4. Automatic clustering (weighted k-means) only for the remaining points
# --------------------------------------------------------------------------- #

# Weighted k-means helper ----------------------------------------------------
def weighted_kmeans(all_real_nodes_coords, n_clusters, weights, max_iter=300):
    """Classic k-means + centroid re-weighting at every iteration."""
    kmeans = KMeans(n_clusters=n_clusters, max_iter=max_iter,
                    n_init=10, random_state=1).fit(all_real_nodes_coords)
    for _ in range(max_iter):
        dist = np.linalg.norm(all_real_nodes_coords[:, None] -
                              kmeans.cluster_centers_, axis=2)
        wdist = dist * weights.values[:, None]
        labels = np.argmin(wdist, axis=1)
        new_centroids = np.array([
            np.average(all_real_nodes_coords[labels == j], axis=0,
                       weights=weights[labels == j])
            for j in range(n_clusters)])
        if np.allclose(kmeans.cluster_centers_, new_centroids):
            break
        kmeans.cluster_centers_ = new_centroids
    return kmeans

# Pre-processing --------------------------------------------------------------
relevant_emissions_gdf['weight'] = (relevant_emissions_gdf['CO2 emissions'] /
                                    relevant_emissions_gdf['CO2 emissions'].sum())

# Projecting to UTM 30N for accurate distance measurements (meters)
relevant_emissions_gdf_proj = relevant_emissions_gdf.to_crs("EPSG:25830")
auto_gdf = relevant_emissions_gdf_proj[relevant_emissions_gdf_proj["Source cluster"].isna()].copy()

auto_coords_proj = np.array(auto_gdf.geometry.apply(
    lambda p: (p.x, p.y)).tolist())
auto_weights = auto_gdf["weight"]

# Total number of final clusters (peninsula_only)
n_auto_clusters = N_SOURCE_CLUSTERS - manual_cluster_id
if n_auto_clusters <= 0:
    raise ValueError("El número de clusters manuales supera el total deseado.")

# Weighted k-means execution (only for points not assigned to manual clusters)
source_kmeans = weighted_kmeans(auto_coords_proj,
                                n_clusters=n_auto_clusters,
                                weights=auto_weights)

# ID assignments for automatic clusters: start from the last manual cluster ID + 1
auto_ids = range(manual_cluster_id, manual_cluster_id + n_auto_clusters)
id_map = dict(zip(range(n_auto_clusters), auto_ids))
auto_gdf["Source cluster"] = [id_map[l] for l in source_kmeans.labels_]

# Integrating the automatic cluster assignments back into the projected GeoDataFrame and then transforming back to WGS84
relevant_emissions_gdf_proj.update(auto_gdf)
relevant_emissions_gdf = relevant_emissions_gdf_proj.to_crs("EPSG:4326")

# --------------------------------------------------------------------------- #
# 5. Rebuilding the complete GeoDataFrame (without the discarded points)
# --------------------------------------------------------------------------- #

non_relevant_emissions_gdf["Source cluster"] = np.nan

# Joining all points except the previously discarded ones
emissions_gdf = pd.concat([relevant_emissions_gdf,
                           non_relevant_emissions_gdf]).sort_index()

# --------------------------------------------------------------------------- #
# 6. Cluster centroids and attributes calculation (for source nodes creation)
# --------------------------------------------------------------------------- #

all_clustered_proj = (emissions_gdf[emissions_gdf["Source cluster"].notna()]
                      .copy().to_crs("EPSG:25830"))

cluster_centroids = (all_clustered_proj.groupby("Source cluster")
                     .geometry.apply(lambda x: MultiPoint(list(x)).centroid))

cluster_emissions = (emissions_gdf.groupby("Source cluster")["CO2 emissions"]
                     .sum().reset_index()
                     .rename(columns={"CO2 emissions": "Total CO2 emissions"}))

source_nodes_gdf_proj = gpd.GeoDataFrame(cluster_emissions,
                                         geometry=cluster_centroids.tolist(),
                                         crs="EPSG:25830")
source_nodes_gdf = source_nodes_gdf_proj.to_crs("EPSG:4326")

source_nodes_gdf = source_nodes_gdf[source_nodes_gdf.geometry.notnull()
                                    & ~source_nodes_gdf.geometry.is_empty]

# --------------------------------------------------------------------------- #
# 7. Line creation between emission points and their cluster centroids (for visualization)
# --------------------------------------------------------------------------- #

source_lines = []
for _, row in emissions_gdf.iterrows():
    cid = row['Source cluster']
    if pd.isna(cid):
        continue
    emission_pt = row['geometry']
    centroid_pt = source_nodes_gdf[source_nodes_gdf['Source cluster'] == cid]\
                  ['geometry'].values
    if len(centroid_pt) > 0:
        source_lines.append(LineString([emission_pt, centroid_pt[0]]))

source_lines_gdf = gpd.GeoDataFrame(geometry=source_lines,
                                    crs=emissions_gdf.crs)




"""

2. Confirmed sinks identification & clustering

"""

sinks_df = pd.read_excel(excel_database, sheet_name="Sinks")
sinks_df["geometry"] = sinks_df.apply(lambda r: Point(r["Longitude"], r["Latitude"]), axis=1)
sinks_gdf = gpd.GeoDataFrame(sinks_df, geometry="geometry", crs="EPSG:4326")

relevant_sinks_gdf = sinks_gdf[sinks_gdf["Capacity"] >= SINK_THRESHOLD].copy()
non_relevant_sinks_gdf = sinks_gdf[sinks_gdf["Capacity"] < SINK_THRESHOLD].copy()

sink_proj = relevant_sinks_gdf.to_crs("EPSG:25830")
sink_coords = np.array([(pt.x, pt.y) for pt in sink_proj.geometry])
sink_weights = sink_proj["Capacity"] / sink_proj["Capacity"].sum()

sink_kmeans = weighted_kmeans(sink_coords, N_SINK_CLUSTERS, sink_weights)

relevant_sinks_gdf["Sink cluster"] = sink_kmeans.labels_
non_relevant_sinks_gdf["Sink cluster"] = np.nan

sinks_gdf = pd.concat([relevant_sinks_gdf, non_relevant_sinks_gdf])

sink_centroids = sink_kmeans.cluster_centers_
centroid_points = [Point(x, y) for x, y in sink_centroids]
sink_cluster_cap = sinks_gdf.groupby("Sink cluster")["Capacity"].sum().reset_index().rename(columns={"Capacity": "Total capacity"})
sink_nodes_gdf_proj = gpd.GeoDataFrame(sink_cluster_cap, geometry=centroid_points, crs="EPSG:25830")
sink_nodes_gdf = sink_nodes_gdf_proj.to_crs("EPSG:4326")

sink_lines = [] # Creating a list for storing the sink_lines

# Iterating on each sink point in sinks_gdf
for _, row in sinks_gdf.iterrows():
    cluster_id = row['Sink cluster'] # Obtaining cluster ID
    sink_point = row['geometry'] # sink point geometry
    
    cluster_point = sink_nodes_gdf[sink_nodes_gdf['Sink cluster'] == cluster_id]['geometry'].values # Finding the corresponding centroid in sink_nodes_gdf
    
    # If the corresponding cluster is found, the line is created
    if len(cluster_point) > 0:
        line = LineString([sink_point, cluster_point[0]])
        sink_lines.append(line)

# Creating a GeoDataFrame with the sink_lines
sink_lines_gdf = gpd.GeoDataFrame(geometry=sink_lines, crs=sinks_gdf.crs)



"""

3. Uncertain sinks (> 1000 Mt) & utilization sites

"""

uncertain_sinks_df = pd.read_excel(excel_database, sheet_name="Uncertain sinks")
uncertain_sinks_df["geometry"] = uncertain_sinks_df.apply(lambda r: Point(r["Longitude"], r["Latitude"]), axis=1)
uncertain_sinks_gdf = gpd.GeoDataFrame(uncertain_sinks_df, geometry="geometry", crs="EPSG:4326")
relevant_uncert_sinks_gdf = uncertain_sinks_gdf[uncertain_sinks_gdf["Capacity (Mt)"] >= UNCERTAIN_SINK_THRESHOLD].copy()

util_df = pd.read_excel(excel_database, sheet_name="Selected CO2 utilization")
util_df["geometry"] = util_df.apply(lambda r: Point(r["Longitude"], r["Latitude"]), axis=1)

util_df["Capacity S1 (Mt/5y)"] = util_df["Capacity S1 (Mt/y)"] * 5
util_df["Capacity S2 (Mt/5y)"] = util_df["Capacity S2 (Mt/y)"] * 5
util_df["Capacity S3 (Mt/5y)"] = util_df["Capacity S3 (Mt/y)"] * 5

util_gdf = gpd.GeoDataFrame(util_df, geometry="geometry", crs="EPSG:4326")





"""

4. Trading nodes -> Not used in the simplified model

"""

# international_nodes_df = pd.read_excel(excel_database, sheet_name="International connections")

# international_nodes_df['geometry'] = international_nodes_df.apply(lambda row: Point(row['Longitude'], row['Latitude']), axis=1) # Creating the geometry column
# international_nodes_gdf = gpd.GeoDataFrame(international_nodes_df, geometry='geometry') # Transforming the dataframe into a geodataframe
# international_nodes_gdf.set_crs('EPSG:4326', inplace=True) # Defining the geographic coordinates system




"""

5. Node assembly -> We put manual nodes first in order to avoid them being altered when we change the number of automatic clusters

"""

# a) Uncertain sinks (manual category)
uncert_sink_nodes = relevant_uncert_sinks_gdf.copy()
uncert_sink_nodes["Node type"] = "S"
uncert_sink_nodes["Stage"] = "First"
uncert_sink_nodes["Total CO2 emissions"] = None
uncert_sink_nodes["Total CO2 capacity"] = uncert_sink_nodes["Capacity (Mt)"]
uncert_sink_nodes["Annual CO2 utilization capacity"] = None

# b) Industrial utilization nodes (manual category)
util_nodes = util_gdf.copy()
util_nodes["Node type"] = "K"
util_nodes["Stage"] = "Second"
util_nodes["Total CO2 emissions"] = None
util_nodes["Total CO2 capacity"] = None
util_nodes["Annual CO2 utilization capacity"] = util_nodes["Capacity S2 (Mt/y)"].round(2)

# c) Automatic source nodes
source_nodes = source_nodes_gdf.copy()
source_nodes["Node type"] = "E"
source_nodes["Stage"] = "First"
source_nodes["Total CO2 emissions"] = round(source_nodes["Total CO2 emissions"] / 1e9, 2)
source_nodes["Total CO2 capacity"] = None
source_nodes["Annual CO2 utilization capacity"] = None

# d) Automatic sink nodes
sink_nodes = sink_nodes_gdf.copy()
sink_nodes["Node type"] = "S"
sink_nodes["Stage"] = "First"
sink_nodes["Total CO2 emissions"] = None
sink_nodes.rename(columns={"Total capacity": "Total CO2 capacity"}, inplace=True)
sink_nodes["Annual CO2 utilization capacity"] = None

# # g) Automatic trade nodes
# trade_nodes = international_nodes_gdf.copy()
# trade_nodes["Node type"] = "M"
# trade_nodes["Stage"] = "First"
# trade_nodes["Total CO2 emissions"] = None
# trade_nodes["Total CO2 capacity"] = None
# trade_nodes["Annual CO2 utilization capacity"] = None

# h) Node identifiers – sequential to preserve manual indices
frames = [uncert_sink_nodes, util_nodes, source_nodes, sink_nodes] # ADD AUXILIARY AND TRADE NODES IF NEEDED
current_id = 1
for df in frames:
    df.reset_index(inplace=True, drop=True)
    df["Node identifier"] = range(current_id, current_id + len(df))
    current_id += len(df)




"""

6. Combine and enrich with common columns

"""

combined = pd.concat(frames, ignore_index=True)
combined["Country"] = "Spain"  # default, will adjust later if needed
combined["Transport method"] = "Onshore"  # default transport mode
combined["Height"] = None

all_nodes_gdf = gpd.GeoDataFrame(combined, geometry="geometry", crs="EPSG:4326")

all_nodes_gdf = all_nodes_gdf[["Node identifier", "Node type", "Total CO2 emissions", "Total CO2 capacity", "Annual CO2 utilization capacity", "Country", "Transport method", "Height", "Stage", "geometry"]]

def inside_bbox(series, bbox):
    """Return boolean mask for GeoSeries of points inside bbox."""
    lon_min, lat_min, lon_max, lat_max = bbox
    x = series.x
    y = series.y
    return (x.between(lon_min, lon_max) & y.between(lat_min, lat_max))

mask_bal = inside_bbox(all_nodes_gdf.geometry, BALEARIC_BBOX)
all_nodes_gdf.loc[mask_bal, "Transport method"] = "Offshore"

all_nodes_gdf.loc[(all_nodes_gdf['Node identifier'] == 1), 'Transport method'] = 'Offshore'
all_nodes_gdf.loc[(all_nodes_gdf['Node identifier'] == 2), 'Transport method'] = 'Offshore'
all_nodes_gdf.loc[(all_nodes_gdf['Node identifier'] == 3), 'Transport method'] = 'Offshore'

# Assigning height values to all nodes

with rasterio.open("merged_srtm.tif") as src: # Reading the merged raster file
    coords = [(point.x, point.y) for point in all_nodes_gdf.geometry]

    # Transformation nach EPSG:3035
    transformer = Transformer.from_crs(
        "EPSG:4326",
        "EPSG:3035",
        always_xy=True
    )
    coords_transformed = [
        transformer.transform(lon, lat)
        for lon, lat in coords
    ]

    heights = list(src.sample(coords_transformed)) # Reading the raster values at the coordinates of the nodes
    heights = [val[0] if val[0] != src.nodata else 0 for val in heights]

all_nodes_gdf["Height"] = [height if mode == "Onshore" else 0 
    for height, mode in zip(heights, all_nodes_gdf["Transport method"])] # Assigning the heights to the nodes based on their transport method. Only onshore nodes are assigned heights

# print(all_nodes_gdf)

"""

7. Auxiliary nodes -> Not used in the simplified model

"""

balearic_box = box(*BALEARIC_BBOX)            # shapely Polygon of the bounding box

# Keep only geometries that DO NOT intersect the Balearic bbox
peninsula = peninsula[~peninsula.geometry.intersects(balearic_box)].reset_index(drop=True)

# ------------------------------------------------------------------
# Auxiliary coastal nodes for offshore points
# ------------------------------------------------------------------
from shapely.ops import nearest_points
from geopy.distance import geodesic
import geopandas as gpd

MAX_DIST_KM = 180

offshore_wgs  = all_nodes_gdf[all_nodes_gdf["Transport method"] == "Offshore"].copy()
offshore_proj = offshore_wgs.to_crs("EPSG:25830")

peninsula_proj = peninsula.to_crs("EPSG:25830")
mainland_union = peninsula_proj.union_all()

aux_records = []

for idx, row in offshore_proj.iterrows():
    off_pt_proj = row.geometry
    nearest_pt_proj = nearest_points(off_pt_proj, mainland_union)[1]

    # 1st filter (flat, fast)
    if off_pt_proj.distance(nearest_pt_proj) > MAX_DIST_KM * 1_000:
        continue

    # 2nd filter (accurate geodesic distance)
    off_pt_wgs     = offshore_wgs.loc[idx, "geometry"]
    nearest_pt_wgs = gpd.GeoSeries([nearest_pt_proj], crs="EPSG:25830")\
                           .to_crs("EPSG:4326").iloc[0]

    if geodesic((off_pt_wgs.y, off_pt_wgs.x),
                (nearest_pt_wgs.y, nearest_pt_wgs.x)).kilometers <= MAX_DIST_KM:

        aux_records.append({
            "geometry" : nearest_pt_proj,
            "Stage"    : row["Stage"]          # heredamos Stage del offshore
        })

# --- Build provisional GeoDataFrame ------------------------------------------------
aux_tmp = gpd.GeoDataFrame(aux_records, crs="EPSG:25830")

if not aux_tmp.empty:
    
    # Unique key based on geometry (WKB → string)
    aux_tmp["geom_key"] = aux_tmp.geometry.apply(lambda g: g.wkb_hex)

    # Priority: 'Second' before 'First'
    aux_tmp["priority"] = aux_tmp["Stage"].map({"Second": 1, "First": 0})

    aux_tmp = aux_tmp.sort_values("priority", ascending=False)

    # Removing duplicates while keeping the row with the highest priority
    aux_nodes_gdf_proj = (
        aux_tmp.drop_duplicates(subset="geom_key")
               .drop(columns=["geom_key", "priority"])
               .reset_index(drop=True)
    )

    # ------------------------------------------------------------------
    # Final atribute assignment for auxiliary nodes (after deduplication and before concatenation)
    # ------------------------------------------------------------------
    aux_nodes_gdf = aux_nodes_gdf_proj.to_crs("EPSG:4326")
    aux_nodes_gdf["Node type"]                       = "A"
    aux_nodes_gdf["Total CO2 emissions"]             = None
    aux_nodes_gdf["Total CO2 capacity"]              = None
    aux_nodes_gdf["Annual CO2 utilization capacity"] = None
    aux_nodes_gdf["Country"]                         = "Spain"
    aux_nodes_gdf["Transport method"]                = "Onshore"
    aux_nodes_gdf["Height"]                          = None

    # Sequential node identifiers
    last_id = all_nodes_gdf["Node identifier"].max()
    aux_nodes_gdf["Node identifier"] = range(last_id + 1,
                                             last_id + 1 + len(aux_nodes_gdf))

    # Join with the master GeoDataFrame, ensuring that auxiliary nodes are added at the end and that indices are reset
    all_nodes_gdf = pd.concat([all_nodes_gdf, aux_nodes_gdf],
                              ignore_index=True)

mask_port = inside_bbox(all_nodes_gdf.geometry, PORTUGUESE_BBOX)
all_nodes_gdf.loc[mask_port, "Country"] = "Portugal"

print(all_nodes_gdf)




####################################################################################################################################################################################################################################################
####################################################################################################################################################################################################################################################
# Pipeline candidates identification and pipeline candidate geodataframe creation
####################################################################################################################################################################################################################################################
####################################################################################################################################################################################################################################################

final_proj = all_nodes_gdf.to_crs(epsg=3857) # Ensure working in metric projection for triangulation

coords = np.array([(geom.x, geom.y) for geom in final_proj.geometry]) # Extracting coordinates and linking them to nodes

tri = Delaunay(coords) # Applying Delaunay
edges = set()
for simplex in tri.simplices:
    for i in range(3):
        edge = tuple(sorted((simplex[i], simplex[(i + 1) % 3])))
        edges.add(edge)

pipeline_candidates = []

for idx, (i, j) in enumerate(edges): # Original coordinates (WGS84)
    pt1 = all_nodes_gdf.geometry.iloc[i]
    pt2 = all_nodes_gdf.geometry.iloc[j]

    line = LineString([pt1, pt2])
    node_ids = (all_nodes_gdf["Node identifier"].iloc[i], all_nodes_gdf["Node identifier"].iloc[j]) # Node identifiers
    node_types = (all_nodes_gdf["Node type"].iloc[i], all_nodes_gdf["Node type"].iloc[j]) # Node types
    node_countries = (all_nodes_gdf["Country"].iloc[i], all_nodes_gdf["Country"].iloc[j]) # Node countries

    methods = [all_nodes_gdf["Transport method"].iloc[i], all_nodes_gdf["Transport method"].iloc[j]] # Transport method
    transport_method = "Offshore" if "Offshore" in methods else "Onshore"

    stage = [all_nodes_gdf["Stage"].iloc[i], all_nodes_gdf["Stage"].iloc[j]] # Decision stage
    decision_stage = "Second" if "Second" in stage else "First"
    
    # height_i = all_nodes_gdf["Height"].iloc[i]
    # height_j = all_nodes_gdf["Height"].iloc[j]

    if transport_method == "Offshore":
        # For marine pipelines force sea-level
        node_heights = (0, 0)
    else:
        # For on-shore pipelines use the node heights if numeric
        def _safe(h):
            try:                     # convierte a float; NaN → ValueError
                return float(h)
            except (TypeError, ValueError):
                return 0           # fallback razonable si faltan datos

        height_i = _safe(all_nodes_gdf["Height"].iloc[i])
        height_j = _safe(all_nodes_gdf["Height"].iloc[j])
        node_heights = (height_i, height_j)

    pt1_proj = gpd.GeoSeries([pt1], crs="EPSG:4326").to_crs("EPSG:3857").iloc[0]
    pt2_proj = gpd.GeoSeries([pt2], crs="EPSG:4326").to_crs("EPSG:3857").iloc[0]
    line_proj = LineString([pt1_proj, pt2_proj])
    distance_km = line_proj.length / 1000 # Distance in kilometres

    pipeline_candidates.append({
        "Pipeline identifier": f"{idx}_a",
        "Node connection": node_ids,
        "Connection type": node_types,
        "Longitude (km)": round(distance_km, 2),
        "Transport method": transport_method,
        "Countries": node_countries,
        "Node heights": node_heights,
        "Stage": decision_stage,
        "Cumul. pos. height (m)": 0,
        "Cumul. neg. height (m)": 0,
        "Max height (m)": 0,
        "Max height distance (km)": 0,
        "Cities crossed": 0,
        "Cities populations": np.nan,
        "geometry": line
    })

    line_reversed = LineString([pt2, pt1])
    pipeline_candidates.append({
        "Pipeline identifier": f"{idx}_b",
        "Node connection": (node_ids[1], node_ids[0]),
        "Connection type": (node_types[1], node_types[0]),
        "Longitude (km)": round(distance_km, 2),
        "Transport method": transport_method,
        "Countries": (node_countries[1], node_countries[0]),
        "Node heights": tuple(reversed(node_heights)),
        "Stage": decision_stage,
        "Cumul. pos. height (m)": 0,
        "Cumul. neg. height (m)": 0,
        "Max height (m)": 0,
        "Max height distance (km)": 0,
        "Cities crossed": 0,
        "Cities populations": np.nan,
        "geometry": line_reversed
    })

pipeline_candidates_gdf = gpd.GeoDataFrame(pipeline_candidates, geometry="geometry", crs="EPSG:4326")


###############################
###############################

# Computing the cumulative positive and negative heights

###############################
###############################

pipeline_candidates_m = pipeline_candidates_gdf.to_crs(epsg=3857) # Transforming to metric projection for height calculation

with rasterio.open("merged_srtm.tif") as src:
    for idx, row in pipeline_candidates_m.iterrows():
        if pipeline_candidates_gdf.at[idx, "Transport method"] == "Offshore": # Skip offshore pipelines
            pipeline_candidates_gdf.at[idx, "Cumul. pos. height (m)"] = 0
            pipeline_candidates_gdf.at[idx, "Cumul. neg. height (m)"] = 0
            continue

        line_m = row.geometry
        length_m = line_m.length
        num_segments = int(length_m // 500) + 1  # 1 km segments for cumulative height calculation
        if num_segments < 2: # Ensure at least 2 segments for interpolation
            num_segments = 2

        points_m = [line_m.interpolate(float(i) / (num_segments - 1), normalized=True) for i in range(num_segments)] # Interpolating points along the line

        gdf_points = gpd.GeoDataFrame(geometry=points_m, crs=3857).to_crs(epsg=4326) # Transforming points back to WGS84 for height sampling
        coords = [(pt.x, pt.y) for pt in gdf_points.geometry]

        heights_raw = list(src.sample(coords)) # Sampling heights from the raster
        heights = [val[0] if val[0] != src.nodata else np.nan for val in heights_raw]
        heights = [h for h in heights if not np.isnan(h)]

        if len(heights) < 2:
            pipeline_candidates_gdf.at[idx, "Cumul. pos. height (m)"] = 0
            pipeline_candidates_gdf.at[idx, "Cumul. neg. height (m)"] = 0
            pipeline_candidates_gdf.at[idx, "Max height (m)"] = 0
            continue

        pos, neg = 0, 0 # Initialize cumulative heights
        for h1, h2 in zip(heights[:-1], heights[1:]):
            delta = h2 - h1
            if delta > 0:
                pos += delta # Cumulative positive height
            else:
                neg += abs(delta) # Cumulative negative height

        pipeline_candidates_gdf.at[idx, "Cumul. pos. height (m)"] = pos
        pipeline_candidates_gdf.at[idx, "Cumul. neg. height (m)"] = neg

        max_height = max(heights)
        pipeline_candidates_gdf.at[idx, "Max height (m)"] = max_height

        max_index = heights.index(max_height)

        max_height_point_wgs = gdf_points.geometry.iloc[max_index] # Point with maximum height in WGS84

        max_height_point_proj = gpd.GeoSeries([max_height_point_wgs], crs="EPSG:4326").to_crs("EPSG:3857").iloc[0] # Transforming to metric projection

        distance_to_max_point_m = line_m.project(max_height_point_proj) # Distance to the maximum height point in metres
        distance_to_max_point_km = distance_to_max_point_m / 1000

        pipeline_candidates_gdf.at[idx, "Max height distance (km)"] = round(distance_to_max_point_km, 2)


###############################
###############################

# Including the pipeline distance to close cities

###############################
###############################

all_cities = pd.read_excel(excel_database, sheet_name="Cities")

all_cities['geometry'] = all_cities.apply(lambda row: Point(row['Longitude'], row['Latitude']), axis=1) # Creating the geometry column
all_cities_gdf = gpd.GeoDataFrame(all_cities, geometry='geometry') # Transforming the dataframe into a geodataframe
all_cities_gdf.set_crs('EPSG:4326', inplace=True) # Defining the geographic coordinates system

filtered_cities_gdf = all_cities_gdf[all_cities_gdf['Population'] >= 20000] # Filtering cities with population >= 20.000

filtered_cities_gdf_m = filtered_cities_gdf.to_crs(epsg=3857) # Transforming to metric projection for distances calculation

def compute_radius_m (population, k=7): # Function to compute the radius of the buffer around the cities
    return k * np.sqrt(population) # Radius in m

filtered_cities_gdf_m['Radius'] = filtered_cities_gdf_m['Population'].apply(compute_radius_m) # Computing the radius of the cities
filtered_cities_gdf_m['Buffer geometry'] = filtered_cities_gdf_m['geometry'].buffer(filtered_cities_gdf_m['Radius']) # Creating the buffer around the cities

city_buffers_gdf_m = filtered_cities_gdf_m.copy() # Extracting the buffers
city_buffers_gdf_m.set_geometry('Buffer geometry', inplace=True) # Setting the geometry to the buffer

def compute_distance_to_cities(line, city_buffers_gdf_m):
    intersected = city_buffers_gdf_m[city_buffers_gdf_m.intersects(line)]
    count = len(intersected)
    populations = intersected['Population'].tolist()
    return count, populations

results = pipeline_candidates_m['geometry'].apply(
    lambda line: compute_distance_to_cities(line, city_buffers_gdf_m)
)

city_buffers_gdf = city_buffers_gdf_m.to_crs(epsg=4326) # Transforming the buffers back to WGS84 for plotting

pipeline_candidates_gdf['Cities crossed'] = results.apply(lambda x: x[0])
pipeline_candidates_gdf['Cities populations'] = results.apply(lambda x: x[1])

# print(all_nodes_gdf.tail(50)) # Displaying the first rows of the pipeline candidates GeoDataFrame
# print(pipeline_candidates_gdf.tail(10))



# ------------------------------------------------------------------
# Enforce rule for offshore storage nodes:
#   If a candidate pipeline is Offshore and one end is a storage node (S),
#   then the other end MUST be an auxiliary node (A), and it MUST be the
#   nearest auxiliary node to that storage. Otherwise, drop the candidate.
# ------------------------------------------------------------------

# --- 1) Build nearest auxiliary mapping for each offshore sink ----------
# (work in metric projection for robust distance computation)
_all_nodes_3857 = all_nodes_gdf.to_crs(epsg=3857)

offshore_sinks = _all_nodes_3857[
    (_all_nodes_3857["Node type"] == "S") &
    (_all_nodes_3857["Transport method"] == "Offshore")
][["Node identifier", "geometry"]].copy()

aux_nodes = _all_nodes_3857[
    (_all_nodes_3857["Node type"] == "A")
][["Node identifier", "geometry"]].copy()

nearest_aux_for_sink = {}  # dict: sink_id -> closest aux_id

if not offshore_sinks.empty and not aux_nodes.empty:
    for _, srow in offshore_sinks.iterrows():
        # Compute distances from this sink to all auxiliaries (meters)
        dists = aux_nodes.geometry.distance(srow.geometry)
        min_idx = dists.idxmin()
        nearest_aux_for_sink[srow["Node identifier"]] = int(aux_nodes.loc[min_idx, "Node identifier"])

# --- 2) Helper to check whether an offshore edge S-? is valid ----------
def _is_valid_offshore_edge(row):
    """Return True if the offshore candidate passes the S-A(nearest) rule."""
    if row["Transport method"] != "Offshore":
        return True  # Only constrain offshore candidates

    (id_i, id_j)     = row["Node connection"]
    (type_i, type_j) = row["Connection type"]

    # Case 1: one end is S and the other is A → must be nearest A for that S
    # (accept both directions: S-A or A-S)
    if (type_i == "S" and type_j == "A"):
        # keep only if j is the nearest A for sink i (when mapping exists)
        nearest_j = nearest_aux_for_sink.get(id_i, None)
        return (nearest_j is None) or (id_j == nearest_j)

    if (type_i == "A" and type_j == "S"):
        nearest_i = nearest_aux_for_sink.get(id_j, None)
        return (nearest_i is None) or (id_i == nearest_i)

    # Any other offshore combination involving a sink (e.g., S-E, S-S, S-K, …) → drop
    if (type_i == "S") or (type_j == "S"):
        return False

    # If no sink is involved (e.g., A-A offshore), do not constrain here
    return True

# --- 3) Filter the GeoDataFrame ----------------------------------------
mask_keep = pipeline_candidates_gdf.apply(_is_valid_offshore_edge, axis=1)
pipeline_candidates_gdf = pipeline_candidates_gdf[mask_keep].reset_index(drop=True)

# If ya habías creado la versión en métrica, sincronízala también:
try:
    pipeline_candidates_m = pipeline_candidates_gdf.to_crs(epsg=3857)
except NameError:
    pass




# ------------------------------------------------------------------
# 1) Remove offshore pipelines longer than 200 km
# ------------------------------------------------------------------
long_offshore = (
    (pipeline_candidates_gdf["Transport method"] == "Offshore") &
    (pipeline_candidates_gdf["Longitude (km)"] > 200)
)
pipeline_candidates_gdf = pipeline_candidates_gdf[~long_offshore].reset_index(drop=True)

# Re-project again to keep metric version in sync
pipeline_candidates_m = pipeline_candidates_gdf.to_crs(epsg=3857)

# ------------------------------------------------------------------
# 2) Remove onshore pipelines that stray >10 km from the mainland
# ------------------------------------------------------------------
# Mainland polygon (union of Spain+Portugal, EPSG:3857)
peninsula_proj   = peninsula.to_crs(epsg=3857)
mainland_union   = peninsula_proj.union_all()          # replaces deprecated unary_union
buffer_10km      = mainland_union.buffer(10_000)       # 10 km buffer

def out_of_buffer(line, buffer_geom):
    """Return True if any part of *line* lies outside *buffer_geom*."""
    return not line.within(buffer_geom)

mask_outside = pipeline_candidates_m.apply(
    lambda row: (
        row["Transport method"] == "Onshore" and
        out_of_buffer(row.geometry, buffer_10km)
    ),
    axis=1
)

pipeline_candidates_gdf = pipeline_candidates_gdf[~mask_outside].reset_index(drop=True)
pipeline_candidates_m = pipeline_candidates_m[~mask_outside].reset_index(drop=True)

# print(f"{len(pipeline_candidates_gdf)} pipeline candidates retained after filtering.")

# ------------------------------------------------------------------
# 3) Eliminar tuberías OFFSHORE de segunda etapa (> 50 km)
# ------------------------------------------------------------------
mask_long_off_2nd = (
    (pipeline_candidates_gdf["Transport method"] == "Offshore") &
    (pipeline_candidates_gdf["Stage"]            == "Second")   &
    (pipeline_candidates_gdf["Longitude (km)"]   >  50)
)

pipeline_candidates_gdf = pipeline_candidates_gdf[~mask_long_off_2nd].reset_index(drop=True)
pipeline_candidates_m   = pipeline_candidates_m[~mask_long_off_2nd].reset_index(drop=True)

# print(f"{len(pipeline_candidates_gdf)} pipeline candidates retained after 3rd filter.")

# print(pipeline_candidates_gdf.tail(10))




####################################################################################################################################################################################################################################################
####################################################################################################################################################################################################################################################
# Plot settings
####################################################################################################################################################################################################################################################
####################################################################################################################################################################################################################################################

# Manually chart boundaries setting for Peninsula + Balearic Islands
ax.set_xlim([-10, 6])
ax.set_ylim([35, 44.3])

# # Manually chart boundaries setting for full system visualization
# ax.set_xlim([-29, 5])
# ax.set_ylim([27.2, 44.3])

# Taking x and y aixs out of the plot
ax.set_xticks([])
ax.set_yticks([])
ax.set_frame_on(False)  # Taking the frame out of the plot

###############################
###############################

# Drawing the CO2 source_nodes on the map

###############################
###############################

# emissions_gdf.plot(ax=ax, facecolor='none', edgecolor='#00008B', marker='o', markersize=emissions_gdf['CO2 emissions']**0.685 * 0.0002, alpha=0.8, label='CO$_2$ sources') # Drawing the CO2 source_nodes on the map

# source_nodes_gdf.plot(ax=ax, facecolor='none', edgecolor='#A30000', marker='o', markersize=source_nodes_gdf['Total CO2 emissions']**0.685 * 0.0002, label="Source nodes") # Drawing the CO2 source nodes on the map
source_nodes.plot(ax=ax, facecolor='#A30000', edgecolor='#A30000', marker='o', markersize=30, label="Source nodes") # Drawing the CO2 source nodes on the map

# for x, y, cluster in zip(emissions_gdf.geometry.x, emissions_gdf.geometry.y, emissions_gdf["Source cluster"]):
#    ax.text(x, y, str(cluster), fontsize=12, ha="center", va="center", color="black") # Labeling the Source clusters on the map

# for x, y, idx in zip(emissions_gdf.geometry.x, emissions_gdf.geometry.y, emissions_gdf.index):
#    ax.text(x, y, str(idx), fontsize=10, ha="center", va="center", color="black") # Labeling the Source clusters on the map

# source_lines_gdf.plot(ax=ax, color='#00008B', linewidth=0.6, linestyle="dashed", alpha=0.7) # Connection source_lines


###############################
###############################

# Drawing the CO2 sinks on the map

###############################
###############################


# sinks_gdf.plot(ax=ax, facecolor='none', edgecolor='#CC5500', marker='o', markersize=sinks_gdf['Capacity']**0.95 * 0.35, alpha=0.8, label='Potential sinks') # Drawing the CO2 source_nodes on the map

# sink_nodes_gdf.plot(ax=ax, facecolor='none', edgecolor='#228B22', marker='o', markersize=sink_nodes_gdf['Total capacity']**0.95 * 0.35, label="Sink nodes") # Drawing the sink nodes on the map
sink_nodes.plot(ax=ax, facecolor='#228B22', edgecolor='#228B22', marker='o', markersize=30, label="Sink nodes") # Drawing the sink nodes on the map

# for x, y, cluster in zip(sinks_gdf.geometry.x, sinks_gdf.geometry.y, sinks_gdf["Sink cluster"]):
#     ax.text(x, y, str(cluster), fontsize=12, ha="center", va="center", color="black") # Labeling the sink clusters on the map

# sink_lines_gdf.plot(ax=ax, color='#CC5500', linewidth=0.6, linestyle="dashed", alpha=0.7) # Connection sink_lines


# ###############################
# ###############################

# # Drawing other elements on the map

# ###############################
# ###############################

aux_nodes_gdf.plot(ax=ax, facecolor='#D4A017', edgecolor='#D4A017', marker='o', markersize=10, label="Auxiliary nodes") # Drawing the auxiliary nodes on the map

# international_nodes_gdf.plot(ax=ax, facecolor='#F28D8C', edgecolor='#F28D8C', marker='o', markersize=30, label="International nodes") # Drawing the trading nodes on the map

relevant_uncert_sinks_gdf.plot(ax=ax, facecolor="#228B22", edgecolor='#228B22', marker='o', markersize=30, label="Uncertain sink nodes") # Drawing the uncertain sinks on the map

util_nodes.plot(ax=ax, facecolor="#5A3A25", edgecolor="#5A3A25", marker='o', markersize=30, label="Uncertain utilization nodes") # Drawing the utilization nodes on the map

city_buffers_gdf.plot(ax=ax, facecolor='#6A5ACD', edgecolor='#6A5ACD', marker='o', markersize=3, label="Cities") # Drawing the cities on the map


first_stage = pipeline_candidates_gdf[pipeline_candidates_gdf['Stage'] == 'First']
second_stage = pipeline_candidates_gdf[pipeline_candidates_gdf['Stage'] == 'Second']

first_stage.plot(ax=ax, color='black', linewidth=0.6, alpha=0.7, linestyle='-') # Drawing the pipeline candidates on the map
second_stage.plot(ax=ax, color='black', linewidth=0.6, alpha=0.7, linestyle=(0, (6, 10))) # Drawing the pipeline candidates on the map


legend_elements = [
    # Line2D([0], [0], marker='o', color='w', label='CO₂ sources', markerfacecolor='none', markeredgecolor= '#00008B', markersize=10),
    Line2D([0], [0], marker='o', color='w', label='Source nodes', markerfacecolor='#A30000', markeredgecolor= '#A30000', markersize=10),
    # Line2D([0], [0], marker='o', color='w', label='Sinks', markerfacecolor='none', markeredgecolor= '#CC5500', markersize=10),
    Line2D([0], [0], marker='o', color='w', label='Sink nodes', markerfacecolor="#228B22", markeredgecolor='#228B22', markersize=10),
    # Line2D([0], [0], marker='o', color='w', label='Uncertain sink nodes', markerfacecolor='#7AAC2F', markeredgecolor='#7AAC2F', markersize=10),
    Line2D([0], [0], marker='o', color='w', label='Uncertain utilization nodes', markerfacecolor='#5A3A25', markeredgecolor='#5A3A25', markersize=10),
    Line2D([0], [0], marker='o', color='w', label='Auxiliary nodes', markerfacecolor='#D4A017', markersize=10),
    # Line2D([0], [0], marker='o', color='w', label='Trade nodes', markerfacecolor='#F28D8C', markersize=10),
    Line2D([0], [0], marker='o', color='w', label='Cities', markerfacecolor='#6A5ACD', markersize=10),
]

ax.legend(handles=legend_elements, loc='lower right')

fig.savefig("cand_pipe_layout.png", dpi=300, bbox_inches='tight')

plt.show() # Showing the plot




####################################################################################################################################################################################################################################################
####################################################################################################################################################################################################################################################
# Data exportation
####################################################################################################################################################################################################################################################
####################################################################################################################################################################################################################################################

pipeline_candidates_gdf['Node connection'] = pipeline_candidates_gdf['Node connection'].apply(lambda x: (int(x[0]), int(x[1])) if isinstance(x, tuple) else x) # Converting node connections to tuples of integers
pipeline_candidates_gdf['Node heights'] = pipeline_candidates_gdf['Node heights'].apply(lambda x: (int(x[0]), int(x[1])) if isinstance(x, tuple) else x) # Converting node heights to tuples of integers

map_cols = ['Facility name', 'Source cluster', 'orig_idx']

output_excel = "iberian_co2_network_data.xlsx"

with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
    all_nodes_gdf.to_excel(writer, sheet_name="Nodes", index=False)
    emissions_gdf[map_cols].to_excel(writer, sheet_name="Emissions clustering map", index=False)
    pipeline_candidates_gdf.to_excel(writer, sheet_name="Pipeline candidates", index=False)