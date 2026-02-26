import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import PowerNorm
from math import radians, cos, sin, asin, sqrt
from datetime import timedelta

# -------------------------------
# 1. LOAD THE DATA
# -------------------------------
df = pd.read_feather("data_indian_ocean.feather")

# Optional: Filter for testing
# df = df.head(50000)

# -------------------------------
# 2. DEFINE THE GRID (KNOWLEDGE DISCOVERY)
# -------------------------------
# [cite_start]The paper suggests a resolution of 0.02 degrees [cite: 242]
step = 0.02

# Calculate boundaries
lat_min, lat_max = df['latitude'].min(), df['latitude'].max()
lon_min, lon_max = df['longitude'].min(), df['longitude'].max()

# Create grid bins
lat_bins = np.arange(lat_min, lat_max, step)
lon_bins = np.arange(lon_min, lon_max, step)

# Create Density Raster (Rt)
# density_raster[i, j] holds the ship count in that cell
density_raster, _, _ = np.histogram2d(
    df['latitude'],
    df['longitude'],
    bins=[lat_bins, lon_bins]
)

print(f"Raster created. Shape: {density_raster.shape}")

# -------------------------------
# 3. BUILD THE GRAPH (OPTIMIZED PATH-FINDER)
# -------------------------------
G = nx.Graph()
rows, cols = density_raster.shape

print("Identifying active cells for graph...")

# Optimization: Only process cells with ships
valid_cells = np.argwhere(density_raster > 0)
print(f"Building graph on {len(valid_cells)} active nodes...")

for r, c in valid_cells:
    node_id = (r, c)

    # Weight: Inverse density (High Density = Low Cost)
    # [cite_start]The paper uses conductance (1/resistance) [cite: 171]
    weight = 1.0 / (density_raster[r, c])

    # Check 4 neighbors
    neighbors = [
        (r - 1, c), (r + 1, c),
        (r, c - 1), (r, c + 1)
    ]

    for nr, nc in neighbors:
        if 0 <= nr < rows and 0 <= nc < cols:
            if density_raster[nr, nc] > 0:
                G.add_edge(node_id, (nr, nc), weight=weight)

print(f"Graph built with {G.number_of_nodes()} nodes.")

# -------------------------------
# 4. HELPER FUNCTIONS
# -------------------------------


def haversine(lon1, lat1, lon2, lat2):
    """Calculate great circle distance between two points."""
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * asin(sqrt(a))
    r = 6371  # Earth radius in km
    return c * r


def get_grid_index(lat, lon, lat_bins, lon_bins):
    """Convert real Lat/Lon to grid Row/Col."""
    row = np.digitize(lat, lat_bins) - 1
    col = np.digitize(lon, lon_bins) - 1
    row = max(0, min(row, len(lat_bins) - 2))
    col = max(0, min(col, len(lon_bins) - 2))
    return (row, col)


# -------------------------------
# 5. CALCULATE PATH AND ETA
# -------------------------------

# Inputs: Start and End points (First and Last in dataset)
start_lat, start_lon = df['latitude'].iloc[0], df['longitude'].iloc[0]
end_lat, end_lon = df['latitude'].iloc[-1], df['longitude'].iloc[-1]

if 'time' in df.columns:
    last_message_time = pd.to_datetime(df['time'].iloc[0])
else:
    last_message_time = pd.Timestamp.now()

start_node = get_grid_index(start_lat, start_lon, lat_bins, lon_bins)
end_node = get_grid_index(end_lat, end_lon, lat_bins, lon_bins)

path_nodes = []
try:
    path_nodes = nx.dijkstra_path(
        G, source=start_node, target=end_node, weight='weight'
    )

    estimated_distance_km = 0
    path_lats = []
    path_lons = []

    for i in range(len(path_nodes)):
        u = path_nodes[i]
        # Store for plotting (center of cell)
        path_lats.append(lat_bins[u[0]] + step / 2)
        path_lons.append(lon_bins[u[1]] + step / 2)

        if i < len(path_nodes) - 1:
            v = path_nodes[i + 1]
            lat1, lon1 = lat_bins[u[0]], lon_bins[u[1]]
            lat2, lon2 = lat_bins[v[0]], lon_bins[v[1]]
            estimated_distance_km += haversine(lon1, lat1, lon2, lat2)

    print(f"Estimated Distance: {estimated_distance_km:.2f} km")

    if 'sog' in df.columns:
        cruise_speeds = df[df['sog'] > 1]['sog']
        v_hat_knots = cruise_speeds.mean()
        v_hat_kmh = v_hat_knots * 1.852
        time_to_arrival = estimated_distance_km / v_hat_kmh
        eta = last_message_time + timedelta(hours=time_to_arrival)

        print(f"ETA: {eta} ({time_to_arrival:.1f} hours)")

except nx.NetworkXNoPath:
    print("No path found.")

# -------------------------------
# 6. VISUALIZATION
# -------------------------------

if path_nodes:
    # Normalize grid for display
    grid_norm = density_raster / density_raster.max()
    alpha = 0.7
    extent = [lon_min, lon_max, lat_min, lat_max]

    cmap = mcolors.LinearSegmentedColormap.from_list("wr", ["white", "red"])

    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(
        grid_norm,
        cmap=cmap,
        norm=PowerNorm(gamma=alpha),
        extent=extent,
        origin="lower"
    )

    # Plot the calculated path
    ax.plot(
        path_lons, path_lats,
        color='blue', linewidth=2, label='Optimal Path'
    )
    ax.scatter(
        path_lons[0], path_lats[0],
        color='green', s=100, label='Start', zorder=5
    )
    ax.scatter(
        path_lons[-1], path_lats[-1],
        color='black', marker='*', s=150, label='End', zorder=5
    )

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.colorbar(im, ax=ax, label="Normalized Density")
    plt.legend()
    plt.title("Vessel ETA Prediction & Optimal Path")
    plt.show()
    # -------------------------------
# 6. VISUALIZATION (SAVED TO FILE)
# -------------------------------
if 'path_nodes' in locals() and path_nodes:
    print("Generating map...")
    
    # Normalize grid for display
    # (Avoid division by zero if max is 0)
    max_val = density_raster.max()
    if max_val > 0:
        grid_norm = density_raster / max_val
    else:
        grid_norm = density_raster

    alpha = 0.7
    extent = [lon_min, lon_max, lat_min, lat_max]

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 10))

    # 1. Plot the Heatmap
    cmap = mcolors.LinearSegmentedColormap.from_list("wr", ["white", "red"])
    im = ax.imshow(
        grid_norm,
        cmap=cmap,
        norm=PowerNorm(gamma=alpha),
        extent=extent,
        origin="lower"
    )

    # 2. Plot the Optimal Path
    ax.plot(
        path_lons, path_lats, 
        color='blue', linewidth=2, label='Optimal Path', alpha=0.8
    )

    # 3. Plot Start and End points
    ax.scatter(
        path_lons[0], path_lats[0], 
        color='green', s=100, edgecolors='black', label='Start', zorder=5
    )
    ax.scatter(
        path_lons[-1], path_lats[-1], 
        color='black', marker='*', s=200, label='End', zorder=5
    )

    # 4. Add Labels and Title
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.colorbar(im, ax=ax, label="Normalized Traffic Density")
    plt.legend(loc="upper right")
    
    # Add the ETA info to the title if available
    title_text = "Vessel ETA Prediction & Optimal Path"
    if 'eta' in locals():
        title_text += f"\nETA: {eta}"
    plt.title(title_text)

    # 5. SAVE THE FILE instead of showing it
    output_filename = "eta_prediction_map.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    
    print(f"\nSUCCESS: Graph saved to '{output_filename}'")
    print("Check your file explorer to open the image.")

else:
    print("No path found to plot.")