"""
Ajout des noms de ports de départ et d'arrivée dans le dataset AIS.

Modifications par rapport à la version précédente :
    - Filtre sur la distance au port attribué (pas seulement à la côte)
    - Coordonnées des ports (lat/lon) ramenées dans le DataFrame final
    - Diagnostics enrichis pour calibrer les seuils

Prérequis :
    pip install geopandas geodatasets
    Fichier "upply-seaports.csv" dans le répertoire courant.
"""

from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point


# ---------------------------------------------------------------------------
# Distance à la côte
# ---------------------------------------------------------------------------

def distance_cote(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajoute une colonne 'dist_cote_km' à chaque point AIS.

    La distance est calculée en degrés puis convertie en km (1° ≈ 111 km).

    Paramètres
    ----------
    df : pd.DataFrame
        DataFrame contenant les colonnes 'latitude' et 'longitude'.

    Retour
    ------
    pd.DataFrame
        DataFrame avec la colonne 'dist_cote_km' ajoutée.
    """
    import geodatasets

    df = df.copy()

    world = gpd.read_file(geodatasets.get_path("naturalearth.land"))
    cotes = world.union_all()

    geometry = [Point(lon, lat) for lon, lat in zip(df["longitude"], df["latitude"])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
    gdf["dist_cote_km"] = gdf.geometry.distance(cotes) * 111

    return gdf.drop(columns="geometry")


# ---------------------------------------------------------------------------
# Association des ports
# ---------------------------------------------------------------------------

def associer_ports(
    df_ais: pd.DataFrame,
    seaports_csv: str | Path,
    seuil_cote_km: float = 100,
    seuil_port_km: float = 50,
) -> pd.DataFrame:
    """
    Associe pour chaque voyage le port de départ et le port d'arrivée
    le plus proche, avec double filtrage et export des coordonnées des ports.

    Paramètres
    ----------
    df_ais : pd.DataFrame
        DataFrame avec les colonnes : 'mmsi', 'voyage number',
        'latitude', 'longitude', 'timestamp'.
    seaports_csv : str | Path
        Chemin vers le fichier CSV des ports ("upply-seaports.csv").
    seuil_cote_km : float
        Distance maximale à la côte (km) pour valider un point comme
        appartenant à une zone portuaire. Défaut : 100 km.
    seuil_port_km : float
        Distance maximale au port attribué (km) pour valider l'attribution.
        Permet d'exclure les voyages dont le premier/dernier point AIS
        est loin de tout port connu. Défaut : 50 km.

    Retour
    ------
    pd.DataFrame
        Une ligne par voyage avec les colonnes :
            voyage number, mmsi,
            lat/lon de départ et d'arrivée,
            dist_cote départ/arrivée,
            port_depart_name, lat_port_depart, lon_port_depart, dist_port_depart_km,
            port_arrivee_name, lat_port_arrivee, lon_port_arrivee, dist_port_arrivee_km.

    Notes
    -----
    Les voyages exclus sont ceux où :
        - le premier OU dernier point AIS est à > seuil_cote_km de la côte
        - le premier OU dernier point AIS est à > seuil_port_km du port attribué
    """
    # ── Distance à la côte ────────────────────────────────────────────────────
    df_ais = distance_cote(df_ais)
    df_sorted = df_ais.sort_values("timestamp")

    # ── Points de départ et d'arrivée par voyage ──────────────────────────────
    ports = df_sorted.groupby(["voyage number", "mmsi"]).agg(
        lat_depart  = ("latitude",     "first"),
        lon_depart  = ("longitude",    "first"),
        dist_depart = ("dist_cote_km", "first"),
        lat_arrivee = ("latitude",     "last"),
        lon_arrivee = ("longitude",    "last"),
        dist_arrivee= ("dist_cote_km", "last"),
    ).reset_index()

    n_avant_cote = len(ports)

    # ── Filtre 1 : proximité côte ─────────────────────────────────────────────
    ports = ports[
        (ports["dist_depart"]  < seuil_cote_km) &
        (ports["dist_arrivee"] < seuil_cote_km)
    ].copy()

    n_apres_cote = len(ports)
    print("=" * 55)
    print("ASSOCIATION DES PORTS")
    print("=" * 55)
    print(f"Voyages total             : {n_avant_cote:,}")
    print(f"Exclus (filtre côte)      : {n_avant_cote - n_apres_cote:,}  "
          f"(>{seuil_cote_km} km de la côte)")

    # ── Chargement de la base de ports ────────────────────────────────────────
    ports_db = pd.read_csv(seaports_csv, sep=";", engine="python")
    ports_db = ports_db[["code", "name", "country_code", "latitude", "longitude"]].dropna()

    gdf_ports = gpd.GeoDataFrame(
        ports_db,
        geometry=gpd.points_from_xy(ports_db["longitude"], ports_db["latitude"]),
        crs="EPSG:4326",
    ).to_crs(epsg=3857)

    # ── GeoDataFrames départ / arrivée ────────────────────────────────────────
    gdf_depart = gpd.GeoDataFrame(
        ports,
        geometry=gpd.points_from_xy(ports["lon_depart"], ports["lat_depart"]),
        crs="EPSG:4326",
    ).to_crs(epsg=3857)

    gdf_arrivee = gpd.GeoDataFrame(
        ports,
        geometry=gpd.points_from_xy(ports["lon_arrivee"], ports["lat_arrivee"]),
        crs="EPSG:4326",
    ).to_crs(epsg=3857)

    # ── Jointure spatiale : port le plus proche + coordonnées du port ─────────
    cols_ports = ["name", "latitude", "longitude", "geometry"]

    gdf_depart = gpd.sjoin_nearest(
        gdf_depart,
        gdf_ports[cols_ports],
        how="left",
        distance_col="dist_port_depart_m",
    ).rename(columns={
        "name":      "port_depart_name",
        "latitude":  "lat_port_depart",
        "longitude": "lon_port_depart",
    })

    gdf_arrivee = gpd.sjoin_nearest(
        gdf_arrivee,
        gdf_ports[cols_ports],
        how="left",
        distance_col="dist_port_arrivee_m",
    ).rename(columns={
        "name":      "port_arrivee_name",
        "latitude":  "lat_port_arrivee",
        "longitude": "lon_port_arrivee",
    })

    # ── Conversion distances en km ────────────────────────────────────────────
    gdf_depart["dist_port_depart_km"]   = gdf_depart["dist_port_depart_m"]   / 1000
    gdf_arrivee["dist_port_arrivee_km"] = gdf_arrivee["dist_port_arrivee_m"] / 1000

    # ── Diagnostic distances au port (avant filtre) ───────────────────────────
    print(f"\nDistance au port attribué (départ) :")
    print(gdf_depart["dist_port_depart_km"].describe().round(1).to_string())
    print(f"\nDistance au port attribué (arrivée) :")
    print(gdf_arrivee["dist_port_arrivee_km"].describe().round(1).to_string())

    # ── Fusion dans le DataFrame final ────────────────────────────────────────
    cols_dep = ["voyage number", "mmsi",
                "port_depart_name", "lat_port_depart", "lon_port_depart",
                "dist_port_depart_km"]
    cols_arr = ["voyage number", "mmsi",
                "port_arrivee_name", "lat_port_arrivee", "lon_port_arrivee",
                "dist_port_arrivee_km"]

    ports = ports.merge(
        gdf_depart[cols_dep].drop_duplicates(["voyage number", "mmsi"]),
        on=["voyage number", "mmsi"], how="left",
    )
    ports = ports.merge(
        gdf_arrivee[cols_arr].drop_duplicates(["voyage number", "mmsi"]),
        on=["voyage number", "mmsi"], how="left",
    )

    n_avant_port = len(ports)

    # ── Filtre 2 : proximité port attribué ────────────────────────────────────
    ports = ports[
        (ports["dist_port_depart_km"]  < seuil_port_km) &
        (ports["dist_port_arrivee_km"] < seuil_port_km)
    ].copy()

    n_apres_port = len(ports)

    print(f"\nExclus (filtre port)      : {n_avant_port - n_apres_port:,}  "
          f"(>{seuil_port_km} km du port attribué)")
    print(f"Voyages conservés         : {n_apres_port:,}  "
          f"({n_apres_port / n_avant_cote * 100:.1f} % du total)")
    print(f"\nPorts de départ identifiés :")
    print(ports["port_depart_name"].value_counts().head(10).to_string())
    print(f"\nPorts d'arrivée identifiés :")
    print(ports["port_arrivee_name"].value_counts().head(10).to_string())
    print("=" * 55)

    return ports