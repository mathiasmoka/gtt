"""
Ajout des noms de ports de départ et d'arrivée dans le dataset AIS.

Prérequis :
    pip install geopandas geodatasets
    Fichier "upply-seaports.csv" dans le répertoire courant.
"""

from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point


# ---------------------------------------------------------------------------
# Chargement des données
# ---------------------------------------------------------------------------

def charger_donnees(chemin: str | Path) -> pd.DataFrame:
    """Charge le fichier feather et retourne un DataFrame."""
    return pd.read_feather(Path(chemin))


# ---------------------------------------------------------------------------
# Fonctions principales
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
    import geodatasets  # import local pour éviter l'erreur si non installé

    df = df.copy()

    world = gpd.read_file(geodatasets.get_path("naturalearth.land"))
    cotes = world.union_all()

    geometry = [Point(lon, lat) for lon, lat in zip(df["longitude"], df["latitude"])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

    gdf["dist_cote_km"] = gdf.geometry.distance(cotes) * 111

    return gdf.drop(columns="geometry")


def associer_ports(
    df_ais: pd.DataFrame,
    seaports_csv: str | Path,
    seuil_cote_km: float = 100,
) -> pd.DataFrame:
    """
    Associe pour chaque voyage le port de départ et le port d'arrivée le plus proche.

    Paramètres
    ----------
    df_ais : pd.DataFrame
        DataFrame avec les colonnes : 'mmsi', 'voyage number',
        'latitude', 'longitude', 'timestamp'.
    seaports_csv : str | Path
        Chemin vers le fichier CSV des ports ("upply-seaports.csv").
    seuil_cote_km : float
        Distance maximale à la côte (km) pour valider un point comme port.
        Défaut : 100 km.

    Retour
    ------
    pd.DataFrame
        DataFrame avec les colonnes : voyage number, mmsi,
        lat/lon de départ et d'arrivée, dist à la côte,
        port_depart_name, port_arrivee_name.
    """
    # Ajout de la distance à la côte
    df_ais = distance_cote(df_ais)
    df_sorted = df_ais.sort_values("timestamp")

    # Points de départ et d'arrivée par voyage
    ports = df_sorted.groupby(["voyage number", "mmsi"]).agg(
        lat_depart=("latitude", "first"),
        lon_depart=("longitude", "first"),
        dist_depart=("dist_cote_km", "first"),
        lat_arrivee=("latitude", "last"),
        lon_arrivee=("longitude", "last"),
        dist_arrivee=("dist_cote_km", "last"),
    ).reset_index()

    # Filtrage : on ne garde que les voyages proches de la côte
    ports = ports[
        (ports["dist_depart"] < seuil_cote_km)
        & (ports["dist_arrivee"] < seuil_cote_km)
    ]

    # Chargement de la base de ports réels
    ports_db = pd.read_csv(seaports_csv, sep=";", engine="python")
    ports_db = ports_db[["code", "name", "country_code", "latitude", "longitude"]].dropna()

    gdf_ports = gpd.GeoDataFrame(
        ports_db,
        geometry=gpd.points_from_xy(ports_db["longitude"], ports_db["latitude"]),
        crs="EPSG:4326",
    ).to_crs(epsg=3857)

    # GeoDataFrames départ / arrivée
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

    # Jointure spatiale : port le plus proche
    gdf_depart = gpd.sjoin_nearest(
        gdf_depart, gdf_ports, how="left", distance_col="dist_port_depart"
    )
    gdf_arrivee = gpd.sjoin_nearest(
        gdf_arrivee, gdf_ports, how="left", distance_col="dist_port_arrivee"
    )

    # Fusion des noms de ports dans le DataFrame final
    ports = ports.merge(
        gdf_depart[["voyage number", "mmsi", "name"]],
        on=["voyage number", "mmsi"],
        how="left",
    ).rename(columns={"name": "port_depart_name"})

    ports = ports.merge(
        gdf_arrivee[["voyage number", "mmsi", "name"]],
        on=["voyage number", "mmsi"],
        how="left",
    ).rename(columns={"name": "port_arrivee_name"})

    return ports
