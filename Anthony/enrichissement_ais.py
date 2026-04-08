"""
Étape 0c — Enrichissement du DataFrame AIS.

Pour chaque point AIS d'un voyage, on ajoute :
    - dist_port_depart_km(t)  : distance haversine au port de départ du voyage
    - dist_port_arrivee_km(t) : distance haversine au port d'arrivée du voyage
    - delta_sog               : variation de SOG entre t-1 et t (proxy accélération)

Dépendances :
    - ajout_ports.py       → associer_ports(), distance_cote()
    - attribuer_secteur.py → attribuer_secteur(), joindre_secteur_ais()

Colonnes attendues dans df_ais :
    mmsi, voyage number, timestamp, latitude, longitude, sog
Colonnes attendues dans df_sectorise (sortie de attribuer_secteur) :
    voyage number, mmsi, secteur,
    lat_port_depart, lon_port_depart,
    lat_port_arrivee, lon_port_arrivee
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Haversine vectorisé
# ---------------------------------------------------------------------------

def haversine_km(lat1: pd.Series, lon1: pd.Series,
                 lat2: pd.Series, lon2: pd.Series) -> pd.Series:
    """
    Distance haversine (km) entre deux séries de coordonnées.

    Paramètres
    ----------
    lat1, lon1 : coordonnées du point de départ (Series ou scalaires)
    lat2, lon2 : coordonnées du point d'arrivée (Series ou scalaires)

    Retour
    ------
    pd.Series de distances en km.
    """
    R = 6371.0  # rayon terrestre moyen (km)

    lat1_r = np.radians(lat1)
    lat2_r = np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))


# ---------------------------------------------------------------------------
# Enrichissement principal
# ---------------------------------------------------------------------------

def enrichir_ais(df_ais: pd.DataFrame, df_sectorise: pd.DataFrame) -> pd.DataFrame:
    """
    Enrichit le DataFrame AIS avec les distances aux ports et le delta_sog.

    Paramètres
    ----------
    df_ais : pd.DataFrame
        Points AIS bruts avec au moins :
        'mmsi', 'voyage number', 'timestamp', 'latitude', 'longitude', 'sog'.
    df_sectorise : pd.DataFrame
        Sortie de attribuer_secteur() — doit contenir :
        'voyage number', 'mmsi', 'secteur',
        'lat_port_depart', 'lon_port_depart',
        'lat_port_arrivee', 'lon_port_arrivee'.

    Retour
    ------
    pd.DataFrame
        df_ais enrichi des colonnes :
            'secteur',
            'dist_port_depart_km',
            'dist_port_arrivee_km',
            'delta_sog'.
        Seuls les voyages présents dans df_sectorise sont conservés (inner join).
    """
    # ── 1. Jointure secteur + coordonnées des ports ──────────────────────────
    cols_ports = [
        "voyage number", "mmsi", "lat_port_depart", "lon_port_depart",
        "secteur",
        "lat_port_arrivee", "lon_port_arrivee",]

    df = df_ais.merge(
        df_sectorise[cols_ports].drop_duplicates(["voyage number", "mmsi"]),
        on=["voyage number", "mmsi"],
        how="inner",
    )

    print(f"Lignes AIS avant enrichissement : {len(df_ais):,}")
    print(f"Lignes AIS après jointure ports : {len(df):,}")

    # ── 2. Tri chronologique (indispensable pour delta_sog) ──────────────────
    df = df.sort_values(["mmsi", "voyage number", "timestamp"]).reset_index(drop=True)

    # ── 3. Distances haversine au port de départ et au port d'arrivée ────────
    df["dist_port_depart_km"] = haversine_km(
        df["latitude"], df["longitude"],
        df["lat_port_depart"], df["lon_port_depart"],
    )

    df["dist_port_arrivee_km"] = haversine_km(
        df["latitude"], df["longitude"],
        df["lat_port_arrivee"], df["lon_port_arrivee"],
    )

    # ── 4. delta_sog : variation de SOG entre deux points consécutifs ────────
    # On ne calcule la différence qu'à l'intérieur d'un même voyage.
    # Le premier point de chaque voyage reçoit NaN (pas de point précédent).
    df["delta_sog"] = (
        df.groupby(["mmsi", "voyage number"])["sog"]
        .diff()
    )

    # ── 5. Nettoyage des colonnes intermédiaires ─────────────────────────────
    df = df.drop(columns=["lat_port_depart", "lon_port_depart",
                           "lat_port_arrivee", "lon_port_arrivee"])

    # ── 6. Diagnostics ───────────────────────────────────────────────────────
    print("\n── Statistiques dist_port_depart_km ──")
    print(df["dist_port_depart_km"].describe().round(1).to_string())
    print("\n── Statistiques dist_port_arrivee_km ──")
    print(df["dist_port_arrivee_km"].describe().round(1).to_string())
    print("\n── Statistiques delta_sog (nœuds/h) ──")
    print(df["delta_sog"].describe().round(3).to_string())
    print(f"\nNaN delta_sog (1er point/voyage) : {df['delta_sog'].isna().sum():,}")

    return df
