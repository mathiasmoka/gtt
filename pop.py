from __future__ import annotations
import os
import pickle
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
from tqdm import tqdm
from pyproj import Transformer

# ================= CONFIG =================

ROOT_DIR = r"C:\Users\dosje\Desktop\ENSAE\ENSAE\ENSAE"
OUT_MODEL = r"C:\Users\dosje\Desktop\ENSAE\model_popularity.pkl"

LAT_COL = "latitude"
LON_COL = "longitude"
VOY_COL = "voyage number"
TS_COL  = "timestamp"
SOG_COL = "sog"

CELL_KM = 10.0

# ==========================================

WEBMERCATOR_HALF = 20037508.342789244
TRANSFORMER_TO_M = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

EdgeKey = Tuple[int, int, int, int]
Node = Tuple[int, int]


def find_feathers(root: str) -> List[str]:
    out = []
    for r, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith(".feather"):
                out.append(os.path.join(r, f))
    return out


def ll_to_cell(lon: np.ndarray, lat: np.ndarray, cell_m: float, x0: float, y0: float):
    x, y = TRANSFORMER_TO_M.transform(lon.astype(float), lat.astype(float))
    i = np.floor((x - x0) / cell_m).astype(np.int32)
    j = np.floor((y - y0) / cell_m).astype(np.int32)
    return i, j


def add_transitions(edges: Dict[EdgeKey, int], i, j):
    if len(i) < 2:
        return

    ij = np.stack([i, j], axis=1)
    keep = np.ones(len(ij), dtype=bool)
    keep[1:] = np.any(ij[1:] != ij[:-1], axis=1)
    ij = ij[keep]
    if len(ij) < 2:
        return

    i1, j1 = ij[:-1].T
    i2, j2 = ij[1:].T
    di = np.abs(i2 - i1)
    dj = np.abs(j2 - j1)
    mask = (di <= 1) & (dj <= 1) & ~((di == 0) & (dj == 0))

    for a, b, c, d in zip(i1[mask], j1[mask], i2[mask], j2[mask]):
        key = (int(a), int(b), int(c), int(d))
        edges[key] = edges.get(key, 0) + 1


def build():
    feathers = find_feathers(ROOT_DIR)
    print("Feathers found:", len(feathers))

    cell_m = CELL_KM * 1000.0
    x0 = -WEBMERCATOR_HALF
    y0 = -WEBMERCATOR_HALF

    edges: Dict[EdgeKey, int] = {}
    speed_sum: Dict[Node, float] = {}
    speed_count: Dict[Node, int] = {}

    for fp in tqdm(feathers, desc="Building model"):
        df = pd.read_feather(fp)
        df[TS_COL] = pd.to_datetime(df[TS_COL], errors="coerce", utc=True)
        df = df.dropna(subset=[LAT_COL, LON_COL, TS_COL])
        df = df.sort_values(TS_COL)

        lat = df[LAT_COL].to_numpy()
        lon = df[LON_COL].to_numpy()
        voyages = df[VOY_COL]
        sog_kn = df[SOG_COL].to_numpy(dtype=float)
        sog_ms = sog_kn * 0.514444  # knots -> m/s

        i, j = ll_to_cell(lon, lat, cell_m=cell_m, x0=x0, y0=y0)

        # vitesse moyenne par cellule
        for ii, jj, v in zip(i, j, sog_ms):
            if not np.isfinite(v) or v <= 0:
                continue
            node = (int(ii), int(jj))
            speed_sum[node] = speed_sum.get(node, 0.0) + float(v)
            speed_count[node] = speed_count.get(node, 0) + 1

        # transitions par voyage
        for _, idx in df.groupby(voyages).groups.items():
            idx = np.asarray(idx)
            add_transitions(edges, i[idx], j[idx])

    speed_mean = {k: speed_sum[k] / speed_count[k] for k in speed_sum.keys()}

    with open(OUT_MODEL, "wb") as f:
        pickle.dump(
            {
                "grid": {"cell_m": cell_m, "x0": x0, "y0": y0},
                "edges": edges,
                "speed_mean_ms": speed_mean,
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    print("Model saved:", OUT_MODEL)
    print("Edges:", len(edges))
    print("Cells with speed:", len(speed_mean))


if __name__ == "__main__":
    build()