"""
=============================================================================
PROJET GTT x ENSAE — Heatmap des Vitesses sur Carte
Océan Indien — Données AIS
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# CHARGEMENT (colonnes minimales pour la heatmap)
# ─────────────────────────────────────────────────────────────────────────────

def charger_donnees(chemin="/home/onyxia/work/data/data_indian_ocean.feather"):
    print("Chargement des donnees...")
    COLONNES = ['latitude', 'longitude', 'sog', 'mmsi']
    df = pd.read_feather(chemin, columns=COLONNES)
    print(f"   Brut : {len(df):,} lignes")

    # Filtres qualite
    df = df[df['sog'].between(0.1, 30)]
    df = df[df['latitude'].between(-40, 35)]
    df = df[df['longitude'].between(30, 120)]

    # Mode dev : 10% — retirer pour analyse finale
    df = df.sample(frac=0.1, random_state=42)
    print(f"   Mode dev : {len(df):,} lignes")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# GRILLE DE BINNING
# ─────────────────────────────────────────────────────────────────────────────

def construire_grille(df, resolution=0.5):
    """
    Découpe l'Océan Indien en cellules de `resolution` degrés.
    Calcule pour chaque cellule : vitesse moyenne, médiane, std, nb observations.
    """
    print(f"Construction de la grille (resolution={resolution} deg)...")

    lat_bins = np.arange(-40, 36, resolution)
    lon_bins = np.arange(30, 121, resolution)

    df['lat_bin'] = pd.cut(df['latitude'],  bins=lat_bins, labels=lat_bins[:-1])
    df['lon_bin'] = pd.cut(df['longitude'], bins=lon_bins, labels=lon_bins[:-1])

    df['lat_bin'] = df['lat_bin'].astype(float)
    df['lon_bin'] = df['lon_bin'].astype(float)

    grille = df.groupby(['lat_bin', 'lon_bin'], observed=True)['sog'].agg(
        sog_mean='mean',
        sog_median='median',
        sog_std='std',
        n_obs='count',
    ).reset_index()

    # Filtrer cellules avec trop peu d'observations
    grille = grille[grille['n_obs'] >= 5]
    print(f"   {len(grille):,} cellules actives")
    return grille, lat_bins, lon_bins, resolution


# ─────────────────────────────────────────────────────────────────────────────
# DESSIN DES TERRES (simplifie)
# ─────────────────────────────────────────────────────────────────────────────

def ajouter_terres(ax):
    """Ajoute des polygones simplifiés des principales masses terrestres."""
    terres = [
        # Afrique
        dict(xy=(-35, 30), width=45, height=75, angle=0),
    ]

    # Contours simplifiés des terres (rectangles approximatifs)
    zones_terre = [
        # (lon_min, lon_max, lat_min, lat_max, label)
        (30,  52,  10,  30, 'Péninsule Arabique'),
        (30,  42, -12,  12, 'Afrique Est'),
        (65,  80,   8,  37, 'Inde'),
        (80,  92,   5,  25, 'Sri Lanka + Bangladesh'),
        (95, 106,  -6,  22, 'Péninsule Indochinoise'),
        (38,  52, -26,  -8, 'Madagascar (approx)'),
        (113, 120, -35,  -5, 'Australie Ouest'),
        (44,  55,  11,  28, 'Corne de l\'Afrique'),
    ]

    for lon_min, lon_max, lat_min, lat_max, label in zones_terre:
        rect = plt.Rectangle(
            (lon_min, lat_min),
            lon_max - lon_min,
            lat_max - lat_min,
            facecolor='#2D4A22',
            edgecolor='#4A7A35',
            linewidth=0.5,
            alpha=0.85,
            zorder=3,
        )
        ax.add_patch(rect)


# ─────────────────────────────────────────────────────────────────────────────
# HEATMAP PRINCIPALE — VITESSE MOYENNE
# ─────────────────────────────────────────────────────────────────────────────

def plot_heatmap_vitesse(grille, resolution=0.5, metric='sog_mean', save=True):
    """
    Heatmap de la vitesse moyenne (ou médiane) sur l'Océan Indien.
    """
    print(f"Generation heatmap ({metric})...")

    # Pivot pour avoir une matrice lat x lon
    pivot = grille.pivot(index='lat_bin', columns='lon_bin', values=metric)

    # Colormap maritime : bleu foncé (lent) -> cyan -> vert -> jaune -> rouge (rapide)
    colors_map = [
        '#0A1628',  # très lent — quasi arrêt
        '#1B4F8A',  # lent
        '#0E7C7B',  # modéré
        '#17BEBB',  # normal
        '#F4A261',  # rapide
        '#E76F51',  # très rapide
    ]
    cmap = LinearSegmentedColormap.from_list('maritime_speed', colors_map, N=256)

    fig, ax = plt.subplots(figsize=(18, 12))
    fig.patch.set_facecolor('#0A1628')
    ax.set_facecolor('#071020')

    # Fond océan
    ocean_rect = plt.Rectangle((30, -40), 90, 75,
                                facecolor='#071020', zorder=0)
    ax.add_patch(ocean_rect)

    # Heatmap
    lons = pivot.columns.values.astype(float)
    lats = pivot.index.values.astype(float)
    Z = pivot.values

    im = ax.pcolormesh(
        lons, lats, Z,
        cmap=cmap,
        vmin=5, vmax=22,
        shading='auto',
        alpha=0.92,
        zorder=1,
    )

    # Terres
    ajouter_terres(ax)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, orientation='vertical',
                        fraction=0.025, pad=0.02, shrink=0.8)
    cbar.set_label('Vitesse SOG (nœuds)', color='white', fontsize=12, labelpad=10)
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white', fontsize=9)

    # Annotations zones
    zones_labels = [
        (57,  22, 'Mer Rouge &\nGolfe Aden',   '#E76F51'),
        (58, -28, 'Océan Indien\nCentral',       '#17BEBB'),
        (75,  20, 'Approches\nIndiennes',        '#F4A261'),
        (55,  25, 'Golfe\nPersique',             '#E9C46A'),
        (100,  5, 'Détroit de\nMalacca',         '#9B5DE5'),
        (44,  -5, 'Côte Est\nAfricaine',         '#2A9D8F'),
        (115, -28, 'Australie &\nPacifique',     '#06D6A0'),
    ]
    for lon, lat, label, color in zones_labels:
        ax.text(lon, lat, label, color=color, fontsize=8,
                ha='center', va='center', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#0A1628',
                          edgecolor=color, alpha=0.75),
                zorder=5)

    # Ports principaux
    ports = [
        (43.14,  11.59, 'Djibouti'),
        (72.82,  18.96, 'Mumbai'),
        (55.27,  25.20, 'Dubaï'),
        (80.28,  13.08, 'Chennai'),
        (103.82,  1.35, 'Singapour'),
        (39.17,  21.48, 'Djeddah'),
        (57.65,  23.58, 'Muscat'),
        (47.98, -18.91, 'Antananarivo'),
    ]
    for lon, lat, nom in ports:
        ax.plot(lon, lat, 'o', color='white', markersize=5,
                zorder=6, markeredgecolor='#F4A261', markeredgewidth=1.2)
        ax.text(lon + 0.8, lat + 0.5, nom, color='white', fontsize=7.5,
                zorder=6, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#0A1628',
                          alpha=0.6, edgecolor='none'))

    # Grille de coordonnées
    ax.set_xticks(np.arange(30, 121, 10))
    ax.set_yticks(np.arange(-40, 36, 10))
    ax.set_xticklabels([f'{x}°E' for x in np.arange(30, 121, 10)],
                       color='#E8F4F8', fontsize=9)
    ax.set_yticklabels([f'{abs(y)}°{"N" if y >= 0 else "S"}' for y in np.arange(-40, 36, 10)],
                       color='#E8F4F8', fontsize=9)
    ax.grid(color='#1A3050', linewidth=0.4, linestyle='--', alpha=0.5, zorder=2)

    # Limites
    ax.set_xlim(30, 120)
    ax.set_ylim(-40, 35)
    ax.set_xlabel('Longitude', color='#E8F4F8', fontsize=11)
    ax.set_ylabel('Latitude',  color='#E8F4F8', fontsize=11)

    titre = 'Vitesse Moyenne' if metric == 'sog_mean' else 'Vitesse Médiane'
    ax.set_title(
        f'Heatmap {titre} des Navires — Océan Indien\n'
        f'Données AIS | Résolution {resolution}° | n={len(grille):,} cellules',
        color='white', fontsize=15, fontweight='bold', pad=15,
    )

    plt.tight_layout()
    fname = f'heatmap_{metric}.png'
    if save:
        plt.savefig(fname, dpi=180, bbox_inches='tight', facecolor='#0A1628')
    plt.show()
    print(f"   Sauvegarde : {fname}")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# HEATMAP DENSITE — NOMBRE D'OBSERVATIONS (routes maritimes)
# ─────────────────────────────────────────────────────────────────────────────

def plot_heatmap_densite(grille, resolution=0.5, save=True):
    """
    Heatmap de densité : révèle les routes maritimes principales.
    """
    print("Generation heatmap densite (routes maritimes)...")

    pivot = grille.pivot(index='lat_bin', columns='lon_bin', values='n_obs')
    lons  = pivot.columns.values.astype(float)
    lats  = pivot.index.values.astype(float)
    Z     = np.log1p(pivot.values)  # log pour mieux voir les routes

    cmap_routes = LinearSegmentedColormap.from_list(
        'routes', ['#071020', '#0E3A5A', '#0E7C7B', '#17BEBB', '#F4A261', '#FFFFFF'], N=256
    )

    fig, ax = plt.subplots(figsize=(18, 12))
    fig.patch.set_facecolor('#0A1628')
    ax.set_facecolor('#071020')

    im = ax.pcolormesh(lons, lats, Z, cmap=cmap_routes, shading='auto',
                       alpha=0.95, zorder=1)

    ajouter_terres(ax)

    cbar = plt.colorbar(im, ax=ax, orientation='vertical',
                        fraction=0.025, pad=0.02, shrink=0.8)
    cbar.set_label('log(Nb observations + 1)', color='white', fontsize=11, labelpad=10)
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white', fontsize=9)

    # Ports
    ports = [
        (43.14,  11.59, 'Djibouti'),
        (72.82,  18.96, 'Mumbai'),
        (55.27,  25.20, 'Dubaï'),
        (80.28,  13.08, 'Chennai'),
        (103.82,  1.35, 'Singapour'),
        (39.17,  21.48, 'Djeddah'),
        (57.65,  23.58, 'Muscat'),
    ]
    for lon, lat, nom in ports:
        ax.plot(lon, lat, '*', color='#F4A261', markersize=9,
                zorder=6, markeredgecolor='white', markeredgewidth=0.5)
        ax.text(lon + 0.8, lat + 0.5, nom, color='white', fontsize=8,
                zorder=6, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#0A1628',
                          alpha=0.65, edgecolor='none'))

    ax.set_xticks(np.arange(30, 121, 10))
    ax.set_yticks(np.arange(-40, 36, 10))
    ax.set_xticklabels([f'{x}°E' for x in np.arange(30, 121, 10)],
                       color='#E8F4F8', fontsize=9)
    ax.set_yticklabels([f'{abs(y)}°{"N" if y >= 0 else "S"}' for y in np.arange(-40, 36, 10)],
                       color='#E8F4F8', fontsize=9)
    ax.grid(color='#1A3050', linewidth=0.4, linestyle='--', alpha=0.5, zorder=2)
    ax.set_xlim(30, 120)
    ax.set_ylim(-40, 35)
    ax.set_xlabel('Longitude', color='#E8F4F8', fontsize=11)
    ax.set_ylabel('Latitude',  color='#E8F4F8', fontsize=11)
    ax.set_title(
        'Densité du Trafic Maritime — Routes Principales — Océan Indien\n'
        f'Données AIS | Résolution {resolution}° | Echelle logarithmique',
        color='white', fontsize=15, fontweight='bold', pad=15,
    )

    plt.tight_layout()
    if save:
        plt.savefig('heatmap_densite_routes.png', dpi=180,
                    bbox_inches='tight', facecolor='#0A1628')
    plt.show()
    print("   Sauvegarde : heatmap_densite_routes.png")


# ─────────────────────────────────────────────────────────────────────────────
# HEATMAP VARIABILITE — ECART-TYPE DE VITESSE
# ─────────────────────────────────────────────────────────────────────────────

def plot_heatmap_variabilite(grille, resolution=0.5, save=True):
    """
    Zones à forte variabilité de vitesse = zones de manœuvre / danger.
    """
    print("Generation heatmap variabilite...")

    pivot = grille.pivot(index='lat_bin', columns='lon_bin', values='sog_std')
    lons  = pivot.columns.values.astype(float)
    lats  = pivot.index.values.astype(float)

    cmap_var = LinearSegmentedColormap.from_list(
        'variabilite', ['#071020', '#1B4F8A', '#E9C46A', '#E76F51', '#FF0000'], N=256
    )

    fig, ax = plt.subplots(figsize=(18, 12))
    fig.patch.set_facecolor('#0A1628')
    ax.set_facecolor('#071020')

    im = ax.pcolormesh(lons, lats, pivot.values, cmap=cmap_var,
                       vmin=0, vmax=8, shading='auto', alpha=0.92, zorder=1)

    ajouter_terres(ax)

    cbar = plt.colorbar(im, ax=ax, orientation='vertical',
                        fraction=0.025, pad=0.02, shrink=0.8)
    cbar.set_label('Écart-type SOG (nœuds)', color='white', fontsize=11, labelpad=10)
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white', fontsize=9)

    ax.set_xticks(np.arange(30, 121, 10))
    ax.set_yticks(np.arange(-40, 36, 10))
    ax.set_xticklabels([f'{x}°E' for x in np.arange(30, 121, 10)],
                       color='#E8F4F8', fontsize=9)
    ax.set_yticklabels([f'{abs(y)}°{"N" if y >= 0 else "S"}' for y in np.arange(-40, 36, 10)],
                       color='#E8F4F8', fontsize=9)
    ax.grid(color='#1A3050', linewidth=0.4, linestyle='--', alpha=0.5, zorder=2)
    ax.set_xlim(30, 120)
    ax.set_ylim(-40, 35)
    ax.set_xlabel('Longitude', color='#E8F4F8', fontsize=11)
    ax.set_ylabel('Latitude',  color='#E8F4F8', fontsize=11)
    ax.set_title(
        'Variabilité des Vitesses (Écart-type) — Zones de Manœuvre\n'
        'Rouge = forte variabilité | Bleu = vitesse stable',
        color='white', fontsize=15, fontweight='bold', pad=15,
    )

    plt.tight_layout()
    if save:
        plt.savefig('heatmap_variabilite.png', dpi=180,
                    bbox_inches='tight', facecolor='#0A1628')
    plt.show()
    print("   Sauvegarde : heatmap_variabilite.png")


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  HEATMAP DES VITESSES — OCEAN INDIEN")
    print("=" * 60)

    # Chargement
    df = charger_donnees()

    # Grille 0.5 deg (changer à 0.25 pour plus de précision, plus lent)
    grille, lat_bins, lon_bins, res = construire_grille(df, resolution=0.5)

    # 3 heatmaps
    print("\nFigure 1 — Vitesse moyenne...")
    plot_heatmap_vitesse(grille, resolution=res, metric='sog_mean')

    print("\nFigure 2 — Densite / Routes maritimes...")
    plot_heatmap_densite(grille, resolution=res)

    print("\nFigure 3 — Variabilite des vitesses...")
    plot_heatmap_variabilite(grille, resolution=res)

    print("\nTermine ! 3 figures generees :")
    print("  - heatmap_sog_mean.png")
    print("  - heatmap_densite_routes.png")
    print("  - heatmap_variabilite.png")
