"""
=============================================================================
PROJET GTT x ENSAE — Analyse Statistique Descriptive
Données AIS — Océan Indien
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.stats import norm, lognorm, gamma, weibull_min, kstest
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# PALETTE COULEURS
# ─────────────────────────────────────────────────────────────────────────────

DEEP  = '#0A1628'
OCEAN = '#1B4F8A'
TEAL  = '#0E7C7B'
CYAN  = '#17BEBB'
GOLD  = '#F4A261'
CORAL = '#E76F51'
LIGHT = '#E8F4F8'

ZONE_COLORS = {
    'Mer Rouge & Golfe Aden': '#E76F51',
    'Cote Est Africaine':     '#2A9D8F',
    'Ocean Indien Central':   '#1B4F8A',
    'Approches Indiennes':    '#F4A261',
    'Detroit de Malacca':     '#9B5DE5',
    'Golfe Persique':         '#E9C46A',
    'Australie & Pacifique':  '#06D6A0',
}

plt.rcParams.update({
    'figure.facecolor': DEEP,
    'axes.facecolor':   '#0D1F3C',
    'axes.edgecolor':   '#2A4060',
    'axes.labelcolor':  LIGHT,
    'xtick.color':      LIGHT,
    'ytick.color':      LIGHT,
    'text.color':       'white',
    'grid.color':       '#2A4060',
    'grid.alpha':       0.4,
    'legend.facecolor': '#0D1F3C',
    'legend.edgecolor': '#2A4060',
})


# ─────────────────────────────────────────────────────────────────────────────
# 1. CHARGEMENT — COLONNES SELECTIONNEES SEULEMENT (économie RAM)
# ─────────────────────────────────────────────────────────────────────────────

def charger_donnees(chemin="/home/onyxia/work/data/data_indian_ocean.feather"):
    print("Chargement des donnees AIS (colonnes selectionnees)...")

    COLONNES = [
        'mmsi', 'latitude', 'longitude', 'timestamp',
        'sog', 'cog', 'nav status code', 'draft',
        'significant wave height Hs (m)',
        'wave period Tp (s)',
        'eastward wind velocity (m/s)',
        'northward wind velocity (m/s)',
        'air temperature at 2m (°K)',
        'sea surface temperature (°K)',
        'mean wave direction (°)',
    ]

    df = pd.read_feather(chemin, columns=COLONNES)
    print(f"   Brut : {len(df):,} lignes x {df.shape[1]} colonnes")

    # Timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(['mmsi', 'timestamp']).reset_index(drop=True)

    # Filtres qualite
    df = df[df['sog'].between(0.1, 30)]
    df = df[df['nav status code'].isin([0, 8])]

    # Mode developpement : retirer pour analyse finale
    df = df.sample(frac=0.1, random_state=42)
    print(f"   Mode dev : {len(df):,} lignes (10%)")

    # Conversions
    df['sog_kmh']    = df['sog'] * 1.852
    df['temp_air_C'] = df['air temperature at 2m (°K)'] - 273.15
    df['temp_mer_C'] = df['sea surface temperature (°K)'] - 273.15
    df['wind_speed'] = np.sqrt(
        df['eastward wind velocity (m/s)'] ** 2 +
        df['northward wind velocity (m/s)'] ** 2
    )

    # Variables temporelles
    df['year']    = df['timestamp'].dt.year
    df['month']   = df['timestamp'].dt.month
    df['weekday'] = df['timestamp'].dt.dayofweek
    df['hour']    = df['timestamp'].dt.hour
    df['season']  = df['month'].map({
        12: 'Ete austral',  1: 'Ete austral',  2: 'Ete austral',
        3:  'Automne',      4: 'Automne',       5: 'Automne',
        6:  'Hiver austral', 7: 'Hiver austral', 8: 'Hiver austral',
        9:  'Printemps',   10: 'Printemps',    11: 'Printemps',
    })

    print(f"   Periode : {df['timestamp'].min().date()} -> {df['timestamp'].max().date()}")
    print(f"   Navires uniques : {df['mmsi'].nunique():,}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. ZONAGE GEOGRAPHIQUE
# ─────────────────────────────────────────────────────────────────────────────

def assigner_zone(lat, lon):
    if lat > 12 and lon < 50:
        return 'Mer Rouge & Golfe Aden'
    elif lat > 20 and 50 <= lon <= 65:
        return 'Golfe Persique'
    elif lat < 12 and lon < 55 and lat > -15:
        return 'Cote Est Africaine'
    elif -10 <= lat <= 25 and 65 <= lon <= 85:
        return 'Approches Indiennes'
    elif lon > 95 and lat > -10:
        return 'Detroit de Malacca'
    elif lat < -20:
        return 'Australie & Pacifique'
    else:
        return 'Ocean Indien Central'


def ajouter_zones(df):
    print("Attribution des zones geographiques...")
    df['zone'] = df.apply(
        lambda r: assigner_zone(r['latitude'], r['longitude']), axis=1
    )
    print(df['zone'].value_counts().to_string())
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. STATISTIQUES DESCRIPTIVES
# ─────────────────────────────────────────────────────────────────────────────

def stats_descriptives(df):
    print("\nStatistiques SOG par zone...")
    grp = df.groupby('zone')['sog']

    table = grp.agg(
        N='count',
        Moyenne='mean',
        Mediane='median',
        Ecart_type='std',
        Min='min',
        P10=lambda x: x.quantile(0.10),
        P25=lambda x: x.quantile(0.25),
        P75=lambda x: x.quantile(0.75),
        P90=lambda x: x.quantile(0.90),
        P95=lambda x: x.quantile(0.95),
        Max='max',
    ).round(3)

    table['CV_pct']    = (table['Ecart_type'] / table['Moyenne'] * 100).round(1)
    table['IQR']       = (table['P75'] - table['P25']).round(3)
    table['Asymetrie'] = grp.apply(lambda x: x.skew()).round(3)
    table['Kurtosis']  = grp.apply(lambda x: x.kurtosis()).round(3)

    print(table.to_string())
    table.to_csv('stats_sog_par_zone.csv', encoding='utf-8-sig')
    return table


def stats_meteo(df):
    print("\nStatistiques meteorologiques...")
    METEO = {
        'significant wave height Hs (m)': 'Hs (m)',
        'wave period Tp (s)':             'Tp (s)',
        'wind_speed':                     'Vent (m/s)',
        'mean wave direction (°)':        'Dir vague (deg)',
        'temp_air_C':                     'T air (C)',
        'temp_mer_C':                     'T mer (C)',
    }
    cols = [c for c in METEO if c in df.columns]
    table = df[cols].rename(columns={c: METEO[c] for c in cols}).describe().T.round(3)
    table['skewness'] = df[cols].skew().values.round(3)
    table['kurtosis'] = df[cols].kurtosis().values.round(3)
    print(table.to_string())
    table.to_csv('stats_meteo.csv', encoding='utf-8-sig')
    return table


# ─────────────────────────────────────────────────────────────────────────────
# 4. AJUSTEMENT DES LOIS DE PROBABILITE
# ─────────────────────────────────────────────────────────────────────────────

LOIS = {
    'Normale':     norm,
    'Log-Normale': lognorm,
    'Gamma':       gamma,
    'Weibull':     weibull_min,
}


def ajuster_lois(df):
    print("\nAjustement des lois par zone...")
    zones = df['zone'].unique()
    resultats = {}

    for zone in zones:
        data = df[df['zone'] == zone]['sog'].dropna().values
        best_name, best_aic = None, np.inf
        best_params, best_ks, best_pv = None, None, None

        for nom, loi in LOIS.items():
            try:
                params = loi.fit(data)
                ll = np.sum(loi.logpdf(data, *params))
                aic = 2 * len(params) - 2 * ll
                ks, pv = kstest(data, loi.cdf, args=params)
                if aic < best_aic:
                    best_aic = aic
                    best_name = nom
                    best_params = params
                    best_ks = ks
                    best_pv = pv
            except Exception:
                pass

        resultats[zone] = {
            'loi':    best_name,
            'AIC':    round(best_aic, 1),
            'KS':     round(best_ks, 4) if best_ks else None,
            'p':      round(best_pv, 4) if best_pv else None,
            'params': best_params,
        }
        print(f"  {zone} -> {best_name} | AIC={best_aic:.0f} | KS={best_ks:.4f} | p={best_pv:.4f}")

    pd.DataFrame({z: {k: v for k, v in r.items() if k != 'params'}
                  for z, r in resultats.items()}).T.to_csv(
        'meilleures_lois.csv', encoding='utf-8-sig'
    )
    return resultats


# ─────────────────────────────────────────────────────────────────────────────
# 5. ESTIMATION TEMPS D'ARRIVEE (ETA) — MONTE CARLO
# ─────────────────────────────────────────────────────────────────────────────

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def estimer_eta(lat_dep, lon_dep, lat_arr, lon_arr, df, n_simulations=10000, seed=42):
    np.random.seed(seed)

    lat_mid = (lat_dep + lat_arr) / 2
    lon_mid = (lon_dep + lon_arr) / 2
    zone = assigner_zone(lat_mid, lon_mid)
    distance_km = haversine(lat_dep, lon_dep, lat_arr, lon_arr)

    vitesses = df[df['zone'] == zone]['sog_kmh'].dropna().values
    if len(vitesses) < 30:
        vitesses = df['sog_kmh'].dropna().values

    mu_log    = np.mean(np.log(vitesses))
    sigma_log = np.std(np.log(vitesses))
    vitesses_sim = np.clip(np.random.lognormal(mu_log, sigma_log, n_simulations), 1, 45)
    temps_sim = distance_km / vitesses_sim

    eta = {
        'zone':          zone,
        'distance_km':   round(distance_km, 1),
        'distance_nm':   round(distance_km / 1.852, 1),
        'ETA_median_h':  round(float(np.median(temps_sim)), 2),
        'ETA_mean_h':    round(float(np.mean(temps_sim)), 2),
        'ETA_P5_h':      round(float(np.percentile(temps_sim, 5)), 2),
        'ETA_P95_h':     round(float(np.percentile(temps_sim, 95)), 2),
        'vitesse_med_kn': round(float(np.median(vitesses_sim) / 1.852), 2),
        'simulations_h': temps_sim,
    }

    print(f"\nETA : {zone}")
    print(f"   Distance    : {eta['distance_nm']} nm")
    print(f"   ETA mediane : {eta['ETA_median_h']:.1f} h")
    print(f"   IC 90%      : [{eta['ETA_P5_h']:.1f} h — {eta['ETA_P95_h']:.1f} h]")
    return eta


# ─────────────────────────────────────────────────────────────────────────────
# 6. VISUALISATIONS
# ─────────────────────────────────────────────────────────────────────────────

def plot_boxplots(df):
    zones = list(df['zone'].value_counts().index)
    data_zones = [df[df['zone'] == z]['sog'].dropna().values for z in zones]
    couleurs = [list(ZONE_COLORS.values())[i % len(ZONE_COLORS)] for i in range(len(zones))]

    fig, ax = plt.subplots(figsize=(14, 5))
    bp = ax.boxplot(
        data_zones, patch_artist=True, notch=True,
        medianprops=dict(color='white', linewidth=2.5),
        whiskerprops=dict(color=LIGHT, linewidth=1.2),
        capprops=dict(color=LIGHT, linewidth=1.5),
        flierprops=dict(marker='.', markersize=2, alpha=0.2, color=GOLD),
    )
    for patch, c in zip(bp['boxes'], couleurs):
        patch.set_facecolor(c)
        patch.set_alpha(0.75)

    ax.set_xticks(range(1, len(zones) + 1))
    ax.set_xticklabels(zones, rotation=22, ha='right', fontsize=9)
    ax.set_ylabel('Vitesse SOG (noeuds)')
    ax.set_title('Distribution des vitesses par zone', fontweight='bold')
    ax.yaxis.grid(True)
    plt.tight_layout()
    plt.savefig('boxplot_vitesse.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("   Sauvegarde : boxplot_vitesse.png")
    return zones, couleurs


def plot_distributions(df, resultats_lois, zones, couleurs):
    ncols = 3
    nrows = int(np.ceil(len(zones) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, nrows * 5))
    axes = axes.flatten()

    for i, zone in enumerate(zones):
        ax = axes[i]
        data = df[df['zone'] == zone]['sog'].dropna().values
        c = couleurs[i]

        ax.hist(data, bins=60, density=True, alpha=0.6, color=c, edgecolor='none')

        info = resultats_lois.get(zone, {})
        loi_name = info.get('loi')
        params = info.get('params')

        if loi_name and params is not None:
            x = np.linspace(data.min(), data.max(), 300)
            y = LOIS[loi_name].pdf(x, *params)
            ax.plot(x, y, color='white', lw=2.5,
                    label=f"{loi_name}\nAIC={info['AIC']:.0f} | KS={info['KS']:.3f}")
            ax.fill_between(x, y, alpha=0.12, color='white')

        for q, ls in [(0.25, '--'), (0.5, '-'), (0.75, '--')]:
            ax.axvline(np.quantile(data, q), color=GOLD, lw=1.1, ls=ls, alpha=0.8)

        ax.set_title(zone, fontweight='bold', fontsize=10)
        ax.set_xlabel('SOG (noeuds)', fontsize=9)
        ax.set_ylabel('Densite', fontsize=9)
        ax.legend(fontsize=8)
        ax.text(0.97, 0.95, f'n={len(data):,}', transform=ax.transAxes,
                ha='right', va='top', color=CYAN, fontsize=9)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle('Distributions de Vitesse par Zone — Meilleures lois (AIC)',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig('distributions_vitesse.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("   Sauvegarde : distributions_vitesse.png")


def plot_meteo(df):
    METEO_VARS = {
        'significant wave height Hs (m)': 'Hs (m)',
        'wave period Tp (s)':             'Tp (s)',
        'wind_speed':                     'Vent (m/s)',
        'temp_air_C':                     'T. air (C)',
        'temp_mer_C':                     'T. mer (C)',
    }
    cols = [c for c in METEO_VARS if c in df.columns]
    colors = [CYAN, TEAL, GOLD, CORAL, '#9B5DE5']

    ncols = 3
    nrows = int(np.ceil(len(cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 5))
    axes = axes.flatten()

    for i, col in enumerate(cols):
        ax = axes[i]
        data = df[col].dropna()
        ax.hist(data, bins=60, density=True, color=colors[i % len(colors)],
                alpha=0.75, edgecolor='none')
        kde_x = np.linspace(data.min(), data.max(), 200)
        kde = stats.gaussian_kde(data.sample(min(5000, len(data)), random_state=42))
        ax.plot(kde_x, kde(kde_x), color='white', lw=2)
        ax.axvline(data.mean(),   color=GOLD,  lw=1.5, ls='--',
                   label=f'Moy={data.mean():.2f}')
        ax.axvline(data.median(), color=CORAL, lw=1.5, ls='-',
                   label=f'Med={data.median():.2f}')
        ax.set_title(METEO_VARS[col], fontweight='bold', fontsize=10)
        ax.set_ylabel('Densite', fontsize=8)
        ax.legend(fontsize=8)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle('Distributions Meteorologiques', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('distributions_meteo.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("   Sauvegarde : distributions_meteo.png")


def plot_draft(df, zones, couleurs):
    draft = df['draft'].replace(0, np.nan)
    draft = draft[draft.between(1, 25)].dropna()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.hist(draft, bins=50, density=True, color=OCEAN, alpha=0.8, edgecolor='none')
    kde_x = np.linspace(draft.min(), draft.max(), 200)
    kde = stats.gaussian_kde(draft.sample(min(5000, len(draft)), random_state=42))
    ax.plot(kde_x, kde(kde_x), color=GOLD, lw=2.5, label='KDE')
    ax.axvline(draft.mean(),   color=CORAL,  lw=2, ls='--',
               label=f'Moy={draft.mean():.1f}m')
    ax.axvline(draft.median(), color='white', lw=2, ls='-',
               label=f'Med={draft.median():.1f}m')
    ax.set_title("Distribution du Draft", fontweight='bold')
    ax.set_xlabel('Draft (m)')
    ax.set_ylabel('Densite')
    ax.legend(fontsize=9)

    ax2 = axes[1]
    draft_df = df[['zone', 'draft']].copy()
    draft_df['draft'] = draft_df['draft'].replace(0, np.nan)
    draft_df = draft_df[draft_df['draft'].between(1, 25)]
    data_z = [draft_df[draft_df['zone'] == z]['draft'].dropna().values for z in zones]
    bp = ax2.boxplot(
        data_z, patch_artist=True,
        medianprops=dict(color='white', lw=2),
        whiskerprops=dict(color=LIGHT),
        capprops=dict(color=LIGHT),
        flierprops=dict(marker='.', alpha=0.2, color=GOLD, markersize=2),
    )
    for patch, c in zip(bp['boxes'], couleurs):
        patch.set_facecolor(c)
        patch.set_alpha(0.75)
    ax2.set_xticks(range(1, len(zones) + 1))
    ax2.set_xticklabels(zones, rotation=22, ha='right', fontsize=8)
    ax2.set_ylabel('Draft (m)')
    ax2.set_title('Draft par zone', fontweight='bold')
    ax2.yaxis.grid(True)

    plt.tight_layout()
    plt.savefig('draft_analyse.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("   Sauvegarde : draft_analyse.png")


def plot_correlation(df):
    corr_vars = [
        'sog', 'draft',
        'significant wave height Hs (m)', 'wave period Tp (s)',
        'wind_speed', 'temp_air_C', 'temp_mer_C',
    ]
    corr_vars = [c for c in corr_vars if c in df.columns]
    labels = ['SOG', 'Draft', 'Hs', 'Tp', 'Vent', 'T.air', 'T.mer']

    corr = df[corr_vars].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(220, 20, as_cmap=True)

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr, mask=mask, cmap=cmap, vmin=-1, vmax=1, center=0,
                annot=True, fmt='.2f', linewidths=0.5, linecolor='#2A4060',
                ax=ax, xticklabels=labels[:len(corr_vars)],
                yticklabels=labels[:len(corr_vars)])
    ax.set_title('Matrice de Correlation', fontweight='bold', pad=12)
    plt.tight_layout()
    plt.savefig('matrice_correlation.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("   Sauvegarde : matrice_correlation.png")

    print("\nCorrelations avec SOG :")
    print(corr['sog'].drop('sog').sort_values(key=abs, ascending=False).round(3))


def plot_series_temporelles(df):
    ts_daily = df.set_index('timestamp')['sog'].resample('D').agg(['mean', 'std', 'count'])
    ts_daily = ts_daily[ts_daily['count'] >= 10]

    fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)

    ax = axes[0]
    ax.plot(ts_daily.index, ts_daily['mean'], color=CYAN, lw=1, alpha=0.8)
    ax.fill_between(ts_daily.index,
                    ts_daily['mean'] - ts_daily['std'],
                    ts_daily['mean'] + ts_daily['std'],
                    alpha=0.2, color=CYAN, label='+-1 ecart-type')
    ax.set_ylabel('SOG moyen (noeuds)')
    ax.set_title('Vitesse moyenne journaliere', fontweight='bold')
    ax.legend(fontsize=9)
    ax.yaxis.grid(True)

    ax2 = axes[1]
    ts_daily['ma30'] = ts_daily['mean'].rolling(30, center=True).mean()
    ts_daily['ma90'] = ts_daily['mean'].rolling(90, center=True).mean()
    ax2.plot(ts_daily.index, ts_daily['mean'], color=CYAN, lw=0.8, alpha=0.4, label='Brut')
    ax2.plot(ts_daily.index, ts_daily['ma30'], color=GOLD,  lw=2,   label='MM 30j')
    ax2.plot(ts_daily.index, ts_daily['ma90'], color=CORAL, lw=2.5, label='MM 90j')
    ax2.set_ylabel('SOG (noeuds)')
    ax2.set_title('Moyennes mobiles — Tendance', fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.yaxis.grid(True)

    ax3 = axes[2]
    ax3.bar(ts_daily.index, ts_daily['count'], color=TEAL, alpha=0.7, width=1, edgecolor='none')
    ax3.set_ylabel("Nb observations")
    ax3.set_title("Volume journalier", fontweight='bold')
    ax3.yaxis.grid(True)

    fig.suptitle("Serie Temporelle — Vitesse SOG", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('serie_temporelle.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("   Sauvegarde : serie_temporelle.png")


def plot_saisonnalite(df):
    MOIS  = ['Jan', 'Fev', 'Mar', 'Avr', 'Mai', 'Jun',
             'Jul', 'Aou', 'Sep', 'Oct', 'Nov', 'Dec']
    JOURS = ['Lun', 'Mar', 'Mer', 'Jeu', 'Ven', 'Sam', 'Dim']

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    ax = axes[0]
    sog_mois = df.groupby('month')['sog'].mean()
    ax.bar(range(1, 13), sog_mois.values, color=CYAN, alpha=0.85, edgecolor='none')
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(MOIS, fontsize=9)
    ax.set_ylabel('SOG moyen (noeuds)')
    ax.set_title('Saisonnalite mensuelle', fontweight='bold')
    ax.axhline(sog_mois.mean(), color=GOLD, lw=1.5, ls='--', label='Moy. globale')
    ax.legend(fontsize=9)
    ax.yaxis.grid(True)

    ax2 = axes[1]
    sog_jour = df.groupby('weekday')['sog'].mean()
    ax2.bar(range(7), sog_jour.values, color=TEAL, alpha=0.85, edgecolor='none')
    ax2.set_xticks(range(7))
    ax2.set_xticklabels(JOURS, fontsize=9)
    ax2.set_title('Saisonnalite hebdomadaire', fontweight='bold')
    ax2.axhline(sog_jour.mean(), color=GOLD, lw=1.5, ls='--')
    ax2.yaxis.grid(True)

    ax3 = axes[2]
    sog_heure = df.groupby('hour')['sog'].mean()
    ax3.plot(sog_heure.index, sog_heure.values,
             color=GOLD, lw=2.5, marker='o', markersize=4)
    ax3.fill_between(sog_heure.index, sog_heure.values, alpha=0.2, color=GOLD)
    ax3.set_xlabel('Heure UTC')
    ax3.set_title('Saisonnalite horaire', fontweight='bold')
    ax3.set_xticks(range(0, 24, 3))
    ax3.yaxis.grid(True)

    fig.suptitle('Patterns Saisonniers de la Vitesse SOG', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('saisonnalite.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("   Sauvegarde : saisonnalite.png")


def plot_decomposition_acf(df):
    ts_weekly = df.set_index('timestamp')['sog'].resample('W').mean().dropna()

    if len(ts_weekly) >= 24:
        decomp = seasonal_decompose(
            ts_weekly, model='additive', period=52, extrapolate_trend='freq'
        )
        fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
        components = [
            (ts_weekly,       'Serie originale', CYAN),
            (decomp.trend,    'Tendance',        GOLD),
            (decomp.seasonal, 'Saisonnalite',    TEAL),
            (decomp.resid,    'Residus',         CORAL),
        ]
        for ax, (series, title, color) in zip(axes, components):
            ax.plot(series.index, series.values, color=color, lw=1.5)
            if title == 'Residus':
                ax.axhline(0, color='white', lw=1, ls='--', alpha=0.5)
            ax.set_ylabel(title, fontsize=9)
            ax.yaxis.grid(True)
        axes[0].set_title('Decomposition STL — SOG hebdomadaire',
                          fontweight='bold', fontsize=13)
        plt.tight_layout()
        plt.savefig('decomposition_stl.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("   Sauvegarde : decomposition_stl.png")
    else:
        print("   Pas assez de semaines pour STL (besoin >= 24)")

    # ACF / PACF
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    plot_acf(ts_weekly.dropna(),  lags=40, ax=axes[0],
             color=CYAN, vlines_kwargs={'colors': CYAN})
    plot_pacf(ts_weekly.dropna(), lags=40, ax=axes[1], method='ywm',
              color=GOLD, vlines_kwargs={'colors': GOLD})
    axes[0].set_title('ACF', fontweight='bold')
    axes[1].set_title('PACF', fontweight='bold')
    for ax in axes:
        ax.set_xlabel('Lag (semaines)')
        ax.yaxis.grid(True)

    fig.suptitle('Structure de Dependance Temporelle', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('acf_pacf.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("   Sauvegarde : acf_pacf.png")

    # Test ADF
    adf_stat, adf_p, _, _, adf_crit, _ = adfuller(ts_weekly.dropna())
    print(f"\nTest Dickey-Fuller Augmente :")
    print(f"   Stat ADF : {adf_stat:.4f} | p-value : {adf_p:.4f}")
    if adf_p < 0.05:
        print("   Serie STATIONNAIRE -> ARMA applicable directement")
    else:
        print("   Serie NON stationnaire -> differenciation necessaire (ARIMA)")


def plot_eta(eta):
    temps = eta['simulations_h']
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.hist(temps, bins=80, density=True, color=TEAL, alpha=0.8, edgecolor='none')
    mu = np.mean(np.log(temps))
    sigma = np.std(np.log(temps))
    x = np.linspace(temps.min(), temps.max(), 300)
    ax.plot(x, lognorm.pdf(x, sigma, scale=np.exp(mu)),
            color=GOLD, lw=2.5, label='Log-Normale ajustee')
    ax.axvline(eta['ETA_P5_h'],    color=CORAL,  lw=1.5, ls='--', label='IC 5%')
    ax.axvline(eta['ETA_median_h'], color='white', lw=2,          label='Mediane')
    ax.axvline(eta['ETA_P95_h'],   color=CORAL,  lw=1.5, ls='--', label='IC 95%')
    ax.set_title("Distribution Monte Carlo du Temps d'Arrivee", fontweight='bold')
    ax.set_xlabel('Temps (heures)')
    ax.set_ylabel('Densite')
    ax.legend(fontsize=9)

    ax2 = axes[1]
    ax2.axis('off')
    lignes = [
        ('Distance',       f"{eta['distance_nm']} nm"),
        ('Zone',            eta['zone']),
        ('',               ''),
        ('ETA Mediane',    f"{eta['ETA_median_h']:.1f} h"),
        ('ETA Moyenne',    f"{eta['ETA_mean_h']:.1f} h"),
        ('IC 5%',          f"{eta['ETA_P5_h']:.1f} h"),
        ('IC 95%',         f"{eta['ETA_P95_h']:.1f} h"),
        ('Vitesse med.',   f"{eta['vitesse_med_kn']:.1f} noeuds"),
        ('Simulations',    f"{eta['n_sim']:,}"),
    ]
    y = 0.92
    for label, val in lignes:
        if not label:
            y -= 0.04
            continue
        color_val = CYAN if 'ETA' in label or 'IC' in label else GOLD
        ax2.text(0.05, y, label + ' :', color=LIGHT, fontsize=11,
                 transform=ax2.transAxes, va='top')
        ax2.text(0.60, y, val, color=color_val, fontsize=11,
                 fontweight='bold', transform=ax2.transAxes, va='top')
        y -= 0.09
    ax2.set_title("Resume ETA", fontweight='bold')

    plt.tight_layout()
    plt.savefig('eta_simulation.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("   Sauvegarde : eta_simulation.png")


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  PROJET GTT x ENSAE — Analyse AIS Ocean Indien")
    print("=" * 60)

    # 1. Chargement
    df = charger_donnees()

    # 2. Zonage
    df = ajouter_zones(df)

    # 3. Stats descriptives
    table_sog  = stats_descriptives(df)
    table_met  = stats_meteo(df)

    # 4. Lois de probabilite
    resultats_lois = ajuster_lois(df)

    # 5. Figures
    print("\nGeneration des figures...")
    zones, couleurs = plot_boxplots(df)
    plot_distributions(df, resultats_lois, zones, couleurs)
    plot_meteo(df)
    plot_draft(df, zones, couleurs)
    plot_correlation(df)
    plot_series_temporelles(df)
    plot_saisonnalite(df)
    plot_decomposition_acf(df)

    # 6. ETA exemple : Djibouti -> Mumbai
    print("\n" + "=" * 60)
    print("  ETA EXEMPLE : Djibouti -> Mumbai")
    eta = estimer_eta(
        lat_dep=11.59, lon_dep=43.14,
        lat_arr=18.96, lon_arr=72.82,
        df=df,
        n_simulations=10000,
    )
    eta['n_sim'] = 10000
    plot_eta(eta)

    print("\nAnalyse terminee !")
    print("Fichiers generes dans le repertoire courant.")