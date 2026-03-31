#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 16:24:49 2025

@author: leopera
"""


 
 ## ==============================================================
 ## 1. Importare Librerie Chiave
 ## ==============================================================

import os
import re
from pathlib import Path
import pandas as pd
import numpy as np
import pandas as pd
import os
from glob import glob
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from statsmodels.api import OLS, add_constant
from statsmodels.stats.stattools import durbin_watson
import statsmodels.api as sm
from math import sqrt

## ==============================================================
## 2. Importare il Dataset Core dei Fattori
## ==============================================================

# Percorso del file CSV (aggiorna se lo hai spostato)
file_path = Path("/Users/leopera/Desktop/dataset_fattori_core.csv")

if not file_path.exists():
    raise FileNotFoundError(f"❌ File non trovato in {file_path}")

# Lettura del dataset
df = pd.read_csv(file_path)

print("✅ dataset_fattori_core importato con successo!")
print(f"Righe totali: {len(df):,}")
print(f"Colonne totali: {len(df.columns):,}")
print("\nColonne disponibili:")
print(df.columns.tolist())

# Conversione Date e YearMonth
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df['YearMonth'] = df['YearMonth'].astype('period[M]')

print("\nIntervallo temporale:",
      df['Date'].min(), "→", df['Date'].max())

print("\nPrime 5 righe:")
print(df.head())


###--------------------
### Funzione per statistiche chiave
###--------------------


def stats(r, freq=12):
    r = r.dropna().astype(float)
    T = len(r)
    if T == 0:
        return pd.Series(dtype=float)

    mean = r.mean()
    vol = r.std(ddof=1)
    ann_ret = (1+r).prod()**(freq/T) - 1
    ann_vol = vol * np.sqrt(freq)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan

    tstat = mean / (vol / np.sqrt(T)) if vol > 0 else np.nan

    cum = (1+r).cumprod()
    running_max = cum.cummax()
    max_dd = (cum / running_max - 1).min()

    hit_ratio = (r > 0).mean()
    skew = r.skew()

    downside_vol = r[r < 0].std(ddof=1) * np.sqrt(freq)
    sortino = ann_ret / downside_vol if downside_vol > 0 else np.nan

    return pd.Series({
        'Obs': T,
        'Mean': mean,
        'Ann_Return': ann_ret,
        'Ann_Vol': ann_vol,
        'Sharpe': sharpe,
        't_stat': tstat,
        'Max_Drawdown': max_dd,
        'Hit_Ratio': hit_ratio,
        'Skewness': skew,
        'Sortino': sortino
    })





###---------------------
### Creazione Benchamrk
###--------------------
df_bench = df[['Date', 'ExcessRet', 'Market Value']].copy()

df_bench['Date'] = pd.to_datetime(df_bench['Date'], errors='coerce')
df_bench['ExcessRet'] = pd.to_numeric(df_bench['ExcessRet'], errors='coerce')
df_bench['Market Value'] = pd.to_numeric(df_bench['Market Value'], errors='coerce')

df_bench = df_bench.dropna(subset=['Date', 'ExcessRet', 'Market Value'])
df_bench = df_bench[df_bench['Market Value'] > 0]

def vw_excess(group):
    r = group['ExcessRet'].values
    w = group['Market Value'].values
    return np.average(r, weights=w)

benchmark_excess = (
    df_bench
    .groupby('Date', sort=True)
    .apply(vw_excess)
    .rename('Benchmark_Excess')
)

benchmark_excess = benchmark_excess.to_frame()
print(benchmark_excess.head())
print("Date range:",
      benchmark_excess.index.min(),
      "→",
      benchmark_excess.index.max())

benchmark_excess['Cumulative'] = (1 + benchmark_excess['Benchmark_Excess']).cumprod() 

# Plot
plt.figure()
plt.plot(benchmark_excess.index, benchmark_excess['Cumulative'])
plt.title("Benchmark Cumulative Excess Return")
plt.xlabel("Date")
plt.ylabel("Cumulative Excess Return")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


print(stats(benchmark_excess['Benchmark_Excess']))


 ## ==============================================================
 ## 2. Sistemare Caratteristiche per la Regressione
 ## ==============================================================



# Funzione di normalizzazione per ranks mensili
def normalize_by_rank(s: pd.Series) -> pd.Series:
    """
    Per ogni mese:
    - calcola il rank delle osservazioni
    - sottrae la media dei rank
    - divide per la std dei rank
    - se tutti NA o std=0 => restituisce 0
    """
    valid = s.dropna()
    # se non ci sono valori validi, tutti 0
    if valid.empty:
        return pd.Series(0.0, index=s.index)

    ranks = valid.rank(method='first')  # 1, 2, ..., N
    mu = ranks.mean()
    sigma = ranks.std(ddof=0)

    if sigma == 0 or np.isnan(sigma):
        z = pd.Series(0.0, index=valid.index)
    else:
        z = (ranks - mu) / sigma

    out = pd.Series(0.0, index=s.index)
    out.loc[valid.index] = z
    return out

# Costruiamo le versioni normalizzate per ciascun fattore
df['CARRY_norm'] = (
    df.groupby('YearMonth', group_keys=False)['Carry_signal']
      .apply(normalize_by_rank)
)

df['DEF_norm'] = (
    df.groupby('YearMonth', group_keys=False)['Defensive_signal']
      .apply(normalize_by_rank)
)

df['MOM_norm'] = (
    df.groupby('YearMonth', group_keys=False)['Mom_signal_adj']
      .apply(normalize_by_rank)
)

df['VALUE_norm'] = (
    df.groupby('YearMonth', group_keys=False)['Value_signal_adj']
      .apply(normalize_by_rank)
)

# Per coerenza con il paper: eventuali NaN residui li poniamo a 0 (valore "medio")
for c in ['CARRY_norm', 'DEF_norm', 'MOM_norm', 'VALUE_norm']:
    df[c] = df[c].fillna(0.0)

print("\nCheck caratteristiche normalizzate (prime righe):")
print(df[['ISIN','YearMonth','CARRY_norm','DEF_norm','MOM_norm','VALUE_norm']].head())


# Verifica dispersione mensile per togliere dalò dataset i primi mesi (mom e value)
disp = df.groupby('YearMonth')[['CARRY_norm','DEF_norm','MOM_norm','VALUE_norm']].std()

# Seleziona mesi dove tutte le std > 0
valid_months = disp[(disp > 0).all(axis=1)].index

start_month = valid_months.min()
print("Start month for full analysis:", start_month)


## ==============================================================
## 4. Regressioni Fama–MacBeth: Next_ExcessRet su 4 caratteristiche
## ==============================================================

# Dataset per la regressione: servono Next_ExcessRet e le 4 caratteristiche normalizzate
reg_cols = ['Next_ExcessRet', 'CARRY_norm', 'DEF_norm', 'MOM_norm', 'VALUE_norm']
df_reg = df[df['YearMonth'] >= start_month].copy()
df_reg = df_reg.dropna(subset=['Next_ExcessRet'])
print("\nNumero di osservazioni totali per regressione:",
      len(df_reg))

# Funzione per eseguire la cross-sectional regressione in un singolo mese
def run_cs_regression(group: pd.DataFrame):
    """
    Regressione cross-section in un mese:
    Next_ExcessRet_i,t+1 = alpha_t + beta1_t*CARRY_norm + ... + beta4_t*VALUE_norm + eps
    """
    y = group['Next_ExcessRet'].astype(float)
    X = group[['CARRY_norm', 'DEF_norm', 'MOM_norm', 'VALUE_norm']].astype(float)

    # Se poche osservazioni, saltiamo
    if len(group) < 20:
        return None

    # Aggiungi costante
    X = sm.add_constant(X)

    try:
        model = sm.OLS(y, X).fit()
    except Exception as e:
        # in caso di problemi numerici (multicollinearità estrema, ecc.)
        return None

    # Ritorniamo i coefficienti + R2 + numero osservazioni
    out = model.params.to_dict()
    out['R2'] = model.rsquared
    out['N'] = len(group)
    return out

# Applichiamo la regressione mese per mese
results = []
for ym, g in df_reg.groupby('YearMonth'):
    res = run_cs_regression(g)
    if res is not None:
        res['YearMonth'] = ym
        results.append(res)

fm_df = pd.DataFrame(results).set_index('YearMonth').sort_index()

print("\nNumero di mesi effettivamente usati (regressione riuscita):", len(fm_df))
print("\nEsempio coefficienti per i primi mesi:")
print(fm_df.head())

# -------------------------------------------------
# 4.1 Calcolo delle statistiche Fama–MacBeth
# -------------------------------------------------

coef_names = ['const', 'CARRY_norm', 'DEF_norm', 'MOM_norm', 'VALUE_norm']
summary_rows = {}

for c in coef_names:
    s = fm_df[c].dropna()
    T = len(s)
    if T <= 1:
        mean_beta = np.nan
        std_beta = np.nan
        t_stat = np.nan
    else:
        mean_beta = s.mean()
        std_beta = s.std(ddof=1)
        t_stat = mean_beta / (std_beta / sqrt(T)) if std_beta > 0 else np.nan

    summary_rows[c] = {
        'Mean_beta': mean_beta,
        'Time_std_beta': std_beta,
        'T_months': T,
        't_FM': t_stat
    }

fm_summary = pd.DataFrame(summary_rows).T

print("\n====================")
print("Fama–MacBeth summary (senza controlli)")
print("====================")
print(fm_summary)

# (Opzionale) Statistiche medie su R2 e numero di obs per mese
print("\nMedia R2 cross-sectional:", fm_df['R2'].mean())
print("Media numero di bond per mese (N):", fm_df['N'].mean())



## ==============================================================
## 5. Fama–MacBeth per singolo fattore (univariate regressions)
## ==============================================================

# Fattori che vogliamo testare singolarmente (usano già le versioni normalizzate)
single_factors = {
    'CARRY':  'CARRY_norm',
    'DEF':    'DEF_norm',
    'MOM':    'MOM_norm',
    'VALUE':  'VALUE_norm'
}

# Controllo che le colonne esistano
needed_cols = ['YearMonth', 'Next_ExcessRet'] + list(single_factors.values())
missing = [c for c in needed_cols if c not in df.columns]
if missing:
    raise KeyError(f"Mancano queste colonne nel dataframe per le regressioni univariate: {missing}")

# Dataset base per le regressioni univariate
df_reg_uni = df.dropna(subset=['Next_ExcessRet', 'YearMonth']).copy()

print("\nOsservazioni totali disponibili per le regressioni univariate:",
      len(df_reg_uni))


def fama_macbeth_single_factor(df_in, factor_col, y_col='Next_ExcessRet',
                               min_obs=20):
    """
    Regressioni Fama–MacBeth univariate:
        y_{i,t+1} = alpha_t + beta_t * factor_{i,t} + eps_{i,t+1}

    Restituisce:
        - fm_df: serie temporale di alpha_t, beta_t, R2, N per ciascun mese
        - summary: dict con Mean_beta, Time_std_beta, T_months, t_FM, Mean_R2, Mean_N
    """
    results = []

    for ym, g in df_in.groupby('YearMonth'):
        g = g.dropna(subset=[y_col, factor_col])
        if len(g) < min_obs:
            continue

        y = g[y_col].astype(float)
        X = g[[factor_col]].astype(float)
        X = sm.add_constant(X)

        try:
            model = sm.OLS(y, X).fit()
        except Exception:
            continue

        res = {
            'YearMonth': ym,
            'alpha': model.params.get('const', np.nan),
            'beta':  model.params.get(factor_col, np.nan),
            'R2':    model.rsquared,
            'N':     len(g)
        }
        results.append(res)

    if not results:
        raise RuntimeError(f"Nessuna regressione riuscita per il fattore {factor_col}")

    fm_df = pd.DataFrame(results).set_index('YearMonth').sort_index()

    # Statistiche Fama–MacBeth sulla serie dei beta_t
    s = fm_df['beta'].dropna()
    T = len(s)
    if T <= 1:
        mean_beta = np.nan
        std_beta = np.nan
        t_stat = np.nan
    else:
        mean_beta = s.mean()
        std_beta = s.std(ddof=1)
        t_stat = mean_beta / (std_beta / sqrt(T)) if std_beta > 0 else np.nan

    summary = {
        'Mean_beta': mean_beta,
        'Time_std_beta': std_beta,
        'T_months': T,
        't_FM': t_stat,
        'Mean_R2': fm_df['R2'].mean(),
        'Mean_N': fm_df['N'].mean()
    }

    return fm_df, summary


# Applichiamo la funzione a tutti i fattori

fm_uni_results = {}   # serie temporali (alpha_t, beta_t, ecc.)
fm_uni_summary = {}   # statistiche riassuntive

for name, col in single_factors.items():
    print(f"\n>>> Fama–MacBeth univariata per il fattore {name} ({col})")
    fm_df_factor, summary_factor = fama_macbeth_single_factor(df_reg_uni, factor_col=col)

    fm_uni_results[name] = fm_df_factor
    fm_uni_summary[name] = summary_factor

    # Piccolo check: primi risultati temporali per il fattore
    print(fm_df_factor.head())
    print("Riassunto:", summary_factor)

# Tabella riassuntiva finale per tutti i fattori
fm_uni_summary_df = pd.DataFrame(fm_uni_summary).T[
    ['Mean_beta', 'Time_std_beta', 'T_months', 't_FM', 'Mean_R2', 'Mean_N']
]

print("\n====================")
print("Fama–MacBeth univariate summary (per fattore)")
print("====================")
print(fm_uni_summary_df)

































## ==============================================================
## 6. Portafogli Long–Short "linear-in-ranks" per ciascun fattore
## ==============================================================

# Costruiamo, per ogni fattore, una serie di ritorni LS:
#   - pesi lineari nel segnale normalizzato (CARRY_norm, DEF_norm, ecc.)
#   - per ogni mese:
#         w_i,t ∝ (score_i,t - media_score_t)
#         normalizzati in modo che somma |w| = 2  (1 long, 1 short)
#         r_factor,t+1 = Σ_i w_i,t * Next_ExcessRet_i,t+1

def build_ls_from_score(df_in, score_col, ret_col='Next_ExcessRet',
                        min_obs=20, gross_exposure=2.0):
    """
    Crea una serie temporale di ritorni long–short per un singolo fattore.

    Per ogni YearMonth:
      - filtra osservazioni valide
      - centra il segnale (score - media)
      - normalizza i pesi perché Σ|w| = gross_exposure
      - calcola il ritorno del portafoglio: Σ w_i * ret_i

    Restituisce:
      - pd.Series indicizzata per YearMonth con i ritorni LS mensili
    """
    rets = []
    idx = []

    for ym, g in df_in.groupby('YearMonth'):
        g = g.dropna(subset=[score_col, ret_col])
        if len(g) < min_obs:
            continue

        s = g[score_col].astype(float)
        # centra il segnale (anche se i *_norm sono già ≈ centrati, qui siamo sicuri)
        s = s - s.mean()

        denom = s.abs().sum()
        if denom == 0 or np.isnan(denom):
            continue

        # pesi long–short con esposizione lorda fissata (Σ|w| = gross_exposure)
        w = gross_exposure * s / denom

        r = (w * g[ret_col].astype(float)).sum()
        rets.append(r)
        idx.append(ym)

    if not rets:
        raise RuntimeError(f"Nessun ritorno costruito per il fattore {score_col}")

    return pd.Series(rets, index=pd.Index(idx, name='YearMonth'),
                     name=score_col + '_LS_rank').sort_index()


# Costruzione delle serie LS "grezze" (prima della scalatura a volatilità costante)
carry_ls_raw  = build_ls_from_score(df, 'CARRY_norm')
def_ls_raw    = build_ls_from_score(df, 'DEF_norm')
mom_ls_raw    = build_ls_from_score(df, 'MOM_norm')
value_ls_raw  = build_ls_from_score(df, 'VALUE_norm')

print("\nPrime osservazioni dei ritorni LS grezzi (CARRY):")
print(carry_ls_raw.head())

## ==============================================================
## 7. Scalare i fattori LS a volatilità costante (5%) con finestra 3–12 mesi
## ==============================================================

def scale_to_constant_vol(ret_series, target_vol_annual=0.05,
                          window=12, min_periods=3):
    """
    Scala i ritorni mensili verso una volatilità TARGET ANNUALE (di default 5%).

    - calcola la volatilità rolling MENSILE su 'window' mesi (min_periods=3)
    - converte la target annua in target mensile: sigma_month = target_vol_annual / sqrt(12)
    - r_scaled_t = r_t * (sigma_month / vol_month_t)
    """
    # volatilità rolling mensile
    vol_month = ret_series.rolling(window=window, min_periods=min_periods).std()
    vol_month = vol_month.replace(0, np.nan)

    # target mensile corrispondente al 5% annuo
    sigma_target_month = target_vol_annual / np.sqrt(12)

    scale = sigma_target_month / vol_month
    scaled = ret_series * scale

    # dove non abbiamo abbastanza storia: uso i ritorni grezzi
    scaled = scaled.where(np.isfinite(scaled), ret_series)

    return scaled, vol_month, scale

carry_ls_scaled, carry_ls_vol, carry_ls_scale   = scale_to_constant_vol(carry_ls_raw)
def_ls_scaled,   def_ls_vol,   def_ls_scale     = scale_to_constant_vol(def_ls_raw)
mom_ls_scaled,   mom_ls_vol,   mom_ls_scale     = scale_to_constant_vol(mom_ls_raw)
value_ls_scaled, value_ls_vol, value_ls_scale   = scale_to_constant_vol(value_ls_raw)

print("\nStatistiche base dei fattori LS (scalati a 5% vol, approx):")
for name, s in {
    'CARRY_LS': carry_ls_scaled,
    'DEF_LS':   def_ls_scaled,
    'MOM_LS':   mom_ls_scaled,
    'VALUE_LS': value_ls_scaled
}.items():
    r = s.dropna()
    if len(r) == 0:
        continue
    ann_ret = (1 + r).prod()**(12/len(r)) - 1
    ann_vol = r.std(ddof=0) * np.sqrt(12)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
    print(f"{name}: AnnRet={ann_ret:.4f}, AnnVol={ann_vol:.4f}, Sharpe={sharpe:.2f}, Obs={len(r)}")


## ==============================================================
## 8. Portafoglio combinato dei quattro fattori (equal-weight di pesi a vol 5%)
## ==============================================================

# Allineiamo le serie dei 4 fattori scalati sullo stesso indice temporale
common_index = carry_ls_scaled.dropna().index
common_index = common_index.intersection(def_ls_scaled.dropna().index)
common_index = common_index.intersection(mom_ls_scaled.dropna().index)
common_index = common_index.intersection(value_ls_scaled.dropna().index)

carry_c = carry_ls_scaled.loc[common_index]
def_c   = def_ls_scaled.loc[common_index]
mom_c   = mom_ls_scaled.loc[common_index]
val_c   = value_ls_scaled.loc[common_index]

# Portafoglio combinato: media semplice dei 4 fattori a vol costante
combined_ls = (carry_c + def_c + mom_c + val_c) / 4.0
combined_ls.name = "Combined_LS"

print("\nPrime osservazioni Combined_LS:")
print(combined_ls.head())

# Statistiche base del portafoglio combinato
r = combined_ls.dropna()
if len(r) > 0:
    ann_ret = (1 + r).prod()**(12/len(r)) - 1
    ann_vol = r.std(ddof=0) * np.sqrt(12)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan

    print("\nStatistiche Combined_LS (equal-weight dei 4 fattori, ognuno a vol 5%):")
    print(f"Annuale Return  : {ann_ret:.4f}")
    print(f"Annuale Vol     : {ann_vol:.4f}")
    print(f"Sharpe Ratio    : {sharpe:.2f}")
    print(f"Numero osservaz.: {len(r)}")
else:
    print("⚠️ Nessuna osservazione valida per Combined_LS.")
    
    
## ==============================================================
## 9. Plot cumulato dei 4 fattori LS + Combined (stesso grafico)
## ==============================================================

import matplotlib.ticker as mticker

# Usiamo le serie tagliate a partire da first_valid
series_dict = {
    "Carry":     carry_c,
    "Defensive": def_c,
    "Momentum":  mom_c,
    "Value":     val_c,
    "Combined":  combined_ls,
}

# Indice comune (dovrebbe essere già identico, ma lo rifacciamo per sicurezza)
common_index = None
for s in series_dict.values():
    idx = s.dropna().index
    common_index = idx if common_index is None else common_index.intersection(idx)

# Costruiamo le curve cumulate rebased a 0% alla prima data comune
cum_series = {}
for name, s in series_dict.items():
    r = s.loc[common_index].astype(float)
    wealth = (1 + r).cumprod()          # indice di ricchezza
    wealth = wealth / wealth.iloc[0]    # parte da 1
    cum = wealth - 1                    # 0% alla prima data
    cum_series[name] = cum

# Asse x (se PeriodIndex → datetime)
idx0 = list(cum_series.values())[0].index
if hasattr(idx0, "to_timestamp"):
    x_axis = idx0.to_timestamp()
else:
    x_axis = idx0

plt.figure(figsize=(10,6))

plt.plot(x_axis, cum_series["Carry"].values,
         label="Carry", color="tab:green", linewidth=1.8)
plt.plot(x_axis, cum_series["Defensive"].values,
         label="Defensive", color="tab:purple", linewidth=1.8)
plt.plot(x_axis, cum_series["Momentum"].values,
         label="Momentum", color="tab:red", linewidth=1.8)
plt.plot(x_axis, cum_series["Value"].values,
         label="Value", color="tab:blue", linewidth=1.8)
plt.plot(x_axis, cum_series["Combined"].values,
         label="Combined", color="black", linewidth=2.4)

plt.title("Factor Long–Short (vol target 5%)\nCrescita cumulata dei portafogli LS", fontsize=12)
plt.ylabel("Cumulative Return", fontsize=11)
plt.xlabel("")
plt.grid(True, alpha=0.3)

ax = plt.gca()
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))

plt.legend(loc="upper left")
plt.tight_layout()
desktop = Path.home() / "Desktop"
fig_path_png = desktop / "Figure_6_ConstantVol_Factors.png"
fig_path_pdf = desktop / "Figure_6_ConstantVol_Factors.pdf"

# English title
plt.title("Cumulative Performance of Volatility-Scaled Long–Short Factor Portfolios (5% Target)",
          fontsize=12)

# Save (PNG for Word + PDF vector quality)
plt.savefig(fig_path_png, dpi=300, bbox_inches="tight")
plt.savefig(fig_path_pdf, bbox_inches="tight")

print(f"✅ Figure saved on Desktop: {fig_path_png}")
print(f"✅ Vector version saved on Desktop: {fig_path_pdf}")
plt.show()

# ==============================================================
# 9.5. Creare segnale COMBINED per bond-mese + quintili Comb_q
# ==============================================================

# 1) segnale combinato (per ogni ISIN e mese)
df["COMB_norm"] = (df["CARRY_norm"] + df["DEF_norm"] + df["MOM_norm"] + df["VALUE_norm"]) / 4.0

# 2) quintili mese per mese (0..4)
def make_monthly_quintiles(s: pd.Series) -> pd.Series:
    try:
        return pd.qcut(s, 5, labels=[0,1,2,3,4])
    except Exception:
        # se in quel mese ci sono pochi bond o valori troppo uguali
        return pd.Series(np.nan, index=s.index)

df["Comb_q"] = (
    df.groupby("YearMonth", group_keys=False)["COMB_norm"]
      .apply(make_monthly_quintiles)
      .astype("float")   # oppure "Int64" se vuoi interi con NA
)

print("✅ Colonna Comb_q creata:", "Comb_q" in df.columns)
print(df[["YearMonth", "ISIN", "COMB_norm", "Comb_q"]].dropna().head())


## ==============================================================
## 10. Portafogli quintili – versione allineata su mesi comuni
## ==============================================================

def build_quintile_portfolios(df_in, quintile_col, ret_col='Next_ExcessRet'):
    """
    Costruisce i portafogli Q1..Q5 e LS (Q5-Q1) per un dato fattore,
    e allinea tutti i mesi al set comune finale.
    
    Restituisce:
        - qret: dataframe q1..q5 per mese
        - ls: serie long-short
    """
    rets = []
    for ym, g in df_in.groupby('YearMonth'):
        g = g.dropna(subset=[quintile_col, ret_col])
        if g.empty:
            continue

        # dentro ogni mese: value-weighted return per Q1..Q5
        q = {}
        for qv in [0,1,2,3,4]:
            sub = g[g[quintile_col] == qv]
            if len(sub) == 0:
                q[f"Q{qv+1}"] = np.nan
            else:
                w = sub['Market Value'].astype(float)
                r = sub[ret_col].astype(float)
                q[f"Q{qv+1}"] = np.average(r, weights=w)

        q['YearMonth'] = ym
        rets.append(q)

    qret = pd.DataFrame(rets).set_index('YearMonth').sort_index()

    # -------------------
    # LS portfolio
    # -------------------
    qret['LS'] = qret['Q5'] - qret['Q1']

    return qret


### Costruzione Q-portfolios per ogni fattore (grezzi)
quintiles_raw = {}
for name, col in {
    "CARRY": "Carry_q",
    "DEFENSIVE": "Def_q",
    "MOMENTUM": "Mom_q",
    "VALUE": "Value_q",
    "COMBINED": "Comb_q"
}.items():
    print(f"Costruzione quintili per {name} ({col})...")
    quintiles_raw[name] = build_quintile_portfolios(df, col)
    

# ==============================================================
# 11. Allineamento su mesi comuni — fatto UNA sola volta
# ==============================================================

# Intersezione degli indici temporali
common_index = None
for q in quintiles_raw.values():
    idx = q.index.dropna()
    common_index = idx if common_index is None else common_index.intersection(idx)

print("\nMesi comuni:", len(common_index))
print(common_index.min(), "→", common_index.max())

# Applica cut ai quintili
quintiles = {name: q.loc[common_index] for name, q in quintiles_raw.items()}


# ==============================================================
# 12. Statistiche finali (tutte su stesso numero di mesi)
# ==============================================================

def qstats(series):
    """Statistiche base"""
    s = series.dropna()
    N = len(s)
    mu = s.mean()
    annret = (1 + s).prod()**(12/N) - 1 if N > 0 else np.nan
    annvol = s.std(ddof=0)*np.sqrt(12) if N > 1 else np.nan
    sharpe = annret / annvol if annvol and annvol>0 else np.nan
    return pd.Series([N, mu, annret, annvol, sharpe],
                     index=["N","MeanRet","AnnRet","AnnVol","Sharpe"])


summary = {}
for name, q in quintiles.items():
    summary[name] = qstats(q['LS'])

summary_df = pd.DataFrame(summary).T

print("\n====================")
print("Riassunto Q5−Q1 ALLINEATO")
print("====================")
print(summary_df.applymap(lambda x: f"{x:.4f}" if isinstance(x,(int,float,np.floating)) else x))

    
print("\n====================")
print("Statistiche per quintili (ALLINEATE) per ogni fattore")
print("====================")

detailed_quintile_stats = {}

for name, q in quintiles.items():
    print(f"\n>>> {name}")
    rows = {}
    for col in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'LS']:
        rows[col] = qstats(q[col])

    stats_df = pd.DataFrame(rows).T[['N', 'MeanRet', 'AnnRet', 'AnnVol', 'Sharpe']]
    detailed_quintile_stats[name] = stats_df

    print(stats_df.applymap(
        lambda x: f"{x:.4f}" if isinstance(x, (int, float, np.floating)) else x
    ))
   
    
    
 ## ==============================================================
## X. Tabella performance fattori LS (linear-in-ranks, vol ~5%)
## ==============================================================

# 1) Raccogliamo le serie LS scalate (quelle usate per il grafico)
factor_ls = {
    "CARRY":     carry_ls_scaled,
    "DEFENSIVE": def_ls_scaled,
    "MOMENTUM":  mom_ls_scaled,
    "VALUE":     value_ls_scaled,
    "COMBINED":  combined_ls,
}

# 2) Allineiamo anche loro sui mesi comuni (gli stessi 58 mesi di prima)
common_index_ls = None
for s in factor_ls.values():
    idx = s.dropna().index
    common_index_ls = idx if common_index_ls is None else common_index_ls.intersection(idx)

print("\nMesi comuni LS (vol target):", len(common_index_ls))
print(common_index_ls.min(), "→", common_index_ls.max())

factor_ls_aligned = {name: s.loc[common_index_ls] for name, s in factor_ls.items()}


# 3) Funzione statistiche (riuso stessa logica di prima)
def ls_stats(r):
    r = r.dropna().astype(float)
    N = len(r)
    if N == 0:
        return pd.Series([np.nan]*5, index=["N","MeanRet","AnnRet","AnnVol","Sharpe"])
    mean_ret = r.mean()
    ann_ret  = (1 + r).prod()**(12/N) - 1
    ann_vol  = r.std(ddof=0) * np.sqrt(12) if N > 1 else np.nan
    sharpe   = ann_ret / ann_vol if (ann_vol and ann_vol > 0) else np.nan
    return pd.Series([N, mean_ret, ann_ret, ann_vol, sharpe],
                     index=["N","MeanRet","AnnRet","AnnVol","Sharpe"])


# 4) Costruiamo la tabella
ls_summary = {}
for name, s in factor_ls_aligned.items():
    ls_summary[name] = ls_stats(s)

ls_summary_df = pd.DataFrame(ls_summary).T

print("\n====================")
print("Factor LS (linear-in-ranks, vol target ≈5%)")
print("====================")
print(ls_summary_df.applymap(
    lambda x: f"{x:.4f}" if isinstance(x, (int,float,np.floating)) else x
))
   
    

## ==============================================================
## X. Correlazione tra i fattori LS (vol target ≈5%)
## ==============================================================

# 1) Raccogliamo i fattori LS scalati
factor_ls = {
    "CARRY":     carry_ls_scaled,
    "DEFENSIVE": def_ls_scaled,
    "MOMENTUM":  mom_ls_scaled,
    "VALUE":     value_ls_scaled,
    "COMBINED":  combined_ls,
}

# 2) Allineiamo sulle stesse date
common_index_corr = None
for s in factor_ls.values():
    idx = s.dropna().index
    common_index_corr = idx if common_index_corr is None else common_index_corr.intersection(idx)

print("\nMesi comuni usati per correlazioni:", len(common_index_corr))
print(common_index_corr.min(), "→", common_index_corr.max())

# 3) Costruiamo un DataFrame con tutti i fattori allineati
corr_df = pd.DataFrame({
    name: s.loc[common_index_corr].values
    for name, s in factor_ls.items()
}, index=common_index_corr)

# 4) Matrice di correlazione
corr_matrix = corr_df.corr()

print("\n====================")
print("Correlazione tra fattori LS (vol target ≈5%)")
print("====================")
print(corr_matrix.round(3))
    







## ==============================================================
## X. Costruzione ConstVol per quintile LS (Q5-Q1)
## ==============================================================

constvol_stats = {}

for name, q in quintiles.items():
    ls_series = q['LS'].dropna()
    
    # scala LS a 5% vol annua
    ls_scaled, _, _ = scale_to_constant_vol(ls_series,
                                            target_vol_annual=0.05,
                                            window=12,
                                            min_periods=3)
    
    stats_cv = ls_stats(ls_scaled)
    constvol_stats[name] = stats_cv

constvol_df = pd.DataFrame(constvol_stats).T

print("\n====================")
print("ConstVol (Q5−Q1 scaled to 5%)")
print("====================")
print(constvol_df[['AnnRet','AnnVol','Sharpe']].round(4))




value_table = detailed_quintile_stats['VALUE'][['AnnRet','AnnVol','Sharpe']].copy()

# aggiungi ConstVol
value_table.loc['ConstVol'] = constvol_df.loc['VALUE'][['AnnRet','AnnVol','Sharpe']]

print(value_table.round(4))



# ==============================================================
# AQR Table 4 replica: Q1..Q5, Q5-Q1, ConstVol (Ret/Vol/SR)
# Uses your existing: quintiles (dict), scale_to_constant_vol(), qstats()
# ==============================================================

factor_order = ["CARRY", "DEFENSIVE", "MOMENTUM", "VALUE", "COMBINED"]
cols = ["Q1", "Q2", "Q3", "Q4", "Q5", "LS", "ConstVol"]

rows = []
table = pd.DataFrame(index=pd.MultiIndex.from_product([factor_order, ["Ret.", "Vol.", "S.R."]]),
                     columns=["Q1","Q2","Q3","Q4","Q5","Q5-Q1","ConstVol"],
                     dtype=float)

# helper: get (AnnRet, AnnVol, Sharpe) from your qstats output
def pick_stats(s):
    st = qstats(s)  # returns N, MeanRet, AnnRet, AnnVol, Sharpe
    return float(st["AnnRet"]), float(st["AnnVol"]), float(st["Sharpe"])

for fac in factor_order:
    q = quintiles[fac].copy()

    # LS already exists in your build_quintile_portfolios, but ensure it:
    if "LS" not in q.columns:
        q["LS"] = q["Q5"] - q["Q1"]

    # ConstVol = scaled version of quintile LS (Q5-Q1), exactly like AQR Table 4
    ls_scaled, _, _ = scale_to_constant_vol(q["LS"], target_vol_annual=0.05, window=12, min_periods=3)

    # Fill the table for this factor
    for bucket in ["Q1","Q2","Q3","Q4","Q5"]:
        annret, annvol, sr = pick_stats(q[bucket])
        table.loc[(fac, "Ret."), bucket] = annret
        table.loc[(fac, "Vol."), bucket] = annvol
        table.loc[(fac, "S.R."), bucket] = sr

    # Q5-Q1 column uses LS
    annret, annvol, sr = pick_stats(q["LS"])
    table.loc[(fac, "Ret."), "Q5-Q1"] = annret
    table.loc[(fac, "Vol."), "Q5-Q1"] = annvol
    table.loc[(fac, "S.R."), "Q5-Q1"] = sr

    # ConstVol column
    annret, annvol, sr = pick_stats(ls_scaled)
    table.loc[(fac, "Ret."), "ConstVol"] = annret
    table.loc[(fac, "Vol."), "ConstVol"] = annvol
    table.loc[(fac, "S.R."), "ConstVol"] = sr

# ---- formatting like the paper: Ret/Vol in %, SR as number
table_fmt = table.copy()

for idx in table_fmt.index:
    metric = idx[1]
    if metric in ["Ret.", "Vol."]:
        table_fmt.loc[idx] = table_fmt.loc[idx].apply(lambda x: "" if pd.isna(x) else f"{100*x:.1f}%")
    else:
        table_fmt.loc[idx] = table_fmt.loc[idx].apply(lambda x: "" if pd.isna(x) else f"{x:.2f}")

print("\n====================")
print("Table 4 — Quintile portfolio tests (AQR-style)")
print("====================")
print(table_fmt)



###------------------
###Creazione dataset per nuova pagina
###------------------

# 1) Selezione colonne utili
cols_needed = [
    "ISIN", "Date", "YearMonth",
    "Market Value",
    "Next_ExcessRet",
    "OAS", "Duration", "Rating_num",
    "CARRY_norm", "DEF_norm", "MOM_norm", "VALUE_norm"
]

missing = [c for c in cols_needed if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns in df: {missing}")

df_lo = df[cols_needed].copy()

# 2) Tipi e pulizia minima
df_lo["Date"] = pd.to_datetime(df_lo["Date"], errors="coerce")
df_lo["YearMonth"] = df_lo["YearMonth"].astype("period[M]")

for c in ["Market Value","Next_ExcessRet","OAS","Duration","Rating_num",
          "CARRY_norm","DEF_norm","MOM_norm","VALUE_norm"]:
    df_lo[c] = pd.to_numeric(df_lo[c], errors="coerce")

# 3) (Opzionale ma consigliato) teniamo solo righe con le variabili chiave per l’ottimizzazione
#    - Market Value > 0 per benchmark weights
#    - segnali non mancanti (se vuoi partire dal mese in cui ci sono tutti)
df_lo = df_lo.dropna(subset=["Date","YearMonth","ISIN","Market Value","OAS","Duration"])
df_lo = df_lo[df_lo["Market Value"] > 0]

# Se vuoi garantire che COMBO sia calcolabile:
df_lo = df_lo.dropna(subset=["CARRY_norm","DEF_norm","MOM_norm","VALUE_norm"])

# 4) Salvataggio su Desktop
desktop_path = Path.home() / "Desktop" / "dataset_long_only_inputs.csv"
df_lo.to_csv(desktop_path, index=False)

print("✅ Dataset long-only salvato su:", desktop_path)
print("Shape:", df_lo.shape)
print("Date range:", df_lo["Date"].min(), "→", df_lo["Date"].max())
print("First rows:")
print(df_lo.head())

# ==============================================================
# EXPORT regressione multivariata (fm_summary) in Excel su Desktop
# ==============================================================

from pathlib import Path

desktop = Path.home() / "Desktop"
out_path = desktop / "FamaMacBeth_Multivariate.xlsx"

# esporta la tabella riassuntiva (quella che ti serve per Word)
fm_summary.to_excel(out_path, sheet_name="FM_Multivariate")

print(f"✅ File Excel salvato qui: {out_path}")

# ==============================================================
# EXPORT tabella ritorni
# ==============================================================

out_path_table4 = desktop / "Table_4_Quintile_Portfolios.xlsx"

# Esporta la versione NON formattata (numerica, meglio per Word/Excel)
table.to_excel(out_path_table4, sheet_name="Table_4")

print(f"✅ Table 4 salvata su Desktop: {out_path_table4}")




import os
import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))

plt.plot(x_axis, cum_series["Carry"].values, label="Carry", color="tab:green", linewidth=1.8)
plt.plot(x_axis, cum_series["Defensive"].values, label="Defensive", color="tab:purple", linewidth=1.8)
plt.plot(x_axis, cum_series["Momentum"].values, label="Momentum", color="tab:red", linewidth=1.8)
plt.plot(x_axis, cum_series["Value"].values, label="Value", color="tab:blue", linewidth=1.8)
plt.plot(x_axis, cum_series["Combined"].values, label="Combined", color="black", linewidth=2.4)

plt.title("Cumulative Performance of Volatility-Scaled Long–Short Factor Portfolios (5% Target)")
plt.ylabel("Cumulative Return")
plt.grid(True, alpha=0.3)
plt.legend(loc="upper left")
plt.tight_layout()

desktop = os.path.join(os.path.expanduser("~"), "Desktop")
plt.savefig(os.path.join(desktop, "combined_portfolio_plot.png"), dpi=300, bbox_inches="tight")

plt.show()




#_______
factor_order = ["CARRY", "DEFENSIVE", "MOMENTUM", "VALUE", "COMBINED"]

table = pd.DataFrame(
    index=pd.MultiIndex.from_product([factor_order, ["Ret.", "Vol.", "S.R."]]),
    columns=["Q1","Q2","Q3","Q4","Q5","Q5-Q1","ConstVol"],
    dtype=float
)

# qui salviamo le serie scaled per poterle poi plottare
constvol_series = {}

def pick_stats(s):
    st = qstats(s)
    return float(st["AnnRet"]), float(st["AnnVol"]), float(st["Sharpe"])

for fac in factor_order:
    q = quintiles[fac].copy()

    if "LS" not in q.columns:
        q["LS"] = q["Q5"] - q["Q1"]

    ls_scaled, _, _ = scale_to_constant_vol(
        q["LS"],
        target_vol_annual=0.05,
        window=12,
        min_periods=3
    )

    # salva la serie scaled
    constvol_series[fac] = ls_scaled.copy()

    for bucket in ["Q1","Q2","Q3","Q4","Q5"]:
        annret, annvol, sr = pick_stats(q[bucket])
        table.loc[(fac, "Ret."), bucket] = annret
        table.loc[(fac, "Vol."), bucket] = annvol
        table.loc[(fac, "S.R."), bucket] = sr

    annret, annvol, sr = pick_stats(q["LS"])
    table.loc[(fac, "Ret."), "Q5-Q1"] = annret
    table.loc[(fac, "Vol."), "Q5-Q1"] = annvol
    table.loc[(fac, "S.R."), "Q5-Q1"] = sr

    annret, annvol, sr = pick_stats(ls_scaled)
    table.loc[(fac, "Ret."), "ConstVol"] = annret
    table.loc[(fac, "Vol."), "ConstVol"] = annvol
    table.loc[(fac, "S.R."), "ConstVol"] = sr
    
# ==============================================================
# Plot cumulato corretto dei portafogli ConstVol Q5-Q1
# ==============================================================

import matplotlib.ticker as mticker
from pathlib import Path
import matplotlib.pyplot as plt

plot_order = ["CARRY", "DEFENSIVE", "MOMENTUM", "VALUE", "COMBINED"]

# 1) Trova indice comune tra tutte le serie scaled
common_idx_plot = None
for fac in plot_order:
    idx = constvol_series[fac].dropna().index
    common_idx_plot = idx if common_idx_plot is None else common_idx_plot.intersection(idx)

print("\nMesi comuni per plot ConstVol:", len(common_idx_plot))
print(common_idx_plot.min(), "→", common_idx_plot.max())

# 2) Costruisci curve cumulative
cum_plot = {}
for fac in plot_order:
    r = constvol_series[fac].loc[common_idx_plot].astype(float)
    wealth = (1 + r).cumprod()
    wealth = wealth / wealth.iloc[0]
    cum_plot[fac] = wealth - 1

# 3) Asse x
idx0 = list(cum_plot.values())[0].index
if hasattr(idx0, "to_timestamp"):
    x_axis = idx0.to_timestamp()
else:
    x_axis = idx0

# 4) Plot
plt.figure(figsize=(10, 6))

plt.plot(x_axis, cum_plot["CARRY"].values,
         label="Carry", color="tab:green", linewidth=1.8)
plt.plot(x_axis, cum_plot["DEFENSIVE"].values,
         label="Defensive", color="tab:purple", linewidth=1.8)
plt.plot(x_axis, cum_plot["MOMENTUM"].values,
         label="Momentum", color="tab:red", linewidth=1.8)
plt.plot(x_axis, cum_plot["VALUE"].values,
         label="Value", color="tab:blue", linewidth=1.8)
plt.plot(x_axis, cum_plot["COMBINED"].values,
         label="Combined", color="black", linewidth=2.4)

plt.title("Cumulative Performance of Volatility-Scaled Long–Short Factor Portfolios (5% Target)")
plt.ylabel("Cumulative Return")
plt.xlabel("")
plt.grid(True, alpha=0.3)

ax = plt.gca()
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))

plt.legend(loc="upper left")
plt.tight_layout()

desktop = Path.home() / "Desktop"
fig_path_png = desktop / "Figure_ConstVol_Q5Q1_AllFactors.png"
fig_path_pdf = desktop / "Figure_ConstVol_Q5Q1_AllFactors.pdf"

plt.savefig(fig_path_png, dpi=300, bbox_inches="tight")
plt.savefig(fig_path_pdf, bbox_inches="tight")

print(f"✅ Figure saved on Desktop: {fig_path_png}")
print(f"✅ Vector version saved on Desktop: {fig_path_pdf}")

plt.show()






