#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  7 13:01:16 2025

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


 ## ==============================================================
 ## 2. Creazione Del Dataset
 ## ==============================================================

# Path Per I Files
folder_path = Path("/Users/leopera/Desktop/Dataset_tesi")

# Individua i file (ordinati per nome)
file_list = sorted(folder_path.glob("iboxx_eur_eod_underlyings_*.csv"))
print(f"File trovati: {len(file_list)}")
print("Esempi:", [p.name for p in file_list[:3]])

if not file_list:
    raise FileNotFoundError("Nessun file trovato: controlla percorso e pattern.")

# Colonne Per Il Dataset Finale
columns_to_keep = [
    'Date', 'ISIN', 'Identifier', 'Notional Amount',
    'Markit iBoxx Rating', 'Annual Yield', 'Duration', 'Daily Return',
    'Month-to-Date Return', 'OAS', 'Z-Spread',
    'Seniority Level 1', 'Time To Maturity', 'Issuer', 'Level 1',
    'Bid_Ask_Spread', 'Year-to-Date Return',
    'Month-to-date Sovereign Curve Swap Return', 'Level 3',
    'Issuer Country', 'Market Value', 
]

# Creazione Del Dataset Completo
def extract_date_from_name(name: str) -> pd.Timestamp:
    m = re.search(r'(\d{8})', name)
    if not m:
        return pd.NaT
    return pd.to_datetime(m.group(1), format="%Y%m%d", errors="coerce")

all_frames = []
for fp in file_list:
    df = pd.read_csv(fp, encoding="ISO-8859-1", low_memory=False)
    df["Source_File"] = fp.name
    df["Date"] = extract_date_from_name(fp.name)

    for col in columns_to_keep:
        if col not in df.columns:
            df[col] = np.nan

    df = df[columns_to_keep].copy()
    all_frames.append(df)

df_all = pd.concat(all_frames, ignore_index=True)

df_clean = df_all.copy()

# Piccolo report 
print("Intervallo date:",
      df_clean["Date"].min(), "→", df_clean["Date"].max())
print("Righe totali:", len(df_clean), "| ISIN unici:", df_clean["ISIN"].nunique())


 ## ==============================================================
 ## 3. Benckmark Di Riferimento Utilizzando I Corporate Bond
 ## ==============================================================
 
# Prendere dal Dataset Soltanto Bond Corporate
df_corp = (
    df_clean.loc[df_clean['Level 1'].eq('Corporates'),
                 ['Date', 'ISIN', 'Month-to-Date Return', 'Market Value',
                  'Month-to-date Sovereign Curve Swap Return']]
            .rename(columns={
                'Month-to-Date Return': 'TR',
                'Market Value': 'Weight',
                'Month-to-date Sovereign Curve Swap Return': 'Swap'
            })
            .copy()
)

for c in ['TR', 'Weight', 'Swap']:
    df_corp[c] = pd.to_numeric(df_corp[c], errors='coerce')

df_corp = df_corp.dropna(subset=['Date', 'ISIN'])

print(f"Righe corporate: {len(df_corp):,}")
print(df_corp.head(3))

# Media Value-Weighted Robusta
def vw_mean(group, value_col, weight_col='Weight'):
    v = group[value_col].astype(float)
    w = group[weight_col].astype(float)
    mask = v.notna() & w.notna() & (w > 0)
    if not mask.any():
        return np.nan
    return np.average(v[mask], weights=w[mask])

# Benchmark TR (EW e VW)
benchmark_corp_eq_tr = (
    df_corp.groupby('Date')['TR']
           .mean()
           .rename('Benchmark_Corp_TR_EW')
)

benchmark_corp_vw_tr = (
    df_corp.groupby('Date')
           .apply(vw_mean, value_col='TR')
           .rename('Benchmark_Corp_TR_VW')
)

# Benchmark Excess = TR - Swap (EW e VW)
df_corp['Excess'] = df_corp['TR'] - df_corp['Swap']

benchmark_corp_eq_ex = (
    df_corp.groupby('Date')['Excess']
           .mean()
           .rename('Benchmark_Corp_Excess_EW')
)

benchmark_corp_vw_ex = (
    df_corp.groupby('Date')
           .apply(vw_mean, value_col='Excess')
           .rename('Benchmark_Corp_Excess_VW')
)


benchmark_corp_eq = benchmark_corp_eq_tr.copy()

print("Mesi nel benchmark (EW TR):", len(benchmark_corp_eq_tr))
print(benchmark_corp_eq_tr.head())

# Excess monthly return: EW vs VW
plt.figure(figsize=(9,3))
benchmark_corp_vw_ex.sort_index().plot(label="VW Excess")
benchmark_corp_eq_ex.sort_index().plot(label="EW Excess")
plt.title("Corporate – Excess monthly return: VW vs EW")
plt.ylabel("Monthly Excess Return")
plt.xlabel("")
plt.axhline(0, lw=1)
plt.legend()
plt.tight_layout()
plt.show()

# Total Return monthly: EW vs VW
plt.figure(figsize=(9,3))
benchmark_corp_vw_tr.sort_index().plot(label="VW TR")
benchmark_corp_eq_tr.sort_index().plot(label="EW TR")
plt.title("Corporate – Total Return monthly: VW vs EW")
plt.ylabel("Monthly TR")
plt.xlabel("")
plt.axhline(0, lw=1)
plt.legend()
plt.tight_layout()
plt.show()

# Cumulata TR: EW vs VW (indice base = 1)
cum_tr_vw = (1 + benchmark_corp_vw_tr.dropna()).cumprod()
cum_tr_eq = (1 + benchmark_corp_eq_tr.dropna()).cumprod()
base = min(cum_tr_vw.index.min(), cum_tr_eq.index.min())
cum_tr_vw = (cum_tr_vw / cum_tr_vw.loc[base])
cum_tr_eq = (cum_tr_eq / cum_tr_eq.loc[base])

plt.figure(figsize=(9,4))
cum_tr_vw.plot(label="VW TR (Index = 1)")
cum_tr_eq.plot(label="EW TR (Index = 1)")
plt.title("Corporate – Cumulative Growth (TR)")
plt.ylabel("Cumulative Growth")
plt.xlabel("")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# Cumulata Excess: EW vs VW (indice base = 1)
cum_ex_vw = (1 + benchmark_corp_vw_ex.dropna()).cumprod()
cum_ex_eq = (1 + benchmark_corp_eq_ex.dropna()).cumprod()
base_ex = min(cum_ex_vw.index.min(), cum_ex_eq.index.min())
cum_ex_vw = (cum_ex_vw / cum_ex_vw.loc[base_ex])
cum_ex_eq = (cum_ex_eq / cum_ex_eq.loc[base_ex])

plt.figure(figsize=(9,4))
cum_ex_vw.plot(label="VW Excess (Index = 1)")
cum_ex_eq.plot(label="EW Excess (Index = 1)")
plt.title("Corporate – Cumulative Growth (Excess)")
plt.ylabel("Cumulative Growth")
plt.xlabel("")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


 ## ==============================================================
 ## 4. Creazione Del Risk Free Rate Utilizzando Come Proxy La Media
 ##    Dei "Month-to-date Sovereign Curve Swap Return"
 ## ==============================================================

# Calcolo utilizzando Equal Weight
rf_eq = (
    df_corp.groupby('Date')['Swap']
           .mean()
           .rename('RiskFree_EW')
)

# Plot dei ritorni mensili del risk-free
plt.figure(figsize=(9,4))
rf_eq.sort_index().plot(color='tab:blue', lw=1.8)
plt.title("Risk-Free Monthly Returns (Equal-Weighted) – Sovereign Curve Swap")
plt.ylabel("Monthly Risk-Free Return")
plt.xlabel("")
plt.grid(True, alpha=0.3)
plt.axhline(0, color='gray', lw=1, alpha=0.6)
plt.tight_layout()
plt.show()

# Calcolo del rendimento cumulativo composto (indice base = 1)
cum_rf_eq = (1 + rf_eq.dropna()).cumprod()
cum_rf_eq = cum_rf_eq / cum_rf_eq.iloc[0]

plt.figure(figsize=(9,4))
cum_rf_eq.plot(color='tab:blue', label='Risk-Free (EW) – Index = 1')
plt.title("Cumulative Growth – Risk-Free (Sovereign Curve Swap, EW)")
plt.ylabel("Cumulative Growth (base = 1)")
plt.xlabel("")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# Plot con il Benchmark Di Riferimento
plt.figure(figsize=(9,4))
cum_rf_eq.plot(label="Risk-Free (EW)", color='tab:blue')
cum_tr_eq.plot(label="Corporate Benchmark (EW)", color='tab:orange')
plt.title("Cumulative Growth – Corporate vs Risk-Free")
plt.ylabel("Cumulative Growth (base = 1)")
plt.xlabel("")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


 ## =========================
 ## 5. Filtraggio Del Dataset Per Creare Il dataset Finale
 ## =========================

dfw = df_clean.copy()
dfw['Date'] = pd.to_datetime(dfw['Date'], errors='coerce')

# Colonne usate nei filtri
num_cols = [
    'Notional Amount', 'Time To Maturity', 'Duration',
    'Annual Yield', 'Bid_Ask_Spread', 'OAS', 'Z-Spread',
    'Month-to-Date Return', 'Year-to-Date Return',
    'Month-to-date Sovereign Curve Swap Return'
]
for c in num_cols:
    if c in dfw.columns:
        dfw[c] = pd.to_numeric(dfw[c], errors='coerce')

# Filtrare per Corporate Bonds
dfw = dfw[dfw['Level 1'].eq('Corporates')]

# Filtrare Per Obbligazioni Senior
if 'Seniority Level 1' in dfw.columns:
    dfw['Seniority Level 1'] = dfw['Seniority Level 1'].astype('string').str.upper().str.strip()
    # alcune basi iBoxx usano "SNR" per senior
    dfw['Seniority Level 1'] = dfw['Seniority Level 1'].replace({'SNR':'SEN'})
    dfw = dfw[dfw['Seniority Level 1'].eq('SEN')]
else:
    raise KeyError("Missing column: 'Seniority Level 1'.")

# Rating dominante per issuer (ponderato per notional)
issuer_rating_sum = (dfw.dropna(subset=['Markit iBoxx Rating', 'Notional Amount'])
                       .groupby(['Date','Issuer','Markit iBoxx Rating'], as_index=False)['Notional Amount']
                       .sum())

dominant_ratings = (issuer_rating_sum.sort_values(['Date','Issuer','Notional Amount'],
                                                  ascending=[True, True, False])
                                   .drop_duplicates(subset=['Date','Issuer'], keep='first')
                                   .rename(columns={'Markit iBoxx Rating':'Dominant Rating'}))

dfw = dfw.merge(dominant_ratings[['Date','Issuer','Dominant Rating']],
                on=['Date','Issuer'], how='inner')
dfw = dfw[dfw['Markit iBoxx Rating'].eq(dfw['Dominant Rating'])].drop(columns=['Dominant Rating'])

# Scelta del bond rappresentativo per issuer prendendo TTM tra 5 e 10 Anni ()
def select_bond_by_maturity(group: pd.DataFrame) -> pd.DataFrame:
    in_range = group[(group['Time To Maturity'] >= 5) & (group['Time To Maturity'] <= 15)]
    cand = in_range if not in_range.empty else group
    cand = cand.sort_values(['Notional Amount', 'Bid_Ask_Spread'],
                            ascending=[False, True])
    return cand.head(1)

df_final = (dfw.groupby(['Date','Issuer'], group_keys=False)
               .apply(select_bond_by_maturity)
               .reset_index(drop=True))

# Rating bucket → numerico (AAA=1, AA=2, A=3, BBB=4)
def bucket_rating(x: str):
    if pd.isna(x):
        return np.nan
    s = str(x).upper().strip()
    if s.startswith('AAA'): return 'AAA'
    if s.startswith('AA'):  return 'AA'   
    if s.startswith('A'):   return 'A'    
    if s.startswith('BBB'): return 'BBB' 
    return np.nan 

df_final['Rating_Bucket'] = df_final['Markit iBoxx Rating'].map(bucket_rating)
rating_map = {'AAA':1, 'AA':2, 'A':3, 'BBB':4}
df_final['Rating_Num'] = df_final['Rating_Bucket'].map(rating_map)

# Pulizia finale
req_cols = ['Bid_Ask_Spread','OAS','Z-Spread','Annual Yield','Duration',
            'Notional Amount','Time To Maturity','Month-to-Date Return']
df_final = df_final.dropna(subset=[c for c in req_cols if c in df_final.columns])

# Scegliere Lo Stesso Paese di Provenienza Per I Bond Americani
if 'Issuer Country' in df_final.columns:
    df_final['Issuer Country'] = (df_final['Issuer Country']
                                  .astype('string')
                                  .str.upper()
                                  .str.strip()
                                  .replace({'UNITED STATES':'USA'}))
else:
    raise KeyError("Missing column: 'Issuer Country'.")

# --- 8) Piccolo report di consistenza ---
print(f"Rows in final dataset: {len(df_final):,}")
print(f"Unique issuers: {df_final['Issuer'].nunique():,} | Unique ISIN: {df_final['ISIN'].nunique():,}")
print("Dates:", df_final['Date'].min(), "→", df_final['Date'].max())
print("Rating buckets distribution:")
print(df_final['Rating_Bucket'].value_counts(dropna=False).sort_index())

# Statistiche Chiave Mensili
metrics = [
    'OAS','Z-Spread','Duration','Annual Yield','Notional Amount',
    'Time To Maturity','Bid_Ask_Spread','Month-to-Date Return',
    'Year-to-Date Return','Rating_Num'
]
summary_stats = {}
for m in metrics:
    if m in df_final.columns:
        s = df_final[m].astype(float)
        summary_stats[m] = {
            'Mean': s.mean(), 'Std': s.std(),
            '5%': s.quantile(0.05), '25%': s.quantile(0.25),
            '50%': s.quantile(0.50), '75%': s.quantile(0.75),
            '95%': s.quantile(0.95)
        }
summary_df = pd.DataFrame(summary_stats).T
print(summary_df.applymap(lambda x: f"{x:,.2f}"))



 ## =========================
 ## 6. DATASET FACTS & TURNOVER
 ## =========================

# Bonds per month
df_final['YearMonth'] = df_final['Date'].dt.to_period('M')
bonds_per_month = df_final.groupby('YearMonth').size()
print("\nRows per month:")
print(bonds_per_month)

plt.figure(figsize=(9,3))
bonds_per_month.astype(int).plot()
plt.title("Rows per Month (after issuer-level selection)")
plt.ylabel("Rows")
plt.xlabel("")
plt.tight_layout()
plt.show()

# Unique ISIN overall and per month Per Check
unique_bonds_total = df_final['ISIN'].nunique()
print(f"\nUnique ISIN overall: {unique_bonds_total:,}")

unique_bonds_month = df_final.groupby('YearMonth')['ISIN'].nunique()
print("\nUnique ISIN per Month:")
print(unique_bonds_month)

plt.figure(figsize=(9,3))
unique_bonds_month.astype(int).plot()
plt.title("Unique ISIN per Month")
plt.ylabel("Count")
plt.xlabel("")
plt.tight_layout()
plt.show()

# Sector & Country composition — counts and value-weighted shares
unique_isin = df_final.drop_duplicates(subset='ISIN')

sector_pct_cnt = unique_isin['Level 3'].value_counts(normalize=True) * 100
print("\nSector composition (by count, unique ISIN):")
print(sector_pct_cnt.round(2))

if 'Notional Amount' in unique_isin.columns:
    sector_pct_vw = (unique_isin.groupby('Level 3')['Notional Amount']
                                .sum()
                                .pipe(lambda s: 100 * s / s.sum()))
    print("\nSector composition (value-weighted, unique ISIN):")
    print(sector_pct_vw.round(2))

if 'Issuer Country' in unique_isin.columns:
    country_pct_cnt = unique_isin['Issuer Country'].value_counts(normalize=True) * 100
    print("\nIssuer Country (by count, unique ISIN):")
    print(country_pct_cnt.round(2))

    if 'Notional Amount' in unique_isin.columns:
        country_pct_vw = (unique_isin.groupby('Issuer Country')['Notional Amount']
                                     .sum()
                                     .pipe(lambda s: 100 * s / s.sum()))
        print("\nIssuer Country (value-weighted, unique ISIN):")
        print(country_pct_vw.round(2))

# 4) ISIN turnover Dove Daremo più peso a quella simmetrica
isin_per_month = df_final.groupby('YearMonth')['ISIN'].apply(set)
months = sorted(isin_per_month.index)

turnover_records = []
for i in range(1, len(months)):
    prev_m, curr_m = months[i-1], months[i]
    A, B = isin_per_month[prev_m], isin_per_month[curr_m]
    entered = B - A
    exited  = A - B
    prev_based = (len(entered) + len(exited)) / max(1, len(A)) 
    inter = len(A & B)
    sym = 1 - (2 * inter) / max(1, (len(A) + len(B)))
    turnover_records.append({
        'YearMonth': curr_m,
        'Entered': len(entered),
        'Exited': len(exited),
        'Prev_Count': len(A),
        'Curr_Count': len(B),
        'Turnover_PrevBased': prev_based,
        'Turnover_Symmetric': sym
    })

turnover_df = pd.DataFrame(turnover_records).set_index('YearMonth')
print("\nMonthly ISIN Turnover (Prev-based & Symmetric):")
print(turnover_df.tail())

# Plots 
plt.figure(figsize=(10,3))
turnover_df['Turnover_Symmetric'].plot()
plt.title("ISIN Turnover (Symmetric, Month over Month)")
plt.ylabel("Turnover")
plt.xlabel("")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,3))
turnover_df['Turnover_PrevBased'].plot()
plt.title("ISIN Turnover (Prev-based, Month over Month)")
plt.ylabel("Turnover")
plt.xlabel("")
plt.tight_layout()
plt.show()

# Key stats 
ts = turnover_df['Turnover_Symmetric']
print("\nTurnover (symmetric) — key stats")
print("Mean   :", f"{ts.mean():.2%}")
print("Median :", f"{ts.median():.2%}")
print("Min/Max:", f"{ts.min():.2%}", "→", ts.idxmin(), "|", f"{ts.max():.2%}", "→", ts.idxmax())

# Implied average holding period (months) ~ 1 / mean_symmetric
if ts.mean() > 0:
    implied_hp = 1.0 / ts.mean()
    print(f"Implied average holding period: ~{implied_hp:.1f} months")
else:
    print("Implied average holding period: n/a (mean turnover is zero)")

# Issuer turnover
issuers_per_month = df_final.groupby('YearMonth')['Issuer'].apply(set)
issuer_records = []
for i in range(1, len(months)):
    prev_m, curr_m = months[i-1], months[i]
    A, B = issuers_per_month[prev_m], issuers_per_month[curr_m]
    entered, exited = B - A, A - B
    prev_based = (len(entered) + len(exited)) / max(1, len(A))
    inter = len(A & B)
    sym = 1 - (2 * inter) / max(1, (len(A) + len(B)))
    issuer_records.append({'YearMonth': curr_m,
                           'Issuer_Turnover_PrevBased': prev_based,
                           'Issuer_Turnover_Symmetric': sym})

issuer_turn_df = pd.DataFrame(issuer_records).set_index('YearMonth')
print("\nIssuer Turnover (symmetric) — mean:",
      f"{issuer_turn_df['Issuer_Turnover_Symmetric'].mean():.2%}")
print(issuer_turn_df.tail())

plt.figure(figsize=(10,3))
issuer_turn_df['Issuer_Turnover_Symmetric'].plot()
plt.title("Issuer Turnover (Symmetric, Month over Month)")
plt.ylabel("Turnover")
plt.xlabel("")
plt.tight_layout()
plt.show()

# Cacolco degli Excess Returns

df_final['Excess Return'] = (
    df_final['Month-to-Date Return'].astype(float)
    - df_final['Month-to-date Sovereign Curve Swap Return'].astype(float)
)

 ## =========================
 ## 7. Download Del Dataset Finale Da Utilizzare Con Un Nuovo Codice
 ## =========================
# Benchmark Corporate TR VW
df_final = df_final.merge(
    benchmark_corp_vw_tr.rename('Benchmark_TR_VW'),
    on='Date',
    how='left'
)

# Benchmark Corporate Excess VW
df_final = df_final.merge(
    benchmark_corp_vw_ex.rename('Benchmark_Excess_VW'),
    on='Date',
    how='left'
)

# Risk-Free VW (da Swap)
df_final = df_final.merge(
    rf_eq,
    on='Date',
    how='left'
)

columns_to_keep = [
    'Date', 'ISIN', 'Identifier', 'Notional Amount',
    'Markit iBoxx Rating', 'Annual Yield', 'Duration', 'Daily Return',
    'Month-to-Date Return', 'OAS', 'Z-Spread',
    'Seniority Level 1', 'Time To Maturity', 'Issuer', 'Level 1',
    'Bid_Ask_Spread', 'Year-to-Date Return',
    'Month-to-date Sovereign Curve Swap Return', 'Level 3',
    'Issuer Country', 'Market Value',
    'Benchmark_TR_VW',
    'Benchmark_Excess_VW',
    'RiskFree_EW'
]

# Seleziona solo le colonne necessarie (quelle definite in columns_to_keep)
# ma mantiene solo quelle effettivamente presenti nel dataset finale
cols_available = [c for c in columns_to_keep if c in df_final.columns]
df_export = df_final[cols_available].copy()

# Percorso di salvataggio (puoi modificare se vuoi)
output_path = Path("/Users/leopera/Desktop/dataset_finale_fattori.csv")

# Esporta in CSV
df_export.to_csv(output_path, index=False, encoding="utf-8-sig")

print(f"\n✅ Dataset finale salvato con successo in:\n{output_path}")
print(f"Numero di righe: {len(df_export):,} | Colonne: {len(df_export.columns)}")
print("Colonne incluse:", list(df_export.columns))







print("Max rows per (YearMonth, Issuer):", df_final.groupby(['YearMonth','Issuer']).size().max())




## Creazione tabella di ISIN unici mensili da mettere su word

table_4_1 = (
    df_final
    .groupby('YearMonth')['ISIN']
    .nunique()
    .reset_index()
    .rename(columns={
        'YearMonth': 'Month',
        'ISIN': 'Number of Bonds'
    })
)

# Arrotondamenti non necessari qui, ma ordiniamo per sicurezza
table_4_1 = table_4_1.sort_values('Month')
table_4_1.head()

# Download
table_4_1.to_excel(
    "/Users/leopera/Desktop/Table_4_1_Number_of_Bonds_per_Month.xlsx",
    index=False
)


print(table_4_1.to_string(index=False))






# =========================
# Table 4.2 – Key Summary Statistics (AQR-style)
# =========================

# Variabili da includere nella tabella
summary_cols = [
    'OAS',
    'Duration',
    'Month-to-Date Return',   # Total Return (monthly)
    'Excess Return',          # Excess Return (monthly)
    'Notional Amount',
    'Time To Maturity'
]

# Teniamo solo le colonne effettivamente presenti
summary_cols = [c for c in summary_cols if c in df_final.columns]

# Percentili (compatti e Word-friendly)
percentiles = [0.05, 0.25, 0.50, 0.75, 0.95]

# =========================
# 1) Statistiche cross-sectional mensili
# =========================
monthly_stats = []

for ym, g in df_final.groupby('YearMonth'):
    stats = (
        g[summary_cols]
        .astype(float)
        .describe(percentiles=percentiles)
        .T
        .reset_index()
        .rename(columns={'index': 'Variable'})
    )
    stats['YearMonth'] = ym
    monthly_stats.append(stats)

monthly_stats = pd.concat(monthly_stats, ignore_index=True)

# =========================
# 2) Media nel tempo delle statistiche mensili
# =========================
table_4_2 = (
    monthly_stats
    .groupby('Variable')[['mean', 'std', '5%', '25%', '50%', '75%', '95%']]
    .mean()
    .reset_index()
)

# =========================
# 3) Rinomina colonne (stile paper)
# =========================
table_4_2 = table_4_2.rename(columns={
    'mean': 'Mean',
    'std': 'Std',
    '5%': 'P5',
    '25%': 'P25',
    '50%': 'P50',
    '75%': 'P75',
    '95%': 'P95'
})

# Ordine delle variabili come definite sopra
table_4_2['Variable'] = pd.Categorical(
    table_4_2['Variable'],
    categories=summary_cols,
    ordered=True
)
table_4_2 = table_4_2.sort_values('Variable').reset_index(drop=True)

# =========================
# 4) Etichette più leggibili per Word
# =========================
table_4_2['Variable'] = table_4_2['Variable'].replace({
    'Month-to-Date Return': 'Total Return (Monthly)',
    'Excess Return': 'Excess Return (Monthly)',
    'Notional Amount': 'Amount Outstanding',
    'Time To Maturity': 'Time to Maturity'
})

# Arrotondamento (puoi cambiare se vuoi)
table_4_2 = table_4_2.round(3)

# =========================
# 5) Export Excel
# =========================
output_path = "/Users/leopera/Desktop/Table_4_2_Key_Bond_Statistics.xlsx"
table_4_2.to_excel(output_path, index=False)

print("Table 4.2 exported to:", output_path)
table_4_2



# ==============================================================
# Benchmark di riferimento (AQR-consistent: one-bond-per-issuer universe)
# ==============================================================

df_bench = df_final[['Date', 'Month-to-Date Return', 'Market Value',
                     'Month-to-date Sovereign Curve Swap Return']].copy()

df_bench = df_bench.rename(columns={
    'Month-to-Date Return': 'TR',
    'Market Value': 'Weight',
    'Month-to-date Sovereign Curve Swap Return': 'Swap'
})

for c in ['TR', 'Weight', 'Swap']:
    df_bench[c] = pd.to_numeric(df_bench[c], errors='coerce')

df_bench = df_bench.dropna(subset=['Date', 'TR', 'Weight', 'Swap'])
df_bench = df_bench[df_bench['Weight'] > 0]

def vw_mean(group, value_col, weight_col='Weight'):
    v = group[value_col].astype(float)
    w = group[weight_col].astype(float)
    mask = v.notna() & w.notna() & (w > 0)
    if not mask.any():
        return np.nan
    return np.average(v[mask], weights=w[mask])

# TR
benchmark_tr_vw = df_bench.groupby('Date').apply(vw_mean, value_col='TR').rename('Benchmark_TR_VW')
benchmark_tr_ew = df_bench.groupby('Date')['TR'].mean().rename('Benchmark_TR_EW')

# Excess
df_bench['Excess'] = df_bench['TR'] - df_bench['Swap']
benchmark_ex_vw = df_bench.groupby('Date').apply(vw_mean, value_col='Excess').rename('Benchmark_Excess_VW')
benchmark_ex_ew = df_bench.groupby('Date')['Excess'].mean().rename('Benchmark_Excess_EW')

plt.figure(figsize=(9,4))
benchmark_ex_vw.sort_index().plot(label="VW Excess")
benchmark_ex_ew.sort_index().plot(label="EW Excess", alpha=0.8)
plt.axhline(0, lw=1)
plt.title("Corporate Bond Benchmark – Monthly Excess Return")
plt.ylabel("Monthly Excess Return")
plt.xlabel("")
plt.legend()
plt.tight_layout()
plt.show()


df_bench = df_final[['Date', 'Excess Return', 'Market Value']].dropna()
df_bench = df_bench[df_bench['Market Value'] > 0]

benchmark_ex_vw = (
    df_bench.assign(wEX=df_bench['Excess Return'] * df_bench['Market Value'])
            .groupby('Date')[['wEX','Market Value']]
            .sum()
            .pipe(lambda x: x['wEX'] / x['Market Value'])
            .sort_index()
)

# Cumulative excess return (Index = 1)
cum_excess = (1 + benchmark_ex_vw).cumprod()
cum_excess = cum_excess / cum_excess.iloc[0]

# Plot
plt.figure(figsize=(9,4))
plt.plot(cum_excess.index, cum_excess.values)
plt.title("Corporate Bond Benchmark – Cumulative Excess Return (VW)")
plt.ylabel("Cumulative Excess Return (Index = 1)")
plt.xlabel("")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# =========================
# PRE-FILTER (df_all)
# =========================

df_all['Date'] = pd.to_datetime(df_all['Date'], errors='coerce')
df_all['YearMonth'] = df_all['Date'].dt.to_period('M')

df_all['Excess Return'] = (
    pd.to_numeric(df_all['Month-to-Date Return'], errors='coerce')
    - pd.to_numeric(df_all['Month-to-date Sovereign Curve Swap Return'], errors='coerce')
)

summary_cols = ['OAS','Duration','Month-to-Date Return','Excess Return','Notional Amount','Time To Maturity']
summary_cols = [c for c in summary_cols if c in df_all.columns]

for c in summary_cols:
    df_all[c] = pd.to_numeric(df_all[c], errors='coerce')

percentiles = [0.05, 0.25, 0.50, 0.75, 0.95]

monthly_stats = []
for ym, g in df_all.groupby('YearMonth'):
    stats = (
        g[summary_cols]
        .describe(percentiles=percentiles)
        .T
        .reset_index()
        .rename(columns={'index': 'Variable'})
    )
    stats['YearMonth'] = ym
    monthly_stats.append(stats)

monthly_stats = pd.concat(monthly_stats, ignore_index=True)

table_pre = (
    monthly_stats
    .groupby('Variable')[['mean', 'std', '5%', '25%', '50%', '75%', '95%']]
    .mean()
    .reset_index()
    .rename(columns={'mean':'Mean','std':'Std','5%':'P5','25%':'P25','50%':'P50','75%':'P75','95%':'P95'})
)

table_pre['Variable'] = pd.Categorical(table_pre['Variable'], categories=summary_cols, ordered=True)
table_pre = table_pre.sort_values('Variable').reset_index(drop=True)

table_pre['Variable'] = table_pre['Variable'].replace({
    'Month-to-Date Return': 'Total Return (Monthly)',
    'Excess Return': 'Excess Return (Monthly)',
    'Notional Amount': 'Amount Outstanding',
    'Time To Maturity': 'Time to Maturity'
})

table_pre = table_pre.round(3)

output_path = "/Users/leopera/Desktop/Table_PRE_Key_Bond_Statistics.xlsx"
table_pre.to_excel(output_path, index=False)
print("PRE table exported to:", output_path)
table_pre


print("\nUnique Level 1 categories:")
print(df_all['Level 1'].dropna().unique())

level1_counts_all = df_all['Level 1'].value_counts()

print("\nLevel 1 distribution (total observations):")
print(level1_counts_all)

df_all['Date'] = pd.to_datetime(df_all['Date'], errors='coerce')
df_all['YearMonth'] = df_all['Date'].dt.to_period('M')

level1_monthly_all = (
    df_all
    .groupby(['YearMonth','Level 1'])['ISIN']
    .nunique()
    .reset_index()
)

level1_monthly_avg_all = (
    level1_monthly_all
    .groupby('Level 1')['ISIN']
    .mean()
    .round(1)
)

print("\nAverage number of bonds per month by Level 1:")
print(level1_monthly_avg_all)



issuer_by_level1 = (
    df_all
    .groupby('Level 1')['Issuer']
    .nunique()
)

print("\nUnique issuers by Level 1:")
print(issuer_by_level1)





