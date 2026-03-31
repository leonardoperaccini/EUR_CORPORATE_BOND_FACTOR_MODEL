#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  3 16:41:52 2025

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
 ## 2. Importare Il Dataset Finale
 ## ==============================================================

# Percorso del file CSV
file_path = Path("/Users/leopera/Desktop/dataset_finale_fattori.csv")

# Controllo esistenza
if not file_path.exists():
    raise FileNotFoundError(f"❌ File non trovato in {file_path}")

# Lettura del dataset
df_fattori = pd.read_csv(file_path)

# Report base
print("✅ Dataset importato con successo!")
print(f"Righe totali: {len(df_fattori):,}")
print(f"Colonne totali: {len(df_fattori.columns):,}")
print("\nColonne principali:")
print(df_fattori.columns.tolist())
print("\nPrime 5 righe:")
print(df_fattori.head())

df_fattori['Date'] = pd.to_datetime(df_fattori['Date'], errors='coerce')
print("\nIntervallo temporale:",
      df_fattori['Date'].min(), "→", df_fattori['Date'].max())
df_fattori['YearMonth'] = df_fattori['Date'].dt.to_period('M')

 ## ==============================================================
 ## 3. Fattore Carry
 ## ==============================================================
  # Utilizzo dell'OAS (Non si usa il BTS per via del fattore che include lo Spread) 
   # Value Weight Con Market Value e Ritorni calcolati come Excess Returns
  
# Calcolo della statistiche mensili
oas_summary = (
    df_fattori.groupby('YearMonth')['OAS']
              .agg(['count', 'min', 'mean', 'max'])
              .round(3)
)

print(oas_summary.head(10))

# Preparazione base
for c in ['OAS', 'Month-to-Date Return', 'Month-to-date Sovereign Curve Swap Return', 'Market Value']:
    df_fattori[c] = pd.to_numeric(df_fattori[c], errors='coerce')

# Excess return
df_fattori['ExcessRet'] = df_fattori['Month-to-Date Return'] - df_fattori['Month-to-date Sovereign Curve Swap Return']
df_fattori = df_fattori.sort_values(['ISIN','Date'])
df_fattori['Next_ExcessRet'] = df_fattori.groupby('ISIN')['ExcessRet'].shift(-1)

# Segnale Carry 
df_fattori['Carry_signal'] = df_fattori['OAS']

# Quintili mensili del segnale
def quintili_mensili(s):
    r = s.rank(method='first')
    return pd.qcut(r, 5, labels=[0,1,2,3,4])

df_fattori['Carry_q'] = df_fattori.groupby('YearMonth', group_keys=False)['Carry_signal'].apply(quintili_mensili)

# Rendimento per quintile (Value-Weighted) e fattore Q4 − Q0
panel = df_fattori.dropna(subset=['Carry_q','Next_ExcessRet','Market Value']).copy()
panel = panel[panel['Market Value'] > 0]
panel['Carry_q'] = panel['Carry_q'].astype(int)

# peso * rendimento, poi somma/somma pesi
panel['wret'] = panel['Market Value'] * panel['Next_ExcessRet']

q_sum = panel.groupby(['YearMonth','Carry_q'])[['wret','Market Value']].sum()
q_ret_vw = (q_sum['wret'] / q_sum['Market Value']).unstack('Carry_q')  # colonne 0..4

carry_ls_vw = (q_ret_vw[4] - q_ret_vw[0]).rename('Carry_LS_VW')





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



print(stats(carry_ls_vw))


cum = (1 + carry_ls_vw.dropna()).cumprod()
cum = cum / cum.iloc[0]

plt.figure(figsize=(9,4))
cum.plot()
plt.title("Carry (OAS) — Long Q4 minus Short Q0 (Value-Weighted)\nCumulative Growth (Index=1)")
plt.ylabel("Cumulative Growth")
plt.xlabel("")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


 ## ==============================================================
 ## 6. DEFENSIVE Factor
 ## ==============================================================
  # Utilizzo di Duration 


# 1) Segnale defensive: duration negativa (low duration = high signal)
df_fattori['Defensive_signal'] = -df_fattori['Duration']

# 2) Quintili mensili sul segnale Defensive (riuso quintili_mensili già definita)
df_fattori['Def_q'] = (
    df_fattori
    .groupby('YearMonth', group_keys=False)['Defensive_signal']
    .apply(quintili_mensili)
)

# 3) Pannello pulito per Defensive
panel_def = df_fattori.dropna(subset=['Def_q', 'Next_ExcessRet', 'Market Value']).copy()
panel_def = panel_def[panel_def['Market Value'] > 0]
panel_def['Def_q'] = panel_def['Def_q'].astype(int)

# 4) Ritorni value-weighted per quintile
panel_def['wret_def'] = panel_def['Market Value'] * panel_def['Next_ExcessRet']

q_def_sum = (
    panel_def
    .groupby(['YearMonth', 'Def_q'])[['wret_def', 'Market Value']]
    .sum()
)

q_def_vw = (q_def_sum['wret_def'] / q_def_sum['Market Value']).unstack('Def_q')  # colonne 0..4

# 5) Fattore Defensive long-short (long low duration, short high duration)
def_ls_vw = (q_def_vw[4] - q_def_vw[0]).rename('Defensive_LS_VW')

# 6) Statistiche e curva cumulata del fattore Defensive
print("\nDefensive (VW) — Q4−Q0 stats:")
print(stats(def_ls_vw))

cum_def = (1 + def_ls_vw.dropna()).cumprod()
cum_def = cum_def / cum_def.iloc[0]

plt.figure(figsize=(9,4))
cum_def.plot()
plt.title("Defensive (Duration) — Long Q4 (Low Dur) minus Short Q0 (High Dur)\nCumulative Growth (Index=1)")
plt.ylabel("Cumulative Growth")
plt.xlabel("")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


   ## ==============================================================
## 5. Fattore Momentum (Credit Excess Return, demean su DTS)
## ==============================================================

# 1) Ordiniamo per ISIN e data (serve per la rolling)
df_fattori = df_fattori.sort_values(['ISIN', 'Date'])
# (A) FORZA reset del segnale (evita colonna stale)
df_fattori = df_fattori.sort_values(['ISIN', 'Date'])
df_fattori.drop(columns=['Mom_signal'], errors='ignore', inplace=True)

# (B) Momentum LAGGED: usa solo info osservabile prima del mese t
exret_lag = df_fattori.groupby('ISIN')['ExcessRet'].shift(1)

df_fattori['Mom_signal'] = (
    exret_lag
    .groupby(df_fattori['ISIN'])
    .rolling(window=6, min_periods=3)
    .sum()
    .reset_index(level=0, drop=True)
) 

# 3) Creiamo il beta ex-ante: DTS = Duration × OAS
df_fattori['DTS'] = df_fattori['Duration'] * df_fattori['OAS']

# 4) Dataset pulito SOLO per il momentum (segnale e DTS non NaN)
df_mom = df_fattori.dropna(subset=['Mom_signal', 'DTS']).copy()

# 5) Quintili mensili di beta (DTS) — proxy di ex-ante beta
df_mom['Beta_q'] = (
    df_mom
    .groupby('YearMonth', group_keys=False)['DTS']
    .apply(quintili_mensili)
)

# 6) Demean del segnale di momentum all’interno dei quintili di DTS
df_mom['Mom_signal_adj'] = (
    df_mom
    .groupby(['YearMonth', 'Beta_q'], group_keys=False)['Mom_signal']
    .transform(lambda x: x - x.mean())
)

# 7) Quintili mensili sul segnale di Momentum normalizzato (Mom_signal_adj)
df_mom['Mom_q'] = (
    df_mom
    .groupby('YearMonth', group_keys=False)['Mom_signal_adj']
    .apply(quintili_mensili)
)

# 8) Pannello pulito per la costruzione dei portafogli Momentum
panel_mom = df_mom.dropna(subset=['Mom_q', 'Next_ExcessRet', 'Market Value']).copy()
first_mom_month = df_fattori.loc[df_fattori['Mom_signal'].notna(), 'YearMonth'].min()
panel_mom = panel_mom[panel_mom['YearMonth'] >= first_mom_month]
panel_mom = panel_mom[panel_mom['Market Value'] > 0]
panel_mom['Mom_q'] = panel_mom['Mom_q'].astype(int)

# 9) Ritorni value-weighted per quintile di Momentum
panel_mom['wret_mom'] = panel_mom['Market Value'] * panel_mom['Next_ExcessRet']

q_mom_sum = (
    panel_mom
    .groupby(['YearMonth', 'Mom_q'])[['wret_mom', 'Market Value']]
    .sum()
)

q_mom_vw = (q_mom_sum['wret_mom'] / q_mom_sum['Market Value']).unstack('Mom_q')  # colonne 0..4

# 10) Fattore Momentum long-short (Long winners Q4, Short losers Q0)
mom_ls_vw = (q_mom_vw[4] - q_mom_vw[0]).rename('Momentum_LS_VW')

mom_ls_vw = (q_mom_vw[4] - q_mom_vw[0]).rename('Momentum_LS_VW')



# 10) Fattore Momentum long-short
mom_ls_vw = (q_mom_vw[4] - q_mom_vw[0]).rename('Momentum_LS_VW')

# --- FIX: allinea i tipi (PeriodIndex mensile) e taglia i mesi non investibili ---
import pandas as pd

# primo mese in cui Mom_signal esiste davvero (dal lato Date, quindi senza ambiguità)
first_mom_month = df_fattori.loc[df_fattori['Mom_signal'].notna(), 'Date'].min().to_period('M')

# assicura che l'indice del fattore sia PeriodIndex mensile
if not isinstance(mom_ls_vw.index, pd.PeriodIndex):
    mom_ls_vw.index = pd.PeriodIndex(mom_ls_vw.index.astype(str), freq='M')

# filtra
mom_ls_vw = mom_ls_vw.loc[mom_ls_vw.index >= first_mom_month]


# 11) Statistiche del fattore Momentum
print("\nMomentum (VW, demean su DTS) — Q4−Q0 stats:")
print(stats(mom_ls_vw))

# 12) Curva cumulata del fattore Momentum
cum_mom = (1 + mom_ls_vw.dropna()).cumprod()
cum_mom = cum_mom / cum_mom.iloc[0]

plt.figure(figsize=(9,4))
cum_mom.plot()
plt.title("Momentum (Excess Return, DTS-normalized)\nLong Q4 (Winners) minus Short Q0 (Losers) — VW, Index=1")
plt.ylabel("Cumulative Growth")
plt.xlabel("")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

   
   
## ==============================================================
## 6. Fattore Value
## ==============================================================


df_fattori = df_fattori.sort_values(['ISIN','Date'])
df_fattori['DTS'] = df_fattori['Duration'] * df_fattori['OAS']


df_fattori['Vol_12m'] = (
    df_fattori
    .groupby('ISIN')['ExcessRet']
    .rolling(window=12, min_periods=3)
    .std()
    .reset_index(level=0, drop=True)
)



rating_map = {'AAA': 1, 'AA': 2, 'A': 3, 'BBB': 4}
df_fattori['Rating_num'] = df_fattori['Markit iBoxx Rating'].map(rating_map)



cols_req = ['OAS', 'Duration', 'Rating_num', 'Vol_12m', 'DTS', 'YearMonth', 'Next_ExcessRet', 'Market Value']
df_val = df_fattori.dropna(subset=cols_req).copy()
df_val = df_val[df_val['Market Value'] > 0]
df_val = df_val[df_val['OAS'] > 0]        # log(OAS) richiede positività
df_val = df_val[df_val['Duration'] > 0]   # log(Duration) richiede positività
df_val = df_val[df_val['Vol_12m'] > 0]    # log(Vol) richiede positività



def value_residuals_month(g):
    y = np.log(g['OAS'].values)
    X = np.column_stack([
        np.ones(len(g)),
        np.log(g['Duration'].values),
        g['Rating_num'].values.astype(float),
        np.log(g['Vol_12m'].values)
    ])
    # OLS via least squares
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ beta
    resid = y - y_hat
    return pd.Series(resid, index=g.index)

df_val['Value_signal'] = (
    df_val
    .groupby('YearMonth', group_keys=False)
    .apply(value_residuals_month)
)



df_val['Beta_q_val'] = (
    df_val
    .groupby('YearMonth', group_keys=False)['DTS']
    .apply(quintili_mensili)
)

df_val['Value_signal_adj'] = (
    df_val
    .groupby(['YearMonth','Beta_q_val'], group_keys=False)['Value_signal']
    .transform(lambda x: x - x.mean())
)



df_val['Value_q'] = (
    df_val
    .groupby('YearMonth', group_keys=False)['Value_signal_adj']
    .apply(quintili_mensili)
)

panel_val = df_val.dropna(subset=['Value_q']).copy()
panel_val['Value_q'] = panel_val['Value_q'].astype(int)

panel_val['wret_val'] = panel_val['Market Value'] * panel_val['Next_ExcessRet']
q_val_sum = panel_val.groupby(['YearMonth','Value_q'])[['wret_val','Market Value']].sum()
q_val_vw = (q_val_sum['wret_val'] / q_val_sum['Market Value']).unstack('Value_q')

val_ls_vw = (q_val_vw[4] - q_val_vw[0]).rename('Value_LS_VW')


print("\nValue (VW, DTS-normalized) — Q4−Q0 stats:")
print(stats(val_ls_vw))

cum_val = (1 + val_ls_vw.dropna()).cumprod()
cum_val = cum_val / cum_val.iloc[0]

plt.figure(figsize=(9,4))
cum_val.plot()
plt.title("Value (Residual log(OAS) vs risk anchors, DTS-normalized)\nLong Q4 (Cheap) minus Short Q0 (Rich) — VW, Index=1")
plt.ylabel("Cumulative Growth")
plt.xlabel("")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


###--------------------------
### MAtrice Di Correlazione
###-------------------------
factors = pd.concat(
    [
        carry_ls_vw.rename('Carry'),
        def_ls_vw.rename('Defensive'),
        mom_ls_vw.rename('Momentum'),
        val_ls_vw.rename('Value')
    ],
    axis=1
)

# Teniamo solo i mesi in cui TUTTI i fattori sono disponibili
factors = factors.dropna()



corr_matrix = factors.corr()
print(corr_matrix)


corr_spearman = factors.corr(method='spearman')
print(corr_spearman)

























## ==============================================================
## 7. Creazione Dataset Core finale (thesis-safe)
## ==============================================================

# 0) Coerenza tipi e chiavi
df_fattori['Date'] = pd.to_datetime(df_fattori['Date'], errors='coerce')
df_fattori['YearMonth'] = df_fattori['Date'].dt.to_period('M')

# (facoltativo ma utile) assicura che le chiavi siano uniche in df_fattori
# se per qualche ragione ci fossero duplicati, li risolve in modo deterministico
df_fattori = df_fattori.sort_values(['ISIN', 'Date'])
df_fattori = df_fattori.drop_duplicates(['ISIN', 'Date'], keep='last')


# 1) Merge Momentum: importa TUTTO ciò che serve in modo consistente
#    (Mom_signal lo vogliamo prendere da df_mom per evitare versioni stale)
mom_merge_cols = ['ISIN', 'Date', 'Mom_signal', 'Mom_signal_adj', 'Mom_q']
df_fattori = df_fattori.merge(
    df_mom[mom_merge_cols].drop_duplicates(['ISIN', 'Date']),
    on=['ISIN', 'Date'],
    how='left',
    validate='one_to_one'   # fallisce se ci sono duplicati -> utile per debugging
)

# 2) Merge Value: segnali e quintili (AQR-style residual)
val_merge_cols = ['ISIN', 'Date', 'Value_signal', 'Value_signal_adj', 'Value_q']
df_fattori = df_fattori.merge(
    df_val[val_merge_cols].drop_duplicates(['ISIN', 'Date']),
    on=['ISIN', 'Date'],
    how='left',
    validate='one_to_one'
)

# 3) Definizione colonne core
#    NOTA: rimuovo Rating_w (legacy) perché nel Value finale usi Rating_num
#    Se vuoi tenerla comunque, puoi reinserirla, ma è sconsigliato per pulizia metodologica.
cols_keep = [
    # Chiavi e tempo
    'ISIN',
    'Date',
    'YearMonth',

    # Ritorni
    'ExcessRet',
    'Next_ExcessRet',
    'Month-to-Date Return',

    # Pesi
    'Market Value',

    # Variabili di base (fondamentali/di mercato)
    'OAS',
    'Duration',
    'Annual Yield',
    'Markit iBoxx Rating',
    'Rating_num',
    'Vol_12m',
    'DTS',

    # Benchmark / risk-free (se presenti nel dataset)
    'Benchmark_TR_VW',
    'Benchmark_Excess_VW',
    'RiskFree_EW',

    # Segnali / quintili fattoriali
    'Carry_signal',
    'Carry_q',
    'Defensive_signal',
    'Def_q',
    'Mom_signal',
    'Mom_signal_adj',
    'Mom_q',
    'Value_signal',
    'Value_signal_adj',
    'Value_q'
]

# 4) Tieni solo le colonne effettivamente presenti (evita KeyError se qualche benchmark manca)
cols_keep_present = [c for c in cols_keep if c in df_fattori.columns]
missing = [c for c in cols_keep if c not in df_fattori.columns]

if len(missing) > 0:
    print("⚠️ Colonne non presenti e quindi escluse dal core dataset:")
    print(missing)

df_core = df_fattori[cols_keep_present].copy()

# 5) Investibilità: tieni solo osservazioni con ritorno mese successivo disponibile
df_core = df_core.dropna(subset=['Next_ExcessRet'])

# 6) Salvataggio su file
out_path = Path("/Users/leopera/Desktop/dataset_fattori_core.csv")
df_core.to_csv(out_path, index=False)

print("📁 Dataset core salvato correttamente!")
print("📌 Percorso:", out_path)
print("🔎 Shape:", df_core.shape)
print("✅ Periodo:", df_core['Date'].min(), "→", df_core['Date'].max())




# --- Carry plot (export for Word) ---
from pathlib import Path

desktop = Path.home() / "Desktop"
png_path = desktop / "Figure_5_Carry_LongShort_Cumulative.png"
pdf_path = desktop / "Figure_5_Carry_LongShort_Cumulative.pdf"

plt.figure(figsize=(10,4.5))
cum.plot()

plt.title("Carry Factor — Cumulative Performance of the Long–Short Portfolio (Value-Weighted)", fontsize=12)
plt.ylabel("Cumulative Return (Index = 1)")
plt.xlabel("")
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save high quality (PNG for Word + PDF vector)
plt.savefig(png_path, dpi=300, bbox_inches="tight")
plt.savefig(pdf_path, bbox_inches="tight")

print(f"✅ Saved on Desktop: {png_path}")
print(f"✅ Saved on Desktop (vector): {pdf_path}")

plt.show()

# --- Defensive plot (export for Word) ---
desktop = Path.home() / "Desktop"
png_path = desktop / "Figure_5_Defensive_LongShort_Cumulative.png"
pdf_path = desktop / "Figure_5_Defensive_LongShort_Cumulative.pdf"

plt.figure(figsize=(10,4.5))
cum_def.plot()

plt.title("Defensive Factor — Cumulative Performance of the Long–Short Portfolio (Value-Weighted)", fontsize=12)
plt.ylabel("Cumulative Return (Index = 1)")
plt.xlabel("")
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(png_path, dpi=300, bbox_inches="tight")
plt.savefig(pdf_path, bbox_inches="tight")

print(f"✅ Saved on Desktop: {png_path}")
print(f"✅ Saved on Desktop (vector): {pdf_path}")

plt.show()

# --- Momentum plot (export for Word) ---
desktop = Path.home() / "Desktop"
png_path = desktop / "Figure_5_Momentum_LongShort_Cumulative.png"
pdf_path = desktop / "Figure_5_Momentum_LongShort_Cumulative.pdf"

plt.figure(figsize=(10,4.5))
cum_mom.plot()

plt.title("Momentum Factor — Cumulative Performance of the Long–Short Portfolio (Value-Weighted)", fontsize=12)
plt.ylabel("Cumulative Return (Index = 1)")
plt.xlabel("")
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(png_path, dpi=300, bbox_inches="tight")
plt.savefig(pdf_path, bbox_inches="tight")

print(f"✅ Saved on Desktop: {png_path}")
print(f"✅ Saved on Desktop (vector): {pdf_path}")

plt.show()

# --- Value plot (export for Word) ---
desktop = Path.home() / "Desktop"
png_path = desktop / "Figure_5_Value_LongShort_Cumulative.png"
pdf_path = desktop / "Figure_5_Value_LongShort_Cumulative.pdf"

plt.figure(figsize=(10,4.5))
cum_val.plot()

plt.title("Value Factor — Cumulative Performance of the Long–Short Portfolio (Value-Weighted)", fontsize=12)
plt.ylabel("Cumulative Return (Index = 1)")
plt.xlabel("")
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(png_path, dpi=300, bbox_inches="tight")
plt.savefig(pdf_path, bbox_inches="tight")

print(f"✅ Saved on Desktop: {png_path}")
print(f"✅ Saved on Desktop (vector): {pdf_path}")

plt.show()

desktop = Path.home() / "Desktop"
excel_path = desktop / "Table_5_Factor_Correlation_Matrix.xlsx"

# Arrotondiamo a 3 decimali per pulizia accademica
corr_export = corr_matrix.round(3)

corr_export.to_excel(excel_path, engine='openpyxl')

print(f"✅ Correlation matrix saved on Desktop: {excel_path}")
# ## ==============================================================
# ## 7. Creazione Dataset Core finale
# ## ==============================================================

# # 1) Portiamo dentro in df_fattori le colonne di Momentum da df_mom
# #    (attenzione a selezionare solo le colonne che non sono già in df_fattori)
# mom_merge_cols = ['ISIN', 'Date', 'Mom_signal_adj', 'Mom_q']
# df_fattori = df_fattori.merge(
#     df_mom[mom_merge_cols].drop_duplicates(['ISIN', 'Date']),
#     on=['ISIN', 'Date'],
#     how='left'
# )

# # 2) Portiamo dentro in df_fattori le colonne di Value da df_val
# val_merge_cols = ['ISIN', 'Date', 'Value_signal', 'Value_signal_adj', 'Value_q']
# df_fattori = df_fattori.merge(
#     df_val[val_merge_cols].drop_duplicates(['ISIN', 'Date']),
#     on=['ISIN', 'Date'],
#     how='left'
# )

# # 3) Ora selezioniamo SOLO le colonne che vogliamo tenere nel dataset core
# cols_keep = [
#     # Chiavi e tempo
#     'ISIN',
#     'Date',
#     'YearMonth',
    
#     # Ritorni
#     'ExcessRet',
#     'Next_ExcessRet',
#     'Month-to-Date Return',
    
#     # Pesi
#     'Market Value',
    
#     # Variabili di base (fondamentali/di mercato)
#     'OAS',
#     'Duration',
#     'Annual Yield',
#     'Markit iBoxx Rating',
#     'Rating_w',
#     'Vol_12m',
#     'DTS',
#     'Benchmark_TR_VW',
#     'Benchmark_Excess_VW',
#     'RiskFree_EW',
    
#     # Segnali / quintili fattoriali
#     'Carry_signal',
#     'Carry_q',
#     'Defensive_signal',
#     'Def_q',
#     'Mom_signal',
#     'Mom_signal_adj',
#     'Mom_q',
#     'Value_signal',
#     'Value_signal_adj',
#     'Value_q'
# ]

# df_core = df_fattori[cols_keep].copy()

# # (Opzionale, ma spesso utile: tieni solo le righe con Next_ExcessRet non NaN)
# df_core = df_core.dropna(subset=['Next_ExcessRet'])

# # 4) Salvataggio su file
# out_path = Path("/Users/leopera/Desktop/dataset_fattori_core.csv")
# df_core.to_csv(out_path, index=False)

# print("📁 Dataset core salvato correttamente!")
# print("📌 Percorso:", out_path)
# print("🔎 Shape:", df_core.shape)