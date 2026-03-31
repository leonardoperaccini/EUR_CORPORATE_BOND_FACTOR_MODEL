[README.md](https://github.com/user-attachments/files/26384256/README.md)
# Factor Investing in EUR Corporate Bond Markets (2020–2025)

**Master's Thesis — Leonardo Peraccini (787811)**  
Empirical replication and adaptation of the AQR factor framework (Israel, Palhares & Richardson, 2018) applied to the EUR-denominated investment-grade corporate bond market over the period April 2020 – March 2025.

> Data provided by Gaetano Romano, Portfolio Manager at Generali Investments.

---

## Overview

This repository contains the Python code used to:

1. Construct and filter the bond-level dataset from iBoxx EUR End-of-Day Underlyings
2. Build four systematic factor signals: **Carry**, **Defensive**, **Momentum**, and **Value**
3. Run **Fama–MacBeth cross-sectional regressions** to assess the predictive power of each signal
4. Construct **quintile portfolios**, **constant-volatility scaled portfolios**, and a **long-only factor-tilted portfolio**, comparing performance against a value-weighted benchmark

---

## Repository Structure

```
├── Codice_Tesi_3_Novembre.py                              # Dataset construction and filtering
├── Codice_Tesi_Creazione_Fattori_3_Novembre.py            # Factor signal construction
└── Codice_Tesi_Regressione_e_creazione_portafoglio_19_novembre.py  # Regressions and portfolios
```

---

## File Descriptions

### 1. `Codice_Tesi_3_Novembre.py` — Dataset Construction

Loads and merges monthly CSV snapshots from the **iBoxx EUR End-of-Day Underlyings** into a single panel dataset. The following filtering steps are applied to construct the final issuer-level universe:

- Restricts to **EUR-denominated corporate bonds** (`Level 1 == 'Corporates'`)
- Retains only **senior unsecured debt** (`Seniority Level 1 == 'SEN'`)
- Identifies the **dominant rating bucket** for each issuer-date (weighted by notional amount outstanding); only bonds belonging to that bucket are retained
- Applies a **conditional maturity screen**: bonds with time-to-maturity between 5 and 15 years are preferred; if none exist, all maturities are considered
- Selects one **representative bond per issuer-month**, prioritising largest notional amount and, as a secondary criterion, tightest bid–ask spread
- Removes observations with missing values in key variables (OAS, Duration, Yield, Bid-Ask Spread, Returns)
- Computes **monthly excess returns** as: `Excess Return = Month-to-Date Return − Month-to-date Sovereign Curve Swap Return`

Additional outputs from this script:
- A **value-weighted corporate benchmark** (both total return and excess return, equal-weighted and value-weighted variants)
- A **risk-free proxy** constructed as the equal-weighted average of the sovereign curve swap returns
- **Descriptive statistics** (Tables 4.1 and 4.2 in the thesis): cross-sectional statistics averaged over time for the filtered universe and the pre-filter universe
- **Turnover analysis** at both ISIN level and issuer level (symmetric and prev-based turnover)
- **Sector and country composition** of the final universe

The final filtered dataset is exported as `dataset_finale_fattori.csv`.

---

### 2. `Codice_Tesi_Creazione_Fattori_3_Novembre.py` — Factor Construction

Loads `dataset_finale_fattori.csv` and constructs the four factor signals. All factor portfolios use **value-weighted excess returns**. At each monthly rebalancing date, bonds are sorted into **five quintiles (Q0–Q4)** and the factor long–short return is defined as Q4 minus Q0.

#### Carry
- Signal: **Option-Adjusted Spread (OAS)**
- Bonds with the highest OAS (Q4) are longed; bonds with the lowest OAS (Q0) are shorted
- No beta-neutralisation is applied, preserving the economic interpretation of carry as compensation for credit risk

#### Defensive
- Signal: **negative of bond duration** (lower duration = higher signal = more defensive)
- Long the lowest-duration quintile (Q4), short the highest-duration quintile (Q0)
- Note: the full AQR defensive measure incorporates leverage and gross profitability; these variables are unavailable in the iBoxx dataset, so duration is used as the sole proxy

#### Momentum
- Signal: **cumulative sum of lagged monthly excess returns** over a 6-month rolling window (minimum 3 observations), constructed using returns lagged by one period to ensure ex-ante consistency
- An **ex-ante beta proxy** is constructed as Duration × OAS (DTS); the raw momentum signal is **demeaned within each DTS quintile** to control for systematic credit risk
- Long recent winners (Q4), short recent losers (Q0)

#### Value
- Signal: **residual from a monthly cross-sectional OLS regression** of log(OAS) on log(Duration), credit rating (numeric), and log(trailing 12-month return volatility, minimum 3 observations)
- Positive residuals indicate bonds that are cheap relative to their fundamental risk profile
- The raw residual is **demeaned within each DTS quintile**, as in Momentum
- Long cheap bonds (Q4), short rich bonds (Q0)

Additional outputs:
- **Performance statistics** for each factor: annualised return, volatility, Sharpe ratio, t-statistic, max drawdown, hit ratio, skewness, Sortino ratio
- **Cumulative performance charts** for each long–short factor (exported as PNG and PDF)
- **Pearson and Spearman correlation matrix** of the four factor return series
- A consolidated **core dataset** (`dataset_fattori_core.csv`) combining all signals and quintile assignments

---

### 3. `Codice_Tesi_Regressione_e_creazione_portafoglio_19_novembre.py` — Regressions and Portfolios

Loads `dataset_fattori_core.csv` and performs all empirical validation and portfolio construction.

#### Fama–MacBeth Cross-Sectional Regressions
- Factor signals are **cross-sectionally normalised by rank** within each month (mean zero, unit variance)
- **Univariate regressions**: each signal is tested individually; next-month excess return is regressed on one normalised characteristic at a time
- **Multivariate regression**: all four signals are included jointly in a single monthly cross-sectional OLS, then averaged over time following the Fama–MacBeth (1973) procedure
- Reported statistics: mean slope coefficient, time-series standard deviation, number of months, FM t-statistic, average R², average number of bonds per month

#### Quintile Portfolio Construction
- Bonds are sorted monthly into five quintiles based on each individual signal and on a **combined signal** (equally weighted average of the four normalised characteristics)
- Portfolio returns are **value-weighted** within each quintile
- The long–short spread (Q5 minus Q1) is computed for each factor and for the combined signal
- Reported statistics per quintile: annualised return, annualised volatility, Sharpe ratio

#### Constant-Volatility Scaling
- Each long–short series is scaled to target an **annualised volatility of 5%**
- Realised volatility is estimated using a **trailing 12-month rolling window** (minimum 3 observations)
- The scaling factor is constructed using only information available up to t−1 (ex-ante)
- The combined portfolio is the **equally weighted average of the four volatility-scaled factor series**

#### Long-Only Portfolio
- A **composite signal** is computed as the equally weighted average of the four normalised characteristics
- Bonds are sorted into quintiles; the long-only portfolio invests exclusively in **quintile Q5**, with value-weighted positions rebalanced monthly
- The **internal benchmark** is the value-weighted portfolio of all eligible bonds in the universe
- Reported metrics: annualised excess return, annualised volatility, Sharpe ratio, maximum drawdown, hit ratio, tracking error, Information Ratio, and alpha from a benchmark regression

---

## Dependencies

```
pandas
numpy
matplotlib
seaborn
statsmodels
pathlib
```

---

## Data

The raw data consists of monthly CSV files from the **iBoxx EUR End-of-Day Underlyings** (April 2020 – March 2025), provided under a data-sharing agreement with Generali Investments. The raw data files are not included in this repository.

---

## Reference

Israel, R., Palhares, D., & Richardson, S. (2018). *Common factors in corporate bond returns*. Journal of Investment Management, 16(3), 17–46.
