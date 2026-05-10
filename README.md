# FRED Economic Monitor

An institutional-grade economic dashboard that downloads data from the [FRED API](https://fred.stlouisfed.org/) and presents it as an interactive Dash web application with red/yellow/green risk scoring across 17 economic indicators.

## Features

- **Crisis Watch** — composite risk scores across six structural dimensions (Treasury Stress, Inflation Persistence, Labor Weakness, Financial Stress, Debt Sustainability, Market Valuation)
- **Summary** — KPI cards for all tracked indicators with as-of dates and risk highlighting
- **Drill-down tabs** — Inflation, Money Supply, Labor Market, Markets & Rates, Fiscal
- **SQL or CSV storage** — writes to SQL Server when configured, falls back to CSV files
- **S&P 500 P/E and CAPE** — sourced from Shiller's Yale dataset, extended with multpl.com for current months

## Setup

### 1. Install dependencies

```
pip install -r requirements.txt
```

> SQL Server connectivity also requires the [ODBC Driver for SQL Server](https://learn.microsoft.com/en-us/sql/connect/odbc/download-odbc-driver-for-sql-server) to be installed on the machine.

### 2. Configure

Edit `config.ini`:

```ini
[fred]
api_key = YOUR_FRED_API_KEY

[Database]
usesqldriver     = True
databaseserver   = your_server
databasename     = your_database
usetrustedconnection = True
databaseusername =
databasepassword =
```

Get a free FRED API key at [https://fred.stlouisfed.org/docs/api/api_key.html](https://fred.stlouisfed.org/docs/api/api_key.html).

Set `usesqldriver = False` (or remove the `[Database]` section entirely) to store data as CSV files under `data/` instead.

### 3. Download data

```
python FREDDownloader.py
```

Downloads all 17 series from FRED plus Shiller P/E and CAPE from Yale/multpl.com. In SQL mode, writes directly to the database. In CSV mode, saves files to `data/`.

### 4. Run the dashboard

```
python FREDDashboard.py
```

Open [http://127.0.0.1:8050](http://127.0.0.1:8050) in a browser.

## Project Structure

```
FREDAPI/
├── FREDDownloader.py        # Data download script
├── FREDDashboard.py         # Dash web application
├── config.ini               # API key and database connection
├── requirements.txt
├── data/                    # CSV storage (CSV mode only)
└── _classes/
    ├── constants.py         # All configuration constants and style tokens
    ├── series_registry.py   # Metadata and risk thresholds for each series
    ├── data_loader.py       # Loads and caches data from SQL or CSV
    ├── sql_storage.py       # SQLAlchemy persistence layer (SQL Server)
    ├── risk_engine.py       # Risk scoring and Crisis Watch dimension logic
    └── chart_factory.py     # Reusable Plotly chart functions
```

## Tracked Series

| Series | Source | Category |
|--------|--------|----------|
| CPI All Items (CPIAUCSL) | FRED | Inflation |
| Core CPI (CPILFESL) | FRED | Inflation |
| Inflation Annual % (FPCPITOTLZGUSA) | FRED | Inflation |
| M1 Money Supply (M1SL) | FRED | Money Supply |
| M2 Money Supply (M2SL) | FRED | Money Supply |
| Real M2 (M2REAL) | FRED | Money Supply |
| VIX Volatility (VIXCLS) | FRED | Financial Stress |
| 10Y Treasury Daily (DGS10) | FRED | Rates |
| 10Y Treasury Monthly (GS10) | FRED | Rates |
| CRE Delinquency Rate (DRCRELEXFACBS) | FRED | Financial Stress |
| Federal Debt / GDP (GFDEGDQ188S) | FRED | Fiscal |
| Unemployment Rate (UNRATE) | FRED | Labor |
| Employment-Population Ratio (EMRATIO) | FRED | Labor |
| Labor Force Participation (CIVPART) | FRED | Labor |
| Nonfarm Payrolls (PAYEMS) | FRED | Labor |
| S&P 500 P/E Ratio (SP500_PE) | Shiller/multpl | Valuation |
| S&P 500 CAPE (SP500_CAPE) | Shiller/multpl | Valuation |

## SQL Schema

When SQL mode is enabled, two tables are created automatically:

```sql
fred_series_data     (series_id, date)  PRIMARY KEY  -- one row per series per date
fred_series_metadata (series_id)        PRIMARY KEY  -- name, category, row count, last updated
```
