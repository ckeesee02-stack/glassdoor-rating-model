# Data

The company-level dataset is not included in this repo because it is derived from Glassdoor's site content. To rerun the analysis, place a CSV at `data/glassdoor_sp500_medians.csv` with one row per company and these columns:

| Column | Description |
|---|---|
| `ticker` | S&P 500 ticker |
| `sector` | GICS sector |
| `Overall` | Overall rating |
| `Comp` | Compensation & Benefits |
| `WLB` | Work/Life Balance |
| `Culture` | Culture & Values |
| `SeniorMgmt` | Senior Management |
| `Career` | Career Opportunities |
| `DI` | Diversity & Inclusion |

Each rating is the median of a company's monthly values from Glassdoor's "Company ratings over time" chart, May to October 2026 (1 to 5 scale).

**Sample:** 503 S&P 500 tickers as of 9/30/26. 5 companies had no ratings trend (HONA, Q, TPL, TKO, VICI), and 3 duplicate share classes were dropped (GOOG, FOX, NWS), leaving 495 companies.
