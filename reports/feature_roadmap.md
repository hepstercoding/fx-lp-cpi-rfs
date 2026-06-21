# Feature Roadmap

## Data Page

- Show source documentation and links.
- Show sample coverage and missing values.
- Plot NSA and SA data where both are available.
- Download selected data columns.

## Baseline LP Page

- Select response variable:
  - headline CPI
  - core CPI 1
  - core CPI 2
- Choose dependent-variable transformation for estimation:
  - y/y inflation
  - m/m inflation
  - cumulative price difference
- Choose displayed transformation independently of the estimated dependent variable:
  - y/y inflation
  - m/m inflation
  - cumulative price difference
- Choose LHS index adjustment independently:
  - seasonally adjusted index
  - non-seasonally adjusted index
- Display confidence intervals:
  - direct 90% HAC confidence intervals when the displayed response is estimated directly
  - 90% diagonal delta-method approximations for algebraically converted responses
- Select CHF shock measure:
  - CHF NEER monthly change [implemented]
  - EURCHF monthly move, CHF-appreciation positive [implemented]
  - USDCHF monthly move, CHF-appreciation positive [implemented]
- Change estimation range with start and end dates.
- Choose projection horizon.
- Choose number of lags.
- Toggle controls:
  - euro area core inflation
  - Brent oil inflation
  - Swiss unemployment
  - energy/fuels CPI
- Toggle forward CHF shocks:
  - standard one-shock FX path [implemented]
  - layered maintained 1% FX path [implemented]
  - forward-shock controls to isolate the initial shock while holding later CHF movements fixed [implemented]
- Always show the CHF response to the CHF shock alongside CPI responses.
- Always show the equation being estimated, updated to match the selected settings.
- Display impulse-response chart with confidence bands.
- Display CHF shock persistence chart:
  - response of future CHF NEER changes or cumulative CHF NEER level to the initial CHF move
  - same sample and specification controls as the CPI LP where feasible
- Display implied REER evolution assuming foreign prices do not react:
  - with forward CHF shocks included, nominal CHF path is held at +1
  - without forward CHF shocks, nominal CHF path uses the estimated CHF persistence response
  - implied REER equals nominal CHF path plus Swiss cumulative price response
  - implied REER is independent of the selected CPI display transformation
- Display a dedicated asymmetry page:
  - positive CHF shocks as +1pp appreciations
  - negative CHF shocks as 1pp depreciations multiplied by -1 for visual comparison
  - no forward-shock controls in the asymmetric regression
  - estimated CHF IRF used to layer shocks into a maintained 1% appreciation/depreciation path
  - maintained symmetric IRF shown as a black dashed reference line
  - one-shock FX appreciation/depreciation IRFs displayed next to the layering weights [implemented]
- Display a dedicated major-groups page:
  - source 14 SNB `plkoprgru` major grouping indexes [implemented]
  - estimate direct y/y LPs for each subgroup [implemented]
  - show headline CPI IRF as a black dashed reference in every subgroup chart [implemented]
  - toggle maintained appreciation/depreciation asymmetry for each subgroup [implemented]
  - choose normal estimated FX path or layered maintained 1% FX path instead of forward-shock controls [implemented]
- Save selected specifications and compare several displayed CPI responses in one chart. [implemented first pass]
- Display coefficient table with standard errors and observations.
- Download results as CSV.

## Period Dummies

- Define one or several period dummies from the dashboard. [implemented first pass]
- Allow each dummy to have: [implemented first pass]
  - name
  - start date
  - end date
  - optional note
- Include selected period dummies as controls in the LP. [implemented as contemporaneous controls]
- Save period-dummy presets for common windows: [implemented first pass]
  - SNB minimum exchange-rate period
  - floor removal period
  - Covid period
  - energy shock period
- Plot selected dummy periods as shaded regions in data and LP charts. [implemented as Baseline LP sample preview]

## Robustness

- Compare full-sample and restricted-sample estimates.
- Compare headline and core CPI responses.
- Estimate appreciation and depreciation responses separately. [implemented first pass]
- Estimate large and small CHF moves separately.
- Compare broad and narrow CHF NEER baskets.
- Add bilateral exchange-rate specifications. [implemented first pass]
- Add direct y/y and cumulative price-level response views.

## Reporting

- Export selected specification summary.
- Export charts as PNG.
- Generate a short markdown interpretation note.
- Keep a run log of selected settings and generated results.
