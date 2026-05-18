"""
Macro Regime Classification Engine
====================================

METHODOLOGY
-----------
The engine classifies the current economic regime by projecting four orthogonal
dimension scores onto a set of pre-defined regime profiles and selecting the
nearest profile in weighted Euclidean space.

Four dimensions, each scored 0 (healthy) → 1 (stressed):

  Dimension          Weight   Inputs
  ─────────────────  ──────   ──────────────────────────────────────────────
  Growth      (g)    0.35     T10Y2Y, AMTMNO YoY, USSLIND MoM, ICSA trend
  Inflation   (i)    0.30     CPI YoY, Core CPI YoY, UMich expectations
  Financial   (f)    0.20     NFCI, HY spread, St. Louis FSI
  Credit      (c)    0.15     C&I lending standards, CC delinquency rate

Component weights within each dimension are defined in COMPONENT_WEIGHTS and
renormalized automatically when a series has no data.

Component sub-scores are produced by calibrated step functions (fixed thresholds
drawn from the macro literature — see each _*_score method for citations).

CLASSIFICATION ALGORITHM
------------------------
1.  Compute (g, i, f, c) from the four dimension methods.
2.  For each of the seven named regimes, compute weighted Euclidean distance to
	its target profile in REGIME_PROFILES:

		d(r) = sqrt( w_g*(g - g_r)^2 + w_i*(i - i_r)^2
				   + w_f*(f - f_r)^2 + w_c*(c - c_r)^2 )

3.  Convert distances to soft probabilities via temperature-scaled softmax
	(SOFTMAX_TEMPERATURE = 4.0):

		P(r) = exp(-τ * d(r)) / Σ exp(-τ * d(k))

4.  The winning regime is argmax P.  If P_winner < MIN_CONFIDENCE (0.30) no
	single regime has clear support and the label falls back to "Uncertain."

WHAT THE ENGINE IS NOT
-----------------------
- Not a hidden Markov model: states are independent across periods.
- Not a clustering algorithm: profiles are pre-specified from macro literature,
  not data-driven.
- Not a probabilistic graphical model: soft probabilities are monotone
  transformations of distance, not posterior probabilities from a generative model.
- Thresholds are static by design: they reflect literature consensus (Estrella &
  Mishkin 1998; Conference Board LEI methodology; NBER dating criteria) rather
  than optimization on historical recession labels.

TRANSITION CRITERIA
-------------------
classify() is stateless — it classifies the current snapshot only.
regime_history() applies a HISTORY_SMOOTHER_MONTHS-month rolling mode filter
to reduce label flickering driven by near-boundary months.

REGIME PROFILES (target dimension scores)
-----------------------------------------
Calibrated against known NBER-dated and ECRI-dated historical episodes:

  Regime                  g     i     f     c   Reference episodes
  ─────────────────────  ────  ────  ────  ────  ─────────────────────────────
  Goldilocks             0.15  0.15  0.20  0.15  2004-06, 2014-18, 2019
  Reflation              0.20  0.65  0.25  0.25  2021, 2010-11, 2017 pick-up
  Stagflation            0.65  0.80  0.55  0.60  2022, 1973-75, 1979-80
  Disinflation           0.50  0.35  0.45  0.45  2023-24, 2015-16, 2012
  Liquidity Boom         0.25  0.25  0.10  0.15  2010-13, Q3 2020 – Q1 2021
  Tightening Cycle       0.40  0.60  0.65  0.55  2022 H1, 2018 Q4, 2006-07
  Balance-Sheet Rec.     0.85  0.25  0.90  0.90  2008-09, 2020-Q1
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import mode as scipy_mode


# ── Classification parameters ───────────────────────────────────────────────

DIMENSION_WEIGHTS: dict[str, float] = {
	"growth":    0.35,
	"inflation": 0.30,
	"financial": 0.20,
	"credit":    0.15,
}

# Component weights within each dimension.
# Keys match the identifiers used in each _*_score method.
# Missing components' weights are redistributed proportionally.
COMPONENT_WEIGHTS: dict[str, dict[str, float]] = {
	"growth": {
		"T10Y2Y":     0.30,  # yield curve as leading growth signal (Estrella & Mishkin 1998)
		"AMTMNO":     0.25,  # manufacturing new orders YoY
		"USSLIND":    0.30,  # Conference Board LEI MoM%
		"ICSA_slope": 0.15,  # trend in initial jobless claims
	},
	"inflation": {
		"CPIAUCSL": 0.40,  # headline CPI YoY
		"CPILFESL": 0.40,  # core CPI YoY (Fed-preferred pre-2012)
		"MICH":     0.20,  # University of Michigan 1-year inflation expectations
	},
	"financial": {
		"NFCI":         0.50,  # Chicago Fed NFCI — broadest available
		"BAMLH0A0HYM2": 0.30,  # HY OAS — credit channel stress
		"STLFSI4":      0.20,  # St. Louis FSI — complementary signal
	},
	"credit": {
		"DRTSCILM": 0.60,  # net % banks tightening C&I standards (SLOOS)
		"DRCCLACBS": 0.40, # credit card delinquency rate
	},
}

# Regime profiles: target (g, i, f, c) for each named regime.
# A regime's profile is the centroid of its expected dimension scores at
# full expression of the regime, calibrated against historical episodes.
REGIME_PROFILES: dict[str, dict[str, float]] = {
	"Goldilocks": {
		"g": 0.15, "i": 0.15, "f": 0.20, "c": 0.15,
	},
	"Reflation": {
		"g": 0.20, "i": 0.65, "f": 0.25, "c": 0.25,
	},
	"Stagflation": {
		"g": 0.65, "i": 0.80, "f": 0.55, "c": 0.60,
	},
	"Disinflation": {
		"g": 0.50, "i": 0.35, "f": 0.45, "c": 0.45,
	},
	"Liquidity Boom": {
		"g": 0.25, "i": 0.25, "f": 0.10, "c": 0.15,
	},
	"Tightening Cycle": {
		"g": 0.40, "i": 0.60, "f": 0.65, "c": 0.55,
	},
	"Balance-Sheet Recession": {
		"g": 0.85, "i": 0.25, "f": 0.90, "c": 0.90,
	},
}

SOFTMAX_TEMPERATURE: float = 4.0   # higher → sharper winner-takes-all assignment
MIN_CONFIDENCE:      float = 0.30  # P_winner below this → "Uncertain"
HISTORY_SMOOTHER_MONTHS: int = 3   # rolling mode window for regime_history()

REGIMES = {
	"Goldilocks":              {"color": "#276749"},
	"Reflation":               {"color": "#d69e2e"},
	"Stagflation":             {"color": "#9b2c2c"},
	"Disinflation":            {"color": "#2b6cb0"},
	"Liquidity Boom":          {"color": "#553c9a"},
	"Tightening Cycle":        {"color": "#744210"},
	"Balance-Sheet Recession": {"color": "#702459"},
	"Uncertain":               {"color": "#718096"},
}

REGIME_DESCRIPTIONS: dict[str, str] = {
	"Goldilocks": (
		"Above-trend growth with contained inflation and neutral-to-easy financial conditions. "
		"The optimal macro backdrop — broad risk appetite, low volatility premium."
	),
	"Reflation": (
		"Above-trend growth with rising inflation. Commodity and cyclical assets outperform; "
		"central banks may be behind the curve. Rotation from duration."
	),
	"Stagflation": (
		"Growth weakness alongside persistent inflation — the worst policy trade-off. "
		"Rate cuts risk entrenching inflation; hikes risk deepening the slowdown."
	),
	"Disinflation": (
		"Growth softening alongside moderating inflation — rate cuts become viable. "
		"Duration assets typically outperform; quality bias in equities."
	),
	"Liquidity Boom": (
		"Easy financial conditions driving asset price inflation and credit expansion. "
		"Duration and risk assets rewarded; watch for late-cycle excess building."
	),
	"Tightening Cycle": (
		"Financial conditions tightening as rate hikes constrain credit and growth. "
		"Watch for overtightening: inversion, spread widening, and claims acceleration."
	),
	"Balance-Sheet Recession": (
		"Severe financial tightening with credit contraction and growth collapse. "
		"Policy options are constrained — the primary risk is deflation or debt-deflation spiral."
	),
	"Uncertain": (
		"Mixed signals across growth, inflation, and financial conditions dimensions. "
		"No single regime has sufficient probability mass to warrant a confident classification."
	),
}

# Regime-specific threshold adjustments for downstream risk scoring.
REGIME_THRESHOLD_ADJUSTMENTS = {
	"Goldilocks": {
		"CPIAUCSL_yoy_red": 3.0,
		"T10Y2Y_red":      -0.3,
	},
	"Stagflation": {
		"CPIAUCSL_yoy_red":  4.0,
		"BAMLH0A0HYM2_red": 500.0,
		"UNRATE_red":        4.5,
	},
	"Tightening Cycle": {
		"T10Y2Y_red":   -0.5,
		"DRCCLACBS_red": 3.0,
	},
	"Balance-Sheet Recession": {
		"BAMLH0A0HYM2_red": 400.0,
		"NFCI_red":          0.5,
		"CPIAUCSL_yoy_red":  1.5,
	},
	"Liquidity Boom": {
		"BAMLH0A0HYM2_yellow": 200.0,
		"NFCI_yellow":         -0.6,
	},
	"Reflation": {
		"CPIAUCSL_yoy_yellow": 3.0,
		"PPIACO_yoy_yellow":   6.0,
	},
}


def _weighted_score(components: dict[str, float], dim: str) -> float:
	"""
	Compute a weighted mean of component sub-scores, renormalising for
	any missing components.  Returns 0.5 if no components are available.
	"""
	weights = COMPONENT_WEIGHTS[dim]
	total_w = sum(weights[k] for k in components)
	if total_w == 0:
		return 0.5
	return sum(components[k] * weights[k] / total_w for k in components)


class RegimeEngine:
	"""
	Classifies the current macro regime and provides regime-aware context.

	See module docstring for full methodology specification.

	All dimension scores are in [0, 1]: 0 = healthy/low-risk, 1 = stressed/recessionary.
	"""

	def __init__(self, dl):
		self._dl = dl
		self._regime_cache: str | None = None

	# ── Internal helpers ───────────────────────────────────────────────────

	def _monthly(self, sid: str) -> pd.Series | None:
		df = self._dl.load(sid)
		if df is None or df.empty:
			return None
		col = df.columns[0]
		try:
			m = df.resample("ME").last()
		except ValueError:
			m = df.resample("M").last()
		return m[col].dropna()

	def _level_last(self, sid: str) -> float | None:
		s = self._monthly(sid)
		return float(s.iloc[-1]) if (s is not None and not s.empty) else None

	def _yoy_last(self, sid: str, periods: int = 12) -> float | None:
		s = self._monthly(sid)
		if s is None or len(s) < periods + 1:
			return None
		pct = (s.pct_change(periods) * 100).dropna()
		return float(pct.iloc[-1]) if not pct.empty else None

	def _trend_slope(self, sid: str, months: int = 6) -> float | None:
		"""OLS slope over the most recent `months` observations."""
		s = self._monthly(sid)
		if s is None or len(s) < months:
			return None
		vals = s.iloc[-months:].values
		if np.any(np.isnan(vals)):
			return None
		x = np.arange(len(vals), dtype=float)
		return float(np.polyfit(x, vals, 1)[0])

	# ── Dimension scoring ──────────────────────────────────────────────────
	#
	# Sub-score breakpoints are calibrated step functions derived from:
	#   - Estrella & Mishkin (1998) for yield curve
	#   - Conference Board LEI methodology for USSLIND / AMTMNO
	#   - Chicago Fed documentation for NFCI breakpoints
	#   - NBER / academic literature for HY spread and lending standards

	def _growth_score(self) -> tuple[float, str]:
		"""
		Growth dimension score.

		Components (weights in COMPONENT_WEIGHTS['growth']):
		  T10Y2Y     0.30 — negative spread signals growth contraction 6-18mo ahead
		  AMTMNO     0.25 — manufacturing orders YoY; < 0 = industrial contraction
		  USSLIND    0.30 — Conference Board LEI MoM%; persistent negative = recession risk
		  ICSA_slope 0.15 — rising claims trend precedes payroll contraction
		"""
		components: dict[str, float] = {}

		t10y2y = self._level_last("T10Y2Y")
		if t10y2y is not None:
			components["T10Y2Y"] = (
				0.85 if t10y2y < -0.5 else   # deep inversion
				0.60 if t10y2y < 0    else   # inversion
				0.35 if t10y2y < 1.0  else   # flat curve
				0.10                          # normal / steep
			)

		mfg_yoy = self._yoy_last("AMTMNO")
		if mfg_yoy is not None:
			components["AMTMNO"] = (
				0.90 if mfg_yoy < -8 else    # severe contraction
				0.60 if mfg_yoy < 0  else    # any contraction
				0.25 if mfg_yoy < 5  else    # slow growth
				0.05                          # strong growth
			)

		lei = self._level_last("USSLIND")
		if lei is not None:
			components["USSLIND"] = (
				0.85 if lei < -1.5 else  # sharp monthly decline
				0.50 if lei < 0    else  # any monthly decline
				0.15                     # monthly increase
			)

		icsa_slope = self._trend_slope("ICSA", 8)
		if icsa_slope is not None:
			components["ICSA_slope"] = (
				0.80 if icsa_slope > 5_000 else  # rapid claims surge
				0.50 if icsa_slope > 1_000 else  # moderate rise
				0.20                              # flat / falling
			)

		score = _weighted_score(components, "growth")
		label = (
			"Contraction" if score > 0.65 else
			"Softening"   if score > 0.40 else
			"Expansion"
		)
		return score, label

	def _inflation_score(self) -> tuple[float, str]:
		"""
		Inflation dimension score.

		Components (weights in COMPONENT_WEIGHTS['inflation']):
		  CPIAUCSL  0.40 — headline CPI YoY; Fed 2% target anchors green/red
		  CPILFESL  0.40 — core CPI YoY; strips food/energy volatility
		  MICH      0.20 — consumer expectations; unanchoring signal
		"""
		components: dict[str, float] = {}

		cpi = self._yoy_last("CPIAUCSL")
		if cpi is not None:
			components["CPIAUCSL"] = (
				0.90 if cpi > 5.0 else   # severely above target
				0.70 if cpi > 3.5 else   # well above target
				0.45 if cpi > 2.5 else   # mildly above target
				0.10                      # at/below target
			)

		core = self._yoy_last("CPILFESL")
		if core is not None:
			components["CPILFESL"] = (
				0.85 if core > 4.0 else  # entrenched
				0.65 if core > 3.0 else  # elevated
				0.40 if core > 2.5 else  # above target
				0.10                      # at/below target
			)

		mich = self._level_last("MICH")
		if mich is not None:
			components["MICH"] = (
				0.85 if mich > 4.5 else  # expectations unanchoring
				0.55 if mich > 3.5 else  # elevated expectations
				0.15                      # anchored
			)

		score = _weighted_score(components, "inflation")
		label = (
			"Entrenched"     if score > 0.70 else
			"Elevated"       if score > 0.45 else
			"At/Below Target"
		)
		return score, label

	def _financial_conditions_score(self) -> tuple[float, str]:
		"""
		Financial conditions dimension score.

		Components (weights in COMPONENT_WEIGHTS['financial']):
		  NFCI         0.50 — Chicago Fed: 0 = historical avg; >1 = tight/crisis
		  BAMLH0A0HYM2 0.30 — HY OAS in bps; >500 = severe stress
		  STLFSI4      0.20 — St. Louis FSI; >1.5 = above-average stress
		"""
		components: dict[str, float] = {}

		nfci = self._level_last("NFCI")
		if nfci is not None:
			components["NFCI"] = (
				0.90 if nfci >  1.0 else   # crisis level
				0.70 if nfci >  0.3 else   # tight
				0.50 if nfci >  0.0 else   # above average
				0.30 if nfci > -0.5 else   # near average
				0.10                        # easy
			)

		hy = self._level_last("BAMLH0A0HYM2")
		if hy is not None:
			components["BAMLH0A0HYM2"] = (
				0.95 if hy > 700 else   # crisis (2008/2020 level)
				0.75 if hy > 500 else   # severe stress
				0.45 if hy > 350 else   # elevated
				0.25 if hy > 250 else   # slightly elevated
				0.10                    # benign
			)

		stlfsi = self._level_last("STLFSI4")
		if stlfsi is not None:
			components["STLFSI4"] = (
				0.90 if stlfsi > 1.5 else  # high stress
				0.65 if stlfsi > 0.5 else  # above average
				0.40 if stlfsi > 0.0 else  # at average
				0.15                        # below average
			)

		score = _weighted_score(components, "financial")
		label = (
			"Crisis"  if score > 0.75 else
			"Tight"   if score > 0.50 else
			"Neutral" if score > 0.30 else
			"Easy"
		)
		return score, label

	def _credit_score(self) -> tuple[float, str]:
		"""
		Credit cycle dimension score.

		Components (weights in COMPONENT_WEIGHTS['credit']):
		  DRTSCILM  0.60 — net % banks tightening C&I standards; >20 = significant tightening
		  DRCCLACBS 0.40 — CC delinquency rate; >4.5% = elevated consumer credit stress
		"""
		components: dict[str, float] = {}

		lending = self._level_last("DRTSCILM")
		if lending is not None:
			components["DRTSCILM"] = (
				0.85 if lending > 50 else  # severe tightening
				0.60 if lending > 20 else  # significant tightening
				0.35 if lending >  0 else  # slight tightening
				0.10                        # easing / neutral
			)

		cc_delinq = self._level_last("DRCCLACBS")
		if cc_delinq is not None:
			components["DRCCLACBS"] = (
				0.80 if cc_delinq > 4.5 else  # elevated delinquency
				0.55 if cc_delinq > 3.0 else  # moderate stress
				0.20                            # normal
			)

		score = _weighted_score(components, "credit")
		label = (
			"Contraction" if score > 0.65 else
			"Tightening"  if score > 0.40 else
			"Expanding"
		)
		return score, label

	# ── Dimension summary ──────────────────────────────────────────────────

	def dimension_scores(self) -> dict:
		"""All four dimension scores and labels."""
		g_score, g_label = self._growth_score()
		i_score, i_label = self._inflation_score()
		f_score, f_label = self._financial_conditions_score()
		c_score, c_label = self._credit_score()
		return {
			"growth":    {"score": g_score, "label": g_label},
			"inflation": {"score": i_score, "label": i_label},
			"financial": {"score": f_score, "label": f_label},
			"credit":    {"score": c_score, "label": c_label},
		}

	# ── Classification core ────────────────────────────────────────────────

	def _regime_distances(self, g: float, i: float,
						  f: float, c: float) -> dict[str, float]:
		"""
		Weighted Euclidean distance from the current dimension point to each
		regime profile.  Lower = closer = more likely.
		"""
		w = DIMENSION_WEIGHTS
		scores = np.array([g, i, f, c])
		result: dict[str, float] = {}
		for regime, profile in REGIME_PROFILES.items():
			target = np.array([profile["g"], profile["i"],
							   profile["f"], profile["c"]])
			diff_sq = np.array([
				w["growth"]    * (scores[0] - target[0]) ** 2,
				w["inflation"] * (scores[1] - target[1]) ** 2,
				w["financial"] * (scores[2] - target[2]) ** 2,
				w["credit"]    * (scores[3] - target[3]) ** 2,
			])
			result[regime] = float(np.sqrt(diff_sq.sum()))
		return result

	def _distances_to_probs(self, distances: dict[str, float]) -> dict[str, float]:
		"""
		Temperature-scaled softmax over negative distances.

		P(r) = exp(-τ * d(r)) / Σ exp(-τ * d(k))

		Numerically stabilised by subtracting the maximum exponent.
		"""
		τ = SOFTMAX_TEMPERATURE
		regimes = list(distances.keys())
		neg_d = np.array([-τ * distances[r] for r in regimes])
		neg_d -= neg_d.max()           # numerical stability
		exp_d = np.exp(neg_d)
		total = exp_d.sum()
		return {r: float(e / total) for r, e in zip(regimes, exp_d)}

	def regime_probabilities(self) -> dict[str, float]:
		"""
		Soft probability distribution over all seven regime types.

		Returns a dict mapping regime name → probability in (0, 1) where
		probabilities sum to 1.0.  "Uncertain" is not included — it is a
		fallback label, not a profile.
		"""
		g, _ = self._growth_score()
		i, _ = self._inflation_score()
		f, _ = self._financial_conditions_score()
		c, _ = self._credit_score()
		distances = self._regime_distances(g, i, f, c)
		return self._distances_to_probs(distances)

	# ── Public classification ──────────────────────────────────────────────

	def classify(self) -> tuple[str, str, str]:
		"""
		Classify the current macro regime.

		Returns (regime_name, hex_color, description).

		Algorithm:
		  1. Score four dimensions: g, i, f, c ∈ [0, 1].
		  2. Compute weighted Euclidean distance to each regime profile.
		  3. Convert distances to soft probabilities via softmax (τ = 4.0).
		  4. Declare winner if P_winner ≥ MIN_CONFIDENCE (0.30); else "Uncertain."
		"""
		g, _ = self._growth_score()
		i, _ = self._inflation_score()
		f, _ = self._financial_conditions_score()
		c, _ = self._credit_score()

		distances = self._regime_distances(g, i, f, c)
		probs     = self._distances_to_probs(distances)

		winner = max(probs, key=probs.__getitem__)
		regime = winner if probs[winner] >= MIN_CONFIDENCE else "Uncertain"

		self._regime_cache = regime
		color = REGIMES.get(regime, REGIMES["Uncertain"])["color"]
		desc  = REGIME_DESCRIPTIONS.get(regime, REGIME_DESCRIPTIONS["Uncertain"])
		return regime, color, desc

	def get_threshold_adjustments(self) -> dict:
		"""Regime-specific threshold overrides for dynamic risk scoring."""
		if self._regime_cache is None:
			self.classify()
		return REGIME_THRESHOLD_ADJUSTMENTS.get(self._regime_cache, {})

	# ── Historical regime timeline ─────────────────────────────────────────

	def regime_history(self, lookback_years: int = 20) -> pd.Series:
		"""
		Monthly time-series of regime labels for charting.

		Uses a three-variable simplified dimension mapping (CPI, T10Y2Y, NFCI /
		HY spread) because full four-dimension scoring requires series with
		complete monthly coverage that some inputs lack over long horizons.

		Simplified dimension mapping:
		  g_proxy = f(T10Y2Y)          — yield curve as growth signal
		  i_proxy = f(CPI YoY)         — headline inflation
		  f_proxy = f(NFCI, HY spread) — financial conditions

		Credit dimension (c) defaults to 0.40 when historical data is sparse
		(SLOOS data begins 1990-Q4).

		A HISTORY_SMOOTHER_MONTHS-month rolling mode filter is applied to reduce
		label flickering at regime boundaries.
		"""
		cutoff = pd.Timestamp.now() - pd.DateOffset(years=lookback_years)

		def _mo_yoy(sid: str, periods: int = 12) -> pd.Series | None:
			df = self._dl.load(sid)
			if df is None or df.empty:
				return None
			col = df.columns[0]
			try:
				m = df.resample("ME").last()
			except ValueError:
				m = df.resample("M").last()
			s = m[col].dropna()
			return (s.pct_change(periods) * 100).dropna() if len(s) > periods else None

		def _mo_lvl(sid: str) -> pd.Series | None:
			df = self._dl.load(sid)
			if df is None or df.empty:
				return None
			col = df.columns[0]
			try:
				m = df.resample("ME").last()
			except ValueError:
				m = df.resample("M").last()
			return m[col].dropna()

		cpi_yoy = _mo_yoy("CPIAUCSL")
		t10y2y  = _mo_lvl("T10Y2Y")
		nfci    = _mo_lvl("NFCI")
		hy      = _mo_lvl("BAMLH0A0HYM2")

		if cpi_yoy is None or t10y2y is None:
			return pd.Series(dtype=str)

		common = cpi_yoy.index.intersection(t10y2y.index)
		common = common[common >= cutoff]
		if len(common) == 0:
			return pd.Series(dtype=str)

		# ── Map simplified inputs to dimension scores ──────────────────────
		# g_proxy thresholds match the T10Y2Y breakpoints in _growth_score
		def _g_proxy(curve: float | None) -> float:
			if curve is None:
				return 0.5
			return (0.85 if curve < -0.5 else
					0.60 if curve <  0.0 else
					0.35 if curve <  1.0 else 0.10)

		def _i_proxy(cpi: float | None) -> float:
			if cpi is None:
				return 0.5
			return (0.90 if cpi > 5.0 else
					0.70 if cpi > 3.5 else
					0.45 if cpi > 2.5 else 0.10)

		def _f_proxy(nfci_val: float | None, hy_val: float | None) -> float:
			sub: dict[str, float] = {}
			if nfci_val is not None:
				sub["NFCI"] = (0.90 if nfci_val >  1.0 else
							   0.70 if nfci_val >  0.3 else
							   0.50 if nfci_val >  0.0 else
							   0.30 if nfci_val > -0.5 else 0.10)
			if hy_val is not None:
				sub["BAMLH0A0HYM2"] = (0.95 if hy_val > 700 else
									   0.75 if hy_val > 500 else
									   0.45 if hy_val > 350 else
									   0.25 if hy_val > 250 else 0.10)
			if not sub:
				return 0.5
			# Use financial component weights, renormalised to NFCI/HY only
			w = {k: COMPONENT_WEIGHTS["financial"][k] for k in sub}
			total = sum(w.values())
			return sum(sub[k] * w[k] / total for k in sub)

		raw_labels: list[str] = []
		for dt in common:
			cpi_v  = cpi_yoy.get(dt)
			t10_v  = t10y2y.get(dt)
			nfci_v = nfci.get(dt)   if nfci is not None else None
			hy_v   = hy.get(dt)     if hy is not None   else None

			g = _g_proxy(t10_v)
			i = _i_proxy(cpi_v)
			f = _f_proxy(nfci_v, hy_v)
			c = 0.40  # credit proxy: neutral default for history (SLOOS partial coverage)

			distances = self._regime_distances(g, i, f, c)
			probs     = self._distances_to_probs(distances)
			winner    = max(probs, key=probs.__getitem__)
			raw_labels.append(winner if probs[winner] >= MIN_CONFIDENCE else "Uncertain")

		raw = pd.Series(raw_labels, index=common, name="regime")

		# ── Rolling mode smoother ─────────────────────────────────────────
		k = HISTORY_SMOOTHER_MONTHS
		if k > 1 and len(raw) >= k:
			smoothed: list[str] = []
			for j in range(len(raw)):
				window = raw.iloc[max(0, j - k + 1): j + 1].dropna()
				if len(window) == 0:
					smoothed.append("")
					continue
				values, counts = np.unique(window.values, return_counts=True)
				mode_value = values[np.argmax(counts)]
				smoothed.append(str(mode_value))
			return pd.Series(smoothed, index=common, name="regime")
		return raw