# Calibrated Probabilistic Air-Quality Forecasting: Reproducing DeepAR and Diagnosing Its Uncertainty

*A reproduction-and-extension study on the UCI Beijing Multi-Site air-quality dataset.*

---

## Abstract

We reproduce **DeepAR** (Salinas et al., 2020), an autoregressive recurrent probabilistic
forecaster, on hourly fine-particulate (PM2.5) air-quality data and evaluate it against three
classical baselines — Seasonal Naive, ARIMA, and ETS — using proper probabilistic scoring rules
(CRPS, interval coverage, Expected Calibration Error, Winkler score). DeepAR is the most accurate
model by every accuracy metric (test CRPS **43.6 ± 1.7** over five seeds, versus 47.4 for ARIMA and
68.4 for Seasonal Naive) but is markedly **overconfident**: its nominal 90 % prediction intervals
cover only 74 % of realised values, and the miscalibration grows roughly five-fold from the one-hour
to the twenty-four-hour horizon. We then apply **post-hoc isotonic recalibration** (Kuleshov et al.,
2018) fitted on a held-out validation period. The headline finding is an honest one: the recalibration
does **not** improve test calibration, because DeepAR is already well-calibrated on the validation
distribution (validation ECE 0.014) while the test period — deep Beijing winter — has shifted beneath
it (test ECE 0.090). An oracle bound (fitting the same map on the test set, used for diagnosis only)
collapses test ECE to 0.020, proving the method is correct and isolating **distribution shift** as the
true obstacle. The practical lesson — always check calibration before deployment, and never assume a
recalibration fit on past data survives a distribution shift — is the contribution we emphasise. Every
figure, table, and number in this report is regenerated from a clean checkout by a single
`make reproduce`.

## 1. Introduction

Decisions made under uncertainty — when to issue a public-health advisory, how much reserve capacity
to hold, whether to trust a forecast at all — depend not on a single predicted number but on a
*calibrated distribution* over outcomes. A model that says "90 % chance PM2.5 stays below 150 µg/m³"
is only useful if, over many such statements, the truth really does fall in range 90 % of the time.
This is the property of **calibration**, and it is distinct from accuracy: a model can have excellent
point error and still be badly miscalibrated, promising more certainty than it delivers.

Deep probabilistic forecasters are now standard tools for this problem. DeepAR, in particular,
popularised the idea of an RNN that emits the parameters of a predictive distribution at each step and
generates forecasts by autoregressive sampling. But a recurring empirical observation — formalised by
Kuleshov et al. (2018) — is that such models are frequently *miscalibrated out of the box*, typically
overconfident, because their distributional parameters are point-estimated by maximum likelihood with
no allowance for epistemic uncertainty.

This project does three things. First, it **reproduces** DeepAR on a real, noisy, seasonal
environmental dataset and benchmarks it honestly against classical statistical baselines with proper
scoring rules — the kind of comparison without which a deep model's numbers are meaningless. Second,
it **diagnoses** DeepAR's calibration in detail, including how miscalibration evolves with the forecast
horizon. Third, it **attempts a post-hoc fix** (isotonic recalibration) under a strict no-leakage
protocol and reports the result faithfully, including the ways in which the fix fails and why. The
emphasis throughout is on correctness of evaluation and reproducibility over headline performance.

Air quality is a fitting domain for this study. PM2.5 concentration is heavy-tailed and strongly
seasonal — winter heating and temperature inversions in Beijing produce extreme episodes that a
Gaussian forecaster handles poorly — and the consequences of mis-stated uncertainty are concrete:
public-health advisories and exposure warnings are issued against thresholds, so an interval that is
too narrow translates directly into missed warnings. The domain also exhibits genuine **temporal
distribution shift** between seasons, which turns out to be the crux of our recalibration result. Two
notions of correctness recur throughout: *calibration*, that stated probabilities match empirical
frequencies, and *sharpness*, that intervals are as tight as calibration allows. Following Gneiting et
al. (2007), the goal is to maximise sharpness subject to calibration — never to trade away honesty about
uncertainty for a tighter-looking band.

## 2. Related work

**DeepAR** (Salinas et al., 2020) is the model under study: an autoregressive RNN that outputs, at each
time step, the parameters of a chosen likelihood (we use Student-*t*, whose heavier tails suit spiky
air-quality data) and forecasts by drawing sample paths. **Probabilistic calibration and sharpness**
were given their modern decision-theoretic footing by Gneiting et al. (2007), who argued for
"maximising sharpness subject to calibration." **Calibration of deep regression models** and a simple
recalibration recipe were introduced by Kuleshov et al. (2018), whose isotonic method we adopt.
**Conformalized quantile regression** (Romano et al., 2019) provides finite-sample coverage guarantees
under exchangeability and is the conformal counterpart to the isotonic approach. The **CRPS**, our
primary distributional score, originates with Matheson & Winkler (1976). Finally, the **Temporal Fusion
Transformer** (Lim et al., 2021) represents the modern attention-based alternative to DeepAR and is
named in our future-work discussion; we deliberately restrict this study to DeepAR to keep the
comparison clean.

## 3. Data

We use the **UCI Beijing Multi-Site Air-Quality** dataset (#501): hourly readings from twelve monitoring
stations over 2013–2017. For the core experiments we select a single station, **Aotizhongxin**, and the
target **PM2.5** (µg/m³); multi-station modelling is left to future work. The raw series has 35,064
hourly rows with 2.6 % missing PM2.5.

Preprocessing builds one **continuous hourly index** (no rows dropped, so the series stays at a regular
frequency that GluonTS requires). Gaps of up to six hours are forward-filled; longer gaps are left as
NaN and excluded from evaluation. We engineer cyclical encodings of hour-of-day, day-of-week and
month; lag features at t−1, t−24 and t−168; 24-hour rolling mean and standard deviation; and the
meteorological/pollutant covariates. Crucially, every engineered feature is **strictly backward-looking**
(lags and rolling windows are shifted by one step), so computing them on the full series before the
temporal split introduces **no future leakage**.

The engineered covariates include the meteorological channels (temperature, pressure, dew point,
rainfall, wind speed and a cyclical encoding of the 16-point wind direction) and the co-pollutants
(PM10, SO₂, NO₂, CO, O₃). These are exogenous and time-interpolated so the feature matrix has no holes,
while the target's own gaps are handled separately (short gaps filled, long gaps excluded) so we never
fabricate a label we then score against. Exploratory analysis confirmed the expected structure: a strong
daily cycle, a weekly component, a heavy right tail in PM2.5 (motivating the Student-*t* likelihood), and
pronounced winter peaks — the seasonal regime that later drives the calibration shift. The choice of a
single station (Aotizhongxin, a central urban site) keeps the reproduction tractable and the narrative
focused; the twelve-station structure of the dataset is what makes multi-site generalisation a natural
future extension rather than a missing piece.

The **temporal split** is strict and date-based: training 2013-03-01 → 2016-06-30 (29,232 h),
validation 2016-07-01 → 2016-12-31 (4,416 h), and test 2017-01-01 → 2017-02-28 (1,416 h). We assert in
the test suite that `max(train) < min(val) < min(test)` with no row overlap, and a `pandera` schema
enforces column types, non-negativity of PM2.5, plausible temperature/pressure ranges, and strict
timestamp monotonicity within each split. The time series with split boundaries is shown in
`docs/images/eda_series.png`. A synthetic data generator with the identical schema serves as a
fallback when the UCI download is unavailable; this run used the **real UCI data**.

## 4. Methods

**Forecast task.** Given 168 hours of history, forecast the next 24 hours as a full predictive
distribution, represented uniformly across all models by **100 sample paths**.

**DeepAR.** We wrap GluonTS's `DeepAREstimator` (PyTorch backend): context length 168, prediction
length 24, two LSTM layers of hidden size 40, dropout 0.1, learning rate 1e-3, Student-*t* output,
non-negative samples, 50 epochs. We train with fixed epochs and **no early stopping**, deliberately
leaving the validation split untouched so it can serve as an unbiased calibration set later. Models are
trained on a single GPU and reported over five seeds (42, 123, 456, 789, 1337).

**Classical baselines.** *Seasonal Naive* repeats the value 24 hours earlier, with intervals from
bootstrapped seasonal-difference residuals. *ARIMA* uses `pmdarima.auto_arima` (seasonal, period 24) to
select an order on a recent training slice, then a `statsmodels` SARIMAX walked forward across the test
set via cheap state extension (`append`). *ETS* uses the statespace `ETSModel` (additive
error/trend/seasonal, period 24) refit on a trailing window, with native simulation for sample paths
and graceful handling of convergence failures. To keep the comparison fair, **every** model — classical
and deep — emits 100 sample paths, so all metrics are computed by the same code.

**Evaluation.** A shared rolling-origin harness tiles the test set with 59 non-overlapping 24-hour
windows; for each window the model sees only history strictly before the origin (no peeking, no
overlap). Windows whose realised values contain a long-gap NaN are excluded by a mask. We report:

- **MAE** and **RMSE** on the predictive median — standard point error, RMSE penalising large misses.
- **CRPS** (Continuous Ranked Probability Score), our primary distributional score. For an ensemble of
  samples it is estimated as `mean(|y − xᵢ|) − ½·mean(|xᵢ − xⱼ|)`; it generalises MAE to distributions
  (lower is better) and rewards forecasts that are both accurate and appropriately spread.
- **Coverage@α**: the fraction of truths inside the central α interval (the (1−α)/2 and (1+α)/2 sample
  quantiles). A calibrated model has coverage ≈ α at every level.
- **ECE** (Expected Calibration Error): the mean absolute gap `|observed − predicted|` across nine
  nominal levels (0.1…0.9) — a single scalar summarising miscalibration.
- **Winkler** interval score: interval width plus a `(2/α)`-scaled penalty for truths that fall
  outside, capturing the sharpness-versus-coverage trade-off in one number.

Because all models emit 100 sample paths, every one of these is computed by the same `summarize_samples`
routine — there is no metric asymmetry between the deep and classical models.

**Recalibration.** We use Kuleshov isotonic recalibration. On the validation set we compute, for every
(window, step), the predictive CDF value at the realised truth (a PIT value); if the model were
calibrated these would be uniform. We fit an isotonic map `R` from PIT value to its empirical CDF and
recalibrate a test forecast by resampling through the inverse map, `x = F⁻¹(R⁻¹(u))`. The map is fit on
validation only and the transform never receives a label — a property checked by a leakage test.

**Reproducibility.** Seeds are set across Python/NumPy/PyTorch/Lightning; the data pipeline is a DVC
stage graph (download → preprocess → split); every run logs parameters, metrics, and provenance
(git SHA, dvc.lock hash, library versions) to MLflow; and `make reproduce` regenerates all artifacts
from a clean state.

## 5. Results

The full comparison (test set, DeepAR as mean ± std over five seeds):

| Model | MAE | RMSE | CRPS | Cov@50 | Cov@80 | Cov@90 | ECE |
|---|---|---|---|---|---|---|---|
| Seasonal Naive | 88.5 | 137.5 | 68.4 | 0.40 | 0.68 | 0.79 | 0.084 |
| ARIMA | 59.9 | 103.5 | 47.4 | 0.55 | 0.74 | 0.80 | 0.072 |
| ETS | 61.3 | 104.9 | 47.5 | 0.66 | 0.84 | 0.88 | 0.109 |
| **DeepAR (raw)** | **59.7** | **99.5** | **43.6 ± 1.7** | 0.36 | 0.62 | 0.74 | 0.125 |
| DeepAR (recalibrated) | 57.7 | 96.6 | 42.9 | 0.38 | 0.66 | 0.80 | 0.098 |

DeepAR has the best CRPS and RMSE of any model and ties ARIMA on MAE, confirming a successful
reproduction. But its coverage is the worst: nominal 90 % intervals capture only 74 % of truths
(`docs/images/calibration_before.png`), and the calibration curve lies below the diagonal at every
level — systematic overconfidence. The per-horizon analysis (`docs/images/calibration_per_horizon.png`)
shows ECE climbing from 0.032 at h+1 to 0.167 at h+24: the autoregressive sampler compounds an
underestimated per-step variance, so the predictive distribution is far too tight deep into the
horizon. Representative forecast windows with sample paths and shaded intervals are in
`docs/images/deepar_forecasts.png`, and point/distributional error growth with lead time in
`docs/images/performance_vs_horizon.png`.

Reading the table as a practitioner would: if you cared only about point accuracy you would pick
DeepAR and stop. If you cared about *trustworthy* 90 % intervals you would be misled — DeepAR's are the
least trustworthy of the five, despite its best CRPS, because CRPS rewards sharpness and DeepAR is sharp
to a fault. ETS, the weakest deep-accuracy competitor, has the best 90 % coverage (0.88) precisely
because its intervals come from an explicit estimated noise process. This tension — accurate but
overconfident versus less accurate but honest — is the entire reason calibration must be measured
separately from accuracy, and it is why a CRPS-only leaderboard would hide the problem this project
exists to surface.

The centerpiece is `docs/images/calibration_comparison.png`, the raw-versus-recalibrated calibration
curves. The recalibrated curve is essentially coincident with the raw one — recalibration moved test
ECE from 0.090 to 0.098 and 90 % coverage from 0.804 to 0.797, i.e. it did not help. The reason is made
precise by two diagnostics: validation ECE before recalibration is already **0.014** (DeepAR is
calibrated in-distribution, so the map is near-identity), while an oracle that fits the map on the test
set itself drives test ECE down to **0.020**. The method works; the validation-fit transfer fails.

## 6. Discussion

**When does DeepAR win, and when lose?** On accuracy it wins everywhere — heavier-tailed Student-*t*
likelihood plus learned temporal structure beat the linear-Gaussian baselines on the spiky winter test
period. On calibration it loses: the explicit statistical models (especially ETS) produce
better-covered intervals because their uncertainty is derived from an estimated noise process rather
than point-estimated network outputs.

**The calibration finding.** DeepAR's overconfidence is real and horizon-dependent. The natural fix —
post-hoc recalibration on held-out data — fails here, and the failure is informative. Because the
validation period (latter half of 2016) is calibrated while the test period (January–February 2017,
deep winter, the heaviest pollution episodes) is not, a recalibration map estimated on validation has
nothing to learn and cannot anticipate the shift. The oracle bound (0.090 → 0.020) confirms that a map
fit on test-like data *would* fix the problem; the obstacle is **distribution shift**, not a broken
method. This is the same lesson that recurs across applied ML — performance and calibration both
degrade when the live distribution drifts from the reference distribution used to tune the system.

**The sharpness trade-off.** Recalibration widened the mean 90 % interval slightly (156 → 164), as
expected, but not enough to lift coverage, since the map was near-identity. Interestingly the
recalibrated median was a touch more accurate (MAE 59.7 → 57.7), a benign side effect of resampling.

**Why the shift defeats the fix — mechanically.** Isotonic recalibration learns a monotone map from the
model's predictive CDF to observed frequencies on the calibration set. If the calibration set is already
calibrated (validation ECE 0.014), that map is essentially the identity, and applying the identity to
the test forecasts leaves them unchanged. The map has no way to encode "but the test period will be
harder," because it never sees the test period. The oracle — fitting the map on the test set's own PIT
values — recovers the correction (ECE 0.090 → 0.020) precisely because it observes the shifted
distribution. The gap between the honest result and the oracle is therefore a direct, quantified measure
of how much the val→test shift costs: it is the entire difference between a near-identity correction and
the one the test period actually needs. A practitioner cannot use the oracle (it requires test labels),
but the comparison localises the failure unambiguously to distribution shift rather than to the
recalibration algorithm.

**Implication for practitioners.** Check calibration before deploying any probabilistic model; report
coverage and ECE alongside accuracy; and treat a recalibration map as a perishable artifact that must be
re-estimated as the data distribution moves. The same principle motivates production drift monitoring:
a calibration map, like a trained model, has a shelf life bounded by the stationarity of the data.

## 7. Limitations & Future Work

This study is deliberately narrow. (1) **Distribution shift / non-stationarity** is the dominant
limitation: marginal recalibration assumes the calibration set is exchangeable with test, which fails
across seasons; **online or shift-aware recalibration** that re-fits as new data arrives is the obvious
next step. (2) The recalibration is **marginal, not conditional** — even a working map would not
guarantee calibration conditional on, say, high- versus low-pollution regimes. (3) We use a **single
station and single target**; cross-station and multi-pollutant generalisation are untested. (4)
**Validation representativeness** bounds recalibration quality, and we have no in-distribution
calibration data for the test period without leaking the future. (5) **Five seeds** is honest but not
exhaustive, and GPU non-determinism produced run-to-run variation in single-seed coverage (0.74–0.80).
(6) Hyperparameters were **fixed, not tuned**, by design. Beyond fixing these, natural extensions include
alternative deep forecasters (Temporal Fusion Transformer, N-BEATS, PatchTST) and conformalized quantile
regression with finite-sample guarantees.

## 8. Conclusion

We reproduced DeepAR on real air-quality data, showed it to be the most accurate but least calibrated
forecaster in a fair probabilistic comparison, diagnosed its horizon-dependent overconfidence, and
applied a correct post-hoc recalibration that nonetheless failed to improve test calibration because of
a validation-to-test distribution shift — a conclusion we substantiate with an oracle bound rather than
assert. The repository reproduces every result from a single command, with strict temporal splits,
proper scoring rules, and experiment tracking throughout.

## 9. References

1. Salinas, D., Flunkert, V., Gasthaus, J., & Januschowski, T. (2020). *DeepAR: Probabilistic
   forecasting with autoregressive recurrent networks.* International Journal of Forecasting, 36(3),
   1181–1191.
2. Gneiting, T., Balabdaoui, F., & Raftery, A. E. (2007). *Probabilistic forecasts, calibration and
   sharpness.* Journal of the Royal Statistical Society: Series B, 69(2), 243–268.
3. Kuleshov, V., Fenner, N., & Ermon, S. (2018). *Accurate uncertainties for deep learning using
   calibrated regression.* ICML 2018.
4. Romano, Y., Patterson, E., & Candès, E. J. (2019). *Conformalized quantile regression.* NeurIPS 2019.
5. Lim, B., Arık, S. Ö., Loeff, N., & Pfister, T. (2021). *Temporal Fusion Transformers for
   interpretable multi-horizon time series forecasting.* International Journal of Forecasting, 37(4),
   1748–1764.
6. Matheson, J. E., & Winkler, R. L. (1976). *Scoring rules for continuous probability distributions.*
   Management Science, 22(10), 1087–1096.
