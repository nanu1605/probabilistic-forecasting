# The Calibration Experiment: Diagnosing and (Trying to) Fix DeepAR's Overconfidence

*Probabilistic PM2.5 forecasting on the Beijing Multi-Site air-quality dataset (UCI #501),
station Aotizhongxin. This document is the technical heart of the project: it diagnoses the
calibration of a reproduced DeepAR model, attempts a post-hoc fix, and reports — honestly — what
worked, what did not, and why.*

---

## 1. Hypothesis

Deep probabilistic forecasters such as DeepAR (Salinas et al., 2020) output the parameters of a
parametric predictive distribution at each step (here a Student-*t*). Those parameters are learned
as point estimates by maximising likelihood, with no mechanism to represent uncertainty *about* the
uncertainty. The widely reported consequence (Kuleshov et al., 2018) is **overconfidence**:
prediction intervals that are too narrow, so their empirical coverage falls short of nominal.

Our working hypothesis was twofold:

1. **Diagnosis.** DeepAR, evaluated on held-out air-quality data, will be miscalibrated —
   specifically overconfident — and the miscalibration will worsen at longer forecast horizons.
2. **Fix.** A post-hoc recalibration map (Kuleshov isotonic recalibration) fitted on a held-out
   validation set, *without retraining and without ever touching test labels*, will move the
   model's intervals toward nominal coverage and reduce the Expected Calibration Error (ECE).

The first half held cleanly. The second half produced a more interesting — and more honest — result
than "it worked," which we document in full.

## 2. Experimental setup

- **Data.** Hourly PM2.5 at one station, 2013-03-01 → 2017-02-28 (35,064 rows). One continuous
  hourly index; gaps ≤6 h forward-filled, longer gaps left NaN and excluded from evaluation.
- **Temporal split (strict, no leakage).** Train 2013-03-01 → 2016-06-30 (29,232 h);
  **Validation** 2016-07-01 → 2016-12-31 (4,416 h); **Test** 2017-01-01 → 2017-02-28 (1,416 h).
  `max(train) < min(val) < min(test)` is asserted in the test suite.
- **Model.** GluonTS DeepAR — context 168 h (1 week), horizon 24 h (1 day), 2 LSTM layers,
  hidden 40, dropout 0.1, Student-*t* output, 50 epochs, trained on GPU. Reported over **5 seeds**.
- **Evaluation protocol.** Rolling origin: 59 non-overlapping 24-hour windows tile the test set;
  every model (classical and deep) emits **100 sample paths** per window, so all metrics — MAE,
  RMSE, CRPS, coverage at nominal levels, ECE, Winkler — are computed identically and the comparison
  is apples-to-apples.
- **Calibration set.** The validation split is reserved *exclusively* for fitting the recalibration
  map; DeepAR uses fixed epochs (no early stopping) so validation is never seen during training.

## 3. Baseline context: classical intervals are better calibrated

| Model | MAE | RMSE | CRPS | Cov@50 | Cov@80 | Cov@90 | ECE |
|---|---|---|---|---|---|---|---|
| Seasonal Naive | 88.5 | 137.5 | 68.4 | 0.40 | 0.68 | 0.79 | 0.084 |
| ARIMA | 59.9 | 103.5 | 47.4 | 0.55 | 0.74 | 0.80 | 0.072 |
| ETS | 61.3 | 104.9 | 47.5 | 0.66 | 0.84 | 0.88 | 0.109 |
| **DeepAR (raw, 5 seeds)** | **59.7** | **99.5** | **43.6 ± 1.7** | 0.36 | 0.62 | 0.74 | 0.125 |

DeepAR is the **most accurate** model on every point and distributional accuracy metric — its CRPS
of 43.6 beats ARIMA/ETS (≈47.4) and crushes Seasonal Naive (68.4). But it is also the **least
calibrated**: its 90 % intervals cover only 74 % of truths, versus 0.80 for ARIMA and 0.88 for ETS.
This is the expected pattern: classical models derive intervals from an explicit statistical noise
model, whereas DeepAR's intervals come from learned point-estimated distribution parameters. Sharp
and accurate, but overconfident.

## 4. The miscalibration diagnosis

The calibration curve (`docs/images/calibration_before.png`) plots observed coverage against
predicted coverage. DeepAR sits **below the diagonal at every level** — the signature of systematic
overconfidence:

| Predicted | 0.5 | 0.8 | 0.9 |
|---|---|---|---|
| Observed (DeepAR raw) | 0.36 | 0.62 | 0.74 |

Every nominal interval is too narrow. The aggregate ECE over the 5 seeds is **0.125**.

Crucially, the miscalibration **worsens monotonically with the forecast horizon**
(`docs/images/calibration_per_horizon.png`):

| Horizon | h+1 | h+6 | h+12 | h+24 |
|---|---|---|---|---|
| ECE | 0.032 | 0.060 | 0.116 | 0.167 |

At the one-hour horizon DeepAR is nearly calibrated (ECE 0.03); by 24 hours ahead the error is five
times larger. This is intuitive: the autoregressive sampler compounds an underestimated per-step
variance, so the predictive distribution is far too tight deep into the horizon. **Why** this happens
is the heart of it — the Student-*t* location, scale and degrees-of-freedom are point estimates; the
model has no way to express "I am unsure about my own scale," so it systematically underestimates σ.

## 5. The recalibration result (the honest part)

**Method.** We use Kuleshov et al. (2018) isotonic recalibration. On the validation set we compute,
for every (window, step), the model's predictive CDF value at the realised truth —
`p = fraction of samples ≤ y` (a PIT value). If the model were calibrated, the `p`'s would be
Uniform[0,1]. We fit an isotonic regression `R` mapping each `p` to its empirical CDF, then recalibrate
a test forecast by resampling: draw `u ~ U(0,1)`, map it through the inverse `R⁻¹`, and read off the
model's empirical quantile at that adjusted level, `x = F⁻¹(R⁻¹(u))`. The map is fit on validation
only; `transform` never receives a label (verified by a leakage test that permuting test labels cannot
change the output).

**What happened.** On the test set, recalibration **did not help**:

| | ECE | Cov@90 | CRPS | Width@90 |
|---|---|---|---|---|
| DeepAR raw (seed 1337) | 0.090 | 0.804 | 42.05 | 156.2 |
| DeepAR recalibrated | 0.098 | 0.797 | 42.87 | 164.3 |

The recalibrated curve in `docs/images/calibration_comparison.png` is essentially on top of the raw
curve — a small *increase* in ECE, not the hoped-for collapse onto the diagonal.

**Why — and how we know it is not a bug.** The decisive numbers are the in-distribution checks:

- **Validation ECE before recalibration is already 0.014.** DeepAR is *well calibrated on the
  validation period*. The isotonic map therefore learns something very close to the identity — there
  is nothing on validation for it to correct — so it can do almost nothing to the test forecasts.
- **An oracle upper bound confirms the method is correct.** If we (illegitimately, for diagnosis only)
  *fit* the same recalibration map on the test set and evaluate on test, ECE drops from **0.090 to
  0.020**. The machinery works; it simply needs a calibration set that resembles the test set.

The gap between validation calibration (ECE 0.014) and test calibration (ECE 0.090) is the whole
story: it is a **distribution shift**. The validation period is the second half of 2016; the test
period is January–February 2017 — deep Beijing winter, the season of the heaviest and most
heavy-tailed PM2.5 episodes. DeepAR is calibrated on the data distribution it was tuned against and
becomes overconfident on the shifted, harder winter regime. A recalibration map estimated on the
calmer validation distribution cannot anticipate that shift. This is precisely the failure mode the
calibration literature warns about: post-hoc recalibration assumes the calibration set is exchangeable
with the test set, and under temporal non-stationarity that assumption breaks.

This mirrors the central lesson of the companion MLOps project — model behaviour drifts when the live
distribution moves away from the reference distribution — here surfacing as a *calibration* failure
rather than an accuracy one.

## 6. The sharpness trade-off

Recalibration is supposed to trade sharpness for coverage: widen intervals until they cover. We do
see the widening — the mean 90 % interval width grows from **156.2 to 164.3** (+5 %) — but because the
map is near-identity the widening is too small to lift test coverage (0.804 → 0.797, within noise).
Interestingly, the recalibrated forecast's **point accuracy improved** (MAE 59.7 → 57.7, RMSE
96.6), a side effect of resampling smoothing the predictive median; CRPS was essentially unchanged
(42.05 → 42.87). The ideal — tight intervals that are also calibrated — is unreachable here not
because of a sharpness/coverage tension but because the calibration signal itself does not transfer.

## 7. Failure modes and what broke during development

- **Recalibration non-transfer (the headline failure).** Documented above: the honest, no-leakage fix
  fails on test under distribution shift, even though the method is provably correct (oracle 0.090 →
  0.020). We report it rather than tune around it.
- **DeepAR evaluation was 50× too slow at first.** The initial design called `predictor.predict` once
  per window (295 calls across 5 seeds), each spinning up Lightning machinery. Batching all 59 window
  contexts into a single multi-series `PandasDataset` per seed cut evaluation from minutes to seconds.
- **MLflow 3.x removed the file-store backend.** `file:./mlruns` raised `MlflowException: filesystem
  tracking backend … is in maintenance mode`. Switched the tracking URI to a local SQLite backend.
- **A non-negativity clip nearly faked a result.** The recalibrator clips samples at 0 (PM2.5 ≥ 0).
  In unit tests on standard-normal synthetic data this clipping destroyed the lower tail and *inflated*
  ECE; the fix was to make non-negativity an explicit, dataset-specific flag rather than a default.

## 8. Honest limitations

1. **Distribution shift / non-stationarity.** The validation and test periods are different seasons;
   marginal recalibration assumes they are exchangeable. They are not, and that is exactly why the fix
   does not transfer. A shift-aware or online recalibration scheme (re-fit as new data arrives) is the
   natural remedy — future work.
2. **Marginal, not conditional, calibration.** The map is fit by pooling across all windows and
   horizons. Even a successful map would only guarantee calibration *on average*, not conditional on,
   say, high- vs low-pollution days or short vs long horizons (where we *know* miscalibration differs).
3. **Single station, single target.** Only Aotizhongxin PM2.5. Cross-station and multi-pollutant
   generalisation is untested.
4. **Validation-set representativeness.** Recalibration quality is bounded by how well validation
   resembles test; here it does not, and we have no in-distribution calibration data for the test
   period without leaking the future.
5. **Five seeds is honest but not exhaustive.** Run-to-run GPU non-determinism is real: single-seed
   test Cov@90 ranged 0.74–0.80 across runs, so single-seed calibration numbers carry meaningful
   variance. Headline metrics are reported as mean ± std; the recalibration before/after pair is
   computed within one self-consistent run.
6. **Fixed hyperparameters.** Per project scope we did not tune DeepAR; better-calibrated
   configurations may exist but searching for them was out of scope.

## 9. Takeaway

DeepAR delivers the best accuracy and sharpness in this study but is overconfident, increasingly so
with horizon. Post-hoc isotonic recalibration is a sound, correctly-implemented fix — proven by the
oracle bound — yet it fails to help on the test set because the model is already calibrated on the
validation distribution and the test period has shifted beneath it. The practitioner lesson is the
one worth carrying: **always check calibration before deploying a probabilistic model, and never
assume a recalibration fit on yesterday's data will hold under tomorrow's distribution.**
