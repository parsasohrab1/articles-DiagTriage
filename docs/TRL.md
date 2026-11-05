# TRL assessment – DiagTriage

**Claimed level: TRL 4** (laboratory validation on one public real dataset). Not TRL 5: no clinical environment, no prospective data, single-source records.

Data: UCI Cardiotocography, n = 2126 (normal 1655, suspect 295, pathologic 176). Mean over 50 random splits. "Auto missed" = share of pathologic cases decided automatically as normal (singleton set = normal).

| Model | Rule | Coverage | Pathologic coverage | Referral rate | Accuracy on automatic | Pathologic auto-missed |
|---|---|---|---|---|---|---|
| logistic | argmax | 0.895 | 0.743 | 0.000 | 0.895 | 0.089 |
| logistic | marginal_0.05 | 0.949 | 0.820 | 0.135 | 0.944 | 0.056 |
| logistic | classwise_0.05 | 0.951 | 0.955 | 0.275 | 0.955 | 0.011 |
| logistic | marginal_0.1 | 0.901 | 0.746 | 0.029 | 0.907 | 0.077 |
| logistic | classwise_0.1 | 0.902 | 0.916 | 0.095 | 0.904 | 0.019 |
| forest | argmax | 0.933 | 0.834 | 0.000 | 0.933 | 0.071 |
| forest | marginal_0.05 | 0.951 | 0.849 | 0.049 | 0.952 | 0.058 |
| forest | classwise_0.05 | 0.954 | 0.960 | 0.189 | 0.952 | 0.006 |
| forest | marginal_0.1 | 0.901 | 0.749 | 0.058 | 0.956 | 0.030 |
| forest | classwise_0.1 | 0.902 | 0.913 | 0.058 | 0.921 | 0.005 |

Plain argmax accuracy: logistic 0.895, forest 0.933.

## Findings and caveats
- Marginal conformal sets reach the nominal average coverage but **not** for the rare pathologic class (0.75 at alpha 0.1; 0.82–0.85 at alpha 0.05) – the average guarantee hides the clinically important failure.
- Class-wise (Mondrian) calibration restores pathologic coverage (0.91–0.96) at the price of more referrals: 5.8–9.5 % at alpha 0.1 and 19–28 % at alpha 0.05.
- With the forest at alpha 0.1, class-wise calibration lowers the pathologic auto-missed rate from 7.1 % (argmax) to 0.5 %, with 5.8 % referrals.
- Logistic regression needs considerably more referrals than the forest for the same guarantee.
- Coverage guarantees assume exchangeable calibration and test records; records come from one source and several CTG tracings may come from the same pregnancy (not identified in the data), so true out-of-source coverage may be lower.
- Class-wise guarantees are per class given the true class; with 176 pathologic records, the per-split calibration set contains only about 44 of them and the reported value is an average over splits.
