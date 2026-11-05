# Software Requirements Specification – DiagTriage

- SR-1 `data.load_ctg` reads the UCI CTG workbook (sheet `Data`), 21 CTG features, target NSP mapped to 0/1/2.
- SR-2 `conformal.calibrate_marginal` – split conformal threshold on 1 - p(true class), pooled; `calibrate_classwise` – one threshold per true class (Mondrian).
- SR-3 `conformal.prediction_sets` builds boolean label sets; `triage` marks empty and multi-label sets as referrals.
- SR-4 `research/run_study.py`: 50 stratified random splits 50 % train / 25 % calibration / 25 % test, logistic regression and a 300-tree random forest, alpha 0.05 and 0.1; baseline = plain argmax.
- SR-5 Three unit tests (marginal coverage, class-wise coverage, triage flags) on synthetic probabilities.
