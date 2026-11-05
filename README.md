# articles-DiagTriage

Conformal triage with abstention: a classifier decides automatically only when its prediction set has one label,
otherwise the case goes to an expert. Real data: UCI Cardiotocography (2126 fetal records, normal / suspect / pathologic).
TRL 4 (public real data, laboratory validation only).

```
pip install -e ".[dev]" && python -m pytest
# data: https://archive.ics.uci.edu/dataset/193 (not committed)
python research/run_study.py /path/to/CTG.xls
```
