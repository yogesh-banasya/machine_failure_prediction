# Machine Failure Prediction (Predictive Maintenance)

A beginner-friendly machine learning project that predicts whether a machine will fail, using sensor readings. Predicting failures early lets maintenance teams fix equipment *before* it breaks, which reduces downtime and cost. This is a common problem in industries such as oil & gas, manufacturing and energy.

## Problem

**Binary classification:** given a machine's sensor readings, predict `1` (will fail) or `0` (will not fail).

## Dataset

`machine_sensor_data.csv` contains 5,000 machines. About 8.9% of them fail, so the data is **imbalanced**.

| Column | Description |
|---|---|
| `temperature_c` | Operating temperature |
| `vibration_mm_s` | Vibration level |
| `pressure_bar` | Pressure |
| `rotation_rpm` | Rotation speed |
| `hours_since_service` | Running hours since last maintenance |
| `failure` | Target: 1 = failed, 0 = ok |

> **Note:** The data is **simulated** so the project is self-contained. The relationships are realistic (hotter, more vibrating, less recently serviced machines fail more often). The same pipeline can be applied to a real dataset such as the public [AI4I 2020 Predictive Maintenance dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset) by changing the column names.

## Approach

1. Load and explore the data (class balance, feature distributions)
2. Split into 80% train / 20% test, **stratified** to keep the failure ratio equal in both
3. Train two models:
   - **Logistic Regression** (baseline, with feature scaling)
   - **Random Forest** (200 trees, max depth 6)
   - Both use `class_weight="balanced"` to handle the rare failure class
4. Evaluate on the unseen test set with accuracy, precision, recall, F1-score and a confusion matrix
5. Inspect feature importance
6. Predict on new machines

## Results (test set: 1,000 machines, 89 real failures)

| Model | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|
| Logistic Regression | 0.877 | 0.410 | 0.865 | 0.556 |
| **Random Forest** | **0.917** | **0.524** | 0.742 | **0.614** |

**Random Forest confusion matrix:** 66 failures caught, 23 missed, 60 false alarms, 851 healthy machines correctly left alone.

### Key takeaways

- **Accuracy alone is misleading.** A model that always predicts "no failure" would already score about 91%, so precision, recall and F1 matter more here.
- **Precision vs recall trade-off:** Logistic Regression catches more real failures (higher recall) but raises many more false alarms (lower precision). Random Forest gives a better overall balance.
- **Most important sensors:** vibration, temperature and hours since service drive the Random Forest's predictions. Pressure and rotation speed matter much less.

## How to run

```bash
pip install pandas scikit-learn seaborn matplotlib jupyter
jupyter notebook machine_failure_beginner.ipynb
```

Keep `machine_sensor_data.csv` in the same folder as the notebook.

## Project structure

```
├── machine_failure_beginner.ipynb   # full walkthrough with explanations
├── machine_sensor_data.csv          # dataset
└── README.md
```

## Possible improvements

- Use a real dataset (e.g. AI4I 2020)
- Add cross-validation and hyperparameter tuning
- Try gradient boosting models
- Adjust the decision threshold to favour recall
- Use time-series sensor history instead of single snapshots

## Tech stack

Python, pandas, scikit-learn, matplotlib, seaborn, Jupyter

## Author

Yogesh
