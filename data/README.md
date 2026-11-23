# Dataset

## Dataset Source

This project uses the **FICO Explainable Machine Learning Challenge HELOC dataset**, which contains anonymised credit-risk information used to study credit-scoring models.

The dataset includes:

- **Original observations:** 10,459
- **Predictor variables:** 23 numerical features
- **Target variable:** `RiskPerformance`
- **Target classes:** Good / Bad

## Special Values

The dataset uses specific numerical codes to represent unavailable or unmet credit-information conditions:

| Code | Meaning |
|---|---|
| `-9` | No Bureau Record |
| `-8` | No Usable/Valid Trades |
| `-7` | Condition not Met |

These values were considered during data preparation and preprocessing.

## Analytical Sample

The final analytical sample contains **9,870 observations**.

The reduction from the original dataset resulted from excluding:

- 588 observations with no bureau record across all 23 predictors.
- 1 duplicate observation.

## Data Availability

The original dataset is not included in this repository.

Users wishing to reproduce the analysis must obtain the HELOC dataset from its original provider and comply with the applicable access and usage terms.

## Usage in This Project

The dataset was used for:

- Exploratory data analysis
- Credit-risk classification
- Logistic Regression, Random Forest and XGBoost modelling
- Predictive performance evaluation
- SHAP-based explainability analysis
- Statistical equivalence and sensitivity analysis

Refer to [`../scripts/README.md`](../scripts/README.md) for the analysis scripts and [`../results/README.md`](../results/README.md) for the generated outputs.
