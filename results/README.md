# Research Results

This directory contains the visualisations and numerical outputs generated during the dissertation research.

## Figures

The `figures/` directory contains five visualisations:

1. **ROC Curves:** Comparison of predictive discrimination across the three models.
2. **AUC Equivalence:** Bootstrap-based comparison of model AUC differences against the selected equivalence margin.
3. **Repeated Partition Analysis:** Test AUC performance across twenty stratified data partitions.
4. **Global SHAP Importance:** Comparison of relative feature attribution across the three models.
5. **Local SHAP Comparison:** Individual applicant explanations across the three models.

## Tables

The `tables/` directory contains the numerical outputs supporting the research, including:

- Model performance metrics and XGBoost tuning results.
- AUC intervals and equivalence testing.
- Repeated partition and sensitivity analysis.
- Global SHAP importance, explanation agreement and stability.
- Local applicant explanations.
- Approval-rate simulations and decision overlap.
- Dataset sample construction and special-code evidence.

## Interpretation

The results should be interpreted in the context of the research methodology and limitations described in the dissertation. Simulated lending decisions are not observed real-world lending outcomes, and SHAP values represent model attributions rather than causal explanations.
