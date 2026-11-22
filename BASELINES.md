

# Baseline protocol

The standalone tree baselines use XGBoost 2.1.4, LightGBM 4.6.0, and CatBoost
1.2.10. The deep tabular baselines are run from TALENT commit
`08301d670a7c854bcf3a73298763484ba58eecdb`; T-MLP uses the authors' released
implementation and recommended fixed configuration.

No method receives dataset-specific hyperparameter search. Model checkpointing
and early stopping use only the validation subset of each outer fold. Every
method uses the same five outer folds, fold-local preprocessing scope, and
fold-specific random seed. For each tree backbone, the standalone tree and its
MedRIFT counterpart use the same fixed tree configuration.

Third-party source code is not redistributed here. It should be obtained from
the respective authors' repositories under their original licenses.

