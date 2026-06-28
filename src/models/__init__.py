"""Phase 2 Intelligence Layer — model training framework.

Module overview
---------------
labels.py              — Label enum, TripleBarrierLabeler, purge_and_embargo
dataset.py             — WalkForwardSplitter, DatasetBuilder
quantile_cone.py       — QuantileConeTrainer (LightGBM quantile regression)
scenario_classifier.py — ScenarioClassifier (LightGBM multi-class)
calibration.py         — ModelCalibrator, UncalibratedConfidenceError
training_pipeline.py   — run_training() — single triggerable step

Build status
------------
Framework: built and tested on synthetic data (Checkpoints 2.3 + 2.4).
Training:  deferred until real Dhan data lands. Call run_training() once.

See DESIGN.md §5 (model architecture) and §13 (anti-leakage table).
"""
