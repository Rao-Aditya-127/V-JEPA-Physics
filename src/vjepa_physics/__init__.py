"""Probing physical variables in a frozen V-JEPA 2 encoder.

Pipeline (see think/01-SPEED-TASK.md):

    video.py     mp4 -> RGB uint8 frames           (decode)
    encoder.py   frames -> 25 pooled hidden states (preprocess + forward + pool)
    extract.py   whole dataset -> cached features on disk
    features.py  cached features -> arrays, layer indexing
    folds.py     grouped cross-validation splits
    probes.py    linear probes and the hyperparameter sweep
    plots.py     figures
"""
