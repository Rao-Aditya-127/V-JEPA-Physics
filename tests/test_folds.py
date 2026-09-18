import numpy as np

from vjepa_physics.folds import check_folds, grouped_folds, random_folds

# Same structure as the speed dataset: 64 values x 24 clips each.
SPEEDS = np.repeat(np.linspace(0.25, 4.0, 64), 24)


def test_grouped_folds_never_leak_a_speed_value():
    folds = grouped_folds(SPEEDS, n_folds=5, val_every=5)
    check_folds(folds, len(SPEEDS), SPEEDS)     # raises on any leak or overlap


def test_every_test_fold_spans_the_full_label_range():
    """Interleaving: a fold that got only slow clips would have a deflated R^2."""
    for fold in grouped_folds(SPEEDS):
        held_out = np.unique(SPEEDS[fold.test])
        assert held_out.min() < 0.6 and held_out.max() > 3.7
        assert 12 <= len(held_out) <= 13


def test_inner_validation_is_about_a_fifth_of_training():
    for fold in grouped_folds(SPEEDS):
        n_train = len(fold.fit) + len(fold.val)
        assert 0.15 < len(fold.val) / n_train < 0.25


def test_random_folds_partition_but_share_values():
    folds = random_folds(len(SPEEDS), n_folds=5, seed=0)
    check_folds(folds, len(SPEEDS))
    shared = set(SPEEDS[folds[0].fit]) & set(SPEEDS[folds[0].test])
    assert len(shared) > 50        # the point of this control: test values were seen in training
