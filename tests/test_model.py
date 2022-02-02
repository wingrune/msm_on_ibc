import pytest

from msm import model
from nilearn import datasets
import numpy as np

# from sklearn.utils.estimator_checks import check_estimator


def test_create_model():
    """Instantiate model without crash"""

    try:
        _ = model.MSM()
    except Exception:
        pytest.fail("Could not instantiate model")


def test_align_random_data():
    """Instantiate model without crash"""

    m = model.MSM()

    # generate random maps
    fs3 = datasets.fetch_surf_fsaverage(mesh="fsaverage3")
    n_voxels = 642
    source_train_data = np.ones((2, n_voxels))
    target_train_data = np.ones((2, n_voxels))

    # align maps
    m.fit(source_train_data, target_train_data, mesh_file=fs3.sphere_left)

    # align new data from source to target
    source_test_data = np.random.rand(4, n_voxels)
    predicted_data = m.transform(source_test_data)
    assert source_test_data.shape == predicted_data.shape

    # check score
    s = m.score(source_test_data, predicted_data)
    assert isinstance(s, int) or isinstance(s, float)


# def test_model_is_sklearn_estimator():
#     """Model should have sklearn compatible API"""
#
#     check_estimator(model.MSM())
