from nilearn import datasets
import os
from tempfile import TemporaryDirectory

from msm import utils


def test_check_fsl():
    """Should set env variables concerning FSL without failing"""

    FSLDIR, FSL_CONFIG_PATH = utils.check_fsl()
    assert os.getenv("FSLDIR") == FSLDIR
    assert os.getenv("FSL_CONFIG_PATH") == FSL_CONFIG_PATH


def test_ungzip():
    """Should create ungzipped file from nilearn"""

    with TemporaryDirectory() as tmp_dir:
        fs5 = datasets.fetch_surf_fsaverage()
        mesh_path = fs5.pial_right
        tmp_mesh_path = os.path.join(tmp_dir, os.path.basename(mesh_path[:-3]))
        utils.ungzip(mesh_path, tmp_mesh_path)
        assert os.path.exists(tmp_mesh_path)


def test_gifti_from_file():
    """Should correctly load fsaverage5 gifti file from nilearn."""

    fs5 = datasets.fetch_surf_fsaverage()
    mesh = utils.gifti_from_file(fs5.pial_left)
    assert mesh.darrays[0].data.shape == (10242, 3)
