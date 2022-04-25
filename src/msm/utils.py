import gzip
import logging
import nibabel as nib
import os
from tempfile import TemporaryDirectory
import shutil


def log_subprocess_output(pipe, err=False):
    logging.getLogger("msm")
    for line in iter(pipe.readline, b""):
        if err:
            logging.warning(line.decode("utf-8"))
        else:
            logging.info(line.decode("utf-8"))


def check_fsl():
    fsl_bin_path = shutil.which("fsl")
    if fsl_bin_path is None:
        raise ("FSL is not installed or is not in PATH")
    else:
        fsl_path = os.path.join(os.path.dirname(fsl_bin_path), "../")
        fsl_config_path = os.path.join(
            fsl_path, "config/basic_configs/config_standard_MSM_strain"
        )
        os.environ["FSLDIR"] = fsl_path
        os.environ["FSL_CONFIG_PATH"] = fsl_config_path

        return fsl_path, fsl_config_path


def gifti_from_file(mesh_path):
    """Load nibabel Gifti object from file path."""

    with TemporaryDirectory() as tmp_dir:
        # If mesh is gzipped, create a new file
        # with uncompressed data and load mesh from this file instead
        if mesh_path.endswith(".gz"):
            tmp_mesh_path = os.path.join(
                tmp_dir, os.path.basename(mesh_path[:-3])
            )
            ungzip(mesh_path, tmp_mesh_path)
            mesh = nib.load(tmp_mesh_path)
        else:
            mesh = nib.load(mesh_path)

        return mesh


def ungzip(input_path, output_path):
    with gzip.open(input_path, "rb") as f_in:
        with open(output_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

    return output_path
