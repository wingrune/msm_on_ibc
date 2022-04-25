import logging
import nibabel as nib
import numpy as np
import os
import pandas as pd
from pathlib import Path
from tempfile import TemporaryDirectory

from msm import utils


def is_same_coordsys(c1, c2):
    return (
        c1.dataspace == c2.dataspace
        and c1.xformspace == c2.xformspace
        and np.all(c1.xform == c2.xform)
    )


def prepare_darrays(darrays, coordsys):
    for d in darrays:
        d.data = d.data.astype(np.float32)
        d.datatype = nib.nifti1.data_type_codes.code["NIFTI_TYPE_FLOAT32"]
        d.intent = nib.nifti1.intent_codes.code["NIFTI_INTENT_POINTSET"]
        if d.coordsys is not None and not is_same_coordsys(
            d.coordsys, coordsys
        ):
            raise ValueError(
                "Provided data is in different coordsys than the mesh."
            )
        d.coordsys = coordsys

    return darrays


def run_msm(
    source_contrasts_list,
    source_mesh,
    target_contrasts_list,
    target_mesh=None,
    epsilon=None,
    iterations=None,
    **kwargs,
):
    """Run MSM on a list of contrast between in data and ref data

    Parameters
    ----------
    source_contrasts_list, target_contrasts_list : list of str
        Data used as features for the registration. The data should be provided
        as a list of GIFTI files, that will be merged to perform
        the multimodale registration.
    source_mesh, target_mesh : str
        Spherical mesh on which all data from source_contrasts_list or
        target_contrasts_list live. The mesh should be given as a GIFTI file.
        Note that if target_mesh is not specified,
        the source_mesh will be used for all input data.
    epsilon: float or None
        Regularization parameter
    iterations: int or str or None
        Number of iterations
        examples: 5 or "5,2,3,4"

    Returns
    -------
    mesh_gii : nibabel.gifti.GiftiImage
        Image holding the transformed mesh.
    transformed_gii : nibabel.gifti.GiftiImage
        Image holding the transformed data in the target_mesh.
    """
    FSLDIR, FSL_CONFIG_PATH = utils.check_fsl()
    logger = logging.getLogger("msm")
    logger.info(f"FSLDIR: {FSLDIR}")
    logger.info(f"FSL_CONFIG_PATH: {FSL_CONFIG_PATH}")

    if target_mesh is None:
        target_mesh = source_mesh

    contrasts_to_load = {
        # Source subject data
        "source_subject": (source_contrasts_list, source_mesh),
        # Target subject data
        "target_subject": (target_contrasts_list, target_mesh),
    }

    contrasts_gifti_file = {}

    with TemporaryDirectory() as tmp_dir:
        # Write temporary MSM config file, used to specify hyperparams
        # Default config is taken from
        # https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/MSM/UserGuide
        config_path = os.path.join(tmp_dir, "msm_config")

        iteration_line = "--it=50,3,3,3,3"
        if iterations is not None:
            if isinstance(iterations, int):
                it = str(iterations)
                iteration_line = f"--it={it},{it},{it},{it}"
            elif isinstance(iterations, str):
                iteration_line = f"--it={iterations}"

        lambda_line = "--lambda=0,0.1,0.2,0.3,0.4"
        if epsilon is not None:
            lambda_line = \
                f"--lambda={epsilon},{epsilon},{epsilon},{epsilon},{epsilon}"

        lines = "\n".join(
            [
                "--simval=3,2,2,2,2",
                "--sigma_in=2,2,2,2,1",
                "--sigma_ref=2,2,2,2,1",
                lambda_line,
                iteration_line,
                "--opt=AFFINE,DISCRETE,DISCRETE,DISCRETE,DISCRETE",
                "--CPgrid=6,1,2,3,4",
                "--SGgrid=6,3,4,5,6",
                "--datagrid=6,4,4,5,6",
                # "--regoption=1", # use the 2014 or 2018 version
                "--regexp=2",
                "--VN",
                "--rescaleL",
            ]
        )

        with open(config_path, "w") as f:
            f.write(lines)

        # For source and target subjects (denoted as in and ref subjects
        # respectively in msm), create a gifti image with all their
        # contrast maps (denoted as "data" in msm).
        # These maps will previously be set to use the same
        # coordinate system as the subject mesh
        for subject, (contrast_paths, mesh_path) in contrasts_to_load.items():
            # Load the coordsys from the mesh associated to the data
            # in order to make sure it is well specified
            mesh = utils.gifti_from_file(mesh_path)
            mesh_coordsys = mesh.darrays[0].coordsys
            contrast_maps = nib.load(contrast_paths[0])
            contrast_maps.darrays = prepare_darrays(
                contrast_maps.darrays, mesh_coordsys
            )

            # Add other contrast maps to gifti file
            for contrast_path in contrast_paths[1:]:
                extra_data = nib.load(contrast_path)
                contrast_maps.darrays.extend(
                    prepare_darrays(extra_data.darrays, mesh_coordsys)
                )

            # Save contrast map
            filename = str(Path(tmp_dir) / f"{subject}.func.gii")
            contrast_maps.to_filename(filename)
            contrasts_gifti_file[subject] = filename

        # If input meshes are compressed, decompress them
        # in temporary files and update mesh path
        if source_mesh.endswith(".gz"):
            tmp_source_mesh = os.path.join(
                tmp_dir, os.path.basename(source_mesh[:-3])
            )
            utils.ungzip(source_mesh, tmp_source_mesh)
            source_mesh = tmp_source_mesh
        if target_mesh.endswith(".gz"):
            tmp_target_mesh = os.path.join(
                tmp_dir, os.path.basename(target_mesh[:-3])
            )
            utils.ungzip(target_mesh, tmp_target_mesh)
            target_mesh = tmp_target_mesh

        # Run MSM
        cmd = " ".join(
            [
                os.path.join(FSLDIR, "bin/msm"),
                f"--inmesh={source_mesh}",
                f"--refmesh={target_mesh}",
                f"--indata={contrasts_gifti_file['source_subject']}",
                f"--refdata={contrasts_gifti_file['target_subject']}",
                f"--conf={config_path}",
                f"-o {tmp_dir}/",
                "-f ASCII",
                "--verbose" if verbose else "",
                "--debug --levels=1" if debug else "",
            ]
        )

        exit_code = os.system(cmd)
        if exit_code != 0:
            raise RuntimeError(f"Failed to run MSM with command:\n{cmd}")

        mesh_ascii_path = Path(tmp_dir) / "sphere.reg.asc"
        mesh_gii_path = Path(tmp_dir) / "transformed_in_mesh.surf.gii"

        cmd = " ".join(
            [
                "surf2surf",
                f"-i {mesh_ascii_path}",
                f"-o {mesh_gii_path}",
                "--outputtype=GIFTI_BIN_GZ",
            ]
        )
        exit_code = os.system(cmd)
        if exit_code != 0:
            raise RuntimeError(
                f"Failed to convert ASCII output to GIFTI with comand:\n{cmd}"
            )

        # Create a transformed GIFTI image with all attributes
        # indentical to target GIFTI image.

        # Transfomed and reprojected data are stored in temporary directory
        # in dpv (data per voxel) format.
        reprojected_dpv = Path(tmp_dir) / "transformed_and_reprojected.dpv"
        transformed_data = pd.read_csv(reprojected_dpv, sep=" ", header=None)

        # Data of interest (scalar data per voxel) are stored in 4th column
        # of dataset (0 - voxel index; 1, 2, 3 - voxel coordinates)
        transformed_data = transformed_data[4].to_numpy()

        # Use first image from target_contrasts_list as
        # a template for the transformed GIFTI image
        # Target data will be replaced by transformed data
        reprojected_contrasts = nib.load(target_contrasts_list[0])

        # Assure data arrays to have one dimension as dpv is always
        # one-dimensional
        # Use [:1] instead of [0] to preserve variable type (list)
        reprojected_contrasts.darrays = reprojected_contrasts.darrays[:1]

        # Replace target data by transformed data
        reprojected_contrasts.darrays[0].data = transformed_data
        mesh_gii = nib.load(mesh_gii_path)

        return mesh_gii, reprojected_contrasts
