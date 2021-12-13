from dotenv import load_dotenv
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

import nibabel as nib

# Load environment variables
ENV = os.getenv("ENV")

if ENV == "production":
    load_dotenv(".env.production")
elif ENV == "staging":
    load_dotenv(".env.staging")
elif ENV == "development":
    load_dotenv(".env.development")
load_dotenv(".env")

FSL_PATH = os.getenv("FSL_PATH")
FSL_CONFIG_PATH = os.getenv("FSL_CONFIG_PATH")

print(FSL_PATH, FSL_CONFIG_PATH)


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
    debug=False,
    verbose=False,
    fsl_config_path=FSL_CONFIG_PATH,
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
    output_dir : str or Path
        Directory in which to save the outputs. It will contain in GIFTI files
        for the transformed data (in the target_mesh) and the transformed_mesh
        in which the source_subject are aligned to target_subject.
    debug : bool
        Flag to run quickly first level of the optimization.
    verbose : bool
        Whether the algorithm is verbose or not.


    Returns
    -------
    mesh_gii : nibabel.gifti.GiftiImage
        Image holding the transformed mesh.
    transformed_gii : nibabel.gifti.GiftiImage
        Image holding the transformed data in the target_mesh.
    """

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
        # For source and target subjects (denoted as in and ref subjects
        # respectively in msm), create a gifti image with all their
        # contrast maps (denoted as "data" in msm).
        # These maps will previously be set to use the same
        # coordinate system as the subject mesh
        for subject, (contrast_paths, mesh_path) in contrasts_to_load.items():
            # Load the coordsys from the mesh associated to the data
            # in order to make sure it is well specified
            mesh = nib.load(mesh_path)
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

        cmd = " ".join(
            [
                f"{FSL_PATH}/bin/msm",
                f"--inmesh={source_mesh}",
                f"--refmesh={target_mesh}",
                f"--indata={contrasts_gifti_file['source_subject']}",
                f"--refdata={contrasts_gifti_file['target_subject']}",
                # f"--conf={FSL_CONFIG_PATH}",
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

        reprojected_dpv = Path(tmp_dir) / "transformed_and_reprojected.dpv"
        transformed_data = pd.read_csv(reprojected_dpv, sep=" ", header=None)
        transformed_data = transformed_data[4].to_numpy()

        reprojected_contrasts = nib.load(target_contrasts_list[0])
        reprojected_contrasts.darrays = reprojected_contrasts.darrays[:1]
        reprojected_contrasts.darrays[0].data = transformed_data

        mesh_gii = nib.load(mesh_gii_path)

        return mesh_gii, reprojected_contrasts
