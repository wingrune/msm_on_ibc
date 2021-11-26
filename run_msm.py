import dotenv
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

import nibabel as nib

dotenv.load_dotenv()
FSL_PATH = os.getenv("FSL_PATH")
FSL_CONFIG_PATH = os.getenv("FSL_CONFIG_PATH")


def is_same_coordsys(c1, c2):
    return (c1.dataspace == c2.dataspace and c1.xformspace == c2.xformspace
            and np.all(c1.xform == c2.xform))


def prepare_darrays(darrays, coordsys):
    for d in darrays:
        d.data = d.data.astype(np.float32)
        d.datatype = nib.nifti1.data_type_codes.code['NIFTI_TYPE_FLOAT32']
        d.intent = nib.nifti1.intent_codes.code['NIFTI_INTENT_POINTSET']
        if (d.coordsys is not None
                and not is_same_coordsys(d.coordsys, coordsys)):
            raise ValueError(
                "Provided data is in different coordsys than the mesh."
            )
        d.coordsys = coordsys

    return darrays


def run_msm(in_data_list, in_mesh, ref_data_list, ref_mesh=None,
            output_dir=None, debug=False, verbose=False,
            fsl_config_path=FSL_CONFIG_PATH):
    """Run MSM on a list of contrast between in data and ref data

    Parameters
    ----------
    in_data_list, ref_data_list : list of str
        Data used as features for the registration. The data should be provided
        as a list of GIFTI files, that will be merged to perform
        the multimodale registration.
    in_mesh, ref_mesh : str
        Spherical mesh on wich all data from in_data_list or ref_data_list
        live. The mesh should be given as a GIFTI file. Not that if ref_mesh is
        not specified, the in_mesh will be used for all input data.
    output_dir : str or Path
        Directory in which to save the outputs. It will contain in GIFTI files
        for the transformed data (in the ref_mesh) and the transformed_mesh in
        which the in_data are aligned to ref_data.
    debug : bool
        Flag to run quickly first level of the optimization.
    verbose : bool
        Whether the algorithm is verbose or not.


    Returns
    -------
    mesh_gii : pathlib.Path
        file holding the transformed mesh.
    transformed_gii : pathlib.Path
        file holding the transformed data in the ref_mesh.
    """

    if ref_mesh is None:
        ref_mesh = in_mesh

    if output_dir is None:
        output_dir = 'outputs'
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    data_to_load = {
        'ref_data': (ref_data_list, ref_mesh),
        'in_data': (in_data_list, in_mesh)
    }

    data_files = {}
    with TemporaryDirectory(prefix='./') as dir_name:
        dir_name = 'test_outputs'

        for sub, (datafiles, mesh_file) in data_to_load.items():

            # Load the coordsys from the mesh associated to the data to make
            # sure it is well specified
            mesh = nib.load(mesh_file)
            coordsys = mesh.darrays[0].coordsys
            data = nib.load(datafiles[0])
            data.darrays = prepare_darrays(data.darrays, coordsys)
            for fname in datafiles[1:]:
                extra_data = nib.load(fname)
                data.darrays.extend(prepare_darrays(
                    extra_data.darrays, coordsys
                ))

            filename = str(Path(dir_name) / f'{sub}.func.gii')
            data.to_filename(filename)
            data_files[sub] = filename

    cmd = ' '.join([
        f"{FSL_PATH}/fsl/bin/msm",
        f"--inmesh={in_mesh}",
        f"--refmesh={ref_mesh}",
        f"--indata={data_files['in_data']}",
        f"--refdata={data_files['ref_data']}",
        f"--conf={FSL_CONFIG_PATH} ",
        f"-o {output_dir}/",
        "-f ASCII",
        "--verbose" if verbose else '',
        "--debug --levels=1" if debug else '',
    ])
    exit_code = os.system(cmd)
    if exit_code != 0:
        raise RuntimeError(f"Failed to run MSM with command:\n{cmd}")

    mesh_ascii = output_dir / 'sphere.reg.asc'
    mesh_gii = output_dir / 'transformed_in_mesh.surf.gii'

    cmd = ' '.join([
            "surf2surf",
            f"-i {mesh_ascii}",
            f"-o {mesh_gii}",
            "--outputtype=GIFTI_BIN_GZ"
    ])
    exit_code = os.system(cmd)
    if exit_code != 0:
        raise RuntimeError(
            f"Failed to convert ASCII output to GIFTI with comand:\n{cmd}"
        )
    mesh_ascii.unlink()

    reprojected_dpv = output_dir / 'transformed_and_reprojected.dpv'
    transformed_data = pd.read_csv(reprojected_dpv, sep=' ', header=None)
    transformed_data = transformed_data[4].to_numpy()

    data = nib.load(ref_data_list[0])
    data.darrays = data.darrays[:1]
    data.darrays[0].data = transformed_data

    reprojected_gii = str(output_dir / "transformed_and_reprojected.func.gii")
    data.to_filename(reprojected_gii)
    reprojected_dpv.unlink()

    return mesh_gii, reprojected_gii
