import nibabel as nib
from nibabel.gifti.gifti import GiftiDataArray, GiftiImage
from nilearn import datasets
import numpy as np
import os
from tempfile import TemporaryDirectory

from msm import utils
from msm import run


def test_run():
    """Main function run_msm should run without error."""

    with TemporaryDirectory() as tmp_dir:
        # Load fsaverage mesh
        fs3 = datasets.fetch_surf_fsaverage(mesh="fsaverage3")
        mesh_path = fs3.sphere_left
        tmp_mesh_path = os.path.join(tmp_dir, os.path.basename(mesh_path[:-3]))
        # Ungzip mesh to new file
        utils.ungzip(mesh_path, tmp_mesh_path)

        n_voxels = 642
        mesh_coordsys = nib.load(tmp_mesh_path).darrays[0].coordsys

        # Generate random contrast maps
        # for source and target individuals
        # with fsaverage coordinate system
        source_contrasts_list = [
            os.path.join(tmp_dir, "source1.gii"),
            os.path.join(tmp_dir, "source2.gii"),
        ]
        for source_contrast in source_contrasts_list:
            img = GiftiImage()
            data = GiftiDataArray(
                np.random.rand(n_voxels), coordsys=mesh_coordsys
            )
            img.add_gifti_data_array(data)
            img.add_gifti_data_array(data)
            nib.save(img, source_contrast)

        target_contrasts_list = [
            os.path.join(tmp_dir, "target1.gii"),
            os.path.join(tmp_dir, "target2.gii"),
        ]
        for target_contrast in target_contrasts_list:
            img = GiftiImage()
            data = GiftiDataArray(
                np.random.rand(n_voxels), coordsys=mesh_coordsys
            )
            img.add_gifti_data_array(data)
            img.add_gifti_data_array(data)
            nib.save(img, target_contrast)

        mesh_gii, transformed_gii = run.run_msm(
            source_contrasts_list,
            tmp_mesh_path,
            target_contrasts_list,
        )

        print(mesh_gii)
        print(transformed_gii)

        assert mesh_gii.darrays[0].data.shape[0] == n_voxels
        assert transformed_gii.darrays[0].data.shape[0] == n_voxels
