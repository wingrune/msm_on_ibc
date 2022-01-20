import nibabel as nib
from nilearn import datasets
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import r2_score
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from msm.run import run_msm
from msm import utils

fsaverage5 = datasets.fetch_surf_fsaverage(mesh="fsaverage5")


class MSM(BaseEstimator, TransformerMixin):
    def __init__(self, epsilon=0.1, **kwargs):
        """
        Initialize MSM object.

        Parameters
        ----------
        epsilon: scalar
            Regularization parameter used in MSM.
            In the MSM documentation, the parameter is often denoted
            as lambda
        """

        self.epsilon = epsilon

    def fit(
        self,
        source_data,
        target_data,
        mesh_file=fsaverage5.sphere_left,
        verbose=False,
        debug=False,
        **kwargs,
    ):
        """
        Fit MSM alignment between source and target datasets.

        Parameters
        ----------
        source_data: ndarray(n_samples, n_features)
            Contrast maps for source subject.
            In our neuroscience context, n_samples would be the number
            of conditions, and n_features the number of voxels.
        target_data: ndarray(n_samples, n_features)
            Contrast maps for target subject.
            Length should match that of source_data
        mesh_file: str
            Path to mesh used for source and target
        output_dir: str
            Path to outputed files

        Returns
        -------
        self: object
            Fitted alignment
        """
        with TemporaryDirectory() as tmp_dir:
            source_filenames = []
            target_filenames = []

            self.mesh_path = mesh_file
            mesh = utils.gifti_from_file(self.mesh_path)
            coordsys = mesh.darrays[0].coordsys
            self.coordsys = coordsys

            # All inputed contrast maps need to be written
            # as gifti files in order to be used with MSM

            # Write all source contrast maps
            for i, contrast in enumerate(source_data):
                filename = str(Path(tmp_dir) / f"source_{i}.func.gii")
                source_filenames.append(filename)

                contrast_data_array = nib.gifti.gifti.GiftiDataArray(
                    data=contrast,
                    datatype=nib.nifti1.data_type_codes.code[
                        "NIFTI_TYPE_FLOAT32"
                    ],
                    intent=nib.nifti1.intent_codes.code[
                        "NIFTI_INTENT_POINTSET"
                    ],
                    coordsys=coordsys,
                )
                contrast_image = nib.gifti.gifti.GiftiImage()
                contrast_image.add_gifti_data_array(contrast_data_array)
                contrast_image.to_filename(filename)

            # Write all target contrast maps
            for i, contrast in enumerate(target_data):
                filename = str(Path(tmp_dir) / f"target_{i}.func.gii")
                target_filenames.append(filename)
                contrast_data_array = nib.gifti.gifti.GiftiDataArray(
                    data=contrast,
                    datatype=nib.nifti1.data_type_codes.code[
                        "NIFTI_TYPE_FLOAT32"
                    ],
                    intent=nib.nifti1.intent_codes.code[
                        "NIFTI_INTENT_POINTSET"
                    ],
                    coordsys=coordsys,
                )
                contrast_image = nib.gifti.gifti.GiftiImage()
                contrast_image.add_gifti_data_array(contrast_data_array)
                contrast_image.to_filename(filename)

            # Run msm
            transformed_mesh, _ = run_msm(
                source_contrasts_list=source_filenames,
                source_mesh=self.mesh_path,
                target_contrasts_list=target_filenames,
                epsilon=self.epsilon,
                debug=debug,
                verbose=verbose,
            )

            # Save computed transformation in model
            self.transformed_mesh = transformed_mesh

        return self

    def transform(self, source_data):
        """
        Map source contrast maps onto target mesh.

        Parameters
        ----------
        source_data: ndarray(n_samples, n_features)
            Contrast maps for source subject.

        Returns
        -------
        predicted_contrast_maps: ndarray(n_samples, n_features)
            Contrast map transformed from source space to target space.
            n is the number of voxels of the target mesh
            use during the fitting phase
        """
        FSLDIR, _ = utils.check_fsl()

        predicted_contrast_maps = []

        # Assure source_data to be 2-dimensional
        if source_data.ndim == 1:
            source_data = np.array([source_data])

        with TemporaryDirectory() as tmp_dir:
            # Write transformed_mesh to gifti file
            transformed_mesh_path = str(Path(tmp_dir) / "transformed_mesh.gii")
            self.transformed_mesh.to_filename(transformed_mesh_path)

            # Write each source contrast map to a gifti file
            for i, contrast in enumerate(source_data):
                source_contrast_filename = str(
                    Path(tmp_dir) / f"source_{i}.func.gii"
                )

                contrast_data_array = nib.gifti.gifti.GiftiDataArray(
                    data=contrast,
                    datatype=nib.nifti1.data_type_codes.code[
                        "NIFTI_TYPE_FLOAT32"
                    ],
                    intent=nib.nifti1.intent_codes.code[
                        "NIFTI_INTENT_POINTSET"
                    ],
                    coordsys=self.coordsys,
                )

                contrast_image = nib.gifti.gifti.GiftiImage()
                # Duplicate contrast map
                # in order to cope with a bug of MSM
                # (MSM doesn't accept 1-dimensional maps)
                contrast_image.add_gifti_data_array(contrast_data_array)
                contrast_image.add_gifti_data_array(contrast_data_array)
                contrast_image.to_filename(source_contrast_filename)

                predicted_contrast_path = str(
                    Path(tmp_dir) / "predicted_contrast"
                )

                # If input mesh is compressed, decompress it
                # in temporary files and update mesh path
                mesh_path = self.mesh_path
                if mesh_path.endswith(".gz"):
                    tmp_mesh_path = os.path.join(
                        tmp_dir, os.path.basename(mesh_path[:-3])
                    )
                    utils.ungzip(mesh_path, tmp_mesh_path)
                    mesh_path = tmp_mesh_path

                # Map source_data onto target mesh
                cmd = " ".join(
                    [
                        os.path.join(FSLDIR, "bin/msmresample"),
                        f"{transformed_mesh_path}",
                        predicted_contrast_path,
                        f"-labels {source_contrast_filename}",
                        f"-project {mesh_path}",
                    ]
                )

                exit_code = os.system(cmd)
                if exit_code != 0:
                    raise RuntimeError(
                        f"Failed to run MSM with command:\n{cmd}"
                    )

                # Load predicted contrast map ndarray and append it
                # to result list
                predicted_contrast_map = (
                    nib.load(f"{predicted_contrast_path}.func.gii")
                    .darrays[0]
                    .data
                )
                predicted_contrast_maps.append(predicted_contrast_map)

        return np.vstack(predicted_contrast_maps)

    def score(self, source_data, target_data):
        """
        Transform source contrast maps using fitted MSM
        and compute cosine distance with actual target constrast maps.

        Parameters
        ----------
        source_data: ndarray(n_samples, n)
            Contrast maps for source subject
        target_data: ndarray(n_samples, n)
            Contrast maps for target subject

        Returns
        -------
        score: float
            cosine distance between
            self.transform(source_data) with target_data
        """

        transformed_data = self.predict(source_data)
        score = r2_score(transformed_data.T, target_data.T)

        return score

    def load_model(self, model_path, mesh_path):
        """
        Load fitted model from file

        Parameters
        ----------
        model_path: str
            Path to saved fitted model.
            After fitting the model is usually saved in
            Path(output_dir) / "transformed_in_mesh.surf.gii"

        mesh_path: str
            Path to mesh used for source and target

        Returns
        -------
        self: object
            Loaded fitted alignment
        """
        self.transformed_mesh = nib.load(model_path)
        self.mesh_path = mesh_path
        mesh = utils.gifti_from_file(self.mesh_path)
        self.coordsys = mesh.darrays[0].coordsys

        return self
