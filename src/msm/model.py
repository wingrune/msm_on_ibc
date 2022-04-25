import logging
import nibabel as nib
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.stats import pearsonr
import os
from pathlib import Path
import shlex
import subprocess
from tempfile import TemporaryDirectory

from msm.run import run_msm
from msm import utils


class MSM(BaseEstimator, TransformerMixin):
    def __init__(self, epsilon=0.1, **kwargs):
        """
        Initialize MSM object.

        Parameters
        ----------
        epsilon: scalar,
            Regularization parameter used in MSM.
            In the MSM documentation, the parameter is often denoted
            as lambda
        """

        self.epsilon = epsilon

    def fit(
        self,
        source_data,
        target_data,
        mesh_file=None,
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
        logger = logging.getLogger("msm")
        if verbose:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.WARNING)

        with TemporaryDirectory() as tmp_dir:
            source_filenames = []
            target_filenames = []

            # Save mesh as nifti image
            # (this is needed because transform calls msm resampling
            # which requires the mesh, but transform can be called on
            # a different machine than fit if the model is saved and loaded)
            self.mesh = utils.gifti_from_file(mesh_file)
            coordsys = self.mesh.darrays[0].coordsys

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
                source_mesh=mesh_file,
                target_contrasts_list=target_filenames,
                epsilon=self.epsilon,
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
        one_dimensional = False
        if source_data.ndim == 1:
            one_dimensional = True
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
                    coordsys=self.mesh.darrays[0].coordsys,
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

                # Create temporary file containing mesh
                mesh_path = str(Path(tmp_dir) / "mesh.gii")
                self.mesh.to_filename(mesh_path)

                # Map source_data onto target mesh
                cmd = shlex.split(
                    " ".join(
                        [
                            os.path.join(FSLDIR, "bin/msmresample"),
                            f"{transformed_mesh_path}",
                            predicted_contrast_path,
                            f"-labels {source_contrast_filename}",
                            f"-project {mesh_path}",
                        ]
                    )
                )

                process = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
                )

                with process.stdout:
                    utils.log_subprocess_output(process.stdout)
                with process.stderr:
                    utils.log_subprocess_output(process.stderr, err=True)

                exit_code = process.wait()

                if exit_code != 0:
                    raise RuntimeError(
                        f"Failed to run msmresample with command:\n{cmd}"
                    )

                # Load predicted contrast map ndarray and append it
                # to result list
                predicted_contrast_map = (
                    nib.load(f"{predicted_contrast_path}.func.gii")
                    .darrays[0]
                    .data
                )
                predicted_contrast_map = predicted_contrast_map.astype(
                    source_data.dtype
                )
                predicted_contrast_maps.append(predicted_contrast_map)

        predicted_data = np.vstack(predicted_contrast_maps)

        # If source data is 1-dim, return 1-dim array
        if one_dimensional:
            return predicted_data.flatten()
        else:
            return predicted_data

    def score(self, source_data, target_data):
        """
        Transform source contrast maps using fitted MSM
        and compute a Pearson correlation coefficient with
        actual target constrast maps.

        Parameters
        ----------
        source_data: ndarray(n_samples, n)
            Contrast maps for source subject
        target_data: ndarray(n_samples, n)
            Contrast maps for target subject

        Returns
        -------
        score: float
            Pearson correlation coefficient between
            self.transform(source_data) with target_data
        """

        transformed_data = self.transform(source_data)
        score = np.mean(
            [
                pearsonr(transformed_data[i, :], target_data[i, :])[0]
                for i in range(transformed_data.shape[0])
            ]
        )

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
        self.mesh = utils.gifti_from_file(mesh_path)

        return self
