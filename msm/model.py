from dotenv import load_dotenv
import nibabel as nib
from nilearn import datasets
from scipy.spatial.distance import cosine
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from msm.run import prepare_darrays, run_msm

ENV = os.getenv("ENV")

if ENV == "production":
    load_dotenv(".env.production")
elif ENV == "staging":
    load_dotenv(".env.staging")
elif ENV == "development":
    load_dotenv(".env.development")
load_dotenv(".env")

FSL_PATH = os.getenv("FSL_PATH")
# FSL_CONFIG_PATH = os.getenv("FSL_CONFIG_PATH")

fsaverage5 = datasets.fetch_surf_fsaverage(mesh="fsaverage5")


class MSMModel:
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
        mesh=fsaverage5.sphere_left,
        output_dir=".",
        verbose=False,
        debug=False,
    ):
        """
        Fit MSM alignment between source and target datasets.

        Parameters
        ----------
        source_data: list of str
            List of paths of all contrast maps for source subject
        target_data: list of str
            List of paths of all contrast maps for target subject.
            Length should match that of source_data
        mesh: str
            Path to mesh used for source and target
        output_dir: str
            Path to outputed files

        Returns
        -------
        self: object
            Fitted alignment
        """

        transformed_mesh, transformed_func = run_msm(
            in_data_list=source_data,
            in_mesh=mesh,
            ref_data_list=target_data,
            debug=debug,
            verbose=verbose,
            output_dir=output_dir,
        )

        self.transformed_mesh = transformed_mesh
        self.transformed_mesh_path = (
            Path(output_dir) / "transformed_in_mesh.surf.gii"
        )
        self.transformed_func = transformed_func
        self.mesh_path = mesh

        mesh_loaded = nib.load(mesh)
        self.coordsys = mesh_loaded.darrays[0].coordsys

        return self

    def transform(self, source_data):
        """
        Map source contrast map onto target mesh.

        Parameters
        ----------
        source_data: str
            Path to source contrast map

        Returns
        -------
        transformed_contrast_map: ndarray(n)
            Contrast map transformed from source space to target space.
            n is the number of voxels of the target mesh
            use during the fitting phase
        """

        data_input = nib.load(source_data)
        data_input.darrays = prepare_darrays(data_input.darrays, self.coordsys)

        # Duplicate contrast map
        # in order to cope with a bug of MSM
        # (MSM doesn't accept 1-dimensional maps)
        data_input.darrays.extend(
            prepare_darrays(data_input.darrays, self.coordsys)
        )

        with TemporaryDirectory(dir="./") as dir_name:
            # Export generated map to temp file
            # because MSM needs file paths
            source_filename = str(Path(dir_name) / "input_test.func.gii")
            data_input.to_filename(source_filename)

            transformed_path = str(Path(dir_name) / "transformed_contrast")

            # Map source_data onto target mesh
            cmd = " ".join(
                [
                    f"{FSL_PATH}/bin/msmresample",
                    f"{self.transformed_mesh_path} ",
                    transformed_path,
                    f"-labels {source_filename} ",
                    f"-project {self.mesh_path}",
                ]
            )

            exit_code = os.system(cmd)
            if exit_code != 0:
                raise RuntimeError(f"Failed to run MSM with command:\n{cmd}")

            # Load saved contrast map
            transformed_contrast_map = (
                nib.load(f"{transformed_path}.func.gii").darrays[0].data
            )

            return transformed_contrast_map

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

        transformed_data = self.transform(source_data)
        score = cosine(transformed_data.T, target_data.T)

        return score

    def load_model(self, model_filename, mesh):
        """
        Load fitted model from file

        Parameters
        ----------
        model_filename: str
            Path to saved fitted model.
            After fitting the model is usually saved in
            Path(output_dir) / "transformed_in_mesh.surf.gii"

        mesh: str
            Path to mesh used for source and target

        Returns
        -------
        self: object
            Loaded fitted alignment
        """
        self.transformed_mesh_path = model_filename
        self.mesh_path = mesh

        mesh_loaded = nib.load(mesh)
        self.coordsys = mesh_loaded.darrays[0].coordsys
        return self
