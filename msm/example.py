# An example for running MSM_Alignment.py
import numpy as np
import nibabel as nib
import os

import sys

# Define env variables manually since they won't be
# picked up from the env file
# os.environ["FSL_PATH"] = "/usr/local/fsl"
# os.environ[
#     "FSL_CONFIG_PATH"
# ] = "/usr/local/fsl/config/basic_configs/config_standard_MSM_strain"

os.environ["FSL_PATH"] = "/mnt/e/usr/local/fsl"
os.environ[
    "FSL_CONFIG_PATH"
] = "/mnt/e/usr/local/fsl/config/basic_configs/config_standard_MSM_strain"

# Add MSM package to sys in order to import it
# sys.path.append("/storage/store2/work/athual/repo/msm_on_ibc_wingrune")
sys.path.append("/mnt/e/Ecole Polytechnique/Parietal/code/msm_on_ibc")

from msm import model  # noqa: E402

# %% Define model
msm = model.MSM()

# %% Define data used for training
source_data = [
    "../data/bold_sub-04_ses-03_rsvp_language_consonant_string_lh.gii",
    "../data/bold_sub-04_ses-04_archi_social_non_speech_sound_lh.gii",
    # "/storage/store2/data/ibc/derivatives/sub-04/ses-03/"
    # "res_fsaverage5_language_ffx/stat_surf/sentence-consonant_string_lh.gii",
    # "/storage/store2/data/ibc/derivatives/sub-04/ses-30/"
    # "res_fsaverage5_mathlang_ffx/stat_surf/visual-auditory_lh.gii",
]

target_data = [
    "../data/bold_sub-07_ses-03_rsvp_language_consonant_string_lh.gii",
    "../data/bold_sub-07_ses-04_archi_social_non_speech_sound_lh.gii",
    # "/storage/store2/data/ibc/derivatives/sub-07/ses-03/"
    # "res_fsaverage5_language_ffx/stat_surf/sentence-consonant_string_lh.gii",
    # "/storage/store2/data/ibc/derivatives/sub-07/ses-30/"
    # "res_fsaverage5_mathlang_ffx/stat_surf/visual-auditory_lh.gii",
]

# %% Fit model
# print("Fitting model...")
# model.fit(source_data, target_data, mesh="./data/lh.sphere.gii")

# making a ndarray of data
source_array = np.stack(
    [nib.load(source).darrays[0].data for source in source_data],
    axis=0,
)

target_array = np.stack(
    [nib.load(target).darrays[0].data for target in target_data],
    axis=0,
)

# msm.fit(
#     source_array,
#     target_array,
#     # mesh_file="/storage/store2/work/athual/fsaverage/lh.sphere.gii",
#     # output_dir="/storage/store2/work/athual/outputs/"
#     # "_051_alignment_method_comparison",
#      mesh_file="../data/lh.sphere.gii",
#      output_dir="../outputs/" "_051_alignment_method_comparison",
# )

msm.load_model(model_path="../outputs/_051_alignment_method_comparison/transformed_in_mesh.surf.gii", mesh_path="../data/lh.sphere.gii")
# %% Evaluate model

# Define data used for test
source_test_data = "../data/bold_sub-04_ses-01_hcp_motor_tongue_lh.gii"
target_test_data = "../data/bold_sub-07_ses-01_hcp_motor_tongue_lh.gii"
# source_test_data = (
#     "/storage/store2/data/ibc/derivatives/sub-04/"
#     "ses-01/res_fsaverage5_hcp_motor_ffx/stat_surf/tongue_lh.gii"
# )

# target_test_data = (
#     "/storage/store2/data/ibc/derivatives/sub-07/"
#     "ses-01/res_fsaverage5_hcp_motor_ffx/stat_surf/tongue_lh.gii"
# )

# %% Load source map
source_map = nib.load(source_test_data).darrays[0].data

# %% Transform source data
print("Transforming contrast map...")
transformed_map = msm.predict(source_map)

# %% Load target map
target_map = nib.load(target_test_data).darrays[0].data

# %% Compute R2 score before and after transformation
print(f"R2 score before transformation: {msm.score(source_map, target_map)}")
print(f"R2 score after transformation: {msm.score(transformed_map, target_map)}")
