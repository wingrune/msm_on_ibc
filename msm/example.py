# An example for running MSM_Alignment.py

import nibabel as nib
import os
from scipy.stats import pearsonr
from scipy.spatial.distance import cosine
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
MSM = model.MSMModel()

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
"""
MSM.fit(
    source_data,
    target_data,
    # mesh="/storage/store2/work/athual/fsaverage/lh.sphere.gii",
    # output_dir="/storage/store2/work/athual/outputs/"
    # "_051_alignment_method_comparison",
    mesh="../data/lh.sphere.gii",
    output_dir="../outputs/"
    "_051_alignment_method_comparison",
)
"""

# %% Fit model

print("Loading model...")
MSM.load_model(
    model_filename="../test_outputs_lambda_39_0.1/transformed_in_mesh.surf.gii",
    mesh="../data/lh.sphere.gii"
)

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

# %% Transform source data
print("Transforming contrast map...")
transformed_map = MSM.transform(source_test_data)

# %% Load target map
target_map = nib.load(target_test_data).darrays[0].data

# %% Compute correlation
score = pearsonr(transformed_map, target_map)
print(score)
score = cosine(transformed_map, target_map)
print(score)
print(MSM.score(source_test_data, target_map))
