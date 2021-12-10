# An example for running MSM_Alignment.py

import nibabel as nib
import importlib.util
from scipy.stats import pearsonr

# %%
# run_msm_spec = importlib.util.spec_from_file_location(
#     ".run_msm",
#     "/storage/store2/work/athual/repo/msm_on_ibc_wingrune/run_msm.py",
# )
# run_msm = importlib.util.module_from_spec(run_msm_spec)
# run_msm_spec.loader.exec_module(run_msm)

spec = importlib.util.spec_from_file_location(
    ".msm_alignment",
    "/mnt/e/Ecole Polytechnique/Parietal/code/msm_on_ibc/msm_alignment.py",
)
msm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(msm)
model = msm.MSM_Alignment()

# %% Define data used for training
source_data = [
    './data/bold_sub-04_ses-03_rsvp_language_consonant_string_lh.gii',
    './data/bold_sub-04_ses-04_archi_social_non_speech_sound_lh.gii'
]

target_data = [
    "./data/bold_sub-07_ses-03_rsvp_language_consonant_string_lh.gii",
    "./data/bold_sub-07_ses-04_archi_social_non_speech_sound_lh.gii",
]

# %% Fit model
print("Fitting model...")
model.fit(source_data, target_data, mesh="./data/lh.sphere.gii")

# Define data used for test
source_test_data = "./data/bold_sub-04_ses-01_hcp_motor_tongue_lh.gii"
target_test_data = "./data/bold_sub-07_ses-01_hcp_motor_tongue_lh.gii"

# %% Transform source data
print("Transforming contrast map...")
transformed_map = model.transform(source_test_data)

# %% Load target map
target_map = nib.load(target_test_data).darrays[0].data

# Compute correlation
score = pearsonr(transformed_map, target_map)
print(score)
