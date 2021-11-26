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
    "/storage/store2/work/athual/repo/msm_on_ibc_wingrune/msm_alignment.py",
)
msm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(msm)
model = msm.MSM_Alignment()

# %% Define data used for training
source_data = [
    "/storage/store2/data/ibc/derivatives/sub-04/ses-03/res_fsaverage5_language_ffx/stat_surf/sentence-consonant_string_lh.gii",
    "/storage/store2/data/ibc/derivatives/sub-04/ses-30/res_fsaverage5_mathlang_ffx/stat_surf/visual-auditory_lh.gii",
]

target_data = [
    "/storage/store2/data/ibc/derivatives/sub-07/ses-03/res_fsaverage5_language_ffx/stat_surf/sentence-consonant_string_lh.gii",
    "/storage/store2/data/ibc/derivatives/sub-07/ses-30/res_fsaverage5_mathlang_ffx/stat_surf/visual-auditory_lh.gii",
]

# %% Fit model
print("Fitting model...")
model.fit(source_data, target_data)

# Define data used for test
source_test_data = "/storage/store2/data/ibc/derivatives/sub-04/ses-01/res_fsaverage7_hcp_motor_ffx/stat_surf/tongue-avg_lh.gii"
target_test_data = "/storage/store2/data/ibc/derivatives/sub-07/ses-01/res_fsaverage7_hcp_motor_ffx/stat_surf/tongue-avg_lh.gii"

# %% Transform source data
print("Transforming contrast map...")
transformed_map = model.transform(source_test_data)

# %% Load target map
target_map = nib.load(target_test_data).darrays[0].data

# Compute correlation
score = pearsonr(transformed_map, target_map)
