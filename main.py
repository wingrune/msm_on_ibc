import subprocess
from pathlib import Path

import matplotlib.pyplot as plt

import nilearn
import nibabel as nib
from nilearn import plotting
from nilearn import datasets

import nibabel


# Display the spher
fsaverage_sphere = 'data/freesurfer/subjects/fsaverage5/surf/lh.sphere'
nilearn.plotting.plot_surf(fsaverage_sphere)


# Display data from an IBS subject
subject_path = Path('data') / 'sub-01' / 'sess-00'
fsaverage5 = datasets.fetch_surf_fsaverage('fsaverage5')
plotting.plot_surf(
    fsaverage5.pial_left, str(subject_path / 'video_left_button_press_lh.gii')
    # fsaverage_sphere, str(subject_path / 'video_left_button_press_lh.gii')
)

plt.show()

output = subprocess.check_output()

# Load spherical mesh produced with `mris_convert`
spherical_mesh = 'data/freesurfer/subjects/fsaverage5/surf/lh.sphere.gii'
sphere = nib.load(spherical_mesh)
coordsys = sphere.darrays[0].coordsys

# We need to set the intent of the data file to
# nib.nifti1.intent_codes.code['NIFTI_INTENT_POINTSET']
data = nib.load(
    '/data/parietal/store2/data/ibc/derivatives/sub-01/ses-00/freesurfer/'
    'rdcsub-01_ses-00_task-ArchiStandard_dir-ap_bold_fsaverage5_lh.gii'
)
data.darrays = data.darrays[:2]
for d in data.darrays:
    d.intent = nib.nifti1.intent_codes.code['NIFTI_INTENT_POINTSET']
    d.coordsys = coordsys

data_filename = 'data.func.gii'
data.to_filename(data_filename)

exit_code, output = subprocess.check_output(
    [
        "/usr/local/fsl/bin/msm",
        f'--inmesh={spherical_mesh}',
        f'--indata={data_filename}',
        f'--refdata={data_filename}',
        '-o outputs/L.',
        '-t ASCII'
    ]
)

# convert with surf2surf -i outputs/L.sphere.reg.asc
# -o outputs/L.sphere.reg.gii --outputtype=GIFTI_BIN_GZ
if exit_code != 0:
    raise SystemExit(exit_code)
