from pathlib import Path

import matplotlib.pyplot as plt

import nilearn
from nilearn import plotting
from nilearn import datasets

import nibabel


# Display the spher
fsaverage_sphere = 'data/freesurfer/subjects/fsaverage5/surf/lh.sphere'
nilearn.plotting.plot_surf(fsaverage_sphere)


# Display data from an IBS subject
subject_path = Path('data') / 'sub-01' / 'sess-00'
fsaverage5 = nilearn.datasets.fetch_surf_fsaverage('fsaverage5')
nilearn.plotting.plot_surf(
    fsaverage5.pial_left, str(subject_path / 'video_left_button_press_lh.gii')
)

plt.show()
