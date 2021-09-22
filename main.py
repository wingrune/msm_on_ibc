from nilearn import plotting

from run_msm import run_msm

# Load spherical mesh produced with `mris_convert`
spherical_mesh = 'data/freesurfer/subjects/fsaverage5/surf/lh.sphere.gii'


data_to_load = {
    'sub-07': [
        '/storage/store2/data/ibc/derivatives/sub-07/ses-01/'
        'res_fsaverage5_hcp_language_ffx/stat_surf/story-math_lh.gii',
        '/storage/store2/data/ibc/derivatives/sub-07/ses-02/'
        'res_fsaverage5_hcp_relational_ffx/stat_surf/relational_lh.gii',
    ],
    'sub-04': [
        '/storage/store2/data/ibc/derivatives/sub-04/ses-01/'
        'res_fsaverage5_hcp_language_ffx/stat_surf/story-math_lh.gii',
        '/storage/store2/data/ibc/derivatives/sub-04/ses-02/'
        'res_fsaverage5_hcp_relational_ffx/stat_surf/relational_lh.gii',
    ]
}

transformed_mesh, transformed_func = run_msm(
    in_data_list=data_to_load['sub-04'], in_mesh=spherical_mesh,
    ref_data_list=data_to_load['sub-07'],
    debug=False, verbose=True, output_dir='test_outputs'
)
print(transformed_func)


##################
# plotting
import matplotlib.pyplot as plt  # noqa: E402

plotting.plot_surf(
    'data/freesurfer/subjects/fsaverage5/surf/lh.inflated',
    transformed_func, title='Transformed Data - story - math'
)
plt.savefig('transformed.pdf')


plotting.plot_surf(
    'data/freesurfer/subjects/fsaverage5/surf/lh.inflated',
    data_to_load['sub-04'][0],
    title='Origin Data - story - math'
)
plt.savefig('origin.pdf')

plotting.plot_surf(
    'data/freesurfer/subjects/fsaverage5/surf/lh.inflated',
    data_to_load['sub-07'][0],
    title='Reference Data - story - math'
)
plt.savefig('reference.pdf')
