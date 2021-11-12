from nilearn import plotting

from run_msm import run_msm

# Load spherical mesh produced with `mris_convert`
spherical_mesh = './data/lh.sphere.gii'


data_to_load = {
    'sub-07': [
        './data/sub-07_relational_lh.gii',
        './data/sub-07_relational_lh.gii',
    ],
    'sub-04': [
        './data/sub-04_relational_lh.gii',
        './data/sub-04_relational_lh.gii',
    ]
}

transformed_mesh, transformed_func = run_msm(
    in_data_list=data_to_load['sub-07'], in_mesh=spherical_mesh,
    ref_data_list=data_to_load['sub-04'],
    debug=False, verbose=True, output_dir='test_outputs'
)


##################
# plotting
import matplotlib.pyplot as plt  # noqa: E402

plotting.plot_surf(
    './data/lh.sphere',
    transformed_func, title='Transformed Data - relational',
    cmap="RdBu",
    threshold=3
)
plt.savefig('transformed_relational_7_to_4.png')


plotting.plot_surf(
    './data/lh.sphere',
    data_to_load['sub-07'][1],
    title='Origin Data - relational',
    cmap="RdBu",
    threshold=3
)
plt.savefig('origin_relational_7_to_4.png')

plotting.plot_surf(
    './data/lh.sphere',
    data_to_load['sub-04'][1],
    title='Reference Data - relational',
    cmap="RdBu",
    threshold=3
)
plt.savefig('reference_relational_7_to_4.png')
