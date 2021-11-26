import dotenv
import os
import json
import matplotlib.pyplot as plt
from lambda_steadiness import FSL_CONFIG_PATH  # noqa: E402
import nibabel as nib
import numpy as np
from pathlib import Path
from scipy import stats
import random

from run_msm import run_msm

dotenv.load_dotenv()
FSL_PATH = os.getenv("FSL_PATH")


def is_same_coordsys(c1, c2):
    return (c1.dataspace == c2.dataspace and c1.xformspace == c2.xformspace
            and np.all(c1.xform == c2.xform))


def prepare_darrays(darrays, coordsys):
    for d in darrays:
        d.data = d.data.astype(np.float32)
        d.datatype = nib.nifti1.data_type_codes.code['NIFTI_TYPE_FLOAT32']
        d.intent = nib.nifti1.intent_codes.code['NIFTI_INTENT_POINTSET']
        if (d.coordsys is not None
                and not is_same_coordsys(d.coordsys, coordsys)):
            raise ValueError(
                "Provided data is in different coordsys than the mesh."
            )
        d.coordsys = coordsys

    return darrays


# Load spherical mesh produced with `mris_convert`
spherical_mesh = './data/lh.sphere.gii'

# load data
subject_input = 'sub-07'
subject_reference = 'sub-04'
data_path = './data/'

# parameters
random.seed(42)
lambd = [0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1]
# if true, a new split will be created, if false, data will be loaded
new_split = False
# if true will run MSM, if false will use an existing run_MSM output
train_mode = False

# data - dictionary of type: input_fname -> reference_fname

data = {}
sessions = ['ses-00', 'ses-01', 'ses-02', 'ses-04']
data_input = [
    os.path.join(data_path, file)
    for file in os.listdir(data_path)
    if subject_input in file and 'lh' in file and (
        any(ses in file for ses in sessions)
    )
]

data_reference = [
    os.path.join(data_path, file)
    for file in os.listdir(data_path)
    if subject_reference in file and 'lh' in file and (
        any(ses in file for ses in sessions)
    )
]

for input_fname in data_input:
    for reference_fname in data_reference:
        reference_contrast = reference_fname.split(subject_reference)[1]
        input_contrast = input_fname.split(subject_input)[1]
        if reference_contrast == input_contrast:
            data[input_fname] = reference_fname

print(len(data))
# split data to train and test

if new_split:
    train_data_input = random.sample(list(data.keys()), int(len(data)*0.8))

    train_data = {
        input_fname: data[input_fname]
        for input_fname in train_data_input
    }

    test_data = {
        input_fname: data[input_fname]
        for input_fname in list(data.keys())
        if input_fname not in train_data_input
    }

    with open("train.json", "w") as outfile:
        json.dump(train_data, outfile)

    with open("test.json", "w") as outfile:
        json.dump(test_data, outfile)
else:
    train_data_fname = "train.json"
    test_data_fname = "test.json"
    with open(train_data_fname) as json_file:
        train_data = json.load(json_file)

    with open(test_data_fname) as json_file:
        test_data = json.load(json_file)

epsilons = [0.01, 0.1, 1, 10, 100, 1000]

cross_correlation = {}
base_cross_correlation = {}  # cross-correlation without deformation

for epsilon in epsilons:

    # replacing lambda values in config file
    with open(f"{FSL_CONFIG_PATH}", "r") as f:
        list_of_lines = f.readlines()
        list_of_lines[3] = f"--lambda={epsilon},{epsilon},{epsilon},{epsilon}\n"

    with open(f"{FSL_CONFIG_PATH}", "w") as f:
        f.writelines(list_of_lines)

    # running MSM with train data
    if train_mode:
        transformed_mesh, transformed_func = run_msm(
            in_data_list=list(train_data.keys()), in_mesh=spherical_mesh,
            ref_data_list=list(train_data.values()),
            debug=False, verbose=True,
            output_dir='test_outputs_lambda' + str(epsilon)
        )

    # testing MSM with test data

    output_dir = 'test_outputs_lambda' + str(epsilon)
    deformed = os.path.join(output_dir, "transformed_in_mesh.surf.gii")
    cross_correlation[epsilon] = 0
    base_cross_correlation[epsilon] = 0
    # Load the coordsys from the mesh associated to the data to make
    # sure it is well specified
    mesh = nib.load(spherical_mesh)
    coordsys = mesh.darrays[0].coordsys

    for input_fname, reference_fname in test_data.items():
        data_input = nib.load(input_fname)
        data_input.darrays = prepare_darrays(
            data_input.darrays, coordsys
        )
        # copy dataset in order to cope with a bug of MSM
        extra_data = nib.load(input_fname) 
        data_input.darrays.extend(prepare_darrays(
            extra_data.darrays, coordsys
        ))
        filename = str(Path(output_dir) / 'input_test.func.gii')
        data_input.to_filename(filename)
        input_fname_preprocessed = filename

        input_contrast_name = input_fname.split(subject_input)[1]
        test_transformed = os.path.join(
            output_dir,
            f"test_transformed_and_projected{input_contrast_name}"
        )
        cmd = ' '.join([
            f"{FSL_PATH}/fsl/bin/msmresample",
            f"{deformed} ",
            test_transformed,
            f"-labels {input_fname_preprocessed} ",
            f"-project {spherical_mesh}",
        ])
        exit_code = os.system(cmd)
        if exit_code != 0:
            raise RuntimeError(f"Failed to run MSM with command:\n{cmd}")

        # compute cross_correlation
        transformed_data_path = f"{test_transformed}.func.gii"
        transformed_data = nib.load(transformed_data_path).darrays[0].data
        reference_data = nib.load(reference_fname).darrays[0].data

        cross_correlation[epsilon] += stats.pearsonr(
            transformed_data, reference_data
        )[0]/len(test_data)
        base_cross_correlation[epsilon] += stats.pearsonr(
            data_input, reference_data
        )[0]/len(test_data)
print(cross_correlation)
print(base_cross_correlation)
f1 = plt.figure()
plt.xlabel("Lambda")
plt.ylabel("Pearson correlation")
plt.title(f"Test data, data number: {len(data)}")
plt.plot(
    epsilons, list(cross_correlation.values()),
    '-o', label="Pearson correlation after transformation"
)
plt.plot(
    epsilons, list(base_cross_correlation.values()),
    '-o', label="Pearson correlation before transformation"
)
plt.legend()
plt.savefig("lambda_optimization_test_data.png")
