import os
import json
import sys
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import random

# Add MSM package to sys in order to import it
# sys.path.append("/storage/store2/work/athual/repo/msm_on_ibc_wingrune")
sys.path.append("/mnt/e/Ecole Polytechnique/Parietal/code/msm_on_ibc")

from msm import model  # noqa: E402
from msm import utils  # noqa: E402

FSLDIR, _ = utils.check_fsl()


# Load spherical mesh produced with `mris_convert`
spherical_mesh = "./data/lh.sphere.gii"

# load data
subject_source = "sub-07"
subject_target = "sub-04"
data_path = "./data/"

# parameters
random.seed(42)
lambd = [0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1]
# if true, a new split will be created, if false, data will be loaded
new_split = False
# if true will run MSM, if false will use an existing run_MSM output
train_mode = True

# data - dictionary of type: source_fname -> target_fname

data = {}
sessions = ["ses-00", "ses-01", "ses-02", "ses-04"]
data_source = [
    os.path.join(data_path, file)
    for file in os.listdir(data_path)
    if subject_source in file
    and "lh" in file
    and (any(ses in file for ses in sessions))
]

data_target = [
    os.path.join(data_path, file)
    for file in os.listdir(data_path)
    if subject_target in file
    and "lh" in file
    and (any(ses in file for ses in sessions))
]

for source_fname in data_source:
    for target_fname in data_target:
        target_contrast = target_fname.split(subject_target)[1]
        source_contrast = source_fname.split(subject_source)[1]
        if target_contrast == source_contrast:
            data[source_fname] = target_fname

# Split data to train and test

if new_split:
    train_data_source = random.sample(list(data.keys()), int(len(data) * 0.8))

    train_data = {
        "source": np.stack(
            [
                nib.load(source_fname).darrays[0].data
                for source_fname in train_data_source
            ],
            axis=0,
        ),
        "target": np.stack(
            [
                nib.load(data[source_fname]).darrays[0].data
                for source_fname in train_data_source
            ],
            axis=0,
        )
    }

    test_data = {
        "source": np.stack(
            [
                nib.load(source_fname).darrays[0].data
                for source_fname in data
                if source_fname not in train_data_source
            ],
            axis=0,
        ),
        "target": np.stack(
            [
                nib.load(data[source_fname]).darrays[0].data
                for source_fname in data
                if source_fname not in train_data_source
            ],
            axis=0,
        )
    }

    with open("train.json", "w") as outfile:
        train_data["source"] = train_data["source"].tolist()
        train_data["target"] = train_data["target"].tolist()
        json.dump(train_data, outfile)

    with open("test.json", "w") as outfile:
        test_data["source"] = test_data["source"].tolist()
        test_data["target"] = test_data["target"].tolist()
        json.dump(test_data, outfile)
else:
    train_data_fname = "train.json"
    test_data_fname = "test.json"
    with open(train_data_fname) as json_file:
        train_data = json.load(json_file)
        train_data["source"] = np.asarray(train_data["source"])
        train_data["target"] = np.asarray(train_data["target"])

    with open(test_data_fname) as json_file:
        test_data = json.load(json_file)
        test_data["source"] = np.asarray(test_data["source"])
        test_data["target"] = np.asarray(test_data["target"])

epsilons = [0.01, 0.1, 1, 10, 100, 1000]

score_model = {}
base_score = {}  # cross-correlation without deformation

for epsilon in epsilons:

    # Define model
    msm = model.MSM(epsilon=epsilon)

    # running MSM with train data
    if train_mode:
        msm.fit(
            train_data["source"],
            train_data["target"],
            mesh_file=spherical_mesh,
            output_dir="../outputs/" f"test_outputs_lambda_{epsilon}",
        )

    # testing MSM with test data

    output_dir = "../outputs/" f"test_outputs_lambda_{epsilon}"
    msm.load_model(
        model_path=os.path.join(output_dir, "transformed_in_mesh.surf.gii"),
        mesh_path=spherical_mesh
    )

    # Transform map
    transformed_map = msm.predict(test_data["source"])

    # Compute R2 score before and after transformation

    score_model[epsilon] = np.mean(
        msm.score(transformed_map, test_data["target"])
    )
    base_score[epsilon] = np.mean(
        msm.score(test_data["source"], test_data["target"])
    )

print(score_model)
print(base_score)
f1 = plt.figure()
plt.xlabel("Lambda")
plt.ylabel("R2 score")
plt.title(f"Test data, data number: {len(data)}")
plt.plot(
    epsilons,
    list(score_model.values()),
    "-o",
    label="R2 score after transformation",
)
plt.plot(
    epsilons,
    list(base_score.values()),
    "-o",
    label="R2 score before transformation",
)
plt.legend()
plt.savefig("lambda_optimization_test_data.png")
