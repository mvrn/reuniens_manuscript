#!/usr/bin/env python

import os
import nipype.interfaces.io as nio
import nipype.interfaces.fsl as fsl
import nipype.interfaces.utility as util
import nipype.pipeline.engine as pe
import nipype.interfaces.freesurfer as fs
import nipype.interfaces.fsl as fsl
from nipype.interfaces.io import DataGrabber, DataSink
from nipype.interfaces import ants
from nipype.pipeline.engine import Workflow, Node, MapNode, JoinNode
from nipype.interfaces.c3 import C3dAffineTool
from nipype.interfaces.utility import IdentityInterface, Function, Merge
from mattfeld_utility_workflows.fs_skullstrip_util import (
    create_freesurfer_skullstrip_workflow,
)

n_clus = 8


def kmeans_func(subject_id, n_clus, thal_masks):
    n_clus = 8
    import os
    import numpy as np
    import nibabel as nb
    import scipy as sp
    import pandas as pd
    from sklearn.cluster import KMeans
    from glob import glob

    root_dir = "/home/data/madlab/McMakin_EMU/derivatives/dwi/mvrn"

    # setting up the empty list that will hold the output files,
    # target clusters, and output features from the hemispheric division
    out_files = []
    target_clusters = []
    out_features = []

    # Iterating over each hemisphere target so we can analyze them separately.
    for hemi in ["0", "1"]:
        if hemi == "0":
            out_file = os.path.join(os.getcwd(), f"kmeans_thallh_{n_clus}-clus.nii.gz")
            curr_thal_mask = thal_masks[0]
        else:
            out_file = os.path.join(os.getcwd(), f"kmeans_thalrh_{n_clus}-clus.nii.gz")
            curr_thal_mask = thal_masks[1]

        # Getting the list of targets for every hemisphere
        targets = sorted(
            glob(
                root_dir
                + f"/probtrakx_thal_v2/{subject_id}/emuprobX/targets"
                + f"/_subject_id_{subject_id}/_pbx2{hemi}/*.nii.gz"
            )
        )
        print(targets)

        # Loading mask for and getting affine and header (info about data dimensions).
        # Then reshaping
        hemi_mask_img = nb.load(curr_thal_mask)

        hemi_mask_data_affine = hemi_mask_img.affine
        hemi_mask_data_header = hemi_mask_img.header

        hemi_mask_data = hemi_mask_img.get_data()
        hemi_mask_data_dims = hemi_mask_data.shape

        # Replace posterior slices of mask with 0 to avoid incorporating
        # voxels that are physically adjacent to the HPC
        # These voxels could artifically inflate target hits.
        hemi_mask_data[:, 0:60, :] = 0

        # Iterate over files that will hold target files (see above)
        for i, file_name in enumerate(targets):
            # Loading and renaming the current target files
            curr_targ_img = nb.load(file_name)
            # Getting affine and header
            curr_targ_affine = curr_targ_img.affine
            curr_targ_header = curr_targ_img.header
            # Getting target data
            curr_targ_img_data = curr_targ_img.get_data()

            # input_thal_array = np.concatenate([(nb.load(lh).get_data()+nb.load(rh).get_data())[hemi_mask_data>0].
            #                               reshape(np.sum(hemi_mask_data), 1) \
            #                            for lh, rh in bilat_targ_files], axis = 1)

            # input_thal_array = curr_targ_img_data[hemi_mask_data > 0].reshape(np.sum(hemi_mask_data), 0)

            # Mask the curr_targ_img_data by the current hemisphere mask
            curr_targ_img_data_onlythal = curr_targ_img_data[hemi_mask_data > 0]

            # Put the masked thalamus data into a n-by-y  matrix
            # n = number of voxels
            # y = number of features (i.e., connections to targets)
            if i == 0:
                input_array = curr_targ_img_data_onlythal
            else:
                input_array = np.column_stack(
                    (input_array, curr_targ_img_data_onlythal)
                )

        rl_thal = hemi_mask_data_dims[0]
        ap_thal = hemi_mask_data_dims[1]
        is_thal = hemi_mask_data_dims[2]

        # Run the kmeans algorithm
        thal_kmeans = KMeans(init="k-means++", n_clusters=n_clus, n_init=10)
        # adds 1 to the cluster assignment to prevent 0 cluster from fading
        # in to non-thal volume
        thal_kmeans_out = thal_kmeans.fit_predict(input_array) + 1
        # making a vector out of the thal mask
        thal_kmeans_results = hemi_mask_data.copy().reshape(
            rl_thal * ap_thal * is_thal, 1
        )

        # Replacing kmeans values wherever there is a 1
        counter = 0
        for idx, value in enumerate(thal_kmeans_results):
            if value > 0:
                thal_kmeans_results[idx] = thal_kmeans_out[counter]
                counter += 1
        # Reshape the column label data back into the original n x y x n shape
        thal_kmeans_results = thal_kmeans_results.reshape(hemi_mask_data_dims)

        # Save the newly labeled thal voxels as a .nii.gz file
        thal_kmeans_img = nb.Nifti1Image(
            thal_kmeans_results, curr_targ_affine, header=curr_targ_header
        )
        thal_kmeans_img.to_filename(out_file)
        out_files.append(out_file)

        for x in range(len(thal_kmeans.cluster_centers_[:, 0])):
            curr_feature_z = []
            for y in range(len(thal_kmeans.cluster_centers_[0, :])):
                curr_z = (
                    thal_kmeans.cluster_centers_[x, y]
                    - np.mean(thal_kmeans.cluster_centers_[:, y])
                ) / np.std(thal_kmeans.cluster_centers_[:, y])
                curr_feature_z.append(curr_z)
            if x == 0:
                all_features_z = curr_feature_z
            else:
                all_features_z = np.vstack((all_features_z, curr_feature_z))

        if hemi == "0":
            hemi_name = "lh"
        elif hemi == "1":
            hemi_name = "rh"
        fname = "{0}_all_features_{1}-clus.csv".format(hemi_name, n_clus)
        np.savetxt(fname, all_features_z)
        out_features.append(os.path.abspath(fname))

        curr_hemi_limbicthal_targ_clusters = []
        for i, curr_feat in enumerate(all_features_z):

            # LABELS:
            # MTL: EC=1, pHC=9, Amyg = 16, HC=17, Nucleus Accumbens=18
            # mPFC:  mOrb=5, rACC = 13
            # controls: Paracentral=8, postcentral= 11, precentral=12, [for later script parietal=10, Superfront = 14]

            if (
                (
                    curr_feat[1] > 0.0
                    and curr_feat[10] > 0.0
                    and curr_feat[17] > 0.0
                    and curr_feat[18] > 0.0
                    and curr_feat[19] > 0.0
                )
                and (curr_feat[7] > 0.0 and curr_feat[15] > 0.0)
            ) and not (
                curr_feat[9] > 0.0
                or curr_feat[12] > 0.0
                or curr_feat[13] > 0.0
                or curr_feat[11] > 0.0
                or curr_feat[15] > 0.0
            ):
                curr_hemi_limbicthal_targ_clusters.append(i + 1)
            elif (
                (
                    curr_feat[1] > 0.0
                    or curr_feat[10] > 0.0
                    or curr_feat[17] > 0.0
                    or curr_feat[18] > 0.0
                    or curr_feat[19] > 0.0
                )
                and (curr_feat[7] > 0.0 or curr_feat[15] > 0.0)
            ) and not (
                curr_feat[9] > 0.0
                or curr_feat[12] > 0.0
                or curr_feat[13] > 0.0
                or curr_feat[11] > 0.0
                or curr_feat[15] > 0.0
            ):
                curr_hemi_limbicthal_targ_clusters.append(i + 1)
            elif (
                (
                    curr_feat[1] > 0.0
                    or curr_feat[10] > 0.0
                    or curr_feat[17] > 0.0
                    or curr_feat[18] > 0.0
                    or curr_feat[19] > 0.0
                )
                or (curr_feat[7] > 0.0 or curr_feat[15] > 0.0)
            ) and not (
                curr_feat[9] > 0.0
                or curr_feat[12] > 0.0
                or curr_feat[13] > 0.0
                or curr_feat[11] > 0.0
                or curr_feat[15] > 0.0
            ):
                curr_hemi_limbicthal_targ_clusters.append(i + 1)

            # CaudalAntCing = 0
            # Entorhinal = 1
            # Fusiform = 2
            # Insula = 3
            # LatFront = 4
            # LatOccipital = 5
            # MedOccipital = 6
            # MedOrbFront = 7
            # MedialPost = 8
            # Paracentral = 9
            # Parahipp = 10
            # Parietal = 11
            # Postcentral = 12
            # Precentral = 13
            # RAcc = 14
            # SupFrontal = 15
            # Temporal = 16
            # Amygdala = 17
            # HPC = 18
            # NucleusAcc = 19

            if len(curr_hemi_limbicthal_targ_clusters) == 0:
                curr_hemi_limbicthal_targ_clusters.append(999)

        target_clusters.append(curr_hemi_limbicthal_targ_clusters)
    return out_files, target_clusters, out_features


def extract_lh_values(target_clusters):
    match_values = target_clusters[0]
    for i, item in enumerate(match_values):
        if len(item) == 0:
            match_values[i] = [999]
    return match_values


def extract_rh_values(target_clusters):
    match_values = target_clusters[1]
    for i, item in enumerate(match_values):
        if len(item) == 0:
            match_values[i] = [999]
    return match_values


def select_lh_file(out_files):
    lh_file = out_files[0]
    return lh_file


def select_rh_file(out_files):
    rh_file = out_files[1]
    return rh_file


def pickfirst(in_file):
    if isinstance(in_file, list):
        return in_file[0]
    else:
        return in_file


root_dir = "/home/data/madlab/McMakin_EMU/derivatives/dwi/mvrn"
sink_directory = f"{root_dir}/mvrn_preed_thal_hemikmeans_82021_v2"
work_directory = "/scratch/madlab/crash/Vanessa/mvrn_thal_hemikmeans_82021_v2"
data_dir = f"{root_dir}/probtrakx_thal_v2"
subjects_dir = "/home/data/madlab/McMakin_EMU/derivatives/freesurfer/"
proj_dir = "/home/data/madlab/McMakin_EMU/derivatives"

subject_id = os.listdir(data_dir)

# Set the working directory
if not os.path.exists(work_directory):
    os.makedirs(work_directory)

emu_thal_kmeans_wf = pe.Workflow(name="emu_thal_kmeans_wf")
emu_thal_kmeans_wf.base_dir = work_directory

# registering subjects and dictionary for the datasource node
info = dict(
    b0=[["subject_id", "eddy_corrected_roi"]],
    ants_warp=[["subject_id", "subject_id", "output_Composite"]],
    aparcaseg=[["subject_id", "aparc+aseg_thresh_warped_thresh"]],
    thal_masks=[["subject_id", "subject_id"]],
)

thal_subjs_iterable = pe.Node(
    IdentityInterface(fields=["subject_id"], mandatory_inputs=True),
    name="thal_subjs_iterable",
)
thal_subjs_iterable.iterables = ("subject_id", subject_id)

# creating node to grab the data files for each subject to get the standard space brain
datasource = Node(
    DataGrabber(infields=["subject_id"], outfields=list(info.keys())), name="datasource"
)
datasource.inputs.base_directory = os.path.abspath(proj_dir)
# loading mask
datasource.inputs.field_template = dict(
    b0="dwi/mvrn/probtrakx_thal_v2/%s/emuprobX/b0/%s.nii.gz",
    ants_warp="norm_anat/%s/anat2targ_xfm/_subject_id_%s/%s.h5",
    aparcaseg="dwi/bedpostx/%s/emubpX/mask/_fs_threshold20/%s.nii",
    thal_masks="dwi/mvrn/probtrakx_thal_v2/%s/emuprobX/thal_mask/_subject_id_%s/*/*",
)
datasource.inputs.template = "*"
datasource.inputs.sort_filelist = True
datasource.inputs.template_args = info
emu_thal_kmeans_wf.connect(thal_subjs_iterable, "subject_id", datasource, "subject_id")

# Creating BBReg node
# Calculate the transformation matrix from DWI space to FreeSurfer space (from PBX)
# coregistering a volume from DWI space to FS anatomical space
# bbreg coregisters the mean functional image created by the realignment of subjecs' surfaces
fs_register = Node(fs.BBRegister(init="fsl"), name="fs_register")
fs_register.inputs.contrast_type = "t2"
fs_register.inputs.out_fsl_file = True
fs_register.inputs.subjects_dir = subjects_dir
emu_thal_kmeans_wf.connect(datasource, "b0", fs_register, "source_file")
emu_thal_kmeans_wf.connect(thal_subjs_iterable, "subject_id", fs_register, "subject_id")

# Extract thalamus seed masks from aparc+aseg.nii.gz file
whole_thal_mask = pe.MapNode(
    fs.Binarize(), iterfield=["match", "binary_file"], name="whole_thal_mask"
)
whole_thal_mask.inputs.match = [[10], [49]]
whole_thal_mask.inputs.binary_file = ["lft_thal.nii.gz", "rt_thal.nii.gz"]
emu_thal_kmeans_wf.connect(datasource, "aparcaseg", whole_thal_mask, "in_file")

# create a function node that runs the kmeans algorithm
run_kmeans = Node(
    Function(
        input_names=["subject_id", "n_clus", "thal_masks"],
        output_names=["out_files", "target_clusters", "out_features"],
        function=kmeans_func,
    ),
    name="run_kmeans",
)
run_kmeans.inputs.n_clus = n_clus
run_kmeans.plugin_args = {
    "sbatch_args": ("-p IB_40C_1.5T --qos pq_madlab --account iacc_madlab -N 1 -n 1")
}
emu_thal_kmeans_wf.connect(thal_subjs_iterable, "subject_id", run_kmeans, "subject_id")
emu_thal_kmeans_wf.connect(datasource, "thal_masks", run_kmeans, "thal_masks")

# create a node to binarize LH and RH targeted clusters (aka midline clusters)
midline_thal_bin = pe.MapNode(
    fs.Binarize(),
    iterfield=["binary_file", "in_file", "match"],
    name="midline_thal_bin",
)
midline_thal_bin.inputs.subjects_dir = subjects_dir
midline_thal_bin.inputs.binary_file = [
    "lft_limbic_thal_bin.nii.gz",
    "rt_limbic_thal_bin.nii.gz",
]
# emu_thal_kmeans_wf.connect(thal_subjs_iterable, 'subject_id', midline_thal_bin, 'subject_id')
emu_thal_kmeans_wf.connect(run_kmeans, "out_files", midline_thal_bin, "in_file")
emu_thal_kmeans_wf.connect(run_kmeans, "target_clusters", midline_thal_bin, "match")

# Add together the left and right hemisphere masks for a single mask
bilat_limbic_thal_mask_combine = pe.Node(
    fsl.ImageMaths(op_string="-add"), name="bilat_limbic_thal_mask_combine"
)
bilat_limbic_thal_mask_combine.inputs.out_file = "bilat_limbic_thal_bin.nii.gz"
emu_thal_kmeans_wf.connect(
    midline_thal_bin,
    ("binary_file", select_lh_file),
    bilat_limbic_thal_mask_combine,
    "in_file",
)
emu_thal_kmeans_wf.connect(
    midline_thal_bin,
    ("binary_file", select_rh_file),
    bilat_limbic_thal_mask_combine,
    "in_file2",
)

# Skull strip the freesurfer brain
fs_skullstrip_wf = create_freesurfer_skullstrip_workflow()
fs_skullstrip_wf.inputs.inputspec.subjects_dir = subjects_dir
emu_thal_kmeans_wf.connect(
    thal_subjs_iterable, "subject_id", fs_skullstrip_wf, "inputspec.subject_id"
)

# Node: convert2itk
convert2itk = Node(C3dAffineTool(), name="convert2itk")
convert2itk.inputs.fsl2ras = True
convert2itk.inputs.itk_transform = True
emu_thal_kmeans_wf.connect(fs_register, "out_fsl_file", convert2itk, "transform_file")
emu_thal_kmeans_wf.connect(datasource, "b0", convert2itk, "source_file")
emu_thal_kmeans_wf.connect(
    fs_skullstrip_wf, "outputspec.skullstripped_file", convert2itk, "reference_file"
)

# Node: concatenate the affine and ants transforms into a list
merge_xfm = Node(Merge(2), iterfield=["in2"], name="merge_xfm")
emu_thal_kmeans_wf.connect(convert2itk, "itk_transform", merge_xfm, "in2")
emu_thal_kmeans_wf.connect(datasource, "ants_warp", merge_xfm, "in1")

# MapNode: Warp binary limbic thalamus kmeans masks to target
bin2targ = MapNode(ants.ApplyTransforms(), iterfield="input_image", name="bin2targ")
bin2targ.inputs.input_image_type = 3
bin2targ.inputs.interpolation = "NearestNeighbor"
bin2targ.inputs.invert_transform_flags = [False, False]
bin2targ.inputs.args = "--float"
bin2targ.inputs.reference_image = (
    "/home/data/madlab/McMakin_EMU/derivatives/study_template/antsTMPL_template.nii.gz"
)
emu_thal_kmeans_wf.connect(midline_thal_bin, "binary_file", bin2targ, "input_image")
emu_thal_kmeans_wf.connect(merge_xfm, "out", bin2targ, "transforms")


bilat_mask_warp2targ = Node(ants.ApplyTransforms(), name="bilat_mask_warp2targ")
bilat_mask_warp2targ.inputs.input_image_type = 3
bilat_mask_warp2targ.inputs.interpolation = "NearestNeighbor"
bilat_mask_warp2targ.inputs.invert_transform_flags = [False, False]
bilat_mask_warp2targ.inputs.args = "--float"
bilat_mask_warp2targ.inputs.reference_image = (
    "/home/data/madlab/McMakin_EMU/derivatives/study_template/antsTMPL_template.nii.gz"
)
emu_thal_kmeans_wf.connect(
    bilat_limbic_thal_mask_combine, "out_file", bilat_mask_warp2targ, "input_image"
)
emu_thal_kmeans_wf.connect(merge_xfm, "out", bilat_mask_warp2targ, "transforms")


# create a node to merge the binarized files in standard space
lh_thal_merge = pe.JoinNode(
    fsl.Merge(),
    joinsource="thal_subjs_iterable",
    joinfield="in_files",
    name="lh_thal_merge",
)
lh_thal_merge.inputs.dimension = "t"
lh_thal_merge.inputs.output_type = "NIFTI_GZ"
emu_thal_kmeans_wf.connect(
    bin2targ, ("output_image", select_lh_file), lh_thal_merge, "in_files"
)

# create a node to merge the binarized files in standard space
rh_thal_merge = pe.JoinNode(
    fsl.Merge(),
    joinsource="thal_subjs_iterable",
    joinfield="in_files",
    name="rh_thal_merge",
)
rh_thal_merge.inputs.dimension = "t"
rh_thal_merge.inputs.output_type = "NIFTI_GZ"
emu_thal_kmeans_wf.connect(
    bin2targ, ("output_image", select_rh_file), rh_thal_merge, "in_files"
)

# create a node to merge the binarized files in standard space
bilat_thal_merge = pe.JoinNode(
    fsl.Merge(),
    joinsource="thal_subjs_iterable",
    joinfield="in_files",
    name="bilat_thal_merge",
)
bilat_thal_merge.inputs.dimension = "t"
bilat_thal_merge.inputs.output_type = "NIFTI_GZ"
emu_thal_kmeans_wf.connect(
    bilat_mask_warp2targ, "output_image", bilat_thal_merge, "in_files"
)

# create a node to calculate the mean image for LH thal mask
lh_thal_mean = Node(fsl.MeanImage(), name="lh_thal_mean")
lh_thal_mean.inputs.dimension = "T"
lh_thal_mean.inputs.output_type = "NIFTI_GZ"
emu_thal_kmeans_wf.connect(lh_thal_merge, "merged_file", lh_thal_mean, "in_file")

# create a node to calculate the mean image for RH thal mask
rh_thal_mean = Node(fsl.MeanImage(), name="rh_thal_mean")
rh_thal_mean.inputs.dimension = "T"
rh_thal_mean.inputs.output_type = "NIFTI_GZ"
emu_thal_kmeans_wf.connect(rh_thal_merge, "merged_file", rh_thal_mean, "in_file")

# create a node to calculate the mean image for COMBINED thal mask
bilat_thal_mean = Node(fsl.MeanImage(), name="bilat_thal_mean")
bilat_thal_mean.inputs.dimension = "T"
bilat_thal_mean.inputs.output_type = "NIFTI_GZ"
emu_thal_kmeans_wf.connect(bilat_thal_merge, "merged_file", bilat_thal_mean, "in_file")

# create a datasink node to save everything
datasink = pe.Node(nio.DataSink(), name="datasink")
datasink.inputs.base_directory = os.path.abspath(sink_directory)
datasink.inputs.substitutions = [("_subject_id_", "")]

emu_thal_kmeans_wf.connect(thal_subjs_iterable, "subject_id", datasink, "container")
emu_thal_kmeans_wf.connect(
    midline_thal_bin, "binary_file", datasink, "dmri_space.@limbic_thal"
)
emu_thal_kmeans_wf.connect(
    bilat_limbic_thal_mask_combine, "out_file", datasink, "dmri_space.@bilimbic_thal"
)
emu_thal_kmeans_wf.connect(
    bilat_mask_warp2targ, "output_image", datasink, "bilat_warp.@bilimbic_thal"
)
emu_thal_kmeans_wf.connect(lh_thal_mean, "out_file", datasink, "avgmasks.@lhmask")
emu_thal_kmeans_wf.connect(rh_thal_mean, "out_file", datasink, "avgmask.@rhmask")
emu_thal_kmeans_wf.connect(bilat_thal_mean, "out_file", datasink, "avgmask.@bimask")
emu_thal_kmeans_wf.connect(run_kmeans, "out_features", datasink, "kmeans.@features")
emu_thal_kmeans_wf.connect(run_kmeans, "out_files", datasink, "kmeans.@masks")

emu_thal_kmeans_wf.config["execution"][
    "crashdump_dir"
] = "/scratch/madlab/crash/Vanessa/mvrn_thal_hemikmeans"
# Bless Adam.....
emu_thal_kmeans_wf.config["execution"]["crashfile_format"] = "txt"
emu_thal_kmeans_wf.run(
    plugin="SLURM",
    plugin_args={
        "sbatch_args": (
            "--partition IB_40C_512G --qos pq_madlab --account iacc_madlab -N 1 -n 1"
        ),
        "overwrite": True,
    },
)
