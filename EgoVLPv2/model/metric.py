# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


"""Module for computing performance metrics

"""
import math
import numbers
from pathlib import Path
# import ipdb
import numpy as np
import torch
import scipy.stats
from sklearn.metrics import average_precision_score
# import ipdb
# import pdb
import os
import pandas as pd
import pickle
from utils import nDCG, mAP  # Assuming these are correctly imported from your utils/ directory


def t2v_metrics(sims, query_masks=None):
    """Compute retrieval metrics from a similiarity matrix.

    Args:
        sims (th.Tensor): N x M matrix of similarities between embeddings, where
             x_{i,j} = <text_embd[i], vid_embed[j]>
        query_masks (th.Tensor): mask any missing queries from the dataset (two videos
             in MSRVTT only have 19, rather than 20 captions)

    Returns:
        (dict[str:float]): retrieval metrics
    """
    assert sims.ndim == 2, "expected a matrix"
    num_queries, num_vids = sims.shape
    dists = -sims
    sorted_dists = np.sort(dists, axis=1)

    # The indices are computed such that they slice out the ground truth distances
    # from the psuedo-rectangular dist matrix
    queries_per_video = num_queries // num_vids
    gt_idx = [[np.ravel_multi_index([ii, jj], (num_queries, num_vids))
               for ii in range(jj * queries_per_video, (jj + 1) * queries_per_video)]
              for jj in range(num_vids)]
    gt_idx = np.array(gt_idx)
    gt_dists = dists.reshape(-1)[gt_idx.reshape(-1)]
    gt_dists = gt_dists[:, np.newaxis]
    rows, cols = np.where((sorted_dists - gt_dists) == 0)  # find column position of GT

    break_ties = "optimistically"
    if rows.size > num_queries:
        assert np.unique(rows).size == num_queries, "issue in metric evaluation"
        if break_ties == "optimistically":
            _, idx = np.unique(rows, return_index=True)
            cols = cols[idx]
        elif break_ties == "averaging":
            locs = np.argwhere((sorted_dists - gt_dists) == 0)
            steps = np.diff(locs[:, 0])
            splits = np.nonzero(steps)[0] + 1
            splits = np.insert(splits, 0, 0)
            summed_cols = np.add.reduceat(locs[:, 1], splits)
            counts = np.diff(np.append(splits, locs.shape[0]))
            avg_cols = summed_cols / counts
            cols = avg_cols

    msg = "expected ranks to match queries ({} vs {}) "
    if cols.size != num_queries:
        pass
    assert cols.size == num_queries, msg

    if query_masks is not None:
        assert query_masks.size == num_queries, "invalid query mask shape"
        cols = cols[query_masks.reshape(-1).astype(np.bool)]
        assert cols.size == query_masks.sum(), "masking was not applied correctly"
        num_queries = query_masks.sum()

    return cols2metrics(cols, num_queries)


def v2t_metrics(sims, query_masks=None):
    """Compute retrieval metrics from a similiarity matrix.

    Args:
        sims (th.Tensor): N x M matrix of similarities between embeddings, where
             x_{i,j} = <text_embd[i], vid_embed[j]>
        query_masks (th.Tensor): mask any missing captions from the dataset

    Returns:
        (dict[str:float]): retrieval metrics

    NOTES: We find the closest "GT caption" in the style of VSE, which corresponds
    to finding the rank of the closest relevant caption in embedding space:
    github.com/ryankiros/visual-semantic-embedding/blob/master/evaluation.py#L52-L56
    """
    sims = sims.T

    if False:
        sims = np.ones((3, 3))
        sims[0, 0] = 2
        sims[1, 1:2] = 2
        sims[2, :] = 2
        query_masks = None

    assert sims.ndim == 2, "expected a matrix"
    num_queries, num_caps = sims.shape
    dists = -sims
    caps_per_video = num_caps // num_queries
    break_ties = "averaging"

    MISSING_VAL = 1E8
    query_ranks = []
    for ii in range(num_queries):
        row_dists = dists[ii, :]
        if query_masks is not None:
            row_dists[np.logical_not(query_masks.reshape(-1))] = MISSING_VAL

        sorted_dists = np.sort(row_dists)

        min_rank = np.inf
        for jj in range(ii * caps_per_video, (ii + 1) * caps_per_video):
            if row_dists[jj] == MISSING_VAL:
                continue
            ranks = np.where((sorted_dists - row_dists[jj]) == 0)[0]
            if break_ties == "optimistically":
                rank = ranks[0]
            elif break_ties == "averaging":
                rank = ranks.mean()
            if rank < min_rank:
                min_rank = rank
        query_ranks.append(min_rank)
    query_ranks = np.array(query_ranks)

    if False:
        sorted_dists = np.sort(dists, axis=1)
        gt_dists_old = np.diag(dists)
        gt_dists_old = gt_dists_old[:, np.newaxis]
        rows_old, cols_old = np.where((sorted_dists - gt_dists_old) == 0)
        if rows_old.size > num_queries:
            _, idx = np.unique(rows_old, return_index=True)
            cols_old = cols_old[idx]
        num_diffs = (1 - (cols_old == query_ranks)).sum()
        msg = f"new metric doesn't match in {num_diffs} places"
        assert np.array_equal(cols_old, query_ranks), msg

        import sys
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        sys.path.insert(0, str(Path.home() / "coding/src/zsvision/python"))
        from zsvision.zs_iterm import zs_dispFig
        plt.matshow(dists)
        zs_dispFig()

    return cols2metrics(query_ranks, num_queries)


def egomcq_accuracy_metrics_ensemble(preds, labels, types):
    metrics = {}
    type_list = torch.unique(types)
    group_list = ["Inter-video", "Intra-video"]
    for type_i, group_i in zip(type_list, group_list):
        correct = 0
        total = 0
        for pred, label, type in zip(preds, labels, types):
            if type == type_i:
                pred_ = torch.argmax(pred)
                if pred_.item() == label.item():
                    correct += 1
                total += 1
        accuracy = correct / total
        metrics[group_i] = accuracy * 100
    return metrics


def egomcq_accuracy_metrics_vtm(preds, labels, types):
    metrics = {}
    type_list = torch.unique(types)
    group_list = ["Inter-video", "Intra-video"]
    for type_i, group_i in zip(type_list, group_list):
        correct = 0
        total = 0
        for pred, label, type in zip(preds, labels, types):
            if type == type_i:
                pred_ = torch.argmax(pred)
                if pred_.item() == label.item():
                    correct += 1
                total += 1
        accuracy = correct / total
        metrics[group_i] = accuracy * 100
    return metrics


def initialise_nDCG_values(relevancy_matrix):
    vis_k_counts = nDCG.calculate_k_counts(relevancy_matrix)
    txt_k_counts = nDCG.calculate_k_counts(relevancy_matrix.T)

    vis_IDCG = nDCG.calculate_IDCG(relevancy_matrix, vis_k_counts)
    txt_IDCG = nDCG.calculate_IDCG(relevancy_matrix.T, txt_k_counts)

    k_counts_dict = {'v': vis_k_counts, 't': txt_k_counts}
    IDCG_dict = {'v': vis_IDCG, 't': txt_IDCG}

    return IDCG_dict, k_counts_dict


def initialise_jpose_nDCG_values(relevancy_matrix):
    action_IDCG, action_k_values = initialise_nDCG_values(relevancy_matrix)

    dataset = {}
    dataset['action'] = {}
    dataset['action']['IDCG'] = action_IDCG
    dataset['action']['k_values'] = action_k_values
    return dataset


# --- MODIFIED mir_metrics_vtc START ---
# This path must be correct for your generated PKL
ACTUAL_META_DIR = "/mnt/arc/yygx/pkgs_baselines/EgoVLPv2/EgoVLPv2/data/my_simulated_ek100_data/EK100/epic-kitchens-100-annotations/retrieval_annotations"


def mir_metrics_vtc(similarity_matrix, idx_arr):  # idx_arr are numerical video IDs
    metrics = {}

    # --- Load Relevancy Matrix (your generated PKL) ---
    relevancy_pkl_path = os.path.join(ACTUAL_META_DIR, "relevancy/caption_relevancy_EPIC_100_retrieval_test.pkl")
    try:
        with open(relevancy_pkl_path, 'rb') as pkl_file:
            relevancy = pickle.load(pkl_file)  # This is your (num_queries x num_clips) numpy array
    except FileNotFoundError:
        print(f"Error: Relevancy PKL file not found at {relevancy_pkl_path}. Cannot calculate metrics.")
        return {'nDCG_V2T': 0.0, 'nDCG_T2V': 0.0, 'nDCG_AVG': 0.0,
                'mAP_V2T': 0.0, 'mAP_T2V': 0.0, 'mAP_AVG': 0.0}  # Return 0s if file missing

    # Ensure relevancy is a numpy array for calculations
    if not isinstance(relevancy, np.ndarray):
        print("Warning: Loaded relevancy is not a NumPy array. Attempting conversion.")
        relevancy = np.array(relevancy)

    # Validate shapes before passing to metric functions
    if similarity_matrix.shape != relevancy.shape:
        print(
            f"Error: Similarity matrix shape {similarity_matrix.shape} does not match relevancy shape {relevancy.shape}.")
        print("Metric calculation will be incorrect. Please check your metadata generation.")
        return {'nDCG_V2T': 0.0, 'nDCG_T2V': 0.0, 'nDCG_AVG': 0.0,
                'mAP_V2T': 0.0, 'mAP_T2V': 0.0, 'mAP_AVG': 0.0}

    # Similarity matrix needs to be normalized to 0-1 range for mAP/nDCG (if not already)
    normalized_similarity_matrix = (similarity_matrix + 1) / 2

    # --- DIRECTLY PASS ALIGNED MATRICES TO METRIC FUNCTIONS ---
    # The `idx_arr` (numerical video IDs) is not directly used for `nDCG`/`mAP` *calculation* here,
    # as the matrices are assumed to be pre-aligned.

    # The `initialise_jpose_nDCG_values` function is called to get IDCG and k_values.
    # This requires `relevancy` to be passed.
    dataset = initialise_jpose_nDCG_values(relevancy)

    vis_nDCG = nDCG.calculate_nDCG(normalized_similarity_matrix, relevancy, dataset['action']['k_values']['v'],
                                   IDCG=dataset['action']['IDCG']['v'])
    txt_nDCG = nDCG.calculate_nDCG(normalized_similarity_matrix.T, relevancy.T, dataset['action']['k_values']['t'],
                                   IDCG=dataset['action']['IDCG']['t'])
    metrics['nDCG_V2T'] = vis_nDCG * 100
    metrics['nDCG_T2V'] = txt_nDCG * 100
    metrics['nDCG_AVG'] = 100 * (vis_nDCG + txt_nDCG) / 2

    vis_mAP = mAP.calculate_mAP(normalized_similarity_matrix, relevancy)
    txt_mAP = mAP.calculate_mAP(normalized_similarity_matrix.T, relevancy.T)
    metrics['mAP_V2T'] = vis_mAP * 100
    metrics['mAP_T2V'] = txt_mAP * 100
    metrics['mAP_AVG'] = 100 * (vis_mAP + txt_mAP) / 2
    return metrics


# --- MODIFIED mir_metrics_vtc END ---


def map(submission_array, gt_array):
    """ Returns mAP, weighted mAP, and AP array """
    m_aps = []
    n_classes = submission_array.shape[1]
    for oc_i in range(n_classes):
        sorted_idxs = np.argsort(-submission_array[:, oc_i])
        tp = gt_array[:, oc_i][sorted_idxs] == 1
        fp = np.invert(tp)
        n_pos = tp.sum()
        if n_pos < 0.1:
            m_aps.append(float('nan'))
            continue
        fp.sum()
        f_pcs = np.cumsum(fp)
        t_pcs = np.cumsum(tp)
        prec = t_pcs / (f_pcs + t_pcs).astype(float)
        avg_prec = 0
        for i in range(submission_array.shape[0]):
            if tp[i]:
                avg_prec += prec[i]
        m_aps.append(avg_prec / n_pos.astype(float))
    m_aps = np.array(m_aps)
    m_ap = np.nanmean(m_aps)
    w_ap = (m_aps * gt_array.sum(axis=0) / gt_array.sum().sum().astype(float))
    return m_ap, w_ap, m_aps


def charades_metrics_vtc(submission_array, gt_array):
    """
    Approximate version of the charades evaluation function
    For precise numbers, use the submission file with the official matlab script
    """
    metrics = {}
    fix = submission_array.copy()
    empty = np.sum(gt_array, axis=1) == 0
    fix[empty, :] = np.NINF
    m_ap, w_ap, m_aps = map(fix, gt_array)
    metrics['mAP'] = m_ap
    return metrics


def charades_metrics_vtm(submission_array, gt_array):
    """
    Approximate version of the charades evaluation function
    For precise numbers, use the submission file with the official matlab script
    """
    metrics = {}
    fix = submission_array.copy()
    empty = np.sum(gt_array, axis=1) == 0
    fix[empty, :] = np.NINF
    m_ap, w_ap, m_aps = map(fix, gt_array)
    metrics['mAP'] = m_ap
    return metrics


def oscc_metrics(preds, labels):
    metrics = {}
    correct = 0
    total = 0
    for pred, label in zip(preds, labels):
        pred_ = torch.argmax(pred)
        if pred_.item() == label.item():
            correct += 1
        total += 1
    accuracy = correct / total
    metrics['accuracy'] = accuracy * 100
    return metrics


def pnr_metrics(
        preds,
        labels,
        sc_labels,
        fps,
        parent_start_frames,
        parent_end_frames,
        parent_pnr_frames,
):
    metrics = {}
    distance_list = list()
    for pred, label, sc_label, \
            parent_start_frame, parent_end_frame, parent_pnr_frame, \
            ind_fps in zip(
        preds,
        labels,
        sc_labels,
        parent_start_frames,
        parent_end_frames,
        parent_pnr_frames,
        fps
    ):
        if sc_label.item() == 1:
            keyframe_loc_pred = torch.argmax(pred).item()
            keyframe_loc_pred_mapped = (parent_end_frame - parent_start_frame) / 16 * keyframe_loc_pred
            keyframe_loc_pred_mapped = keyframe_loc_pred_mapped.item()
            gt = parent_pnr_frame.item() - parent_start_frame.item()
            err_frame = abs(keyframe_loc_pred_mapped - gt)
            err_sec = err_frame / ind_fps.item()
            distance_list.append(err_sec)
    if len(distance_list) == 0:
        metrics['keyframe_distance'] = np.mean(0.0)
    metrics['keyframe_distance'] = np.mean(distance_list)
    return metrics