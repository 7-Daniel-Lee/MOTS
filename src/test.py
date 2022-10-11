'''
1. establish best mapping between the gt cluster and the hypothesis cluster? 
Do we really need?
'''
import numpy as np
from scipy.optimize import linear_sum_assignment

# 只接受id，没有输入点，新的代码需要改过来！


def eval_sequence(fields, data, threshold):
    """Calculates CLEAR metrics for one sequence"""
    # Initialise results
    res = {}
    for field in fields:
        res[field] = 0

    # Return result quickly if tracker or gt sequence is empty
    if data['num_tracker_dets'] == 0:
        # res['CLR_FN'] = data['num_gt_dets']
        # res['ML'] = data['num_gt_ids']
        # res['MLR'] = 1.0
        return res
    if data['num_gt_dets'] == 0:
        # res['CLR_FP'] = data['num_tracker_dets']
        # res['MLR'] = 1.0
        return res

    # Variables counting global association
    num_gt_ids = data['num_gt_ids']

    # Note that IDSWs are counted based on the last time each gt_id was present (any number of frames previously),
    # but are only used in matching to continue current tracks based on the gt_id in the single previous timestep.
    prev_tracker_id = np.nan * np.zeros(num_gt_ids)  # For scoring IDSW
    prev_timestep_tracker_id = np.nan * np.zeros(num_gt_ids)  # For matching IDSW

    # Calculate scores for each timestep
    for t, (gt_ids_t, tracker_ids_t) in enumerate(zip(data['gt_ids'], data['tracker_ids'])): 
        # Deal with the case that there are no gt_det/tracker_det in a timestep.
        if len(gt_ids_t) == 0:
            res['CLR_FP'] += len(tracker_ids_t)
            continue
        if len(tracker_ids_t) == 0:
            res['CLR_FN'] += len(gt_ids_t)
            continue

        #######################################################################################################
        # Calc score matrix to first minimise IDSWs from previous frame, and then maximise MOTP secondarily????

        # calculate similarity

        similarity = data['similarity_scores'][t]   ### what is similarity??   should be calculated with IOU between bounding boxes
        score_mat = (tracker_ids_t[np.newaxis, :] == prev_timestep_tracker_id[gt_ids_t[:, np.newaxis]]) # 
        score_mat = 1000 * score_mat + similarity
        score_mat[similarity < threshold - np.finfo('float').eps] = 0

        # Hungarian algorithm to find best matches ????
        match_rows, match_cols = linear_sum_assignment(-score_mat)
        actually_matched_mask = score_mat[match_rows, match_cols] > 0 + np.finfo('float').eps
        match_rows = match_rows[actually_matched_mask]
        match_cols = match_cols[actually_matched_mask]

        ######################################################################################################

        matched_gt_ids = gt_ids_t[match_rows]
        matched_tracker_ids = tracker_ids_t[match_cols]

        # Calc IDSW for MOTA
        prev_matched_tracker_ids = prev_tracker_id[matched_gt_ids]
        is_idsw = (np.logical_not(np.isnan(prev_matched_tracker_ids))) & (
            np.not_equal(matched_tracker_ids, prev_matched_tracker_ids))
        res['IDSW'] += np.sum(is_idsw)

        # Calculate and accumulate basic statistics
        num_matches = len(matched_gt_ids)
        res['CLR_TP'] += num_matches
        res['CLR_FN'] += len(gt_ids_t) - num_matches
        res['CLR_FP'] += len(tracker_ids_t) - num_matches
        if num_matches > 0:
            res['MOTP_sum'] += sum(similarity[match_rows, match_cols])

    res['CLR_Frames'] = data['num_timesteps']

    # Calculate final CLEAR scores
    res['MOTA'] = (res['CLR_TP'] - res['CLR_FP'] - res['IDSW']) / np.maximum(1.0, res['CLR_TP'] + res['CLR_FN'])
    return res
