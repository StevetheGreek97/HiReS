from shapely.strtree import STRtree
import numpy as np

def nms(parser, iou_matrix, iou_thresh=0.7):
    """
    Apply NMS and return indices to discard (suppress).
    """
    keep = set(range(len(parser)))

    for (i, j), iou_val in iou_matrix.items():
        if i >= j:  # Avoid duplicate/reversed pairs (0, 1) and (1, 0)
            continue
        if i not in keep or j not in keep:
            continue
        #if parser[i].class_id != parser[j].class_id: # Maybe I would do somthing with it?
        #    continue
        if iou_val > iou_thresh:
            conf_i = parser[i].confidence
            conf_j = parser[j].confidence
            if conf_i is None or conf_j is None:
                raise ValueError("Confidence scores are required for NMS.")

            if conf_i >= conf_j:
                keep.discard(j)
            else:
                keep.discard(i)

    # discarded = sorted(set(range(len(parser))) - keep) # Do you need the discarded ones?
    return keep

def compute_iou_matrix(matches, parser, return_dense=True):
    iou_matrix = {}

    for match_group in matches:
        for i in range(len(match_group)):
            for j in range(i + 1, len(match_group)):
                idx_i = match_group[i]
                idx_j = match_group[j]

                poly_i = parser[idx_i].polygon
                poly_j = parser[idx_j].polygon

                val = iou(poly_i, poly_j)

                iou_matrix[(idx_i, idx_j)] = val
                iou_matrix[(idx_j, idx_i)] = val  # optional symmetry

    if return_dense:
        N = len(parser)
        dense_iou = np.zeros((N, N))
        for (i, j), val in iou_matrix.items():
            dense_iou[i, j] = val
        return iou_matrix, dense_iou

    return iou_matrix, None

def iou(a, b):
    inter = a.intersection(b).area
    if inter == 0.0:
        return 0.0
    # union via inclusion–exclusion 
    return inter / (a.area + b.area - inter)

def get_matches(parser):
    tree = STRtree([item.polygon for item in parser])
    matches = set()
    parsed_items = list(parser.parse())

    for i in range(len(parsed_items)):
        intersecting = tree.query(parser[i].polygon)

        if len(intersecting) > 1:
            inter = list(intersecting)
            inter.sort()
            matches.add(tuple(inter))
    return matches

def run_nms(data, iou_thresh=0.7):
    matches = get_matches(data)
    iou_mtrx, _ = compute_iou_matrix(matches, data, return_dense=True)
    kept = nms(data, iou_mtrx, iou_thresh=iou_thresh)
    return [data[i] for i in sorted(kept)]
