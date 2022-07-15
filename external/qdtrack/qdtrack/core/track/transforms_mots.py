from collections import defaultdict


def track2result(bboxes, labels, segms, ids):
    valid_inds = ids > -1
    bboxes = bboxes[valid_inds].cpu().numpy()
    labels = labels[valid_inds].cpu().numpy()
    segms = [segms[i] for i in range(len(segms)) if valid_inds[i] == True]
    ids = ids[valid_inds].cpu().numpy()

    outputs = defaultdict(list)
    for bbox, label, segm, id in zip(bboxes, labels, segms, ids):
        outputs[id] = dict(bbox=bbox, label=label, segm=segm)
    return outputs

def segtrack2result(bboxes, labels, segms, ids):
    outputs = track2result(bboxes, labels, segms, ids)
    return outputs

