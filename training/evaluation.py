import csv
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageOps
from scipy.ndimage import label
from skimage.morphology import skeletonize
from data.paired_pseudo import load_sample, restore_probability
from data.fundus_manifest import atomic_json
from pseudo_labels.quality import save_png


def ratio(n, d):
    return float(n/d) if d else None


def pr_auc(truth, scores):
    truth, scores = np.asarray(truth, bool), np.asarray(scores, np.float64)
    positives = int(truth.sum())
    if not positives:
        return None
    order = np.argsort(-scores, kind='stable')
    y, p = truth[order], scores[order]
    idx = np.r_[np.flatnonzero(np.diff(p)), len(y)-1]
    tp = np.cumsum(y)[idx]
    precision = np.r_[1., tp/(idx+1)]
    recall = np.r_[0., tp/positives]
    # Trapezoidal PR curve with grouped ties; not mislabeled average precision.
    return float(np.sum((precision[1:]+precision[:-1])*.5*np.diff(recall)))


def segmentation_metrics(prob, mask, valid, threshold):
    truth, pred = mask > 0, prob >= threshold
    v = valid > 0
    tp = int((pred & truth & v).sum())
    fp = int((pred & ~truth & v).sum())
    fn = int((~pred & truth & v).sum())
    tn = int((~pred & ~truth & v).sum())
    p, g = pred & v, truth & v
    skp, skg = skeletonize(p), skeletonize(g)
    topology_precision = ratio((skp & g).sum(), skp.sum())
    topology_recall = ratio((skg & p).sum(), skg.sum())
    cldice = None if topology_precision is None or topology_recall is None else ratio(2*topology_precision*topology_recall, topology_precision+topology_recall)
    dice = ratio(2*tp, 2*tp+fp+fn)
    return dict(dice=dice, score=None if dice is None else 100*dice, iou=ratio(tp, tp+fp+fn),
                recall=ratio(tp, tp+fn), precision=ratio(tp, tp+fp), fnr=ratio(fn, tp+fn),
                fdr=ratio(fp, tp+fp), fpr=ratio(fp, fp+tn), tp=tp, fp=fp, fn=fn, tn=tn,
                pr_auc=pr_auc(truth[v], prob[v]) if v.any() else None, cldice=cldice,
                predicted_components=int(label(p)[1]), target_components=int(label(g)[1]))


@torch.no_grad()
def validate_student(student, rows, size, device, threshold):
    student.eval()
    values = []
    for row in rows:
        b = load_sample(row, size, device, target='gt_mask_path')
        p = restore_probability(student.forward_probability(b['image']), b['info'])
        with Image.open(row['gt_mask_path']) as im:
            m = np.asarray(im.convert('L'))
        with Image.open(row['fov_path']) as im:
            v = np.asarray(im.convert('L'))
        score = segmentation_metrics(p, m, v, threshold)['dice']
        if score is not None:
            values.append(score)
    if not values:
        raise ValueError('验证集没有定义良好的Dice，不能选择检查点')
    return float(np.mean(values))


@torch.no_grad()
def evaluate(student, rows, size, device, threshold, folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    student.eval()
    results = []
    for row in rows:
        b = load_sample(row, size, device, target='gt_mask_path')
        p = restore_probability(student.forward_probability(b['image']), b['info'])
        with Image.open(row['gt_mask_path']) as im:
            m = np.asarray(im.convert('L'))
        with Image.open(row['fov_path']) as im:
            v = np.asarray(im.convert('L'))
        metrics = segmentation_metrics(p, m, v, threshold)
        results.append(dict(sample_id=row['sample_id'], **metrics))
        sid = row['sample_id']
        save_png(folder/(sid+'.png'), (p >= threshold).astype(np.uint8)*255)
        np.save(folder/(sid+'.prob.npy'), p.astype(np.float32))
        with Image.open(row['image_path']) as im:
            rgb = np.asarray(ImageOps.exif_transpose(im).convert('RGB')).copy()
        pred, gt, valid = p >= threshold, m > 0, v > 0
        errors = np.zeros_like(rgb)
        errors[pred & gt & valid] = [0, 255, 0]
        errors[~pred & gt & valid] = [255, 0, 0]
        errors[pred & ~gt & valid] = [0, 0, 255]
        overlay = rgb.copy()
        overlay[pred] = (.5*rgb[pred]+.5*np.array([0, 255, 0])).astype(np.uint8)
        panel = np.concatenate([rgb, np.repeat((pred*255).astype(np.uint8)[..., None], 3, 2),
                                np.repeat(m[..., None], 3, 2), overlay, errors], axis=1)
        Image.fromarray(panel).save(folder/(sid+'.comparison.png'))
    if not results:
        raise ValueError('指定划分没有人工标注评估图')
    with (folder/'metrics.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    means, defined = {}, {}
    for k in results[0]:
        if k == 'sample_id':
            continue
        values = [r[k] for r in results if r[k] is not None]
        means[k] = float(np.mean(values)) if values else None
        defined[k] = len(values)
    atomic_json(folder/'summary.json', dict(equal_image_mean=means, defined_samples=defined, total_samples=len(rows),
                                            threshold=threshold, undefined_policy='零分母记录null/CSV空值，均值仅统计已定义图并报告样本数',
                                            connectivity='clDice和连通分量；不等价于临床拓扑正确性'))
    return dict(samples=len(results), dice=means['dice'], output=str(folder))
