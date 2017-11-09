
import os
import torch as T
from inspect import signature

def cuda(obj):
    if os.getenv('USE_CUDA', None):
        if isinstance(obj, tuple):
            return tuple(cuda(o) for o in obj)
        elif hasattr(obj, 'cuda'):
            return obj.cuda()
    return obj

def tovar(*arrs, **kwargs):
    tensors = [(T.from_numpy(a) if isinstance(a, np.ndarray) else a) for a in arrs]
    vars_ = [T.autograd.Variable(t, **kwargs) for t in tensors]
    if os.getenv('USE_CUDA', None):
        vars_ = [v.cuda() for v in vars_]
    return vars_[0] if len(vars_) == 1 else vars_


def tonumpy(*vars_):
    arrs = [(v.data.cpu().numpy() if isinstance(v, T.autograd.Variable) else
             v.cpu().numpy() if T.is_tensor(v) else v) for v in vars_]
    return arrs[0] if len(arrs) == 1 else arrs


def normalize_contrast(x):
    '''
    x: ND Tensor (..., nchannels, nrows, ncols)
    '''
    max_x = x.max(-1, keepdim=True).max(-2, keepdim=True).max(-3, keepdim=True)
    min_x = x.min(-1, keepdim=True).min(-2, keepdim=True).min(-3, keepdim=True)
    return (x - min_x) / (max_x - min_x + 1e-5)


def intersection(a, b):
    assert np.all(tonumpy(a[..., 2]) >= 0)
    assert np.all(tonumpy(a[..., 3]) >= 0)
    assert np.all(tonumpy(b[..., 2]) >= 0)
    assert np.all(tonumpy(b[..., 3]) >= 0)

    x1 = T.max(a[..., 0], b[..., 0])
    y1 = T.max(a[..., 1], b[..., 1])
    x2 = T.min(a[..., 0] + a[..., 2], b[..., 0] + b[..., 2])
    y2 = T.min(a[..., 1] + a[..., 3], b[..., 1] + b[..., 3])
    w = (x2 - x1).clamp(min=0)
    h = (y2 - y1).clamp(min=0)
    return w * h


def intersection_within(bbox, within):
    assert np.all(tonumpy(bbox[..., 2]) >= 0)
    assert np.all(tonumpy(bbox[..., 3]) >= 0)
    assert np.all(tonumpy(within[..., 2]) >= 0)
    assert np.all(tonumpy(within[..., 3]) >= 0)

    x1 = T.max(bbox[..., 0], within[..., 0])
    y1 = T.max(bbox[..., 1], within[..., 1])
    x2 = T.min(bbox[..., 0] + bbox[..., 2], within[..., 0] + within[..., 2])
    y2 = T.min(bbox[..., 1] + bbox[..., 3], within[..., 1] + within[..., 3])
    w = (x2 - x1).clamp(min=0)
    h = (y2 - y1).clamp(min=0)

    x = x1 - within[..., 0]
    y = y1 - within[..., 1]

    area = h * w
    y = y.clamp(min=0)
    x = x.clamp(min=0)

    return T.stack([x, y, w, h], -1)


def iou(a, b):
    i_area = intersection(a, b)
    a_area = a[..., 2] * a[..., 3]
    b_area = b[..., 2] * b[..., 3]
    return i_area / (a_area + b_area - i_area)


def nll(x, eps=1e-8):
    dx = ((x - eps) < 0).float() * eps
    return -T.log(x + dx)


def masked_nll(x, presence, weight=None):
    nll_x = nll(x)
    if weight is not None:
        nll_x = nll_x * weight
    nll_x = nll_x * (presence != 0).float()
    p = (presence != 0).float().sum(1)
    nll = nll_x.sum(1) / p.clamp(min=1) * (p != 0).float()
    return nll.sum() / (p != 0).float().sum()


def iou_loss(a, b, presence):
    '''
    a, b: (batch_size, nobjs, 4)
    presence: (batch_size, nobjs)
    '''
    i = iou(a, b)
    return masked_nll(i, presence)


def intersection_loss(pred, target, presence):
    area = target[..., 2] * target[..., 3]
    i = intersection(pred, target)
    i = i / (area + (area == 0).float())
    return masked_nll(i, presence)


def area_loss(pred, nrows, ncols, presence):
    '''
    Prevent the prediction from covering the whole image.
    If it goes WAY beyond the whole image, then *HEAVILY* penalize it.
    '''
    area = pred[..., 2] * pred[..., 3]
    ratio = area / (nrows * ncols)
    weight = T.clamp(ratio, 1, 10)
    ratio = T.clamp(ratio, 0, 1)
    return masked_nll(1 - ratio, presence, weight)


def _bbox_to_mask(yy, region_size, output_size):
    neg_part = (-yy[:2]).clamp(min=0)
    core_shape = T.round(yy[2:] - neg_part).data.int()
    core = tovar(T.ones(core_shape[1], core_shape[0]))

    y1 = yy[1].clamp(min=0)
    x1 = yy[0].clamp(min=0)
    y2 = (yy[1] + yy[3]).clamp(max=region_size[0])
    x2 = (yy[0] + yy[2]).clamp(max=region_size[1])

    padspace = (x1, region_size[1] - x2, y1, region_size[0] - y2)
    core = core.unsqueeze(0).unsqueeze(1)
    padded = F.pad(core, padspace).squeeze(1).squeeze(0)

    region_size = region_size.round().int()
    mask = tonumpy(padded[:region_size[0], :region_size[1]])
    resized_mask = tovar(scipy.misc.imresize(mask, output_size))
    return resized_mask


def bbox_to_mask(bbox, region_size, output_size):
    leading_shape = bbox.size()[:-1]
    bbox_flat = bbox.view(-1, 4)
    region_size_flat = region_size_flat.view(-1, 2)

    masks = []
    for b, rs in zip(bbox_flat, region_size):
        masks.append(_bbox_to_mask(b, rs, output_size))

    masks = T.stack(masks, 0)
    return masks.view(*leading_shape, *output_size)


def addbox(ax, b, ec):
    ax.add_patch(PA.Rectangle((b[0] - b[2] / 2, b[1] - b[3] / 2), b[2], b[3],
                 ec=ec, fill=False, lw=1))
