
import torch as T
import hart
import attention
import alexnet
import adaptive_loss

import kth
import argparse

from util import *


def get_presence(batch_size, seqlen, n_glims, lengths):
    presences = cuda(T.ones(batch_size, seqlen, n_glims))
    for i in range(batch_size):
        if lengths.data[i] >= seqlen:
            continue
        presences[i, lengths.data[i]:] = 0
    return presences


parser = argparse.ArgumentParser()
parser.add_argument('--batchsize', type=int, default=8)
parser.add_argument('--seqlen', type=int, default=30)
parser.add_argument('--num-workers', type=int, default=0)
parser.add_argument('--statesize', type=int, default=128)
parser.add_argument('--n-dfn-channels', type=int, default=10)

args = parser.parse_args()

image_size = (120, 160)
glim_size = (40, 40)
n_glims = 2     # DEBUG

train_dataset = kth.KTHDataset(
        'frames', 'KTHBoundingBoxInfoTrain.txt', seqlen=args.seqlen)
valid_dataset = kth.KTHDataset(
        'frames', 'KTHBoundingBoxInfoValidation.txt', seqlen=None)
train_dataloader = kth.KTHDataLoader(
        train_dataset, args.batchsize, num_workers=args.num_workers)
valid_dataloader = kth.KTHDataLoader(
        valid_dataset, 1, num_workers=args.num_workers)

feature_extractor = cuda(alexnet.AlexNetModel(n_out_feature_maps=args.n_dfn_channels))
attender = cuda(attention.RATMAttention(image_size, glim_size))
cell = cuda(attention.AttentionCell(
        args.statesize,
        image_size,
        glim_size,
        args.statesize,
        feature_extractor,
        attender,
        0.5,
        n_glims=n_glims,
        n_dfn_channels=args.n_dfn_channels))
tracker = cuda(hart.HART(cell))
al = cuda(adaptive_loss.AdaptiveLoss(4))

params = sum([list(m.parameters()) for m in [tracker, al]], [])
named_params = sum([list(m.named_parameters()) for m in [tracker, al]], [])
opt = T.optim.Adam(params, lr=1e-4)
itr = 0

for train_item, valid_item in zip(train_dataloader, valid_dataloader):
    tracker.train()
    al.train()
    itr += 1

    _images, _bboxes, _lengths = train_item
    images, bboxes, lengths = tovar(_images, _bboxes, _lengths)
    bboxes = bboxes.unsqueeze(2).expand(args.batchsize, args.seqlen, n_glims, 4)
    presences = tovar(
            get_presence(args.batchsize, args.seqlen, n_glims, lengths))

    bbox_pred, atts, mask_logits, bbox_from_att, bbox_from_att_nobias, pres = \
            tracker(images, bboxes[:, 0], presences[:, 0])

    bbox_loss, att_intersection_loss, att_area_loss, obj_mask_xe = \
            tracker.losses(
                    bbox_pred,
                    bbox_from_att,
                    bboxes,
                    pres,
                    presences,
                    mask_logits,
                    image_size[0],
                    image_size[1],
                    )

    loss = al(bbox_loss, att_intersection_loss, att_area_loss, obj_mask_xe)

    opt.zero_grad()
    loss.backward()
    opt.step()

    tracker.eval()

    _images, _bboxes, _lengths = valid_item
    images, bboxes, lengths = tovar(_images, _bboxes, _lengths, volatile=True)
    seqlen = images.size()[1]
    bboxes = bboxes.unsqueeze(2).expand(1, seqlen, n_glims, 4)
    presences = tovar(
            get_presence(1, seqlen, n_glims, lengths))

    bbox_pred, atts, mask_logits, bbox_from_att, bbox_from_att_nobias, pres = \
            tracker(images, bboxes[:, 0], presences[:, 0])
    print(toscalar(loss), toscalar(iou(bbox_pred, bboxes).mean()))
