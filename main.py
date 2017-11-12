
import torch as T
import hart
import attention
import alexnet
import adaptive_loss
import viz
import kth
import argparse
import matplotlib.pyplot as PL
import cv2
import numpy as np

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
parser.add_argument('--l2reg', type=float, default=1e-4)
parser.add_argument('--n-viz', type=int, default=1)
parser.add_argument('--gradclip', type=float, default=1)

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
        valid_dataset, 1, num_workers=args.num_workers, shuffle=False)

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
weights = [1, 1, 1, 1]
al = cuda(adaptive_loss.AdaptiveLoss(weights=weights))

params = sum([list(m.parameters()) for m in [tracker, al]], [])
named_params = sum([list(m.named_parameters()) for m in [tracker, al]], [])
opt = T.optim.RMSprop(params, lr=3e-5)

epoch = 0

wm = viz.VisdomWindowManager()

while True:
    epoch += 1

    itr = 0

    tracker.train()
    al.train()

    for train_item in train_dataloader:
        itr += 1

        _images, _bboxes, _lengths = train_item
        images, bboxes, lengths = tovar(_images, _bboxes, _lengths)
        bboxes = bboxes.unsqueeze(2).expand(args.batchsize, args.seqlen, n_glims, 4)
        presences = tovar(
                get_presence(args.batchsize, args.seqlen, n_glims, lengths))

        bbox_pred, atts, mask_logits, bbox_from_att, bbox_from_att_nobias, \
                pres, dfn_l2, raw_glims = tracker(
                        images, bboxes[:, 0], presences[:, 0])

        bbox_loss, att_intersection_loss, att_area_loss, obj_mask_xe, iou_mean = \
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

        losses = [bbox_loss, att_intersection_loss, att_area_loss, obj_mask_xe]
        loss = al(*losses)

        if args.l2reg:
            l2reg = sum(T.norm(p) ** 2 for p in tracker.parameters())
            l2reg += dfn_l2
            loss += args.l2reg * l2reg

        opt.zero_grad()
        loss.backward()
        check_grads(named_params)
        clip_grads(named_params, args.gradclip)
        opt.step()

        print('TRAIN', epoch, toscalar(loss), toscalar(iou_mean))

        # Visualizations
        wm.append_scalar('bbox IOU loss', toscalar(bbox_loss))
        wm.append_scalar(
                'attention intersection loss', toscalar(att_intersection_loss))
        wm.append_scalar('attention area loss', toscalar(att_area_loss))
        wm.append_scalar('mask cross entropy', toscalar(obj_mask_xe))
        wm.append_scalar('overall loss', toscalar(loss))
        wm.append_scalar('average IOU (train)', toscalar(iou_mean))

    print(tonumpy(bboxes))
    print(tonumpy(bbox_pred))

    tracker.eval()
    avg_iou = 0

    for i, valid_item in enumerate(valid_dataloader):
        _images, _bboxes, _lengths = valid_item
        images, bboxes, lengths = tovar(_images, _bboxes, _lengths, volatile=True)
        seqlen = images.size()[1]
        bboxes = bboxes.unsqueeze(2).expand(1, seqlen, n_glims, 4)
        presences = tovar(
                get_presence(1, seqlen, n_glims, lengths))

        bbox_pred, atts, mask_logits, bbox_from_att, bbox_from_att_nobias, \
                pres, _, raw_glims = tracker(
                        images, bboxes[:, 0], presences[:, 0])

        current_iou = toscalar(iou(bbox_pred, bboxes).mean())
        print('VALID', epoch, current_iou)

        avg_iou += current_iou

        # Visualizations
        if i < args.n_viz:
            name = 'video-%d' % i
            wm.reset_mpl_figure_sequence(name)

            for t in range(_images.size()[1]):
                fig, ax = PL.subplots(2, 2)
                fig.set_size_inches((10, 10))
                # (0, 0): Original image and bbox target/prediction
                original_img = tonumpy(_images[0, t].permute(1, 2, 0))
                original_img = torch_unnormalize_image(original_img)
                ax[0, 0].imshow(original_img)
                ax[0, 0].set_title('Original image + bbox/prediction')
                addbox(ax[0, 0], tonumpy(bboxes[0, t, 0]), 'red')
                addbox(ax[0, 0], tonumpy(bbox_pred[0, t, 0]), 'yellow')
                addbox(ax[0, 0], tonumpy(bbox_from_att[0, t, 0]), 'green')
                addbox(ax[0, 0], tonumpy(bbox_from_att_nobias[0, t, 0]), 'cyan')
                # (0, 1): Spatial attention
                raw_glim_img = tonumpy(raw_glims[0, t, 0].permute(1, 2, 0))
                raw_glim_img = torch_unnormalize_image(raw_glim_img)
                ax[0, 1].set_title('Spatial attention glimpse')
                ax[0, 1].imshow(raw_glim_img)
                # (1, 0): Attention mask
                mask = tonumpy(F.sigmoid(mask_logits[0, t, 0]))
                resized_mask = cv2.resize(mask, (glim_size[1], glim_size[0]))
                ax[1, 0].set_title('Appearance mask')
                ax[1, 0].matshow(resized_mask, cmap='gray')
                resized_mask = np.expand_dims(resized_mask, 2)
                # (1, 1): Masked attended image
                attended_glim_img = raw_glim_img * resized_mask
                ax[1, 1].set_title('Masked glimpse')
                ax[1, 1].imshow(attended_glim_img)

                wm.append_mpl_figure_to_sequence(name, fig)

            wm.display_mpl_figure_sequence(
                    name,
                    win=name,
                    opts=dict(title=name, fps=10),
                    )

    avg_iou = avg_iou / len(valid_dataset)
    print('VALID-AVG', epoch, avg_iou)

    wm.append_scalar('Average IOU (validation)', avg_iou)
    print(tonumpy(bboxes))
    print(tonumpy(bbox_pred))
