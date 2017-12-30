
import torch as T
import hart
import attention
import alexnet
import adaptive_loss
import viz
import dataset
import argparse
import matplotlib.pyplot as PL
import cv2
import numpy as np
import logging

from util import *

logging.getLogger().setLevel('INFO')

def get_presence(batch_size, seqlen, n_glims, lengths):
    '''
    Generates a weight tensor indicating whether the object indeed exists.
    In our setting where we always track one single object, the weight is
    1 if the object exists, and 0 otherwise.  Since the object always
    exists in a given sequence, we simply set all the weights beyond the
    sequence length to 0.

    In the case of multi-object tracking & detection, the weight tensor
    should come from the dataset.
    '''
    presences = cuda(T.ones(batch_size, seqlen, n_glims))
    for i in range(batch_size):
        if lengths.data[i] >= seqlen:
            continue
        presences[i, lengths.data[i]:] = 0
    return presences


def update_learning_rate(opt, lr):
    for pg in opt.param_groups:
        pg['lr'] = lr


# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--batchsize', type=int, default=8)
parser.add_argument('--seqlen', type=int, default=30)
parser.add_argument('--validate-complete-sequence', action='store_true')
parser.add_argument('--num-workers', type=int, default=0)
parser.add_argument('--statesize', type=int, default=128)
parser.add_argument('--n-dfn-channels', type=int, default=10)
parser.add_argument('--l2reg', type=float, default=1e-4)
parser.add_argument('--n-viz', type=int, default=1)
parser.add_argument('--gradclip', type=float, default=1)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--lr-scale-ratio', type=float, default=1.2)
parser.add_argument('--lr-min', type=float, default=1e-5)
parser.add_argument('--zoneout', type=float, default=0.05)
parser.add_argument('--opt', type=str, default='SGD')
parser.add_argument('--visdom-host', type=str, default='localhost')
parser.add_argument('--kth-dir', type=str, default='.')
parser.add_argument('--imagenet-dir', type=str, default='.')
parser.add_argument('--dataset', type=str, default='kth',
                    choices=['kth', 'imagenet'])
parser.add_argument('--max-iter-per-epoch', type=int, default=5000)
parser.add_argument('--image-rows', type=int, default=120)
parser.add_argument('--image-cols', type=int, default=160)
parser.add_argument('--glim-size', type=int, default=40)

args = parser.parse_args()

image_size = (args.image_rows, args.image_cols)
glim_size = (args.glim_size, args.glim_size)
n_glims = 1
valid_batch_size = 1 if args.validate_complete_sequence else args.batchsize
valid_seqlen = None if args.validate_complete_sequence else args.seqlen

# Dataset
if args.dataset == 'kth':
    frames_dir = os.path.join(args.kth_dir, 'frames')
    bbox_info_train = os.path.join(
            args.kth_dir, 'KTHBoundingBoxInfoTrain.txt')
    bbox_info_valid = os.path.join(
            args.kth_dir, 'KTHBoundingBoxInfoValidation.txt')
    train_dataset = dataset.KTHDataset(
            frames_dir, bbox_info_train, seqlen=args.seqlen,
            rows=image_size[0], cols=image_size[1])
    valid_dataset = dataset.KTHDataset(
            frames_dir, bbox_info_valid, seqlen=valid_seqlen,
            rows=image_size[0], cols=image_size[1])
elif args.dataset == 'imagenet':
    data_dir = os.path.join(args.imagenet_dir, 'Data/VID')
    anno_dir = os.path.join(args.imagenet_dir, 'Annotations/VID')
    train_dataset = dataset.ImagenetVIDDataset(
            os.path.join(data_dir, 'train'),
            os.path.join(args.imagenet_dir, 'train-pkl'),
            os.path.join(anno_dir, 'train'),
            seqlen=args.seqlen,
            rows=image_size[0], cols=image_size[1])
    valid_dataset = dataset.ImagenetVIDDataset(
            os.path.join(data_dir, 'val'),
            os.path.join(args.imagenet_dir, 'valid-pkl'),
            os.path.join(anno_dir, 'val'),
            seqlen=valid_seqlen,
            rows=image_size[0], cols=image_size[1])

train_dataloader = dataset.VideoDataLoader(
        train_dataset, args.batchsize, num_workers=args.num_workers)
valid_dataloader = dataset.VideoDataLoader(
        valid_dataset, valid_batch_size, num_workers=args.num_workers)

# Model
feature_extractor = cuda(alexnet.AlexNetModel(n_out_feature_maps=args.n_dfn_channels))
attender = cuda(attention.RATMAttention(image_size, glim_size))
cell = cuda(attention.AttentionCell(
        args.statesize,
        image_size,
        glim_size,
        args.statesize,
        feature_extractor,
        attender,
        args.zoneout,
        n_glims=n_glims,
        n_dfn_channels=args.n_dfn_channels))
tracker = cuda(hart.HART(cell))
weights = [1, 1, 1, 1]
al = cuda(adaptive_loss.AdaptiveLoss(weights=weights))
lr = args.lr

# Optimizer
params = sum([list(m.parameters()) for m in [tracker, al]], [])
named_params = sum([list(m.named_parameters()) for m in [tracker, al]], [])
opt = getattr(T.optim, args.opt)(params, lr=lr)

# Main program
epoch = 0

wm = viz.VisdomWindowManager(server='http://' + args.visdom_host)
train_it = iter(train_dataloader)
valid_it = iter(valid_dataloader)

while True:
    epoch += 1

    # Training goes here...
    tracker.train()
    al.train()

    for itr in range(args.max_iter_per_epoch):
        try:
            train_item = next(train_it)
        except StopIteration:
            train_it = iter(train_dataloader)
            train_item = next(train_it)

        # bbox prediction
        _images, _bboxes, _lengths = train_item
        images, bboxes, lengths = tovar(_images, _bboxes, _lengths)
        bboxes = bboxes.unsqueeze(2).expand(args.batchsize, args.seqlen, n_glims, 4)
        presences = tovar(
                get_presence(args.batchsize, args.seqlen, n_glims, lengths))

        bbox_pred, atts, mask_logits, bbox_from_att, bbox_from_att_nobias, \
                pres, dfn_l2, raw_glims, apps = tracker(
                        images, bboxes[:, 0], presences[:, 0])

        # Loss computation
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
        #att_loss = att_intersection_loss + att_area_loss

        losses = [bbox_loss, att_intersection_loss, att_area_loss, obj_mask_xe]
        loss = al(*losses)

        if args.l2reg:
            l2reg = sum(T.norm(p) ** 2 for p in tracker.parameters())
            l2reg += dfn_l2
            loss += args.l2reg * l2reg

        # Step
        opt.zero_grad()
        loss.backward()
        if check_grads(named_params):
            print('Skipping...')
            tracker.attention_cell._att_bias.data.zero_()
            continue
        clip_grads(named_params, args.gradclip)
        opt.step()

        print('TRAIN', epoch, toscalar(loss), toscalar(iou_mean))

        # Visualizations
        wm.append_scalar('bbox IOU loss', toscalar(bbox_loss))
        wm.append_scalar('attention intersection loss',
                         toscalar(att_intersection_loss))
        wm.append_scalar('attention area loss', toscalar(att_area_loss))
        wm.append_scalar('mask cross entropy', toscalar(obj_mask_xe))
        wm.append_scalar('overall loss', toscalar(loss))
        wm.append_scalar('average IOU (train)', toscalar(iou_mean))
        wm.append_scalar('lambda',
                         [toscalar(al.transform(l)) for l in al.lambdas],
                         opts=dict(
                             legend=[
                                 'bbox',
                                 'attention-intersection-loss',
                                 'attention-area-loss',
                                 'mask-cross-entropy',
                                 ]
                             )
                         )
        wm.heatmap(
                tonumpy(apps[0, :, 0]),
                win='app',
                )

    print(tonumpy(bboxes))
    print(tonumpy(bbox_pred))

    # Validation goes here...
    tracker.eval()
    avg_iou = 0

    for i in range(args.max_iter_per_epoch):
        try:
            valid_item = next(valid_it)
        except StopIteration:
            valid_it = iter(valid_dataloader)
            valid_item = next(valid_it)

        _images, _bboxes, _lengths = valid_item
        images, bboxes, lengths = tovar(_images, _bboxes, _lengths, volatile=True)
        seqlen = images.size()[1]
        bboxes = bboxes.unsqueeze(2).expand(valid_batch_size, seqlen, n_glims, 4)
        presences = tovar(
                get_presence(valid_batch_size, seqlen, n_glims, lengths))

        bbox_pred, atts, mask_logits, bbox_from_att, bbox_from_att_nobias, \
                pres, _, raw_glims, apps = tracker(
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
            wm.heatmap(
                    tonumpy(apps[0, :, 0]),
                    win='app',
                    )

    avg_iou = avg_iou / args.max_iter_per_epoch
    print('VALID-AVG', epoch, avg_iou)

    wm.append_scalar('Average IOU (validation)', avg_iou)
    print(tonumpy(bboxes))
    print(tonumpy(bbox_pred))

    # Schedule learning rate
    lr /= args.lr_scale_ratio
    lr = max(lr, args.lr_min)
    update_learning_rate(opt, lr)
