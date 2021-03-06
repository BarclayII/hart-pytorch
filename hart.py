
import torch as T
from torch import nn
import torch.nn.functional as F

from util import *

class HART(nn.Module):
    def __init__(self,
                 attention_cell):
        nn.Module.__init__(self)
        state_size = attention_cell.state_size

        self.attention_cell = attention_cell
        self.bbox_predictor = nn.Sequential(
                nn.Linear(state_size, state_size),
                nn.ELU(),
                nn.Linear(state_size, attention_cell.n_glims * 4),
                nn.Tanh(),
                )

    def forward(self, x, bbox0, presence0):
        '''
        x: 5D (batch_size, nframes, nchannels, nrows, ncols)
        bbox0: (batch_size, nobjects, 4) [cx, cy, w, h]
        presence0: (batch_size, nobjects)

        returns:
        bbox: 4D (batch_size, nframes, nobjs, 4) [cx, cy, w, h]
        atts: 4D (batch_size, nframes, nobjs, att_params)

        '''
        batch_size, nframes, nchannels, nrows, ncols = x.size()
        rnn_output, rnn_state, att0, app0 = self.attention_cell.zero_state(
                x[:, 0], bbox0, presence0)

        outputs = []
        for i in range(nframes):
            output = self.attention_cell(
                    x[:, i], att0, app0, presence0, rnn_state)
            outputs.append(output[:3] + output[4:])     # skip rnn states
            att0, app0, presence0, rnn_state = output[:4]

        _outputs = outputs
        # atts, apps: the attention/appearance features for next frame
        atts, apps, presence_logits, outputs, glims, mask_logits, mask_feats, \
                dfn_l2s, raw_glims = [T.stack(o, 1) for o in zip(*outputs)]

        bbox_delta = self.bbox_predictor(outputs.view(batch_size * nframes, -1))
        bbox_delta = bbox_delta.view(
                batch_size, nframes, self.attention_cell.n_glims, 4)

        # (batch_size, nframes, nobjs, att_params)
        atts = T.cat([att0.unsqueeze(1), atts], 1)
        bbox_from_att = self.attention_cell.attender.att_to_bbox(atts)
        # We need to subtract the bias on attention here, since we impose a
        # bias on the initial spatial attention, hence the bias on all
        # subsequent attentions.
        bbox_from_att_nobias = self.attention_cell.attender.att_to_bbox(
                atts - self.attention_cell.att_bias)

        bbox_delta_scaled = T.cat([
            tovar(T.zeros(bbox_delta[:, 0:1].size())),
            bbox_delta], 1)
        bbox_scaler = tovar(T.Tensor([ncols, nrows, ncols, nrows]))
        bbox_delta_scaled = bbox_delta_scaled * bbox_scaler

        bbox = clamp_bbox(bbox_delta_scaled + bbox_from_att_nobias)

        check_bbox_validness(bbox)
        check_bbox_validness(bbox_from_att)
        check_bbox_validness(bbox_from_att_nobias)

        return (
                bbox[:, :-1],
                atts[:, :-1],
                mask_logits,
                bbox_from_att[:, :-1],
                bbox_from_att_nobias[:, :-1],
                presence_logits,
                dfn_l2s.mean(),
                raw_glims,
                apps,
                )

    def losses(self,
               bbox,
               bbox_from_att,
               bbox_target,
               presence_logits,
               presences_target,
               mask_logits,
               img_rows,
               img_cols,
               lambda_xe=0.):
        bbox_loss = iou_loss(bbox, bbox_target, presences_target)
        masked_iou = iou(bbox, bbox_target) * presences_target
        iou_mean = masked_iou.sum() / presences_target.sum()

        att_intersection_loss = intersection_loss(
                bbox_from_att, bbox_target, presences_target)
        att_area_loss = area_loss(
                bbox_from_att, img_rows, img_cols, presences_target)

        target_mask_bbox = intersection_within(bbox_target, bbox_from_att)
        att_rows = bbox_from_att[..., 3]
        att_cols = bbox_from_att[..., 2]

        target_obj_mask = bbox_to_mask(
                target_mask_bbox,
                att_rows,
                att_cols,
                mask_logits.size()[-2:])

        pos = target_obj_mask.sum(4, keepdim=True).sum(3, keepdim=True)
        neg = (1 - target_obj_mask).sum(4, keepdim=True).sum(3, keepdim=True)
        frac_pos = pos / (pos + neg)
        frac_neg = 1 - frac_pos
        weight = ((target_obj_mask != 0).float() * frac_neg +
                  (target_obj_mask == 0).float() * frac_pos)

        weight = weight * presences_target[:, :, :, np.newaxis, np.newaxis]

        obj_mask_xe = F.binary_cross_entropy_with_logits(
                mask_logits, target_obj_mask, weight)

        return (
                bbox_loss,
                att_intersection_loss,
                att_area_loss,
                obj_mask_xe,
                iou_mean,
                )
