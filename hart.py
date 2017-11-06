
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
                x, bbox0, presence0)

        outputs = []
        for i in range(nframes):
            output = self.attention_cell(
                    x[:, i], att0, app0, presence0, rnn_state)
            outputs.append(output)
            att0, app0, presence0, rnn_state = output[:4]

        atts, apps, presence_logits, states, outputs, \
                glims, mask_logits, mask_feats = [
                        T.stack(o, 1) for o in zip(*outputs)]

        bbox_delta = self.bbox_predictor(outputs.view(batch_size * nframes, -1))
        bbox_delta = bbox_delta.view(
                batch_size, nframes, attention_cell.n_glims, 4)

        # (batch_size, nframes, nobjs, att_params)
        atts = T.cat([att0.unsqueeze(1), atts], 1)
        bbox_from_att = self.attention_cell.attender.att_to_bbox(atts)
        bbox_from_att_nobias = self.attention_cell.attender.att_to_bbox(
                atts - self.attention_cell.att_bias)

        bbox_delta_scaled = T.cat([
            tovar(T.zeros(bbox_delta[:, 0:1].size())),
            bbox_delta], 1)
        bbox_scaler = T.Tensor([ncols, nrows, ncols, nrows])
        bbox_delta_scaled = bbox_delta_scaled * bbox_scaler

        bbox = bbox_delta_scaled + bbox_from_att_nobias

        return (
                bbox,
                atts,
                mask_logits,
                bbox_from_att,
                bbox_from_att_nobias,
                presences_logits,
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
        att_intersection_loss = intersection_loss(
                bbox_from_att, bbox_target, presences_target)
        att_area_loss = area_loss(
                bbox_from_att, img_rows, img_cols, presence_target)

        target_mask_bbox = intersection_within(bbox_target, bbox_from_att)
        att_rows = bbox_from_att[..., 3]
        att_cols = bbox_from_att[..., 2]

        target_obj_mask = bbox_to_mask(
                target_mask_bbox,
                (att_rows, att_cols),
                mask_logits.size()[-2:])

        pos = target_obj_mask.sum(4, keepdim=True).sum(3, keepdim=True)
        neg = (1 - target_obj_mask).sum(4, keepdim=True).sum(3, keepdim=True)
        frac_pos = pos / (pos + neg)
        frac_neg = 1 - frac_pos
        weight = target_obj_mask * frac_pos + (1 - target_obj_mask) * frac_neg

        obj_mask_xe = F.binary_cross_entropy_with_logits(
                mask_logits, target_obj_mask, weight)
        obj_mask_xe = obj_mask_xe.mean(4).mean(3)

        obj_mask_xe = obj_mask_xe * presences_target
        obj_mask_xe = obj_mask_xe.sum() / presences_target.sum()

        return (
                bbox_loss,
                att_intersection_loss,
                att_area_loss,
                obj_mask_xe,
                )
