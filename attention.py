
import torch as T
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch.nn import init

from dfn import DynamicConvFilter, DynamicConvFilterGenerator
from zoneout import ZoneoutLSTMCell
from util import *

def gaussian_masks(c, d, s, len_, glim_len):
    '''
    c, d, s: 2D Tensor (batch_size, n_glims)
    len_, glim_len: int
    returns: 4D Tensor (batch_size, n_glims, glim_len, len_)
        each row is a 1D Gaussian
    '''
    batch_size, n_glims = c.size()

    # The original HART code did not shift the coordinates by
    # glim_len / 2.  The generated Gaussian attention does not
    # correspond to the actual crop of the bbox.
    # Possibly a bug?
    R = tovar(T.arange(0, glim_len)).view(1, 1, 1, -1) - glim_len / 2
    C = tovar(T.arange(0, len_)).view(1, 1, -1, 1)
    C = C.expand(batch_size, n_glims, len_, 1)
    c = c[:, :, np.newaxis, np.newaxis]
    d = d[:, :, np.newaxis, np.newaxis]
    s = s[:, :, np.newaxis, np.newaxis]

    cr = c + R * d
    sr = tovar(T.ones(cr.size())) * s

    mask = C - cr
    mask = (-0.5 * (mask / sr) ** 2).exp()

    mask = mask / (mask.sum(2, keepdim=True) + 1e-8)
    return mask


def extract_gaussian_glims(x, a, glim_size):
    '''
    x: 4D Tensor (batch_size, nchannels, nrows, ncols)
    a: 3D Tensor (batch_size, n_glims, att_params)
        att_params: (cx, cy, dx, dy, sx, sy)
    returns:
        5D Tensor (batch_size, n_glims, nchannels, n_glim_rows, n_glim_cols)
    '''
    batch_size, n_glims, _ = a.size()
    cx, cy, dx, dy, sx, sy = T.unbind(a, -1)
    _, nchannels, nrows, ncols = x.size()
    n_glim_rows, n_glim_cols = glim_size

    # (batch_size, n_glims, nrows, n_glim_rows)
    Fy = gaussian_masks(cy, dy, sy, nrows, n_glim_rows)
    # (batch_size, n_glims, ncols, n_glim_cols)
    Fx = gaussian_masks(cx, dx, sx, ncols, n_glim_cols)

    # (batch_size, n_glims, 1, nrows, n_glim_rows)
    Fy = Fy.unsqueeze(2)
    # (batch_size, n_glims, 1, ncols, n_glim_cols)
    Fx = Fx.unsqueeze(2)

    # (batch_size, 1, nchannels, nrows, ncols)
    x = x.unsqueeze(1)
    # (batch_size, n_glims, nchannels, n_glim_rows, n_glim_cols)
    g = Fy.transpose(-1, -2) @ x @ Fx

    return g

class RATMAttention(nn.Module):
    '''
    [cx, cy, dx, dy, sx, sy]
    0 <= cx, cy <= 1
    0 <= dx, dy <= 1
    '''
    att_params = 6      # no. of parameters for attention parameterization

    def __init__(self, x_size, glim_size):
        '''
        x_size: [n_image_rows, n_image_cols]
        glim_size: [n_glim_rows, n_glim_cols]
        '''
        nn.Module.__init__(self)
        self.x_size = x_size
        self.glim_size = glim_size

    def forward(self, x, spatial_att):
        '''
        x: 4D Tensor (batch_size, nchannels, n_image_rows, n_image_cols)
        spatial_att: 3D Tensor (batch_size, n_glims, att_params) relative scales
        '''
        # (batch_size, n_glims, att_params)
        absolute_att = self._to_absolute_attention(spatial_att)
        glims = extract_gaussian_glims(x, absolute_att, self.glim_size)

        return glims

    def att_to_bbox(self, spatial_att):
        '''
        spatial_att: (..., 6) [cx, cy, dx, dy, sx, sy] relative scales ]0, 1[
        return: (..., 4) [cx, cy, w, h] absolute scales
        '''
        cx = spatial_att[..., 0] * self.x_size[1]
        cy = spatial_att[..., 1] * self.x_size[0]
        w = T.abs(spatial_att[..., 2]) * (self.x_size[1] - 1)
        h = T.abs(spatial_att[..., 3]) * (self.x_size[0] - 1)
        bbox = T.stack([cx, cy, w, h], -1)
        return bbox

    def bbox_to_att(self, bbox):
        '''
        bbox: (..., 4) [cx, cy, w, h] absolute scales
        return: (..., 6) [cx, cy, dx, dy, sx, sy] relative scales ]0, 1[
        '''
        cx = bbox[..., 0] / self.x_size[1]
        cy = bbox[..., 1] / self.x_size[0]
        dx = bbox[..., 2] / (self.x_size[1] - 1)
        dy = bbox[..., 3] / (self.x_size[0] - 1)
        sx = bbox[..., 2] * 0.5 / self.x_size[1]
        sy = bbox[..., 3] * 0.5 / self.x_size[0]
        spatial_att = T.stack([cx, cy, dx, dy, sx, sy], -1)

        return spatial_att

    def _to_axis_attention(self, image_len, glim_len, c, d, s):
        c = c * image_len
        d = d * (image_len - 1) / (glim_len - 1)
        s = (s + 1e-5) * image_len / glim_len
        return c, d, s

    def _to_absolute_attention(self, params):
        '''
        params: 3D Tensor (batch_size, n_glims, att_params)
        '''
        n_image_rows, n_image_cols = self.x_size
        n_glim_rows, n_glim_cols = self.glim_size
        cx, dx, sx = T.unbind(params[..., ::2], -1)
        cy, dy, sy = T.unbind(params[..., 1::2], -1)
        cx, dx, sx = self._to_axis_attention(
                n_image_cols, n_glim_cols, cx, dx, sx)
        cy, dy, sy = self._to_axis_attention(
                n_image_rows, n_glim_rows, cy, dy, sy)

        # ap is now the absolute coordinate/scale on image
        # (batch_size, n_glims, att_params)
        ap = T.stack([cx, cy, dx, dy, sx, sy], -1)
        return ap


class AttentionCell(nn.Module):
    def __init__(self,
                 state_size,
                 image_size,
                 glim_size,
                 app_size,
                 feature_extractor,
                 attender,
                 zoneout_prob,
                 n_glims=1,
                 n_dfn_channels=10,
                 att_scale_logit_init=-2.5,
                 normalize_glimpse=False,
                 mask_feat_size=10):
        nn.Module.__init__(self)
        self.feature_extractor = feature_extractor
        self.attender = attender
        self.normalize_glimpse = normalize_glimpse
        self.state_size = state_size
        self.n_glims = n_glims
        self.app_size = app_size
        self.att_params = attender.att_params
        self.mask_feat_size = mask_feat_size
        glim_flatsize = np.asscalar(np.prod(glim_size))

        self.pre_dfngen = DynamicConvFilterGenerator(
                app_size,
                feature_extractor.n_out_channels,
                n_dfn_channels,
                (1, 1))
        self.pre_dfn = DynamicConvFilter(
                feature_extractor.n_out_channels,
                n_dfn_channels,
                (1, 1))
        self.dfngen = DynamicConvFilterGenerator(
                app_size,
                n_dfn_channels,
                n_dfn_channels,
                (3, 3))
        self.dfn = DynamicConvFilter(
                n_dfn_channels,
                n_dfn_channels,
                (3, 3),
                padding=1)
        self.proj = nn.Sequential(
                nn.Linear(
                    n_glims * (
                        n_dfn_channels *
                        feature_extractor.compute_output_flatsize(glim_size) +
                        self.attender.att_params),
                    state_size),
                nn.ELU(),
                )
        self.app_predictor = nn.Sequential(
                nn.Linear(state_size, n_glims * app_size),
                nn.ELU(),
                )
        self.rnncell = ZoneoutLSTMCell(state_size, state_size, p=zoneout_prob)
        self.masker = nn.Conv2d(n_dfn_channels, 1, (1, 1))
        self.mask_feat_extractor = nn.Sequential(
                nn.Linear(
                    feature_extractor.compute_output_flatsize(glim_size),
                    mask_feat_size
                    ),
                nn.ELU(),
                )
        att_pred_output_layer = nn.Linear(state_size, n_glims * self.att_params)
        init.uniform(att_pred_output_layer.weight, -1e-3, 1e-3)
        init.constant(att_pred_output_layer.bias, 0)
        self.att_pred_layer = nn.Sequential(
                nn.Linear(state_size + mask_feat_size * n_glims, state_size),
                nn.ELU(),
                att_pred_output_layer,
                nn.Tanh(),
                )
        self.att_delta_scale_logit = nn.Parameter(
                T.zeros(n_glims, self.att_params) +
                att_scale_logit_init)
        self._att_bias = nn.Parameter(T.zeros(self.att_params)) # ?
        self._att_scale = .25           # ???

    @property
    def att_bias(self):
        return self._att_scale * T.tanh(self._att_bias)

    def zero_state(self, x, bbox, presence):
        '''
        x: 4D (batch_size, nchannels, nrows, ncols)
        bbox: 3D (batch_size, nobjects, 4) [cx, cy, w, h]
        presence: 2D (batch_size, nobjects)
        '''
        batch_size = x.size()[0]
        # (batch_size, nobjects = nglims, att_params)
        att = self.attender.bbox_to_att(bbox)
        rnn_state = self.rnncell.zero_state(batch_size)

        _, feats, _, _, _ = self.extract_features(x, att, None)
        rnn_output, rnn_state = self.rnncell(feats, rnn_state)

        att += self.att_bias.view(1, 1, -1)

        app = self.app_predictor(rnn_output)
        app = app.view(batch_size, self.n_glims, self.app_size)
        return rnn_output, rnn_state, att, app

    def extract_features(self, x, spatial_att, appearance):
        '''
        x: 4D Tensor (batch_size, nchannels, n_image_rows, n_image_cols)
        spatial_att: 3D Tensor (batch_size, n_glims, att_params)
        appearance: 3D Tensor (batch_size, n_glims, app_size)
            or None (when first called during hidden state initialization)

        returns:
        glims: 5D (batch_size, n_glims, nchannels, n_glim_rows, n_glim_cols)
        projected_feats: 2D
            (batch_size, state_size)
        mask_logit: 3D or None (if appearance is None)
            (batch_size, n_glims, raw_feat_rows, raw_feat_cols)
        dfn_norm: L2 norm of generated DFN parameters
        prenorm_glims: as @glims, but not contrast-normalized (for viz)
        '''
        # (batch_size, n_glims, nchannels, n_glim_rows, n_glim_cols)
        prenorm_glims = glims = self.attender(x, spatial_att)
        batch_size, n_glims, nchannels, n_glim_rows, n_glim_cols = glims.size()

        if self.normalize_glimpse:
            glims = util.normalize_contrast(glims)

        # (batch_size * n_glims, ...)
        glims_reshaped = glims.view(-1, nchannels, n_glim_rows, n_glim_cols)
        raw_feats, readout, feats = self.feature_extractor(glims_reshaped)

        if appearance is not None:
            appearance = appearance.view(-1, self.app_size)
            _, _, raw_feat_rows, raw_feat_cols = raw_feats.size()

            # (batch_size * n_glims, n_dfn_channels,
            #  raw_feat_rows, raw_feat_cols)
            pre_dfn_w, pre_dfn_b = self.pre_dfngen(appearance)
            pre_dfn_feats = F.elu(self.pre_dfn(raw_feats, pre_dfn_w, pre_dfn_b))
            dfn_w, dfn_b = self.dfngen(appearance)
            dfn_feats = F.elu(self.dfn(pre_dfn_feats, dfn_w, dfn_b))

            dfn_l2 = T.norm(pre_dfn_w) ** 2 + T.norm(pre_dfn_b) ** 2
            dfn_l2 += T.norm(dfn_w) ** 2 + T.norm(dfn_b) ** 2

            mask_logit = self.masker(dfn_feats)
            mask = F.sigmoid(mask_logit)
            masked_feats = feats * mask
            mask_logit = mask_logit.squeeze(1).view(
                    batch_size, n_glims, raw_feat_rows, raw_feat_cols)
        else:
            masked_feats = feats
            mask_logit = None
            dfn_l2 = 0

        # Now I'm collapsing all glimpse features of a single sample into the
        # same vector.  Not sure if I need to make a separate one for each
        # glimpse.
        projected_feats = self.proj(
                T.cat([
                    masked_feats.view(batch_size, -1),
                    spatial_att.view(batch_size, -1),
                    ], 1)
                )

        return glims, projected_feats, mask_logit, dfn_l2, prenorm_glims


    def forward(self,
                x,
                spatial_att,
                appearance,
                presence,
                hidden_state):
        '''
        x: 4D (batch_size, nchannels, nrows, ncols)
        spatial_att: 3D (batch_size, n_glims, att_params)
            [cx, cy, dx, dy, sx, sy]
        presence: (batch_size, n_glims)
        appearance: (batch_size, n_glims, app_size)
        hidden_state: something shouldn't care

        returns:
        rnn_output: (batch_size, state_size)
        new_att: (batch_size, n_glims, att_params)
        new_app: (batch_size, n_glims, app_size)
        presence: (batch_size, n_glims)
        glims: (batch_size, n_glims, nchannels, n_glim_rows, n_glim_cols)
        mask_logit: (batch_size, n_glims, raw_feat_rows, raw_feat_cols)
        mask_feats: (batch_size, n_glims, mask_feat_size)
        next_state: something shouldn't care
        '''
        batch_size = x.size()[0]

        glims, feats, mask_logit, dfn_l2, raw_glims = self.extract_features(
                x, spatial_att, appearance)

        rnn_output, next_state = self.rnncell(feats, hidden_state)

        # Combine mask with RNN state
        mask_feats = self.mask_feat_extractor(
                mask_logit.view(batch_size * self.n_glims, -1))
        att_input = T.cat(
                [rnn_output, mask_feats.view(batch_size, -1)],
                1)
        mask_feats = mask_feats.view(
                batch_size, self.n_glims, self.mask_feat_size)

        att_pred = self.att_pred_layer(att_input)
        att_pred = att_pred.view(batch_size, self.n_glims, self.att_params)
        att_delta_scale = F.sigmoid(self.att_delta_scale_logit)
        new_att = spatial_att + att_delta_scale.unsqueeze(0) * att_pred

        # The original HART paper uses RNN state itself as appearance feature
        # vector...
        new_app = self.app_predictor(rnn_output)
        new_app = new_app.view(batch_size, self.n_glims, self.app_size)

        return (
                new_att,
                new_app,
                presence,
                next_state,
                rnn_output,
                glims,
                mask_logit,
                mask_feats,
                dfn_l2,
                raw_glims,
                )
