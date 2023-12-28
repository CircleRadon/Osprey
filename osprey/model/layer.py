import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 num_layers: int) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class MaskExtractor(nn.Module):
    def __init__(self, mask_shape=112, embed_dim=1024, out_dim=4096):
        super(MaskExtractor, self).__init__()
        self.mask_shape = mask_shape
        self.mask_pooling = MaskPooling()
        self.feat_linear = nn.Linear(embed_dim, out_dim)
        self.mask_linear = MLP(mask_shape*mask_shape, embed_dim, out_dim, 3)
        # self.res_linear = {}
        self.feature_name = ['res2', 'res3', 'res4', 'res5']

        # for i, feat in enumerate(self.feature_name):
        #     self.res_linear[feat] = nn.Linear(192*(2**i), embed_dim)

        self.res2 = nn.Linear(192, 1024)
        self.res3 = nn.Linear(384, 1024)
        self.res4 = nn.Linear(768, 1024)
        self.res5 = nn.Linear(1536, 1024)

    def forward(self, feats, masks):
        query_feats = []
        pos_feats = []
        num_imgs = len(masks)

        for idx in range(num_imgs):
            mask = masks[idx].unsqueeze(0).float()

            num_feats = len(self.feature_name)
            mask_feats = mask.new_zeros(num_feats, mask.shape[1], 1024)
            for i, name in enumerate(self.feature_name):
                feat = feats[name][idx].unsqueeze(0)

                raw_dtype = feat.dtype
                feat = feat.to(mask.dtype)
                mask_feat_raw = self.mask_pooling(feat, mask)
                
                mask_feat_flatten = mask_feat_raw.reshape(-1, mask_feat_raw.shape[-1])

                # self.res_linear[name] = self.res_linear[name].to(dtype=mask_feat_flatten.dtype, device=mask_feat_flatten.device)
                # mask_feat = self.res_linear[name](mask_feat_flatten)

                if name=='res2':
                    self.res2 = self.res2.to(device=mask_feat_flatten.device, dtype=mask_feat_flatten.dtype)
                    mask_feat = self.res2(mask_feat_flatten)
                elif name=='res3':
                    self.res3 = self.res3.to(device=mask_feat_flatten.device, dtype=mask_feat_flatten.dtype)
                    mask_feat = self.res3(mask_feat_flatten)
                elif name=='res4':
                    self.res4 = self.res4.to(device=mask_feat_flatten.device, dtype=mask_feat_flatten.dtype)
                    mask_feat = self.res4(mask_feat_flatten)
                else:
                    self.res5 = self.res5.to(device=mask_feat_flatten.device, dtype=mask_feat_flatten.dtype)
                    mask_feat = self.res5(mask_feat_flatten)

                mask_feat = mask_feat.reshape(*mask_feat_raw.shape[:2], -1)
                mask_feat = mask_feat.to(raw_dtype)
               
                mask_feats[i] = mask_feat[0]
            mask_feats = mask_feats.sum(0)
            self.feat_linear = self.feat_linear.to(dtype=mask_feats.dtype, device=mask_feats.device)
            mask_feats_linear = self.feat_linear(mask_feats)
            query_feats.append(mask_feats_linear)

            # position
            mask = F.interpolate(mask, size=self.mask_shape, mode='bilinear', align_corners=False)
            self.mask_linear = self.mask_linear.to(dtype=mask.dtype, device=mask.device)
            pos_feat = self.mask_linear(mask.reshape(mask.shape[1], -1))
            pos_feats.append(pos_feat)

        return query_feats, pos_feats

    
class MaskPooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, mask):

        if not x.shape[-2:] == mask.shape[-2:]:
            # reshape mask to x
            mask = F.interpolate(mask, size=x.shape[-2:], mode='bilinear', align_corners=False)

        b, c, h ,w = x.shape
        b, q, h, w = mask.shape
        mask = (mask > 0).to(mask.dtype)
        denorm = mask.sum(dim=(-1, -2), keepdim=True) + 1e-8

        mask_pooled_x = torch.einsum(
            "bchw,bqhw->bqc",
            x,
            mask / denorm,
        )
        return mask_pooled_x
