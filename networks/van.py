import jittor as jt
from jittor import nn
import warnings

from utils.pyt_utils import DropPath, to_2tuple

from utils.pyt_utils import load_model
from .CC import CC_module as CrissCrossAttention 

# TODO parallel
BatchNorm2d = nn.BatchNorm


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def execute(self, x):
        x = self.dwconv(x)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., linear=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv(in_features, hidden_features, 1)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        # self.apply(self._init_weights)  TODO
        self.linear = linear
        if self.linear:
            self.relu = nn.ReLU(inplace=True)

    def execute(self, x):
        x = self.fc1(x)
        if self.linear:
            x = self.relu(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class AttentionModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv(dim, dim, 1)

    def execute(self, x):
        u = x.clone()       
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)

        return u * attn

class SpatialAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.proj_1 = nn.Conv(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = AttentionModule(d_model)
        self.proj_2 = nn.Conv(d_model, d_model, 1)

    def execute(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x

class Block(nn.Module):
    def __init__(self,
                 dim,
                 mlp_ratio=4.,
                 drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 linear=False
                #  norm_cfg=dict(type='SyncBN', requires_grad=True)
                 ):
        super().__init__()
        # self.norm1 = build_norm_layer(norm_cfg, dim)[1]  # TODO
        self.norm1 = nn.BatchNorm2d(dim)
        self.attn = SpatialAttention(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # self.norm2 = build_norm_layer(norm_cfg, dim)[1]
        self.norm2 = nn.BatchNorm2d(dim)  # TODO
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop, linear=linear)
        layer_scale_init_value = 1e-2

        self.layer_scale_1 = jt.Var(
            layer_scale_init_value * jt.ones((dim))).start_grad()
        self.layer_scale_2 = jt.Var(
            layer_scale_init_value * jt.ones((dim))).start_grad()

        # self.apply(self._init_weights)  TODO

    def execute(self, x, H, W):
        B, N, C = x.shape
        x = x.permute(0, 2, 1).view(B, C, H, W)
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1)
                               * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1)
                               * self.mlp(self.norm2(x)))
        x = x.view(B, C, N).permute(0, 2, 1)
        return x

class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self,
                 patch_size=7,
                 stride=4,
                 in_chans=3,
                 embed_dim=768
                #  norm_cfg=dict(type='SyncBN', requires_grad=True)
                 ):
        super().__init__()
        patch_size = to_2tuple(patch_size)

        self.proj = nn.Conv(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        
        # self.norm = build_norm_layer(norm_cfg, embed_dim)[1]
        self.norm = nn.BatchNorm2d(embed_dim)
        # self.apply(self._init_weights)  TODO

    def execute(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = self.norm(x)

        x = x.flatten(2).transpose(1, 2)

        return x, H, W

class RCCAModule(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes):
        super(RCCAModule, self).__init__()
        inter_channels = in_channels // 4  # 512
        self.conva = nn.Sequential(nn.Conv(in_channels, inter_channels, 3, padding=1, bias=False),
                                   BatchNorm2d(inter_channels), nn.ReLU())
        self.cca = CrissCrossAttention(inter_channels)
        self.convb = nn.Sequential(nn.Conv(inter_channels, inter_channels, 3, padding=1, bias=False),
                                   BatchNorm2d(inter_channels), nn.ReLU())

        self.bottleneck = nn.Sequential(  # 2048+512 = 2560, 512
            nn.Conv(in_channels+inter_channels, out_channels, kernel_size=3, padding=1, dilation=1, bias=False),
            BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv(out_channels, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        )

    def execute(self, x, recurrence=1):
        output = self.conva(x)
        for _ in range(recurrence):
            output = self.cca(output)
        output = self.convb(output)

        output = self.bottleneck(jt.concat([x, output], 1))  # debug
        return output

# @BACKBONES.register_module()  TODO
class VAN(nn.Module):
    def __init__(self, num_classes, criterion, recurrence, 
                in_chans=3,
                embed_dims=[64, 128, 256, 512],
                mlp_ratios=[8, 8, 4, 4],
                drop_rate=0.,
                drop_path_rate=0.,
                depths=[3, 4, 6, 3],
                num_stages=4,
                linear=False,
                pretrained=None,
                init_cfg=None
                # norm_cfg=dict(type='SyncBN', requires_grad=True)
                ):
        super(VAN, self).__init__(init_cfg=init_cfg)


        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be set at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is not None:
            raise TypeError('pretrained must be a str or None')


        self.depths = depths
        self.num_stages = num_stages
        self.linear = linear

        dpr = [x.item() for x in jt.linspace(0, drop_path_rate,
                                                sum(depths))]  # stochastic depth decay rule
        cur = 0

        for i in range(num_stages):
            patch_embed = OverlapPatchEmbed(patch_size=7 if i == 0 else 3,
                                            stride=4 if i == 0 else 2,
                                            in_chans=in_chans if i == 0 else embed_dims[i - 1],
                                            embed_dim=embed_dims[i])

            block = nn.ModuleList([Block(dim=embed_dims[i],
                                         mlp_ratio=mlp_ratios[i],
                                         drop=drop_rate,
                                         drop_path=dpr[cur + j],
                                         linear=linear
                                        #  norm_cfg=norm_cfg
                                         )
                                   for j in range(depths[i])])
            norm = nn.LayerNorm(embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)
        
        self.head = RCCAModule(embed_dims[-1], embed_dims[-1]//4, num_classes)  # TODO Debug
        self.dsn = nn.Sequential(
            nn.Conv(embed_dims[-2], embed_dims[-2]//4, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(embed_dims[-2]//4), 
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv(embed_dims[-2]//4, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        )
        self.criterion = criterion
        self.recurrence = recurrence

    def execute(self, x, labels=None):

        B = x.shape[0]
        outs = []

        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x, H, W = patch_embed(x)
            for blk in block:
                x = blk(x, H, W)
            x = norm(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2)
            outs.append(x)
        
        # CC
        # outs: [1,64,193,193,], [1,128,97,97,], [1,320,49,49,], [1,512,25,25,]

        x_dsn = self.dsn(outs[-2])  # x = outs[-1] # [1,19,49,49,] 
        x = self.head(x, self.recurrence)  # [1,19,25,25,] 

        outs = [x, x_dsn]
        if self.criterion is not None and labels is not None:
            return self.criterion(outs, labels)
        else:
            return outs

def Seg_Model(num_classes, criterion=None, pretrained_model=None, recurrence=0, **kwargs):
    # van_b0
    # model = VAN(num_classes, criterion, recurrence,
    #         embed_dims=[32, 64, 160, 256], 
    #         mlp_ratios=[8, 8, 4, 4], 
    #         depths=[3, 3, 5, 2])
    # van_b1
    # model = VAN(num_classes, criterion, recurrence,
    #         embed_dims=[64, 128, 320, 512], 
    #         mlp_ratios=[8, 8, 4, 4], 
    #         depths=[2, 2, 4, 2])

    # van_b2
    model = VAN(num_classes, criterion, recurrence,
            embed_dims=[64, 128, 320, 512], 
            mlp_ratios=[8, 8, 4, 4], 
            depths=[3, 3, 12, 3])

    if pretrained_model is not None:
        model = load_model(model, pretrained_model)

    return model