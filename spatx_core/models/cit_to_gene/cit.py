from typing import Type, List, Any

import torch
import torch.nn as nn
from torch.nn import Module
from torch import Tensor
from ._cit_building_blocks import (
    PatchEmbed, BasicLayer, FinalPatchExpand_X4, PatchMerging,
    PatchExpand, GhostModule, BasicLayer_up,
    ConvMixer, ConvBlockDDConv, ConvBlock, UpConvDDConv
)
from timm.layers.weight_init import trunc_normal_ #type: ignore

class CIT(nn.Module):
    def __init__(self, img_size : int =224, patch_size : int =4, in_chans : int =3, out_chans : int =1,
                 embed_dim : int =96, depths : list[int] =[2, 2, 6, 2], depths_decoder : list[int] =[1, 2, 2, 2], num_heads : list[int] =[3, 6, 12, 24],
                 window_size : int =7, mlp_ratio : float =4., qkv_bias : bool =True, qk_scale : float | None =None,
                 drop_rate : float =0., attn_drop_rate : float =0., drop_path_rate : List[float] =[0.1],
                 norm_layer : Type[Module] = nn.LayerNorm, ape : bool =True, patch_norm : bool =True,
                 use_checkpoint : bool =False, final_upsample : str ="expand_first", **kwargs : Any):
        super().__init__() #type: ignore
        self.out_channel = out_chans
        self.num_layers = len(depths)
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.size = int(img_size/(2 ** (self.num_layers + 1)))
        self.size_out = int(img_size/4)
        self.num_features_up = int(embed_dim * 2)
        self.mlp_ratio = mlp_ratio
        self.final_upsample = final_upsample

        self.window_size = window_size
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate
        self.norm_layer = nn.LayerNorm

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate[0], sum(depths))]


        if self.final_upsample == "expand_first":
            self.up = FinalPatchExpand_X4(input_resolution=(img_size // patch_size, img_size // patch_size),
                                          dim_scale=4, dim=embed_dim)
            self.output = nn.Conv2d(in_channels=6, out_channels=1, kernel_size=1, bias=False)

        self.apply(self._init_weights)

        self.embed_dim = 96
        self.num_heads = 3
        self.depth35 = 6
        self.drop_path3 = dpr[4:10]
        self.drop_path4 = dpr[10:12]

        print("CiT-Net-T----embed_dim:{}; num_heads:{}; depths:{}".format(self.embed_dim, num_heads, depths))

        self.layer1 = BasicLayer(dim=self.embed_dim * 1,
                            input_resolution=(56, 56),
                            depth=2,
                            num_heads=self.num_heads * 1,
                            window_size=self.window_size,  # 7
                            mlp_ratio=self.mlp_ratio,  # 4.
                            qkv_bias=self.qkv_bias,  # True
                            qk_scale=self.qk_scale,  # None
                            drop=self.drop_rate,  # 0.
                            attn_drop=self.attn_drop_rate,  # 0.
                            drop_path=dpr[0:2],
                            norm_layer=self.norm_layer,
                            downsample=PatchMerging,
                            use_checkpoint=False)

        self.layer2 = BasicLayer(dim=self.embed_dim * 2,
                            input_resolution=(28, 28),
                            depth=2,
                            num_heads=self.num_heads * 2,
                            window_size=self.window_size,  # 7
                            mlp_ratio=self.mlp_ratio,  # 4.
                            qkv_bias=self.qkv_bias,  # True
                            qk_scale=self.qk_scale,  # None
                            drop=self.drop_rate,  # 0.
                            attn_drop=self.attn_drop_rate,  # 0.
                            drop_path=dpr[2:4],
                            norm_layer=self.norm_layer,
                            downsample=PatchMerging,
                            use_checkpoint=False)

        self.layer3 = BasicLayer(dim=self.embed_dim * 4,
                            input_resolution=(14, 14),
                            depth=self.depth35,
                            num_heads=self.num_heads * 4,
                            window_size=self.window_size,  # 7
                            mlp_ratio=self.mlp_ratio,  # 4.
                            qkv_bias=self.qkv_bias,  # True
                            qk_scale=self.qk_scale,  # None
                            drop=self.drop_rate,  # 0.
                            attn_drop=self.attn_drop_rate,  # 0.
                            drop_path=self.drop_path3,
                            norm_layer=self.norm_layer,
                            downsample=PatchMerging,
                            use_checkpoint=False)

        self.layer4 = BasicLayer(dim=self.embed_dim * 8,
                            input_resolution=(7, 7),
                            depth=2,
                            num_heads=self.num_heads * 8,
                            window_size=self.window_size,  # 7
                            mlp_ratio=self.mlp_ratio,  # 4.
                            qkv_bias=self.qkv_bias,  # True
                            qk_scale=self.qk_scale,  # None
                            drop=self.drop_rate,  # 0.
                            attn_drop=self.attn_drop_rate,  # 0.
                            drop_path=self.drop_path4,
                            norm_layer=self.norm_layer,
                            downsample=None,
                            use_checkpoint=False)

        self.norm = norm_layer([self.num_features, self.size, self.size])

        self.Patch_Expand1 = PatchExpand(input_resolution=(7, 7),
                                        dim=self.embed_dim * 8,
                                        dim_scale=2,
                                        norm_layer=norm_layer)


        self.concat_linear1 = GhostModule(inp=self.embed_dim * 8, oup=self.embed_dim * 4)

        self.layer5 = BasicLayer_up(dim=self.embed_dim * 4,
                                    input_resolution=(14, 14),
                                    depth=self.depth35,
                                    num_heads=self.num_heads * 4,

                                    window_size=self.window_size,  # 7
                                    mlp_ratio=self.mlp_ratio,  # 4.
                                    qkv_bias=self.qkv_bias,  # True
                                    qk_scale=self.qk_scale,  # None
                                    drop=self.drop_rate,  # 0.
                                    attn_drop=self.attn_drop_rate,  # 0.

                                    drop_path=self.drop_path3,
                                    norm_layer=norm_layer,
                                    upsample=PatchExpand,
                                    use_checkpoint=False)

        self.concat_linear2 = GhostModule(inp=self.embed_dim * 4, oup=self.embed_dim * 2)

        self.layer6 = BasicLayer_up(dim=self.embed_dim * 2,
                                    input_resolution=(28, 28),
                                    depth=2,
                                    num_heads=self.num_heads * 2,

                                    window_size=self.window_size,  # 7
                                    mlp_ratio=self.mlp_ratio,  # 4.
                                    qkv_bias=self.qkv_bias,  # True
                                    qk_scale=self.qk_scale,  # None
                                    drop=self.drop_rate,  # 0.
                                    attn_drop=self.attn_drop_rate,  # 0.

                                    drop_path=dpr[2:4],
                                    norm_layer=norm_layer,
                                    upsample=PatchExpand,
                                    use_checkpoint=False)

        self.concat_linear3 = GhostModule(inp=self.embed_dim * 2, oup=self.embed_dim * 1)

        self.layer7 = BasicLayer_up(dim=self.embed_dim * 1,
                                    input_resolution=(56, 56),
                                    depth=2,
                                    num_heads=self.num_heads * 1,

                                    window_size=self.window_size,  # 7
                                    mlp_ratio=self.mlp_ratio,  # 4.
                                    qkv_bias=self.qkv_bias,  # True
                                    qk_scale=self.qk_scale,  # None
                                    drop=self.drop_rate,  # 0.
                                    attn_drop=self.attn_drop_rate,  # 0.

                                    drop_path=dpr[0:2],
                                    norm_layer=norm_layer,
                                    upsample=None,
                                    use_checkpoint=False)

        self.norm_up = norm_layer([self.embed_dim, self.size_out, self.size_out])
        self.patch = ConvMixer(dim=48, depth=5)  # 修改_ConvMixer层数
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Conv1e = ConvBlock(ch_in=48, ch_out=self.embed_dim * 1)
        self.Conv1s = ConvBlock(ch_in=48, ch_out=self.embed_dim * 1)
        self.Conv2e = ConvBlockDDConv(ch_in=self.embed_dim * 1, ch_out=self.embed_dim * 2)
        self.Conv3e = ConvBlockDDConv(ch_in=self.embed_dim * 2, ch_out=self.embed_dim * 4)
        self.Conv4e = ConvBlockDDConv(ch_in=self.embed_dim * 4, ch_out=self.embed_dim * 8)
        self.Up4d = UpConvDDConv(ch_in=self.embed_dim * 8, ch_out=self.embed_dim * 4)
        self.Up_conv4d = ConvBlock(ch_in=self.embed_dim * 8, ch_out=self.embed_dim * 4)
        self.Up3d = UpConvDDConv(ch_in=self.embed_dim * 4, ch_out=self.embed_dim * 2)
        self.Up_conv3d = ConvBlock(ch_in=self.embed_dim * 4, ch_out=self.embed_dim * 2)
        self.Up2d = UpConvDDConv(ch_in=self.embed_dim * 2, ch_out=self.embed_dim * 1)
        self.Up_conv2d = ConvBlock(ch_in=self.embed_dim * 2, ch_out=self.embed_dim * 1)
        self.Mid_Conv1 = nn.Conv2d(self.embed_dim * 2, self.embed_dim * 1, kernel_size=1, stride=1, padding=0)
        self.Mid_Conv2 = nn.Conv2d(self.embed_dim * 4, self.embed_dim * 2, kernel_size=1, stride=1, padding=0)
        self.Mid_Conv3 = nn.Conv2d(self.embed_dim * 8, self.embed_dim * 4, kernel_size=1, stride=1, padding=0)
        self.BN = nn.BatchNorm2d(1)
        self.CiT_Conv = nn.Conv2d(2, 1, kernel_size=1, stride=1, padding=0)

    def _init_weights(self, m : Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None: #type:ignore
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.unused
    def _no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.unused
    def _no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}


    def _up_x4(self, x : Tensor):
        if self.final_upsample == "expand_first":
            x = self.up(x)
            x = self.output(x)

        return x

    def forward(self, x : Tensor):   # 1,3,224,224
        x = self.patch(x)
        Cnn = x
        Swin = x

        Cnn = self.Conv1e(Cnn)     # 1,96,56,56
        Swin = self.Conv1s(Swin)   # 1,96,56,56
        Cnn1 = Cnn
        Swin1 = Swin
        Mid1 = torch.cat((Cnn1, Swin1), dim=1)
        Mid1 = self.Mid_Conv1(Mid1)

        Cnn = self.maxpool(Cnn)
        Cnn = self.Conv2e(Cnn)
        Swin = self.layer1(Swin)   # 28,28
        Cnn2 = Cnn
        Swin2 = Swin
        Mid2 = torch.cat((Cnn2, Swin2), dim=1)
        Mid2 = self.Mid_Conv2(Mid2)

        Cnn = self.maxpool(Cnn)
        Cnn = self.Conv3e(Cnn)
        Swin = self.layer2(Swin)  # 14,14
        Cnn3 = Cnn
        Swin3 = Swin
        Mid3 = torch.cat((Cnn3, Swin3), dim=1)
        
        Mid3 = self.Mid_Conv3(Mid3)

        Cnn = self.maxpool(Cnn)
        Cnn = self.Conv4e(Cnn)
        Swin = self.layer3(Swin)  # 7,7
        Swin = self.layer4(Swin)  # 7,7
        Swin = self.norm(Swin)  # B L C  (1, 768, 7, 7)
        Cnn4 = Cnn
        Swin4 = Swin
        return torch.cat((Cnn4, Swin4), dim=1)
