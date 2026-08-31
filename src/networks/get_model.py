from networks.unext import UNext
from networks.ukan import UKAN
from networks.unetv2 import UNetV2
from monai.networks.nets import UNet, UNETR, SwinUNETR
from networks.ukat import UKAT
from networks.ukast import UKAST
from networks.ukast_debug import UKASTdb

def getModel(cnf, device=0):
    """
    Return model instance based on name.
    kwargs can hold things like in_ch, out_ch, etc.
    """
    
    # ----------- UNEXT --------------------
    if cnf.train.model.lower() == "unext":
        model = UNext(
            num_classes=cnf.data.num_class,
            img_size=cnf.train.cs[0],
            in_chans=cnf.data.num_ch
        )
    
    # ----------- UNET --------------------
    elif cnf.train.model.lower() == "unet":
        model = UNet(
            spatial_dims=cnf.model_params.spatial_dims, 
            in_channels=cnf.meta.ch, 
            out_channels=cnf.meta.cls, 
            channels=cnf.model_params.channels,
            strides=cnf.model_params.strides,
            act=cnf.model_params.act, 
            dropout=cnf.model_params.dropout, 
        )  

    # ----------- UNETv2 --------------------
    elif cnf.train.model.lower() == "unetv2":
        model = UNetV2(
            in_chans = cnf.meta.ch,
            n_classes=cnf.meta.cls,
            backbone=cnf.model_params.backbone, 
            deep_supervision=cnf.model_params.deep_sup, 
            pretrained_path=None
        )
        
    # ----------- UNETR --------------------
    elif cnf.train.model.lower() == "unetr":
        model = UNETR(
            in_channels=cnf.meta.ch,
            out_channels=cnf.meta.cls,
            img_size=cnf.train.cs[0],
            feature_size=cnf.model_params.feature_size,
            hidden_size=cnf.model_params.hidden_size,
            mlp_dim=cnf.model_params.mlp_dim,
            num_heads=cnf.model_params.num_heads,
            proj_type=cnf.model_params.proj_type,
            norm_name=cnf.model_params.norm_name,
            conv_block=cnf.model_params.conv_block,
            res_block=cnf.model_params.res_block,
            dropout_rate=cnf.model_params.dropout_rate,
            spatial_dims=cnf.model_params.spatial_dims,
            qkv_bias=cnf.model_params.qkv_bias,
            save_attn=cnf.model_params.save_attn
        )  

    # ----------- SwinUNETR v1 and v2 --------------------
    elif cnf.train.model.lower() in {"swinunetr", "swinunetr_v2"}:
        use_v2 = cnf.train.model.lower() == "swinunetr_v2"
        model = SwinUNETR(
            in_channels=cnf.meta.ch,
            out_channels=cnf.meta.cls,
            patch_size=cnf.model_params.patch_size,
            depths=cnf.model_params.depths,
            num_heads=cnf.model_params.num_heads,
            window_size=cnf.model_params.window_size,
            qkv_bias=cnf.model_params.qkv_bias,
            mlp_ratio=cnf.model_params.mlp_ratio,
            feature_size=cnf.model_params.feature_size,
            norm_name=cnf.model_params.norm_name,
            drop_rate=cnf.model_params.drop_rate,
            attn_drop_rate=cnf.model_params.attn_drop_rate,
            dropout_path_rate=cnf.model_params.dropout_path_rate,
            normalize=cnf.model_params.normalize,
            patch_norm=cnf.model_params.patch_norm,
            use_checkpoint=cnf.model_params.use_checkpoint,
            spatial_dims=cnf.model_params.spatial_dims,
            downsample=cnf.model_params.downsample,
            use_v2=use_v2,
        )

    #----------- UKAST_debug --------------------
    elif "ukast_db" in cnf.train.model.lower():
        model = UKASTdb(
            device=device,
            in_channels=cnf.meta.ch,
            out_channels=cnf.meta.cls,
            patch_size=cnf.model_params.patch_size,
            depths=cnf.model_params.depths,
            num_heads=cnf.model_params.num_heads,
            window_size=cnf.model_params.window_size,
            qkv_bias=cnf.model_params.qkv_bias,
            mlp_ratio=cnf.model_params.mlp_ratio,
            feature_size=cnf.model_params.feature_size,
            norm_name=cnf.model_params.norm_name,
            drop_rate=cnf.model_params.drop_rate,
            attn_drop_rate=cnf.model_params.attn_drop_rate,
            dropout_path_rate=cnf.model_params.dropout_path_rate,
            normalize=cnf.model_params.normalize,
            patch_norm=cnf.model_params.patch_norm,
            use_checkpoint=cnf.model_params.use_checkpoint,
            spatial_dims=cnf.model_params.spatial_dims,
            downsample=cnf.model_params.downsample,
            kan_act_init=cnf.model_params.act_init,
            kan_group_size=cnf.model_params.kan_group_size,
            kan_poly_order=cnf.model_params.kan_poly_order,
            use_resconv=cnf.model_params.use_resconv,
            use_postnorm=cnf.model_params.use_postnorm,
            use_triton=cnf.model_params.use_triton,
            num_registers=cnf.model_params.num_registers,
        )    
    
    #----------- UKAST --------------------
    elif "ukast" in cnf.train.model.lower():
        model = UKAST(
            device=device,
            in_channels=cnf.meta.ch,
            out_channels=cnf.meta.cls,
            patch_size=cnf.model_params.patch_size,
            depths=cnf.model_params.depths,
            num_heads=cnf.model_params.num_heads,
            window_size=cnf.model_params.window_size,
            qkv_bias=cnf.model_params.qkv_bias,
            mlp_ratio=cnf.model_params.mlp_ratio,
            feature_size=cnf.model_params.feature_size,
            norm_name=cnf.model_params.norm_name,
            drop_rate=cnf.model_params.drop_rate,
            attn_drop_rate=cnf.model_params.attn_drop_rate,
            dropout_path_rate=cnf.model_params.dropout_path_rate,
            normalize=cnf.model_params.normalize,
            patch_norm=cnf.model_params.patch_norm,
            use_checkpoint=cnf.model_params.use_checkpoint,
            spatial_dims=cnf.model_params.spatial_dims,
            downsample=cnf.model_params.downsample,
            kan_act_init=cnf.model_params.act_init,
            kan_group_size=cnf.model_params.group_size,
            kan_poly_order=cnf.model_params.poly_order,
            use_resconv=cnf.model_params.use_resconv,
            use_postnorm=cnf.model_params.use_postnorm,
            use_triton=cnf.model_params.use_triton
        )    

    # ----------- UKAN --------------------
    elif cnf.train.model.lower() == "ukan":
        # Get base parameters for UKAN model
        model_kwargs = {
            'num_classes': cnf.meta.cls,
            'img_size': cnf.train.cs[0],
            'in_chans': cnf.meta.ch
        }
        
        # Check if model_params is configured
        if hasattr(cnf, 'model_params') and cnf.model_params:
            # Add embed_dims parameter (if configured)
            if hasattr(cnf.model_params, 'embed_dims'):
                model_kwargs['embed_dims'] = cnf.model_params.embed_dims
                # print(f"Using custom embed_dims: {cnf.model_params.embed_dims}")
            
            # Add other optional parameters
            if hasattr(cnf.model_params, 'no_kan'):
                model_kwargs['no_kan'] = cnf.model_params.no_kan
                
            if hasattr(cnf.model_params, 'drop_rate'):
                model_kwargs['drop_rate'] = cnf.model_params.drop_rate
                
            if hasattr(cnf.model_params, 'drop_path_rate'):
                model_kwargs['drop_path_rate'] = cnf.model_params.drop_path_rate
        else:
            print("Using default embed_dims: [256, 320, 512]")
        
        model = UKAN(**model_kwargs)
        
        # Print model parameter count info
        # total_params = sum(p.numel() for p in model.parameters())
        # print(f"UKAN model total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
        
    # ----------- UKAT  --------------------
    elif cnf.train.model.lower() == "ukat":
        model = UKAT(
            in_channels=cnf.meta.ch,
            out_channels=cnf.meta.cls,
            img_size=(cnf.train.cs[0],cnf.train.cs[1]),
            feature_size=cnf.model_params.feature_size,
            hidden_size=cnf.model_params.hidden_size,
            mlp_ratio=cnf.model_params.mlp_ratio,
            num_heads=cnf.model_params.num_heads,
            depth=cnf.model_params.depth,
            act_init=cnf.model_params.act_init,
            proj_type=cnf.model_params.proj_type,
            norm_name=cnf.model_params.norm_name,
            conv_block=cnf.model_params.conv_block,
            res_block=cnf.model_params.res_block,
            dropout_rate=cnf.model_params.dropout_rate,
            qkv_bias=cnf.model_params.qkv_bias,
            save_attn=cnf.model_params.save_attn,
            spatial_dims=cnf.model_params.spatial_dims,
        )

    else:
        raise ValueError(f"Model {cnf.train.model} currently unsupported.")
    
    return model


# OLD BACKUPS
def getModelOld(cnf):
    """
    Return model instance based on name.
    kwargs can hold things like in_ch, out_ch, etc.
    """
    if cnf.train.model.lower() == "unext":
        model = UNext(
            num_classes=cnf.meta.cls,
            img_size=cnf.train.cs[0],
            in_chans=cnf.meta.ch
        )
    elif cnf.train.model.lower() == "ukan":
        model = UKAN(
            num_classes=cnf.meta.cls,
            img_size=cnf.train.cs[0],
            in_chans=cnf.meta.ch
        )
    else:
        raise ValueError(f"Model {name} currently unsupported.")
    
    return model