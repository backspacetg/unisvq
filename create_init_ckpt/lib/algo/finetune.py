"""
Utilities for fine tuning
"""
import copy
import traceback
from operator import attrgetter

import glog
import torch
from torch import nn

from .. import codebook, utils
from ..linear import *

from . import quip


def finetune_decoder_layer(layer, name, device, train_dl, valid_dl, position_embeddings, args):
    layer = layer.to(device=device, dtype=torch.bfloat16)
    # for n, p in layer.named_parameters():
    #     if p.requires_grad:
    #         glog.info(f"{n} to be tuned")
    #     else:
    #         glog.info(f"{n} fixed")
    susv_params, params = utils.extract_susv_params(layer)
    optim = utils.get_susv_adam(susv_params, params, args)
    best_loss = utils.calculate_mse_loss(layer, valid_dl, device=device, position_embeddings=position_embeddings)
    best_sd = copy.deepcopy(layer.state_dict())
    glog.info(f'layer {name} initial loss {best_loss}')
    # scaler = torch.amp.GradScaler(enabled=True)
    worse_ct = 0
    batch_position_embeddings = None
    for epoch in range(args.ft_epochs):
        for bidx, (source, targets) in enumerate(train_dl):
            # with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
            if batch_position_embeddings is None:
                batch_position_embeddings = (position_embeddings[0][:args.ft_bs], position_embeddings[1][:args.ft_bs])
            output = layer(source.to(device), position_embeddings=batch_position_embeddings)[0]
            loss = nn.MSELoss()(output, targets.to(device))
            loss.backward()
            # scaler.scale(loss).backward()
            if bidx % args.ft_update_freq == args.ft_update_freq - 1 or bidx == len(train_dl) - 1:
                # scaler.step(optim)
                optim.step()
                # scaler.update()
                optim.zero_grad()

        if epoch % args.ft_valid_freq == (args.ft_valid_freq - 1):
            test_loss = utils.calculate_mse_loss(layer, valid_dl, device=device, position_embeddings=position_embeddings)
            if test_loss < best_loss:
                glog.info(f'layer {name} @ epoch {epoch} new loss {test_loss} old loss {best_loss} BETTER')
                best_loss = test_loss
                best_sd = copy.deepcopy(layer.state_dict())
                worse_ct = 0
            else:
                glog.info(f'layer {name} @ epoch {epoch} new loss {test_loss} old loss {best_loss} WORSE')
                worse_ct += 1
                if worse_ct >= args.ft_early_stop:
                    break

    del optim, train_dl, valid_dl

    layer.load_state_dict(best_sd)
    utils.clean()
    layer = layer.cpu()


def quantize_finetune_decoder_layer(mixed_layer, quant_order, idx, cb, args, device, pre_orig_emb, orig_emb, position_embeddings, hessian_data=None):
    torch.manual_seed(idx)
    torch.set_num_threads(args.num_cpu_threads)

    codebook_id = codebook.get_id(args.codebook)
    mixed_layer = mixed_layer.to(dtype=torch.bfloat16)

    train_dl, valid_dl = utils.split_data(X=pre_orig_emb, Y=orig_emb, args=args)

    shared_args = (cb.codesz, cb.packsz, cb.pack_out, str(cb.idx_dtype), cb.version)
    shared_kwargs = {
        'rank': args.lora_rank,
        'rescale_WH': args.rescale_WH,
        'resid_scale_override': args.resid_scale_override,
        'train_mode': args.ft_train_mode,
        'grad_ckpt': args.ft_grad_ckpt,
        'codebook_id': codebook_id,
        'skip_mix_hadk': args.skip_mix_hadk
    }

    for _, (linear_attr, group_name) in enumerate(quant_order):
        orig_linear = attrgetter(linear_attr)(mixed_layer)
        save_path = f'{args.save_path}/{idx}_{linear_attr}.pt'
        with torch.no_grad():
            weights = [orig_linear.weight]
            if hessian_data is None:
                hessian_path = f'{args.hessian_path}/{idx}_{group_name}.pt'
                quip.quantize_linear(weights, save_path, hessian_path, cb, args, device, bias=orig_linear.bias)
            else:
                quip.quantize_linear_with_hessian(weights, save_path, hessian_data[group_name], cb, args, device, bias=orig_linear.bias)
            saved_linear = torch.load(save_path, map_location=torch.device('cpu'), weights_only=True)
            shared_kwargs['bias'] = (orig_linear.bias is not None)
            quant_linear = QuantizedLinear(
                saved_linear['shapes'][0][1],
                saved_linear['shapes'][0][0],
                *shared_args, **shared_kwargs
            )
            utils.unpack_quip(quant_linear, saved_linear, codebook_id, cb.codesz)
        quant_linear.SU = nn.Parameter(quant_linear.SU.to(dtype=torch.bfloat16), requires_grad=True)
        quant_linear.SV = nn.Parameter(quant_linear.SV.to(dtype=torch.bfloat16), requires_grad=True)
        if "tuneable" in args.codebook:
            quant_linear.codebook_class.codebook.linear_proj.to(dtype=torch.bfloat16)
        split_attr = linear_attr.split('.')
        setattr(attrgetter('.'.join(split_attr[:-1]))(mixed_layer), split_attr[-1], quant_linear)

    with torch.enable_grad():
        finetune_decoder_layer(
            layer=mixed_layer, 
            name=f'{idx}_{group_name}', 
            device=device, 
            train_dl=train_dl, 
            valid_dl=valid_dl, 
            position_embeddings=position_embeddings,
            args=args
        )

    with torch.no_grad():
        utils.clean()
        for linear_attr, group_name in quant_order:
            utils.save_susv(attrgetter(linear_attr)(mixed_layer), f'{args.save_path}/{idx}_{linear_attr}.pt', save_tuneable = ("tuneable" in args.codebook))

    # mixed_layer = mixed_layer.to(torch.float16).cpu()
    utils.clean()
    torch.set_grad_enabled(False)
