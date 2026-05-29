import torch
from torch import nn

import glog

def extract_susv_params(module):
    susv_params = []
    params = []
    for name, param in module.named_parameters():
        if param.requires_grad:
            if 'SU' in name or 'SV' in name:
                susv_params.append(param)
            else:
                params.append(param)
            # glog.info(f"tuneable: {name}")
    return susv_params, params


def get_susv_adam(susv_params, params, args):
    return torch.optim.Adam([
        {
            'params': susv_params,
            'lr': args.ft_susv_lr
        },
        {
            'params': params,
            'lr': args.ft_lr
        },
    ])


def save_susv(module, path, save_tuneable=False):
    saved_layer = torch.load(path, map_location=torch.device('cpu'), weights_only=True)
    saved_layer['SU'] = module.SU.data.to(torch.bfloat16)
    saved_layer['SV'] = module.SV.data.to(torch.bfloat16)
    if save_tuneable:
        saved_layer['codebook_class.codebook.linear_proj.weight'] = module.codebook_class.codebook.linear_proj.weight.data.to(dtype=torch.bfloat16)
        saved_layer['codebook_class.codebook.linear_proj.bias'] = module.codebook_class.codebook.linear_proj.bias.data.to(dtype=torch.bfloat16)
    torch.save(saved_layer, path)


def calculate_mse_loss(layer, dataloader, device, position_embeddings):
    layer.eval()
    total_loss = 0
    ct = 0
    batch_position_embeddings = None
    with torch.no_grad():
        for _, (source, target) in enumerate(dataloader):
            if batch_position_embeddings is None:
                batch_size = source.shape[0]
                batch_position_embeddings = (position_embeddings[0][:batch_size], position_embeddings[1][:batch_size])
            # glog.info(f"batch size info: {i} {batch_size}, {source.shape}, {batch_position_embeddings[0].shape} {batch_position_embeddings[1].shape}")
            total_loss += nn.MSELoss()(layer(source.to(device), position_embeddings=batch_position_embeddings)[0], target.to(device))
            ct += 1
    layer.train()
    return (total_loss / ct).cpu().item()


def calculate_ce_loss(layer, position_embeddings, attention_mask, dataloader):
    layer.eval()
    total_loss = 0
    ct = 0
    with torch.no_grad():
        for source, target in dataloader:
            output = layer(
                source,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask.float())[:, :-1].contiguous()
            total_loss += nn.CrossEntropyLoss()(
                output.view(-1, output.shape[-1]),
                target.to(0).view(-1, target.shape[-1]),
            )
            ct += 1
    layer.train()
    return (total_loss / ct).cpu().item()
