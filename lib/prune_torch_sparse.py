import time 
import heapq 
import torch 
import torch.nn as nn 
from .sparsegpt import SparseGPT 
from .layerwrapper import WrappedGPT
from .data import get_loaders 
import numpy as np
import torch.nn as nn

from pdb import set_trace as st 
from .quant import *
import numpy as np
from scipy.optimize import linear_sum_assignment

def quantile_normalization(vector):
    sorted_vector, _ = torch.sort(vector)
    rank_mean = torch.mean(sorted_vector).unsqueeze(0)
    
    sorted_idx = torch.argsort(vector)
    ranks = torch.argsort(sorted_idx)
    normalized_vector = rank_mean.expand_as(vector).gather(0, ranks)
    
    return normalized_vector

def maximize_total_value(matrix):
    # 使用 linear_sum_assignment 函数来解决分配问题
    row_indices, col_indices = linear_sum_assignment(-matrix)  # 使用负值来最大化总体价值
    total_value = -matrix[row_indices, col_indices].sum()
    return total_value, row_indices, col_indices

def pert_strips(idx1, idx2, W_metric, W_strip_value, N, M, T):
    n1 = W_metric.shape[0]
    n2 = W_metric.shape[1]
    
    k1 = idx1 // M
    k2 = idx2 // M
    
    # The first strip
    strip_idx1 = [i for i in range(k1*M, (k1+1)*M) if i!= idx1]
    strip_idx1.append(idx2)
    tmp1 = W_metric[:,strip_idx1].clone()
    W_mask1 = (torch.zeros_like(tmp1)==1)
    W_mask1.scatter_(1,torch.topk(tmp1, N, dim=1, largest=False)[1], True)
    tmp1[W_mask1]=0
    
    # The second strip
    strip_idx2 = [i for i in range(k2*M, (k2+1)*M) if i!= idx2]
    strip_idx2.append(idx1)
    # print(strip_idx2, W_metric.shape[1])
    tmp2 = W_metric[:,strip_idx2].clone()
    W_mask2 = (torch.zeros_like(tmp2)==1)
    W_mask2.scatter_(1,torch.topk(tmp2, N, dim=1, largest=False)[1], True)
    tmp2[W_mask2]=0
#     print(tmp1, tmp2)
    after_pert = torch.sum(tmp1) + torch.sum(tmp2)
    delta = after_pert - (W_strip_value[k1] + W_strip_value[k2])
#     print(after_pert, W_strip_value[k1], W_strip_value[k2])
    
    
    if delta > 0 or np.random.rand()<torch.exp(delta / T):
#         print("T is: ", T, "delta is: ", delta, " excute perturbation!")
        W_strip_value[k1] = torch.sum(tmp1)
        W_strip_value[k2] = torch.sum(tmp2)
        # tmp = W_metric[:, idx1].clone()
        # W_metric[:, idx1] = W_metric[:, idx2].clone()
        # W_metric[:, idx2] = tmp
        return True
    else:
#         print("T is: ", T, "delta is: ", delta, " won't excute perturbation!")
        return False

def find_layers(module, layers=[nn.Linear], name=''):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

def check_sparsity(args, model):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 
    if "llama" in args.model:
        layers = model.model.layers
    elif "opt" in args.model:
        layers = model.model.decoder.layers
    count = 0 
    total_params = 0
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        sub_count = 0
        sub_params = 0
        for name in subset:
            W = subset[name].weight.data
            count += (W==0).sum().item()
            total_params += W.numel()

            sub_count += (W==0).sum().item()
            sub_params += W.numel()

        print(f"layer {i} sparsity {float(sub_count)/sub_params:.6f}")

    model.config.use_cache = use_cache 
    return float(count)/total_params 

def prepare_calibration_input(args, model, dataloader, device):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    if "llama" in args.model:
        layers = model.model.layers
        # dev = model.hf_device_map["model.embed_tokens"]
        if "model.embed_tokens" in model.hf_device_map:
            device = model.hf_device_map["model.embed_tokens"]
    elif "opt" in args.model:
        layers = model.model.decoder.layers

    

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=device)
    inps.requires_grad = False
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, args, module):
            super().__init__()
            self.module = module
            self.model = args.model
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            if "llama" in args.model:
                cache['position_ids'] = kwargs['position_ids']

            raise ValueError
    layers[0] = Catcher(args, layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(device))
        except ValueError:
            pass 
    layers[0] = layers[0].module

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
   
    model.config.use_cache = use_cache
    if "llama" in args.model:
        position_ids = cache['position_ids']
        return inps, outs, attention_mask, position_ids 
    elif "opt" in args.model:
        return inps, outs, attention_mask

def return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before):
    thres_cumsum = sum_before * alpha 
    sort_mask = tmp_metric <= thres_cumsum.reshape((-1,1))
    thres = torch.gather(sort_res[0], dim=1, index=sort_mask.sum(dim=1, keepdims=True)-1)
    W_mask = (W_metric <= thres)
    cur_sparsity = (W_mask==True).sum() / W_mask.numel()
    return W_mask, cur_sparsity

def prune_magnitude(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    if "llama" in args.model:
        layers = model.model.layers
    elif "opt" in args.model:
        layers = model.model.decoder.layers
        
    per_outneuron = False

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        for name in subset:
            W = subset[name].weight.data.clone().type(torch.float64)
            if args.prune_method == "magnitude":
                W_metric = torch.abs(W)
            elif args.prune_method == "magnitude_norm_in":
                W_metric = torch.abs(W)/torch.sum(torch.abs(W), dim=0)
            elif args.prune_method == "magnitude_norm_out":
                W_metric = torch.abs(W)/torch.sum(torch.abs(W), dim=1).reshape(-1, 1)
            elif args.prune_method == "magnitude_norm_in_out":
                W_metric = torch.abs(W)/torch.sum(torch.abs(W), dim=0) + torch.abs(W)/torch.sum(torch.abs(W), dim=1).reshape(-1, 1)
            elif args.prune_method == "magnitude_norm_in_out_norm_sum":
                W_in = torch.abs(W)/torch.sum(torch.abs(W), dim=0)
                W_out = torch.abs(W)/torch.sum(torch.abs(W), dim=1).reshape(-1, 1)
                # print(W_in.shape, W_out.shape)
                W_metric = W_in/torch.sum(W_in) + W_out/torch.sum(W_out)
            elif args.prune_method == "magnitude_norm_in_out_norm_mean":
                W_in = torch.abs(W)/torch.sum(torch.abs(W), dim=0)
                W_out = torch.abs(W)/torch.sum(torch.abs(W), dim=1).reshape(-1, 1)
                W_metric = W_in/torch.mean(W_in) + W_out/torch.mean(W_out)
                
            elif args.prune_method == "magnitude_norm_in_out_norm_z_score":
                W_in = torch.abs(W)/torch.sum(torch.abs(W), dim=0)
                W_out = torch.abs(W)/torch.sum(torch.abs(W), dim=1).reshape(-1, 1)
                def zscore(data):
                    mean = torch.mean(data)
                    std_dev = torch.std(data)
                    return (data - mean) / std_dev
                W_metric = zscore(W_in) + zscore(W_out)
                
            elif args.prune_method == "magnitude_norm_in_out_norm_quantile":
                W_in = torch.abs(W)/torch.sum(torch.abs(W), dim=0)
                W_out = torch.abs(W)/torch.sum(torch.abs(W), dim=1).reshape(-1, 1)

                W_metric = quantile_normalization(W_in.squeeze()).unsqueeze(0) + quantile_normalization(W_out.squeeze()).unsqueeze(1)
                
            elif args.prune_method == "magnitude_divide_norm_in_out":
                W_metric = torch.abs(W)/(torch.sum(torch.abs(W), dim=0) + torch.sum(torch.abs(W), dim=1).reshape(-1, 1))
            elif args.prune_method == "magnitude_norm_in_out_per_outneuron":
                per_outneuron = True
                W_metric = torch.abs(W)/torch.sum(torch.abs(W), dim=0) + torch.abs(W)/torch.sum(torch.abs(W), dim=1).reshape(-1, 1)
            
            elif args.prune_method == "magnitude_norm_in_out_multiply":
                W_metric = torch.abs(W)/torch.sum(torch.abs(W), dim=0) * torch.abs(W)/torch.sum(torch.abs(W), dim=1).reshape(-1, 1)
            
            elif args.prune_method == "magnitude_norm_in_out_mean_std":
                W_norm_in = torch.abs(W)/torch.sum(torch.abs(W), dim=0)
                W_norm_out = torch.abs(W)/torch.sum(torch.abs(W), dim=1).reshape(-1, 1)
                W_metric = (W_norm_in - torch.mean(W_norm_in))/ torch.std(W_norm_in) + (W_norm_out - torch.mean(W_norm_out))/torch.std(W_norm_out)
            
            elif args.prune_method == "magnitude_norm_in_out_max":
                W_metric = torch.abs(W)/torch.max(torch.abs(W), dim=0)[0] + torch.abs(W)/torch.max(torch.abs(W), dim=1)[0].reshape(-1, 1)
                
            elif args.prune_method == "magnitude_norm_in_out_mean":
                W_metric = torch.abs(W)/torch.mean(torch.abs(W), dim=0) + torch.abs(W)/torch.mean(torch.abs(W), dim=1).reshape(-1, 1)
                
            # elif args.prune_method == "magnitude_norm_in_out_mean_std":
            #     W_metric = torch.abs(W)*torch.std(torch.abs(W), dim=0)/torch.mean(torch.abs(W), dim=0) + torch.abs(W)*torch.std(torch.abs(W), dim=1).reshape(-1, 1)/torch.mean(torch.abs(W), dim=1).reshape(-1, 1)
                
            elif "magnitude_norm_in_out_iter" in args.prune_method:
                iters = args.prune_method.split("_")[-1]
                sparsity = torch.linspace(0.1, args.sparsity_ratio, int(iters))
                W_metric = torch.abs(W)/torch.sum(torch.abs(W), dim=0) + torch.abs(W)/torch.sum(torch.abs(W), dim=1).reshape(-1, 1)
                for s in sparsity:
                    thresh = torch.sort(W_metric.flatten().cuda())[0][int(W.shape[0]* W.shape[1]*s)].cpu()
                    W_mask = (W_metric<=thresh)
                    W[W_mask] = 0
                    W_metric = torch.abs(W)/torch.sum(torch.abs(W), dim=0) + torch.abs(W)/torch.sum(torch.abs(W), dim=1).reshape(-1, 1)
            
                
            elif args.prune_method == "magnitude_norm_in_out_scale":
                W_metric = torch.abs(W)/torch.sum(torch.abs(W), dim=0) / torch.max(torch.abs(W)/torch.sum(torch.abs(W), dim=0)) + torch.abs(W)/torch.sum(torch.abs(W), dim=1).reshape(-1, 1) / torch.max(torch.abs(W)/torch.sum(torch.abs(W), dim=1).reshape(-1, 1))
                
            
            elif args.prune_method == "magnitude_norm_in_out_scale_sigm":
                sigmoid = nn.Sigmoid()
                W_metric = sigmoid(-10 + 20*(torch.abs(W)/torch.sum(torch.abs(W), dim=0) / torch.max(torch.abs(W)/torch.sum(torch.abs(W), dim=0)))) + sigmoid(-10 + 20*(torch.abs(W)/torch.sum(torch.abs(W), dim=1).reshape(-1, 1) / torch.max(torch.abs(W)/torch.sum(torch.abs(W), dim=1).reshape(-1, 1))))

            elif args.prune_method == "magnitude_norm_in_out_fix_alpha":
                W_metric = args.alpha * torch.abs(W)/torch.sum(torch.abs(W), dim=0) + (1-args.alpha) * torch.abs(W)/torch.sum(torch.abs(W), dim=1).reshape(-1, 1)
            elif args.prune_method == "magnitude_norm_in_out_fix_alpha_scale":
                W_metric_norm_in = torch.abs(W)/torch.sum(torch.abs(W), dim=0)
                W_metric_norm_out = torch.abs(W)/torch.sum(torch.abs(W), dim=1).reshape(-1, 1)

                W_metric = alpha * W_metric_norm_in * (torch.sum(W_metric_norm_out) / torch.sum(W_metric_norm_in)) + (1-alpha)* W_metric_norm_out

            elif args.prune_method == "magnitude_norm_in_out_alpha":
                alphas = torch.linspace(0, 1, 21)
                max_v = -1
                max_alpha = alphas[0]
                # W = W.type(torch.float64)
                W_metric_norm_in = torch.abs(W)/torch.mean(torch.abs(W), dim=0)
                W_metric_norm_in = (W_metric_norm_in - torch.mean(W_metric_norm_in))/torch.std(W_metric_norm_in)
                W_metric_norm_out = torch.abs(W)/torch.mean(torch.abs(W), dim=1).reshape(-1, 1)
                W_metric_norm_out = (W_metric_norm_out - torch.mean(W_metric_norm_out))/torch.std(W_metric_norm_out)
                # print(torch.sum(W_metric_norm_in * torch.sum(W_metric_norm_out) / torch.sum(W_metric_norm_in)), torch.sum(W_metric_norm_out))
                

                for alpha in alphas:
                    W_metric = alpha * W_metric_norm_in + (1-alpha)* W_metric_norm_out
                    print(torch.sum(W_metric))
                    thresh = torch.sort(W_metric.flatten().cuda())[0][int(W.numel()*args.sparsity_ratio)].cpu()
                    W_mask = (W_metric>thresh)
                    # W_metric = W_metric.type(torch.float64)
                    # print(torch.sum(W_metric[W_mask]), max_v)
                    # print(W_metric[W_mask])
                    # print(torch.sum(W_mask), thresh)
                    if torch.sum(W_metric[W_mask])>max_v:
                        max_alpha = alpha
                        max_v = torch.sum(W_metric[W_mask])
                        # print(max_v)
                
                W_metric = max_alpha * W_metric_norm_in + (1-max_alpha)* W_metric_norm_out
                # W_metric = W_metric.type(torch.float64)
                print("best alpha is: ", max_alpha.item(), " max sum is: ", max_v.item())
                
                
            # elif args.prune_method == "magnitude_caps":
            #     if "gate" in name or "up" in name:
            #         pre_weights = layer.post_attention_layernorm.weight.data.clone().type(torch.float32)
            #         post_weights = layer.mlp.down_proj.weight.data.clone().type(torch.float32)
            #         W_metric = torch.abs(pre_weights)/torch.sum(torch.abs(pre_weights)) * torch.abs(W)/torch.sum(torch.abs(W), dim=0) + torch.abs(W)/torch.sum(torch.abs(W), dim=1).reshape(-1, 1) * torch.mean(torch.abs(post_weights), dim=0).reshape(-1, 1)/torch.sum(torch.mean(torch.abs(post_weights), dim=0).reshape(-1, 1))
            #     else:
            #         W_metric = torch.abs(W)/torch.sum(torch.abs(W), dim=0) + torch.abs(W)/torch.sum(torch.abs(W), dim=1).reshape(-1, 1)
                
            elif args.prune_method == "magnitude_norm_neuron_multiply":
                W_metric = torch.sum(torch.abs(W), dim=0)/torch.sum(torch.abs(W)) * torch.sum(torch.abs(W), dim=1).reshape(-1, 1)/torch.sum(torch.abs(W)) * torch.abs(W)/torch.sum(torch.abs(W), dim=0) * torch.abs(W)/ torch.sum(torch.abs(W), dim=1).reshape(-1, 1)
                
                # if "gate" in name or "up" in name:
                #     pre_weights = layer.post_attention_layernorm.weight.data.clone().type(torch.float32)
                #     post_weights = layer.mlp.down_proj.weight.data.clone().type(torch.float32)
                #     W_metric = torch.abs(pre_weights)/torch.sum(torch.abs(pre_weights)) + torch.abs(W)/torch.sum(torch.abs(W), dim=0) + torch.abs(W)/torch.sum(torch.abs(W), dim=1).reshape(-1, 1) * torch.mean(torch.abs(post_weights), dim=0).reshape(-1, 1)/torch.sum(torch.mean(torch.abs(post_weights), dim=0).reshape(-1, 1))
                #     # W_metric = torch.abs(pre_weights)/torch.sum(torch.abs(pre_weights)) * (torch.abs(W)/torch.sum(torch.abs(W), dim=1).reshape(-1, 1) +  torch.abs(W)/torch.sum(torch.abs(W), dim=0)) + (torch.abs(W)/torch.sum(torch.abs(W), dim=1).reshape(-1, 1) +  torch.abs(W)/torch.sum(torch.abs(W), dim=0)) * torch.mean(torch.abs(post_weights), dim=0).reshape(-1, 1)/torch.sum(torch.mean(torch.abs(post_weights), dim=0).reshape(-1, 1))
                    
                # # elif "down" in name:
                # #     pre_weights = torch.mean(torch.abs(layer.mlp.gate_proj.weight.data.clone().type(torch.float32)), dim=1) * torch.mean(torch.abs(layer.mlp.up_proj.weight.data.clone().type(torch.float32)), dim=1)
                # #     W_metric = torch.sqrt(pre_weights) * torch.abs(W)/torch.sum(torch.abs(W), dim=0) + torch.abs(W)/torch.sum(torch.abs(W), dim=1).reshape(-1, 1)
                # else:
                #     W_metric = torch.abs(W)/torch.sum(torch.abs(W), dim=0) + torch.abs(W)/torch.sum(torch.abs(W), dim=1).reshape(-1, 1)
            
            elif args.prune_method == "magnitude_norm_neuron_sum":
                W_metric = torch.sum(torch.abs(W), dim=0)/torch.sum(torch.abs(W)) + torch.sum(torch.abs(W), dim=1).reshape(-1, 1)/torch.sum(torch.abs(W)) + torch.abs(W)/torch.sum(torch.abs(W), dim=0) + torch.abs(W)/ torch.sum(torch.abs(W), dim=1).reshape(-1, 1)
                    
            elif args.prune_method == "magnitude_norm_caps_multiply":
                if "gate" in name or "up" in name:
                    pre_weights = layer.post_attention_layernorm.weight.data.clone().type(torch.float32)
                    post_weights = layer.mlp.down_proj.weight.data.clone().type(torch.float32)
                    W_metric = torch.abs(pre_weights)/torch.sum(torch.abs(pre_weights)) * torch.abs(W)/torch.sum(torch.abs(W), dim=0) + torch.abs(W)/torch.sum(torch.abs(W), dim=1).reshape(-1, 1) * torch.sum(torch.abs(post_weights), dim=0).reshape(-1, 1)/torch.sum(torch.abs(post_weights))
                    # W_metric = torch.abs(pre_weights)/torch.sum(torch.abs(pre_weights)) * (torch.abs(W)/torch.sum(torch.abs(W), dim=1).reshape(-1, 1) +  torch.abs(W)/torch.sum(torch.abs(W), dim=0)) + (torch.abs(W)/torch.sum(torch.abs(W), dim=1).reshape(-1, 1) +  torch.abs(W)/torch.sum(torch.abs(W), dim=0)) * torch.mean(torch.abs(post_weights), dim=0).reshape(-1, 1)/torch.sum(torch.mean(torch.abs(post_weights), dim=0).reshape(-1, 1))
                    
                # elif "down" in name:
                #     pre_weights = torch.mean(torch.abs(layer.mlp.gate_proj.weight.data.clone().type(torch.float32)), dim=1) * torch.mean(torch.abs(layer.mlp.up_proj.weight.data.clone().type(torch.float32)), dim=1)
                #     W_metric = torch.sqrt(pre_weights7.56) * torch.abs(W)/torch.sum(torch.abs(W), dim=0) + torch.abs(W)/torch.sum(torch.abs(W), dim=1).reshape(-1, 1)
                else:
                    W_metric = torch.abs(W)/torch.sum(torch.abs(W), dim=0) + torch.abs(W)/torch.sum(torch.abs(W), dim=1).reshape(-1, 1)
            elif args.prune_method == "magnitude_norm_caps_sum":
                if "gate" in name or "up" in name:
                    pre_weights = layer.post_attention_layernorm.weight.data.clone().type(torch.float32)
                    post_weights = layer.mlp.down_proj.weight.data.clone().type(torch.float32)
                    W_metric = torch.abs(pre_weights)/torch.sum(torch.abs(pre_weights)) + torch.abs(W)/torch.sum(torch.abs(W), dim=0) + torch.abs(W)/torch.sum(torch.abs(W), dim=1).reshape(-1, 1) + torch.sum(torch.abs(post_weights), dim=0).reshape(-1, 1)/torch.sum(torch.abs(post_weights))
                    # W_metric = torch.abs(pre_weights)/torch.sum(torch.abs(pre_weights)) * (torch.abs(W)/torch.sum(torch.abs(W), dim=1).reshape(-1, 1) +  torch.abs(W)/torch.sum(torch.abs(W), dim=0)) + (torch.abs(W)/torch.sum(torch.abs(W), dim=1).reshape(-1, 1) +  torch.abs(W)/torch.sum(torch.abs(W), dim=0)) * torch.mean(torch.abs(post_weights), dim=0).reshape(-1, 1)/torch.sum(torch.mean(torch.abs(post_weights), dim=0).reshape(-1, 1))
                    
                # elif "down" in name:
                #     pre_weights = torch.mean(torch.abs(layer.mlp.gate_proj.weight.data.clone().type(torch.float32)), dim=1) * torch.mean(torch.abs(layer.mlp.up_proj.weight.data.clone().type(torch.float32)), dim=1)
                #     W_metric = torch.sqrt(pre_weights) + torch.abs(W)/torch.sum(torch.abs(W), dim=0) + torch.abs(W)/torch.sum(torch.abs(W), dim=1).reshape(-1, 1)
                else:
                    W_metric = torch.abs(W)/torch.sum(torch.abs(W), dim=0) + torch.abs(W)/torch.sum(torch.abs(W), dim=1).reshape(-1, 1)
            
            # elif args.prune_method == "magnitude_norm_caps_plus":
            #     if "gate" in name or "up" in name:
            #         pre_weights = layer.post_attention_layernorm.weight.data.clone().type(torch.float32)
            #         post_weights = layer.mlp.down_proj.weight.data.clone().type(torch.float32)
            #         W_metric = torch.abs(pre_weights)/torch.sum(torch.abs(pre_weights)) + torch.abs(W)/torch.sum(torch.abs(W), dim=0) + torch.mean(torch.abs(post_weights), dim=0).reshape(-1, 1)/torch.sum(torch.mean(torch.abs(post_weights), dim=0).reshape(-1, 1)) + torch.abs(W)/ torch.sum(torch.abs(W), dim=1).reshape(-1, 1)
            #     # elif "down" in name:
            #     #     pre_weights = torch.mean(torch.abs(layer.mlp.gate_proj.weight.data.clone().type(torch.float32)), dim=1) * torch.mean(torch.abs(layer.mlp.up_proj.weight.data.clone().type(torch.float32)), dim=1)
            #     #     W_metric = torch.sqrt(pre_weights)  + torch.abs(W)/torch.sum(torch.abs(W), dim=0) + torch.abs(W)/torch.sum(torch.abs(W), dim=1).reshape(-1, 1)
            #     else:
            #         W_metric = torch.abs(W)/torch.sum(torch.abs(W), dim=0) + torch.abs(W)/torch.sum(torch.abs(W), dim=1).reshape(-1, 1)
                        
            # elif args.prune_method == "magnitude_norm_caps_node_plus":
            #     if "gate" in name or "up" in name:
            #         pre_weights = layer.post_attention_layernorm.weight.data.clone().type(torch.float32)
            #         post_weights = layer.mlp.down_proj.weight.data.clone().type(torch.float32)
            #         W_metric = (torch.abs(pre_weights)+torch.abs(W))/(torch.abs(pre_weights)+torch.sum(torch.abs(W), dim=0)) + (torch.abs(W) + torch.mean(torch.abs(post_weights), dim=0).reshape(-1, 1))/(torch.sum(torch.abs(W), dim=1).reshape(-1, 1) + torch.mean(torch.abs(post_weights), dim=0).reshape(-1, 1))
            #     # elif "down" in name:
            #     #     pre_weights = torch.mean(torch.abs(layer.mlp.gate_proj.weight.data.clone().type(torch.float32)), dim=1) * torch.mean(torch.abs(layer.mlp.up_proj.weight.data.clone().type(torch.float32)), dim=1)
            #     #     W_metric = (pre_weights + torch.abs(W))/(pre_weights + torch.sum(torch.abs(W), dim=0)) + torch.abs(W)/torch.sum(torch.abs(W), dim=1).reshape(-1, 1)
            #     else:
            #         W_metric = torch.abs(W)/torch.sum(torch.abs(W), dim=0) + torch.abs(W)/torch.sum(torch.abs(W), dim=1).reshape(-1, 1)
                    
            # elif args.prune_method == "magnitude_norm_caps_node_multiply":
            #     if "gate" in name or "up" in name:
            #         pre_weights = layer.post_attention_layernorm.weight.data.clone().type(torch.float32)
            #         post_weights = layer.mlp.down_proj.weight.data.clone().type(torch.float32)
            #         W_metric = (torch.abs(pre_weights)*torch.abs(W))/(torch.abs(pre_weights)+torch.sum(torch.abs(W), dim=0))**2 + (torch.abs(W) * torch.mean(torch.abs(post_weights), dim=0).reshape(-1, 1))/(torch.sum(torch.abs(W), dim=1).reshape(-1, 1) + torch.mean(torch.abs(post_weights), dim=0).reshape(-1, 1))**2
            #     # elif "down" in name:
            #     #     pre_weights = torch.mean(torch.abs(layer.mlp.gate_proj.weight.data.clone().type(torch.float32)), dim=1) * torch.mean(torch.abs(layer.mlp.up_proj.weight.data.clone().type(torch.float32)), dim=1)
            #     #     W_metric = (pre_weights * torch.abs(W))/(pre_weights + torch.sum(torch.abs(W), dim=0))**2 + torch.abs(W)/torch.sum(torch.abs(W), dim=1).reshape(-1, 1)
            #     else:
            #         W_metric = torch.abs(W)/torch.sum(torch.abs(W), dim=0) + torch.abs(W)/torch.sum(torch.abs(W), dim=1).reshape(-1, 1)
                    
            # elif args.prune_method == "magnitude_norm_caps_multiply":
            #     if "gate" in name or "up" in name:
            #         pre_weights = layer.post_attention_layernorm.weight.data.clone().type(torch.float32)
            #         post_weights = layer.mlp.down_proj.weight.data.clone().type(torch.float32)
            #         W_metric = torch.abs(pre_weights)/torch.sum(torch.abs(pre_weights)) * torch.abs(W)/torch.sum(torch.abs(W), dim=0) + torch.abs(W)/torch.sum(torch.abs(W), dim=1).reshape(-1, 1) * torch.mean(torch.abs(post_weights), dim=0).reshape(-1, 1)/torch.sum(torch.mean(torch.abs(post_weights), dim=0).reshape(-1, 1))
            #     else:
            #         W_metric = torch.abs(W)/torch.sum(torch.abs(W), dim=0) + torch.abs(W)/torch.sum(torch.abs(W), dim=1).reshape(-1, 1)
            W_mask = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False
            if prune_n != 0:
                if args.perturb:
                    print("starting find the optimal perturbs!!!")
                    W_metric_pert = W_metric.clone()
                    W_strip_value = torch.zeros(W_metric.shape[1]//prune_m).cuda()
                    for ii in range(W_metric.shape[1]):
                        if ii % prune_m == 0:
                            tmp = W_metric[:,ii:(ii+prune_m)].float()
                            W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
                            W_metric_strip = W_metric[:, ii:(ii+prune_m)]
                            W_strip_value[ii//prune_m] = torch.sum(W_metric_strip[W_mask[:, ii:(ii+prune_m)]==0])
                    T = 10
                    cool_rate = 0.99
                    perturbs = 20000
                    idx = torch.linspace(0, W_metric.shape[1]-1, W_metric.shape[1])
                    while perturbs >= 0:
                        
                        i1 = np.random.randint(0, W_metric.shape[1])
                        i2 = np.random.randint(0, W_metric.shape[1])
                        while i2//prune_m == i1//prune_m:
                            i2 = np.random.randint(0, W_metric.shape[1])

                        mark = pert_strips(i1, i2, W_metric_pert[:, idx], W_strip_value, prune_n, prune_m, T)

                        if mark:
                            tmp = idx[i1]
                            idx[i1] = idx[i2]
                            idx[i2] = tmp
                            T *= cool_rate
                        perturbs -= 1
                    # W_metric[:, idx] = W_metric_pert
                    W_mask = torch.zeros_like(W_metric)==1
                    W_mask_pert = W_mask.clone()
                    for ii in range(W_metric.shape[1]):
                        if ii % prune_m == 0:
                            tmp = W_metric_pert[:,ii:(ii+prune_m)].float()
                            W_mask_pert.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
                    idx = idx.long()
                    W_mask[:, idx]=W_mask_pert
                elif args.resort:
                    sort_res = torch.sort(W_metric, dim=-1, stable=True)
                    # unstructured pruning
                    indices = sort_res[1][:,:int(W_metric.shape[1]*args.sparsity_ratio)]
                    W_mask.scatter_(1, indices, True)
                    # for ii in range(W_metric.shape[1]):
                    #     if ii % prune_m == 0:
                    #         tmp = W_metric[:,ii:(ii+prune_m)].float()
                    #         W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
                    W_metric[W_mask==0] = W_metric[W_mask==0]
                    sorted_idx = torch.sort(torch.sum(W_mask, dim=0))[1]
                    index = torch.zeros_like(sorted_idx)
                    for ii in range(1, prune_m+1):
                        index[ii-1::prune_m] = sorted_idx[int(W_metric.shape[1]* (ii-1)/prune_m) :int(W_metric.shape[1]* ii/prune_m)]

                    
                    W_metric = W_metric[:, index]
                    W_mask = torch.zeros_like(W_metric)==1
                    W_mask_pert = torch.zeros_like(W_metric)==1
                    for ii in range(W_metric.shape[1]):
                        if ii % prune_m == 0:
                            tmp = W_metric[:,ii:(ii+prune_m)].float()
                            W_mask_pert.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
                    W_mask[:, index]=W_mask_pert
                else: 
                    for ii in range(W_metric.shape[1]):
                        if ii % prune_m == 0:
                            tmp = W_metric[:,ii:(ii+prune_m)].float()
                            W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
            else:
                if per_outneuron:
                    sort_res = torch.sort(W_metric, dim=-1, stable=True)
                    # unstructured pruning
                    indices = sort_res[1][:,:int(W_metric.shape[1]*args.sparsity_ratio)]
                    W_mask.scatter_(1, indices, True)
                else:
                    thresh = torch.sort(W_metric.flatten().cuda())[0][int(W.shape[0]* W.shape[1]*args.sparsity_ratio)].cpu()
                    W_mask = (W_metric<=thresh)

            subset[name].weight.data[W_mask] = 0

def prune_wanda(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 
    print("loading calibdation data")
    dataloader, _ = get_loaders(args.calib_dataset,nsamples=args.nsamples,seed=args.seed,seqlen=args.seqlen,tokenizer=tokenizer)
    print("dataset loading complete")
    with torch.no_grad():
        if "llama" in args.model:
            inps, outs, attention_mask, position_ids = prepare_calibration_input(args, model, dataloader, device)
        elif "opt" in args.model:
            inps, outs, attention_mask= prepare_calibration_input(args, model, dataloader, device)
    if "llama" in args.model:
        layers = model.model.layers
    elif "opt" in args.model:
        layers = model.model.decoder.layers

    if "rec" in args.prune_method:
        reconstruct = True
    else:
        reconstruct = False

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        if "llama" in args.model:
            if f"model.layers.{i}" in model.hf_device_map:   ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
                dev = model.hf_device_map[f"model.layers.{i}"]
                # print(inps)
                inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)
        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(args, subset[name], layer_name=name, reconstruct=reconstruct)
            if args.gptq:
                wrapped_layers[name].quantizer = Quantizer()
                wrapped_layers[name].quantizer.configure(
                        args.wbits, perchannel=True, sym=args.sym, mse=False
                    )

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        
        # args.nsamples = 1
        for j in range(args.nsamples):
            with torch.no_grad():
                if "llama" in args.model:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
                elif "opt" in args.model:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]

        for h in handles:
            h.remove()

        for name in subset:
            if args.gptq:
                print('Quantizing ...')
                wrapped_layers[name].fasterquant(
                    percdamp=args.percdamp, groupsize=args.groupsize, actorder=args.act_order, static_groups=args.static_groups
                )
                # quantizers['model.layers.%d.%s' % (i, name)] = wrapped_layers[name].quantizer
                # wrapped_layers[name].free()
            per_outneuron = False
            print(f"pruning layer {i} name {name}")
            W = subset[name].weight.data.clone().type(torch.float32)
            if args.prune_method == "wanda":
                per_outneuron = True
                W_metric = torch.abs(W) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))
                
            if args.prune_method == "wanda_part":
                if "mlp" in name:
                    W_metric = torch.abs(W) / torch.sum(torch.abs(W), dim=0) + torch.abs(W) / torch.sum(torch.abs(W), dim=1).reshape(-1, 1)
                else:
                    W_metric = (torch.abs(W) / torch.sum(torch.abs(W), dim=0) + torch.abs(W) / torch.sum(torch.abs(W), dim=1).reshape(-1, 1)) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))
                
            elif args.prune_method == "wanda_l1":
                per_outneuron = True
                W_metric = torch.abs(W) * wrapped_layers[name].scaler_row_l1.reshape((1,-1))

            elif args.prune_method == "wanda_no_per_out":
                W_metric = torch.abs(W) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))
            elif args.prune_method == "wanda_norm_in_per_out":
                per_outneuron = True
                W_metric = torch.abs(W) / torch.sum(torch.abs(W), dim=0) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))


            elif args.prune_method == "wanda_sqrt":
                per_outneuron = True
                W_metric = torch.abs(W) * torch.sqrt(torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1))))
            elif args.prune_method == "wanda_norm_in":
                W_metric = torch.abs(W) / torch.sum(torch.abs(W), dim=0) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))
            elif args.prune_method == "wanda_norm_out":
                W_metric = torch.abs(W) / torch.sum(torch.abs(W), dim=1).reshape(-1, 1) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))
            elif args.prune_method == "wanda_norm_in_out_ones":
                W_metric = (torch.abs(W) / torch.sum(torch.abs(W), dim=0) + torch.abs(W) / torch.sum(torch.abs(W), dim=1).reshape(-1, 1)) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))

            elif args.prune_method == "wanda_norm_in_out_rate":
                W_metric = (torch.abs(W) / torch.sum(torch.abs(W), dim=0) + torch.abs(W) / torch.sum(torch.abs(W), dim=1).reshape(-1, 1)) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1))) + torch.abs(W) * torch.sqrt(torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1))))/torch.sqrt(wrapped_layers[name].scaler_col.reshape((-1,1)))
 
            elif args.prune_method == "wanda_norm_in_out_l1":
                W_metric = (torch.abs(W) / torch.sum(torch.abs(W), dim=0) + torch.abs(W) / torch.sum(torch.abs(W), dim=1).reshape(-1, 1)) * wrapped_layers[name].scaler_row_l1.reshape((1,-1))

            elif args.prune_method == "wanda_norm_in_out_sqrt":
                W_metric = (torch.abs(W) / torch.sum(torch.abs(W), dim=0) + torch.abs(W) / torch.sum(torch.abs(W), dim=1).reshape(-1, 1)) * torch.sqrt(torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1))))
            elif args.prune_method == "wanda_norm_in_out_sqrt_layer":
                if i < 4 or i >= 30:
                    break
                W_metric = (torch.abs(W) / torch.sum(torch.abs(W), dim=0) + torch.abs(W) / torch.sum(torch.abs(W), dim=1).reshape(-1, 1)) * torch.sqrt(torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1))))

            
            elif args.prune_method == "wanda_layer":
                if i < 4 or i >= 30:
                    break
                per_outneuron = True
                W_metric = torch.abs(W) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))
                
            
            elif args.prune_method == "wanda_activation_layer":
                if i < 4 or i >= 30:
                    break
                W_metric = wrapped_layers[name].scaler_row.reshape(-1)
                W_metric = W_metric.unsqueeze(0).expand(W.shape[0], *W_metric.shape)
                
            elif args.prune_method == "wanda_norm_in_out_sqrt_all":
                W_in = torch.abs(W) * torch.sqrt(torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1))))
                W_metric = (torch.abs(W_in) / torch.sum(torch.abs(W_in), dim=0) + torch.abs(W_in) / torch.sum(torch.abs(W_in), dim=1).reshape(-1, 1)) 
                
            # elif args.prune_method == "wanda_norm_in_var_out_sqrt":
            #     W_metric = torch.abs(W) / torch.sum(torch.abs(W), dim=0) * wrapped_layers[name].scaler_var.reshape((1,-1)) + torch.abs(W) / torch.sum(torch.abs(W), dim=1).reshape(-1, 1) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))
            #     # pass

                
            elif args.prune_method == "wanda_var":
                per_outneuron = True
                W_metric = W**2 * wrapped_layers[name].scaler_var.reshape((1,-1))
            
            elif args.prune_method == "wanda_norm_ab":
                W_metric = (torch.abs(W) / torch.sum(torch.abs(W), dim=0) + torch.abs(W) / torch.sum(torch.abs(W), dim=1).reshape(-1, 1))**(args.a) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))**(args.b)

                
                
            elif args.prune_method == "wanda_var_x":
                per_outneuron = True
                # print(torch.max(wrapped_layers[name].scaler_var[:10]), torch.min(wrapped_layers[name].scaler_var[:10]))
                W_metric = (torch.abs(W) / torch.sum(torch.abs(W), dim=0) + torch.abs(W) / torch.sum(torch.abs(W), dim=1).reshape(-1, 1)) * (torch.sqrt(wrapped_layers[name].scaler_var) + torch.sqrt(torch.sqrt(wrapped_layers[name].scaler_row)))
                # W_metric = (torch.abs(W) / torch.sum(torch.abs(W), dim=0) + torch.abs(W) / torch.sum(torch.abs(W), dim=1).reshape(-1, 1)) * wrapped_layers[name].scaler_var.reshape((1,-1))
            elif args.prune_method == "wanda_var_norm":
                # per_outneuron = True
                W_metric = (W**2)/torch.sum(W**2, dim=0) * wrapped_layers[name].scaler_var + torch.abs(W)/torch.sum(torch.abs(W), dim=1).reshape(-1, 1) * torch.sqrt(wrapped_layers[name].scaler_row)
                # W_metric = (torch.abs(W) / torch.sum(torch.abs(W), dim=0) + torch.abs(W) / torch.sum(torch.abs(W), dim=1).reshape(-1, 1)) * wrapped_layers[name].scaler_var.reshape((1,-1))

            elif args.prune_method == "wanda_norm_in_out_sqrt_float64":
                W_f64 = W.clone().type(torch.float64)
                W_metric = (torch.abs(W) / torch.sum(torch.abs(W), dim=0) + torch.abs(W) / torch.sum(torch.abs(W), dim=1).reshape(-1, 1)) * torch.sqrt(torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1))))

            elif args.prune_method == "wanda_norm_in_out_log":
                W_metric = (torch.abs(W) / torch.sum(torch.abs(W), dim=0) + torch.abs(W) / torch.sum(torch.abs(W), dim=1).reshape(-1, 1)) * torch.log(torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1))))
          
            elif args.prune_method == "wanda_norm_in_out_sqrt_sqrt":
                W_metric = (torch.abs(W) / torch.sum(torch.abs(W), dim=0) + torch.abs(W) / torch.sum(torch.abs(W), dim=1).reshape(-1, 1)) * torch.sqrt(torch.sqrt(torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))))
                
            elif args.prune_method == "wanda_norm_in_out_sqrt_sqrt_scale":
                W_metric = (torch.abs(W) / torch.sum(torch.abs(W), dim=0) + torch.abs(W) / torch.sum(torch.abs(W), dim=1).reshape(-1, 1)) * torch.sqrt(torch.sqrt(torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))))

            elif args.prune_method == "wanda_activation":
                W_metric = wrapped_layers[name].scaler_row.reshape(-1)
                 
                W_metric = W_metric.unsqueeze(0).expand(W.shape[0], *W_metric.shape)

            
            elif args.prune_method == "wanda_norm_in_out_sqrt_alpha":
                W_metric = (args.alpha * torch.abs(W) / torch.sum(torch.abs(W), dim=0) + (1 - args.alpha) * torch.abs(W) / torch.sum(torch.abs(W), dim=1).reshape(-1, 1)) * torch.sqrt(torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1))))

            elif args.prune_method == "wanda_norm_all":
                W_metric = torch.abs(W) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))
                W_metric = W_metric / torch.sum(W_metric, dim=0) + W_metric/torch.sum(W_metric, dim=1).reshape(-1, 1)
            elif args.prune_method == "wanda_norm_in_out_sqrt_rec":
                W_metric = (torch.abs(W) / torch.sum(torch.abs(W), dim=0) + torch.abs(W) / torch.sum(torch.abs(W), dim=1).reshape(-1, 1)) * torch.sqrt(torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1))))
            elif args.prune_method == "wanda_norm_all_sqrt":
                W_metric = torch.abs(W) * torch.sqrt(torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1))))
                W_metric = W_metric / torch.sum(W_metric, dim=0) + W_metric/torch.sum(W_metric, dim=1).reshape(-1, 1)
            elif args.prune_method == "wanda_norm_all_sqrt_alpha":
                W_metric = torch.abs(W) * torch.sqrt(torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1))))
                W_metric = args.alpha * W_metric / torch.sum(W_metric, dim=0) + (1-args.alpha)*W_metric/torch.sum(W_metric, dim=1).reshape(-1, 1)
            elif args.prune_method == "wanda_norm_all_alpha":
                W_metric = torch.abs(W) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))
                W_metric = args.alpha * W_metric / torch.sum(W_metric, dim=0) + (1-args.alpha)*W_metric/torch.sum(W_metric, dim=1).reshape(-1, 1)
            elif args.prune_method == "wanda_norm_caps":    
                if "gate" in name:
                    post_weights = wrapped_layers["mlp.up_proj"].out.type(torch.float32)
                    W_metric = torch.sqrt(wrapped_layers[name].scaler_row) * torch.abs(W)* post_weights.reshape(-1, 1) * torch.mean(torch.abs(layer.mlp.down_proj.weight.data), dim=0).reshape(-1, 1)
                    # print(post_weights[:20])
                    # W_metric = torch.abs(pre_weights)/torch.sum(torch.abs(pre_weights)) * torch.abs(W)/torch.sum(torch.abs(W), dim=0) + torch.abs(W)/torch.sum(torch.abs(W), dim=1).reshape(-1, 1) * torch.mean(torch.abs(post_weights), dim=0).reshape(-1, 1)/torch.sum(torch.mean(torch.abs(post_weights), dim=0).reshape(-1, 1))
                    # W_metric = torch.abs(pre_weights)/torch.sum(torch.abs(pre_weights)) * (torch.abs(W)/torch.sum(torch.abs(W), dim=1).reshape(-1, 1) +  torch.abs(W)/torch.sum(torch.abs(W), dim=0)) + (torch.abs(W)/torch.sum(torch.abs(W), dim=1).reshape(-1, 1) +  torch.abs(W)/torch.sum(torch.abs(W), dim=0)) * torch.mean(torch.abs(post_weights), dim=0).reshape(-1, 1)/torch.sum(torch.mean(torch.abs(post_weights), dim=0).reshape(-1, 1))
                elif "up" in name:
                    post_weights = wrapped_layers["mlp.gate_proj"].out.type(torch.float32)
                    W_metric = torch.sqrt(wrapped_layers[name].scaler_row) * torch.abs(W) * post_weights.reshape(-1, 1) * torch.mean(torch.abs(layer.mlp.down_proj.weight.data), dim=0).reshape(-1, 1)
                    # print(post_weights[:20])
                    # print(post_weights[:20])
                # elif "down" in name:
                #     pre_weights = torch.mean(torch.abs(layer.mlp.gate_proj.weight.data.clone().type(torch.float32)), dim=1) * torch.mean(torch.abs(layer.mlp.up_proj.weight.data.clone().type(torch.float32)), dim=1)
                #     W_metric = torch.sqrt(pre_weights) * torch.abs(W)/torch.sum(torch.abs(W), dim=0) + torch.abs(W)/torch.sum(torch.abs(W), dim=1).reshape(-1, 1)
                else:
                    W_metric = torch.sqrt(wrapped_layers[name].scaler_row) * (torch.abs(W)/torch.sum(torch.abs(W), dim=0) + torch.abs(W)/torch.sum(torch.abs(W), dim=1).reshape(-1, 1))
            
            elif args.prune_method == "wanda_caps":    
                
                if "gate" in name:
                    post_weights = torch.sqrt(wrapped_layers["mlp.up_proj"].out).clone().type(torch.float32)
                    W_metric = torch.sqrt(wrapped_layers[name].scaler_row) * torch.abs(W) * post_weights.reshape(-1, 1)
                    print(post_weights[:20])
                    # W_metric = torch.abs(pre_weights)/torch.sum(torch.abs(pre_weights)) * torch.abs(W)/torch.sum(torch.abs(W), dim=0) + torch.abs(W)/torch.sum(torch.abs(W), dim=1).reshape(-1, 1) * torch.mean(torch.abs(post_weights), dim=0).reshape(-1, 1)/torch.sum(torch.mean(torch.abs(post_weights), dim=0).reshape(-1, 1))
                    # W_metric = torch.abs(pre_weights)/torch.sum(torch.abs(pre_weights)) * (torch.abs(W)/torch.sum(torch.abs(W), dim=1).reshape(-1, 1) +  torch.abs(W)/torch.sum(torch.abs(W), dim=0)) + (torch.abs(W)/torch.sum(torch.abs(W), dim=1).reshape(-1, 1) +  torch.abs(W)/torch.sum(torch.abs(W), dim=0)) * torch.mean(torch.abs(post_weights), dim=0).reshape(-1, 1)/torch.sum(torch.mean(torch.abs(post_weights), dim=0).reshape(-1, 1))
                elif "up" in name:
                    post_weights = torch.sqrt(wrapped_layers["mlp.gate_proj"].out).clone().type(torch.float32)
                    W_metric = torch.sqrt(wrapped_layers[name].scaler_row) * torch.abs(W) * post_weights.reshape(-1, 1)
                    print(post_weights[:20])
                # elif "down" in name:
                #     pre_weights = torch.mean(torch.abs(layer.mlp.gate_proj.weight.data.clone().type(torch.float32)), dim=1) * torch.mean(torch.abs(layer.mlp.up_proj.weight.data.clone().type(torch.float32)), dim=1)
                #     W_metric = torch.sqrt(pre_weights) * torch.abs(W)/torch.sum(torch.abs(W), dim=0) + torch.abs(W)/torch.sum(torch.abs(W), dim=1).reshape(-1, 1)
                else:
                    per_outneuron = True
                    W_metric = torch.sqrt(wrapped_layers[name].scaler_row) * torch.abs(W)

            W_mask = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False
            if prune_n != 0:
                # structured n:m sparsity
                if args.perturb:
                    print("starting find the optimal perturbs!!!")
                    W_metric_pert = W_metric.clone()
                    W_strip_value = torch.zeros(W_metric.shape[1]//prune_m).to(device)
                    for ii in range(W_metric.shape[1]):
                        if ii % prune_m == 0:
                            tmp = W_metric[:,ii:(ii+prune_m)].float()
                            W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
                            W_metric_strip = W_metric[:, ii:(ii+prune_m)]
                            W_strip_value[ii//prune_m] = torch.sum(W_metric_strip[W_mask[:, ii:(ii+prune_m)]==0])
                    T = 100
                    cool_rate = 0.95
                    perturbs = 20000
                    idx = torch.LongTensor(torch.linspace(0, W_metric.shape[1]-1, W_metric.shape[1]))
                    while perturbs >= 0:
                        # k = np.random.randint(1, prune_m + 1)
                        i1 = np.random.randint(0, W_metric.shape[1]//prune_m)
                        i2 = np.random.randint(0, W_metric.shape[1]//prune_m)
                        while i2//prune_m == i1//prune_m:
                            i2 = np.random.randint(0, W_metric.shape[1])


                        mark = pert_strips(i1, i2, W_metric_pert[:, idx], W_strip_value, prune_n, prune_m, T)

                        if mark:
                            tmp = idx[i1].clone()
                            idx[i1] = idx[i2].clone()
                            idx[i2] = tmp
                        perturbs -= 1

                        if perturbs % 50 == 0:
                            T *= cool_rate
                        if perturbs % 10000 == 0:
                            print("Perturbs: {0}, Current T: {1}, Values: {2}".format(perturbs, T, torch.sum(W_strip_value).type(torch.float32)))
                    # W_metric[:, idx] = W_metric_pert
                    print("Final Values: ", torch.sum(W_strip_value).type(torch.float32))
                    W_mask = torch.zeros_like(W_metric)==1
                    W_mask_pert = W_mask.clone()
                    for ii in range(W_metric.shape[1]):
                        if ii % prune_m == 0:
                            tmp = W_metric_pert[:,ii:(ii+prune_m)].float()
                            W_mask_pert.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
                    idx = idx.long()
                    W_mask[:, idx]=W_mask_pert

                elif args.resort:
                    for ii in range(W_metric.shape[1]):
                        if ii % prune_m == 0:
                            tmp = W_metric[:,ii:(ii+prune_m)].float()
                            W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
                    print("The total value before resort: ", torch.sum(W_metric[W_mask==0].type(torch.float32)).item())

                    sort_res = torch.sort(W_metric, dim=-1, stable=True)
                    # unstructured pruning
                    W_mask = torch.zeros_like(W_metric)==1
                    indices = sort_res[1][:,:int(W_metric.shape[1]*args.sparsity_ratio)]
                    W_mask.scatter_(1, indices, True)
                    
                    # sorted_idx = torch.sort(torch.sum(W_mask==0, dim=0))[1]
                    sorted_idx = torch.sort(torch.sum(W_mask==0, dim=0))[1]
                    index = torch.zeros_like(sorted_idx)
                    for ii in range(1, prune_m+1):
                        index[ii-1::prune_m] = sorted_idx[int(W_metric.shape[1]* (ii-1)/prune_m) :int(W_metric.shape[1]* ii/prune_m)]

                    
                    W_metric_resort = W_metric[:, index].clone()

                    if args.resort_pert:
                        W_strip_value = torch.zeros(W_metric.shape[1]//prune_m).to(device)
                        W_mask = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False
                        for ii in range(W_metric.shape[1]):
                            if ii % prune_m == 0:
                                tmp = W_metric_resort[:,ii:(ii+prune_m)].float()
                                W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
                                W_metric_strip = W_metric_resort[:, ii:(ii+prune_m)]
                                W_strip_value[ii//prune_m] = torch.sum(W_metric_strip[W_mask[:, ii:(ii+prune_m)]==0])
                        print("The total value after resort: ", torch.sum(W_strip_value).type(torch.float32).item())
                        T = 100
                        cool_rate = 0.95
                        perturbs = 0
                        idx = torch.linspace(0, W_metric.shape[1]-1, W_metric.shape[1]).long()
                        while perturbs <= 50000:
                            k1 = np.random.randint(1, prune_m + 1)
                            k2 = np.random.randint(0, prune_m)

                            i1 = np.random.randint(0, W_metric.shape[1]//prune_m)
                            i2 = np.random.randint(0, W_metric.shape[1]//prune_m)
                            while i2 == i1:
                                i2 = np.random.randint(0, W_metric.shape[1]//prune_m)
                            i1 = i1 * k1 + k2
                            i2 = i2 * k1 + k2

                            mark = pert_strips(i1, i2, W_metric_resort[:, idx], W_strip_value, prune_n, prune_m, T)

                            if mark:
                                tmp = idx[i1].clone()
                                idx[i1] = idx[i2].clone()
                                idx[i2] = tmp
                            perturbs += 1

                            if perturbs % 50 == 0:
                                T *= cool_rate
                            if perturbs % 10000 == 0:
                                print("Perturbs: {0}, Current T: {1}, Values: {2}".format(perturbs, T, torch.sum(W_strip_value).type(torch.float32).item()))
                        
                        # print("Final Values: ", torch.sum(W_strip_value).type(torch.float32))
                        W_metric_resort_pert = W_metric_resort[:, idx]
                        W_mask = torch.zeros_like(W_metric)==1
                        W_mask_resort = W_mask.clone()
                        W_mask_resort_pert = W_mask.clone()
                        for ii in range(W_metric.shape[1]):
                            if ii % prune_m == 0:
                                tmp = W_metric_resort_pert[:,ii:(ii+prune_m)].float()
                                W_mask_resort_pert.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
                        # idx = idx.long()
                        # print(idx, index, torch.sum(W_mask_pert))
                        W_mask_resort[:, idx]=W_mask_resort_pert
                        W_mask[:, index] = W_mask_resort
                        print("Final Values: ", torch.sum(W_metric[W_mask==0].type(torch.float32)).item())
                        # print(torch.sum(W_mask))
                    elif args.resort_lsa:
                        if "o_proj" in name:
                            W_strip_value = torch.zeros(W_metric.shape[1]//prune_m).to(device)
                            W_mask = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False
                            for ii in range(W_metric.shape[1]):
                                if ii % prune_m == 0:
                                    tmp = W_metric_resort[:,ii:(ii+prune_m)].float()
                                    W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
                                    W_metric_strip = W_metric_resort[:, ii:(ii+prune_m)]
                                    W_strip_value[ii//prune_m] = torch.sum(W_metric_strip[W_mask[:, ii:(ii+prune_m)]==0])
                            print("The total value after resort: ", torch.sum(W_strip_value).type(torch.float32).item())

                            # permute the first col
                            W_metric_shape = W_metric.shape[1] // prune_m
                            score_matrix = torch.zeros(W_metric_shape, W_metric_shape)

                            rows = torch.arange(W_metric_shape)
                            for row in rows:
                                strip_idx = torch.zeros(W_metric_shape, prune_m)
                                strip_idx[:, 0] = (rows * prune_m).reshape(-1)
                                strip_idx[:, 1:] = torch.arange(1, prune_m) + row * prune_m
                                strip_idx = strip_idx.long()
                                tmp = W_metric_resort[:, strip_idx].transpose(1, 0)
                                W_mask = torch.zeros_like(tmp)
                                tmp_index = torch.sort(tmp, dim=-1)[1]
                                W_mask.scatter_(dim=-1, index=tmp_index[:, :, :prune_n], value=1)
                                score_matrix[:, row] = torch.sum(torch.sum((tmp*(W_mask==0)).type(torch.float32), dim=-1), dim=-1).reshape(-1)

                            _, _, col_indices1 = maximize_total_value(score_matrix.cpu())
                            col_indices1 = torch.LongTensor(col_indices1)
                            idx1 = torch.linspace(0, W_metric.shape[1]//prune_m - 1, W_metric.shape[1]//prune_m).long()

                            W_metric_resort_lsa1 = W_metric_resort.clone()
                            W_metric_resort_lsa1[:, idx1 * prune_m] = W_metric_resort[:, col_indices1*prune_m].clone()


                            # permute the last col
                            score_matrix = torch.zeros(W_metric_shape, W_metric_shape)

                            rows = torch.arange(W_metric_shape)
                            for row in rows:
                                strip_idx = torch.zeros(W_metric_shape, prune_m)
                                strip_idx[:, -1] = (rows * prune_m + 3).reshape(-1)
                                strip_idx[:, :prune_m-1] = torch.arange(prune_m-1) + row * prune_m
                                strip_idx = strip_idx.long()
                                tmp = W_metric_resort_lsa1[:, strip_idx].transpose(1, 0)
                                W_mask = torch.zeros_like(tmp)
                                tmp_index = torch.sort(tmp, dim=-1)[1]
                                W_mask.scatter_(dim=-1, index=tmp_index[:, :, :prune_n], value=1)
                                score_matrix[:, row] = torch.sum(torch.sum((tmp*(W_mask==0)).type(torch.float32), dim=-1), dim=-1).reshape(-1)

                            _, _, col_indices2 = maximize_total_value(score_matrix.cpu())
                            col_indices2 = torch.LongTensor(col_indices2)
                            idx2 = torch.linspace(0, W_metric.shape[1]//prune_m - 1, W_metric.shape[1]//prune_m).long()

                            W_metric_resort_lsa2 = W_metric_resort_lsa1.clone()
                            W_metric_resort_lsa2[:, idx2 * prune_m + prune_m - 1] = W_metric_resort_lsa1[:, col_indices2*prune_m + prune_m - 1].clone()


                            W_mask = torch.zeros_like(W_metric)==1
                            # W_mask_resort = torch.zeros_like(W_metric)==1
                            # W_mask_resort_lsa1 = torch.zeros_like(W_metric)==1
                            W_mask_resort_lsa2 = torch.zeros_like(W_metric)==1
                            for ii in range(W_metric.shape[1]):
                                if ii % prune_m == 0:
                                    tmp = W_metric_resort_lsa2[:,ii:(ii+prune_m)].float()
                                    W_mask_resort_lsa2.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)

                            W_mask_resort_lsa1 = W_mask_resort_lsa2.clone()
                            # print(col_indices2)
                            # print(col_indices2 * prune_m + prune_m - 1)
                            # print(W_mask_resort_lsa1[: (col_indices2 * prune_m + prune_m - 1).long()])
                            # print(W_metric_resort_lsa2[:, (idx2 * prune_m + prune_m - 1).long()])
                            W_mask_resort_lsa1[:, col_indices2 * prune_m + prune_m - 1] = W_mask_resort_lsa2[:, idx2 * prune_m + prune_m - 1]

                            W_mask_resort = W_mask_resort_lsa1.clone()

                            W_mask_resort[:, col_indices1 * prune_m] = W_mask_resort_lsa1[:, idx1 * prune_m]
                            W_mask[:, index] = W_mask_resort
                            print("The total value after resort: ", torch.sum(W_metric[W_mask==0].type(torch.float32)).item())

                            W_mask_best =  torch.zeros_like(W_metric)==1
                            sort_res = torch.sort(W_metric, dim=-1, stable=True)
                            # unstructured pruning
                            indices = sort_res[1][:,:int(W_metric.shape[1]*args.sparsity_ratio)]
                            W_mask_best.scatter_(1, indices, True)
                            print("The Best value: ", torch.sum(W_metric[W_mask_best==0].type(torch.float32)).item())


                        else:
                            W_mask = torch.zeros_like(W_metric)==1
                            W_mask_resort = torch.zeros_like(W_metric)==1
                            for ii in range(W_metric.shape[1]):
                                if ii % prune_m == 0:
                                    tmp = W_metric_resort[:,ii:(ii+prune_m)].float()
                                    W_mask_resort.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
                            
                            W_mask[:, index]=W_mask_resort
                            print("The total value after resort: ", torch.sum(W_metric[W_mask==0].type(torch.float32)).item())
                    else:
                        W_mask = torch.zeros_like(W_metric)==1
                        W_mask_resort = torch.zeros_like(W_metric)==1
                        for ii in range(W_metric.shape[1]):
                            if ii % prune_m == 0:
                                tmp = W_metric_resort[:,ii:(ii+prune_m)].float()
                                W_mask_resort.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
                        
                        W_mask[:, index]=W_mask_resort
                        print("The total value after resort: ", torch.sum(W_metric[W_mask==0].type(torch.float32)).item())
                    
                    

                else:
                    for ii in range(W_metric.shape[1]):
                        if ii % prune_m == 0:
                            tmp = W_metric[:,ii:(ii+prune_m)].float()
                            W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
            else:
                if per_outneuron:
                    sort_res = torch.sort(W_metric, dim=-1, stable=True)
                    # unstructured pruning
                    indices = sort_res[1][:,:int(W_metric.shape[1]*args.sparsity_ratio)]
                    W_mask.scatter_(1, indices, True)
                else:
                    thresh = torch.sort(W_metric.flatten().cuda())[0][int(W.shape[0]*W.shape[1]*args.sparsity_ratio)].cpu()
                    W_mask = (W_metric<=thresh)
                    # print(thresh)
            if reconstruct:
                wrapped_layers[name].fasterprune(args.sparsity_ratio, mask= W_mask)
                
            else:
                subset[name].weight.data[W_mask] = 0  ## set weights to zero 
            wrapped_layers[name].free()

        for j in range(args.nsamples):
            with torch.no_grad():
                if "llama" in args.model:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
                elif "opt" in args.model:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        inps, outs = outs, inps

    model.config.use_cache = use_cache 
    torch.cuda.empty_cache()


@torch.no_grad()
def prune_sparsegpt(args, model, tokenizer, dev, prune_n=0, prune_m=0):
    ## SparseGPT code available at: https://github.com/IST-DASLab/sparsegpt/tree/f5c25005a61f96a0933ca2f95705a963585aafaa
    print('Starting ...')
    dataloader, _ = get_loaders(args.calib_dataset, nsamples=args.nsamples,seed=args.seed,seqlen=args.seqlen,tokenizer=tokenizer)

    use_cache = model.config.use_cache
    model.config.use_cache = False
    if "llama" in args.model:
        layers = model.model.layers
    elif "opt" in args.model:
        layers = model.model.decoder.layers

    if "model.embed_tokens" in model.hf_device_map:
        dev = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            if "llama" in args.model:
                cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    print('Ready.')

    for i in range(len(layers)):
        layer = layers[i]
        if "llama" in args.model:
            if f"model.layers.{i}" in model.hf_device_map:
                dev = model.hf_device_map[f"model.layers.{i}"]
                print(f"layer {i} device {dev}")
                inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

        subset = find_layers(layer)

        gpts = {}
        for name in subset:
            gpts[name] = SparseGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            if "llama" in args.model:
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
            else:
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]

        for h in handles:
            h.remove()

        for name in gpts:
            print(i, name)
            print('Pruning ...')
            if "norm" in args.model:
                gpts[name].fasterprune(args.sparsity_ratio, prune_n=prune_n, prune_m=prune_m, percdamp=0.01, blocksize=128, norm=True)
            else:
                gpts[name].fasterprune(args.sparsity_ratio, prune_n=prune_n, prune_m=prune_m, percdamp=0.01, blocksize=128)
            gpts[name].free()

        for j in range(args.nsamples):
            if "llama" in args.model:
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
            else:
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        layers[i] = layer 
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()


