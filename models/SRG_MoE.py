from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np
import copy

from layers.PatchTST_backbone import PatchTST_backbone
from layers.PatchTST_layers import series_decomp

from models.DLinear import Model as DLinearModel
from models.TCN import Model as TCNModel
from models.NLinear import Model as NLinearModel

#more efficient version
# class SC_EMA(nn.Module):
#     def __init__(self, alpha):
#         super(SC_EMA, self).__init__()
#         self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float32))
#         self.alpha.data.clamp_(0, 1)

#     def parallel_ema_conv(self, x, alpha, direction='forward'):
#         # x: [batch_size, node, time]
#         batch_size, node, time = x.shape
        
#         if direction == 'forward':
#          
#             kernel = torch.tensor([alpha * ((1 - alpha) ** i) for i in range(time)], device=x.device)
#             kernel = kernel.view(1, 1, -1)  # [1, 1, time]
            
#             
#             x_reshaped = x.view(batch_size * node, 1, time)  # [batch*node, 1, time]
#             ema = F.conv1d(x_reshaped, kernel, padding=time-1)
#             ema = ema[:, :, :time]  # 取前time个输出
#             ema = ema.view(batch_size, node, time)
            
#         else:  # backward
#        
#             x_flipped = torch.flip(x, dims=[2])
#             ema_flipped = self.parallel_ema_conv(x_flipped, alpha, 'forward')
#             ema = torch.flip(ema_flipped, dims=[2])
        
#         return ema

#     def parallel_sc_ema(self, x, alpha):
#         # x: [batch_size, node, time]
#         batch_size, node, time = x.shape
        
#      
#         forward_ema = self.parallel_ema_conv(x, alpha, 'forward')
#         backward_ema = self.parallel_ema_conv(x, alpha, 'backward')
        
#        
#         sc_ema = torch.zeros_like(x)
        
#        
#         sc_ema[:, :, 0] = x[:, :, 0]
        
#         if time > 1:
#             #α * x + (1-α)/2 * forward_ema + (1-α)/2 * backward_ema
#             sc_ema[:, :, 1:] = alpha * x[:, :, 1:] + \
#                               (1 - alpha)/2 * forward_ema[:, :, :-1] + \
#                               (1 - alpha)/2 * backward_ema[:, :, 1:]
        
#         return sc_ema

#     def forward(self, x):
#         x = x.permute(0, 2, 1)  # [batch_size, node, time]
#         sc_ema_output = self.parallel_sc_ema(x, self.alpha)
#         return sc_ema_output.permute(0, 2, 1)


class SC_EMA(nn.Module):
    def __init__(self, alpha):
        super(SC_EMA, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float32))
        self.alpha.data.clamp_(0, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        
        batch_size, node, time = x.shape
        
        # Compute forward EMA
        forward_ema_list = []
        forward_ema_t = x[:, :, 0]
        forward_ema_list.append(forward_ema_t)
        
        for t in range(1, time):
            forward_ema_t = self.alpha * x[:, :, t] + (1 - self.alpha) * forward_ema_t
            forward_ema_list.append(forward_ema_t)
        
        forward_ema = torch.stack(forward_ema_list, dim=2)  # [batch_size, node, time]
        
        # Compute backward EMA
        reversed_x = torch.flip(x, dims=[2])
        backward_ema_list = []
        backward_ema_t = reversed_x[:, :, 0]
        backward_ema_list.append(backward_ema_t)
        
        for t in range(1, time):
            backward_ema_t = self.alpha * reversed_x[:, :, t] + (1 - self.alpha) * backward_ema_t
            backward_ema_list.append(backward_ema_t)
        
        backward_ema = torch.stack(backward_ema_list, dim=2)
        backward_ema = torch.flip(backward_ema, dims=[2]) 
        
        # Compute SC-EMA using bidirectional fusion
        sc_ema_list = []
        sc_ema_list.append(x[:, :, 0:1])

        # Compute SC-EMA for all other points
        for t in range(1, time):
                next_val = self.alpha * x[:, :, t:t+1] + \
                    (1 - self.alpha)/2 * forward_ema[:, :, t-1:t] + \
                    (1 - self.alpha)/2 * backward_ema[:, :, t+1:t+2]
            sc_ema_list.append(next_val)
        
        sc_ema_output = torch.cat(sc_ema_list, dim=2)
        
        return sc_ema_output.permute(0, 2, 1)


class decom_Gate(nn.Module):
    def __init__(self, alpha):
        super(BEDF_Gate, self).__init__()
        self.ma = SC_EMA(alpha)
    
    def forward(self, x):
        trend = self.ma(x)
        seasonal = x - trend
        return seasonal, trend


class SRG_Gate(nn.Module):
    def __init__(self, configs):
        super(SREMC_Gate, self).__init__()
        self.decom = decom_Gate(configs.alpha)
        self.mc_dropout = True
        self.dropout_rate = getattr(configs, 'mc_dropout', 0.1)
        self.num_samples = configs.num_samples
        
        self.linear1 = nn.Sequential(
            nn.Linear(configs.seq_len * configs.enc_in, configs.mlp_hidden1),
            nn.ReLU(),
            nn.Linear(configs.mlp_hidden1, configs.num_experts)
        )
        self.linear2 = nn.Linear(configs.seq_len * configs.enc_in, configs.num_experts)
        self.last_seasonal = nn.Linear(configs.num_experts, configs.num_experts)
        self.last_trend = nn.Linear(configs.num_experts, configs.num_experts)

    def normal(self, x):
        with torch.no_grad():
            means = x.mean(1, keepdim=True)
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        
        x_norm = (x - means) / stdev
        return x_norm

    def calculate_mc_entropy(self, weights):
        max_vals = torch.max(weights, dim=0, keepdim=True)[0]
        exp_weights = torch.exp(weights - max_vals)
        prob = exp_weights / torch.sum(exp_weights, dim=0, keepdim=True)
        
        epsilon = 1e-12
        entropy = -torch.sum(prob * torch.log(prob + epsilon), dim=0)
        
        return torch.mean(entropy)

    def forward(self, x):
        seasonal, trend = self.decom(x)
        
        batch_size = seasonal.size(0)
        seasonal_flat = seasonal.reshape(batch_size, -1)
        trend_flat = trend.reshape(batch_size, -1)
        
        # Monte Carlo Dropout
        if self.mc_dropout:
            final_weights = []
            
            #more efficient version
            # seasonal_flat_expand = seasonal_flat.repeat(self.num_samples, 1)  # [num_samples*batch, ...]
            # trend = self.normal(trend)
            # trend_flat = trend.reshape(batch_size, -1)
            # trend_flat_expand = trend_flat.repeat(self.num_samples, 1)

         
            # seasonal_weight = self.linear1(seasonal_flat_expand)
            # seasonal_weight = F.dropout(seasonal_weight, p=self.dropout_rate, training=True)
            # gatesum_seasonal = F.softmax(seasonal_weight, dim=1)

            # trend_weight = self.linear2(trend_flat_expand)
            # trend_weight = F.dropout(trend_weight, p=self.dropout_rate, training=True)
            # gatesum_trend = F.softmax(trend_weight, dim=1)

            # gatesum_seasonal = self.last_seasonal(gatesum_seasonal)
            # gatesum_trend = self.last_trend(gatesum_trend)
            # weight_sum = gatesum_trend + gatesum_seasonal
            # weight = F.softmax(weight_sum, dim=1)  # [num_samples*batch, num_experts]

            # # reshape为[num_samples, batch, num_experts]
            # weight = weight.view(self.num_samples, batch_size, -1)
            # total_entropy = self.calculate_mc_entropy(weight)
            # weight = weight.mean(dim=0)  # [batch, num_experts]
            for _ in range(self.num_samples):
                seasonal_weight = self.linear1(seasonal_flat)
                seasonal_weight = F.dropout(seasonal_weight, p=self.dropout_rate, training=True)
                gatesum_seasonal = F.softmax(seasonal_weight, dim=1)
                
                trend_norm = self.normal(trend)
                trend_flat = trend_norm.reshape(batch_size, -1)
                trend_weight = self.linear2(trend_flat)
                trend_weight = F.dropout(trend_weight, p=self.dropout_rate, training=True)
                gatesum_trend = F.softmax(trend_weight, dim=1)
                
                gatesum_seasonal = self.last_seasonal(gatesum_seasonal)
                gatesum_trend = self.last_trend(gatesum_trend)
                weight_sum = gatesum_trend + gatesum_seasonal
                weight = F.softmax(weight_sum, dim=1)  # [batch_size, num_experts]
                final_weights.append(weight)
            
            final_weights = torch.stack(final_weights)
            
            total_entropy = self.calculate_mc_entropy(final_weights)
            
            weight = final_weights.mean(dim=0)
        else:
            seasonal_weight = self.linear1(seasonal_flat)
            gatesum_seasonal = F.softmax(seasonal_weight, dim=1)
            
            trend_norm = self.normal(trend)
            trend_flat = trend_norm.reshape(batch_size, -1)
            trend_weight = self.linear2(trend_flat)
            gatesum_trend = F.softmax(trend_weight, dim=1)
            
            gatesum_seasonal = self.last_seasonal(gatesum_seasonal)
            gatesum_trend = self.last_trend(gatesum_trend)
            weight_sum = gatesum_trend + gatesum_seasonal
            weight = F.softmax(weight_sum, dim=1)
            total_entropy = torch.tensor(0.0, device=x.device)
        
        return weight, total_entropy


class DLinearExpert(nn.Module):
    def __init__(self, configs):
        super(DLinearExpert, self).__init__()
        self.dlinear = DLinearModel(configs)
    
    def forward(self, x):
        x_input = x.permute(0, 2, 1)
        output = self.dlinear.forecast(x_input)  # [batch_size, pred_len, node]
        
        return output


class TCNExpert(nn.Module):
    def __init__(self, configs):
        super(TCNExpert, self).__init__()
        self.tcn = TCNModel(configs)
    
    def forward(self, x, x_mark, dec_inp, batch_y_mark):
        x_input = x.permute(0, 2, 1)
        output = self.tcn.forward(x_input, x_mark, dec_inp, batch_y_mark)  # [batch_size, pred_len, node]
        
        return output


class NLinearExpert(nn.Module):
    def __init__(self, configs):
        super(NLinearExpert, self).__init__()
        self.nlinear = NLinearModel(configs)
    
    def forward(self, x):
        x_input = x.permute(0, 2, 1)
        output = self.nlinear.forward(x_input)  # [batch_size, pred_len, node]
        
        return output


class Model(nn.Module):
    def __init__(self, configs, max_seq_len:Optional[int]=1024, d_k:Optional[int]=None, d_v:Optional[int]=None, norm:str='BatchNorm', attn_dropout:float=0., 
                 act:str="gelu", key_padding_mask:bool='auto',padding_var:Optional[int]=None, attn_mask:Optional[Tensor]=None, res_attention:bool=True, 
                 pre_norm:bool=False, store_attn:bool=False, pe:str='zeros', learn_pe:bool=True, pretrain_head:bool=False, head_type = 'flatten', verbose:bool=False, **kwargs):
        
        super().__init__()

        self.expert_type = getattr(configs, 'expert_type', 'PatchTST')
        
        # load parameters
        c_in = configs.enc_in
        context_window = configs.seq_len
        target_window = configs.pred_len
        
        n_layers = configs.e_layers
        n_heads = configs.n_heads
        d_model = configs.d_model
        d_ff = configs.d_ff
        dropout = configs.dropout
        fc_dropout = configs.fc_dropout
        head_dropout = configs.head_dropout
        
        individual = configs.individual
    
        patch_len = configs.patch_len
        stride = configs.stride
        padding_patch = configs.padding_patch
        
        revin = configs.revin
        affine = configs.affine
        subtract_last = configs.subtract_last
        
        decomposition = configs.decomposition
        kernel_size = configs.kernel_size

        # MoE parameters
        self.num_experts = configs.num_experts
        self.activated_experts = configs.activated_experts
        self.seed = configs.seed
        print(f"[Debug] Using SEED = {self.seed}")


        # model
        self.decomposition = decomposition
        if self.decomposition:
            self.decomp_module = series_decomp(kernel_size)
            # Create experts for trend
            self.experts_trend = nn.ModuleList()
            for i in range(self.num_experts):
                expert = self._create_expert(configs, i, c_in, context_window, target_window, patch_len, stride, 
                                  max_seq_len, n_layers, d_model, n_heads, d_k, d_v, d_ff, norm, attn_dropout,
                                  dropout, act, key_padding_mask, padding_var, attn_mask, res_attention, pre_norm, store_attn,
                                  pe, learn_pe, fc_dropout, head_dropout, padding_patch, pretrain_head, head_type, individual, revin, affine,
                                  subtract_last, verbose, **kwargs)
                torch.manual_seed(self.seed + i)
                expert.apply(self._init_weights)
                self.experts_trend.append(expert)

            # Create experts for residual
            self.experts_res = nn.ModuleList()
            for i in range(self.num_experts):
                expert = self._create_expert(configs, i, c_in, context_window, target_window, patch_len, stride, 
                                  max_seq_len, n_layers, d_model, n_heads, d_k, d_v, d_ff, norm, attn_dropout,
                                  dropout, act, key_padding_mask, padding_var, attn_mask, res_attention, pre_norm, store_attn,
                                  pe, learn_pe, fc_dropout, head_dropout, padding_patch, pretrain_head, head_type, individual, revin, affine,
                                  subtract_last, verbose, **kwargs)
                torch.manual_seed(self.seed + i + self.num_experts)
                expert.apply(self._init_weights)
                self.experts_res.append(expert)
        else:
            # Create experts
            self.experts = nn.ModuleList()
            for i in range(self.num_experts):
                expert = self._create_expert(configs, i, c_in, context_window, target_window, patch_len, stride, 
                                  max_seq_len, n_layers, d_model, n_heads, d_k, d_v, d_ff, norm, attn_dropout,
                                  dropout, act, key_padding_mask, padding_var, attn_mask, res_attention, pre_norm, store_attn,
                                  pe, learn_pe, fc_dropout, head_dropout, padding_patch, pretrain_head, head_type, individual, revin, affine,
                                  subtract_last, verbose, **kwargs)
                torch.manual_seed(self.seed + i)
                expert.apply(self._init_weights)
                self.experts.append(expert)

        # Restore the original seed to not affect other parts of the model
        torch.manual_seed(self.seed)

     
        self.gate = SRG_Gate(configs)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.fill_(0.01)


    def _create_expert(self, configs, expert_id, c_in, context_window, target_window, patch_len, stride, 
                      max_seq_len, n_layers, d_model, n_heads, d_k, d_v, d_ff, norm, attn_dropout,
                      dropout, act, key_padding_mask, padding_var, attn_mask, res_attention, pre_norm, store_attn,
                      pe, learn_pe, fc_dropout, head_dropout, padding_patch, pretrain_head, head_type, individual, revin, affine,
                      subtract_last, verbose, **kwargs):
        
        if self.expert_type == 'PatchTST':
            return PatchTST_backbone(c_in=c_in, context_window=context_window, target_window=target_window, patch_len=patch_len, stride=stride, 
                            max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                            n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                            dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, 
                            attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                            pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch=padding_patch,
                            pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                            subtract_last=subtract_last, verbose=verbose, **kwargs)
        elif self.expert_type == 'DLinear':
            expert_configs = copy.deepcopy(configs)
            return DLinearExpert(expert_configs)
        elif self.expert_type == 'NLinear':
            expert_configs = copy.deepcopy(configs)
            return NLinearExpert(expert_configs)
        elif self.expert_type == 'TCN':
            expert_configs = copy.deepcopy(configs)
            return TCNExpert(expert_configs)
        else:
            raise ValueError(f"Unsupported expert type: {self.expert_type}")

    def _normalize_expert_output(self, expert_output, expert_type):
        if expert_type == 'PatchTST':
            return expert_output.permute(0, 2, 1)
        elif expert_type == 'DLinear':
            return expert_output
        elif expert_type == 'NLinear':
            return expert_output
        elif expert_type == 'TCN':
            return expert_output
        else:
            return expert_output


  
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        batch_size = x_enc.shape[0]
        x_enc = x_enc.permute(0,2,1)  # [batch_size, node, seq_len]

        if torch.isnan(x_enc).any() or torch.isinf(x_enc).any():
            print(f"Warning: Input contains NaN or Inf values")
            x_enc = torch.nan_to_num(x_enc, nan=0.0, posinf=1e6, neginf=-1e6)

        # Get gate weights using SREMC_Gate
        weight, entropy = self.gate(x_enc)
        
        if torch.isnan(weight).any() or torch.isinf(weight).any():
            print(f"Warning: Gate weights contain NaN or Inf values")
            weight = torch.nan_to_num(weight, nan=1.0/self.num_experts, posinf=1.0, neginf=0.0)
        
        if self.decomposition:
            # Decompose the input
            seasonal_init, trend_init = self.decomp_module(x_enc.permute(0,2,1))  # [batch_size, seq_len, node]
            seasonal_init = seasonal_init.permute(0,2,1)  # [batch_size, node, seq_len]
            trend_init = trend_init.permute(0,2,1)  # [batch_size, node, seq_len]
            
            if torch.isnan(seasonal_init).any() or torch.isinf(seasonal_init).any():
                print(f"Warning: Seasonal component contains NaN or Inf values")
                seasonal_init = torch.nan_to_num(seasonal_init, nan=0.0, posinf=1e6, neginf=-1e6)
            if torch.isnan(trend_init).any() or torch.isinf(trend_init).any():
                print(f"Warning: Trend component contains NaN or Inf values")
                trend_init = torch.nan_to_num(trend_init, nan=0.0, posinf=1e6, neginf=-1e6)
            
            # Get expert outputs for trend
            trend_outputs = []
            for expert in self.experts_trend:
                expert_output = expert(trend_init)  # [batch_size, pred_len, node]
                if torch.isnan(expert_output).any() or torch.isinf(expert_output).any():
                    print(f"Warning: Trend expert output contains NaN or Inf values")
                    expert_output = torch.nan_to_num(expert_output, nan=0.0, posinf=1e6, neginf=-1e6)
                trend_outputs.append(expert_output)
            trend_outputs = torch.stack(trend_outputs, dim=1)  # [batch_size, num_experts, pred_len, node]
            
            # Get expert outputs for residual
            res_outputs = []
            for expert in self.experts_res:
                expert_output = expert(seasonal_init)  # [batch_size, pred_len, node]
                if torch.isnan(expert_output).any() or torch.isinf(expert_output).any():
                    print(f"Warning: Residual expert output contains NaN or Inf values")
                    expert_output = torch.nan_to_num(expert_output, nan=0.0, posinf=1e6, neginf=-1e6)
                res_outputs.append(expert_output)
            res_outputs = torch.stack(res_outputs, dim=1)  # [batch_size, num_experts, pred_len, node]
            
            # Select top-k experts
            topk_values, topk_indices = torch.topk(weight, self.activated_experts, dim=1)  # [batch_size, activated_experts]
            expert_weights = F.softmax(topk_values, dim=1)  # [batch_size, activated_experts]
            
            if torch.isnan(expert_weights).any() or torch.isinf(expert_weights).any():
                print(f"Warning: Expert weights contain NaN or Inf values")
                expert_weights = torch.ones_like(expert_weights) / self.activated_experts
        
            # Process trend and residual separately
            def process_outputs(expert_outputs):
                batch_size = expert_outputs.size(0)
                pred_len = expert_outputs.size(2)
                node = expert_outputs.size(3)
                
                # Reshape indices to match expert_outputs dimensions
                indices = topk_indices.unsqueeze(-1).unsqueeze(-1)  # [batch_size, activated_experts, 1, 1]
                indices = indices.expand(-1, -1, pred_len, node)  # [batch_size, activated_experts, pred_len, node]
                
                # Gather selected expert outputs
                selected_outputs = torch.gather(expert_outputs, dim=1, index=indices)
                
                # Apply weights
                weights = expert_weights.view(batch_size, -1, 1, 1)  # [batch_size, activated_experts, 1, 1]
                weighted_outputs = selected_outputs * weights
                
                # Combine outputs
                result = weighted_outputs.sum(dim=1)  # [batch_size, pred_len, node]
                
                if torch.isnan(result).any() or torch.isinf(result).any():
                    print(f"Warning: Final output contains NaN or Inf values")
                    result = torch.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)
                
                return result
            
            trend_final = process_outputs(trend_outputs)
            res_final = process_outputs(res_outputs)
        
            # Combine trend and residual
            outputs = trend_final + res_final  # [batch_size, pred_len, node]
            
            if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                print(f"Warning: Combined output contains NaN or Inf values")
                outputs = torch.nan_to_num(outputs, nan=0.0, posinf=1e6, neginf=-1e6)
            
            return outputs, entropy  # [batch_size, pred_len, node] = [batch_size, 96, 7]

        else:
            # Get expert outputs
            expert_outputs = []
 
            if self.expert_type == 'TCN':
                # Special handling for TCN expert
                for i, expert in enumerate(self.experts):
                    expert_output = expert(x_enc, x_mark_enc, x_dec, x_mark_dec)
                    expert_outputs.append(expert_output)
            else:
                # Original logic for other experts
                for i, expert in enumerate(self.experts):
                    expert_output = expert(x_enc)
                    # Apply normalization for different expert types
                    expert_output = self._normalize_expert_output(expert_output, self.expert_type) # add
                    expert_outputs.append(expert_output)
            
            expert_outputs = torch.stack(expert_outputs, 1)  # (batch_size, num_experts, pred_len, c_out)
        
        # Select top-k experts
        topk_values, topk_indices = torch.topk(weight, self.activated_experts, dim=1)  # [batch_size, activated_experts]
        expert_weights = F.softmax(topk_values, dim=1)  # [batch_size, activated_experts]
        
        if torch.isnan(expert_weights).any() or torch.isinf(expert_weights).any():
            print(f"Warning: Expert weights contain NaN or Inf values")
            expert_weights = torch.ones_like(expert_weights) / self.activated_experts
        
        # Reshape indices for gathering
        batch_size = expert_outputs.size(0)
        pred_len = expert_outputs.size(2)
        node = expert_outputs.size(3)
        
        # Reshape indices to match expert_outputs dimensions
        topk_indices = topk_indices.unsqueeze(-1).unsqueeze(-1)  # [batch_size, activated_experts, 1, 1]
        topk_indices = topk_indices.expand(-1, -1, pred_len, node)  # [batch_size, activated_experts, pred_len, node]
        
        # Gather selected expert outputs
        selected_expert_outputs = torch.gather(
            expert_outputs, 
            dim=1, 
            index=topk_indices
        )
        
        # Apply weights
        expert_weights = expert_weights.view(batch_size, -1, 1, 1)  # [batch_size, activated_experts, 1, 1]
        weighted_outputs = selected_expert_outputs * expert_weights
        
        # Combine outputs
        outputs = weighted_outputs.sum(dim=1)  # [batch_size, pred_len, node]
        
        if torch.isnan(outputs).any() or torch.isinf(outputs).any():
            print(f"Warning: Final output contains NaN or Inf values")
            outputs = torch.nan_to_num(outputs, nan=0.0, posinf=1e6, neginf=-1e6)
        
        return outputs, entropy  # [batch_size, pred_len, node] = [batch_size, 96, 7]

