from collections import Counter
import contextlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import *

# Handle mixed precision compatibility.
if not hasattr(torch.cuda, 'amp') or not hasattr(torch.cuda.amp, 'autocast'):
    def autocast(enabled=False):
        return contextlib.nullcontext()
else:
    from torch.cuda.amp import autocast

class SharedParameters(nn.Module):
    def __init__(self, upsample_type, emb_size, num_params, init, init_blocks):
        super(SharedParameters, self).__init__()
        self._upsample_type = upsample_type
        self._emb_size = emb_size
        params = torch.Tensor(num_params)
        init(params[:(num_params // 9) * 9].view(-1, 3, 3))
        if init_blocks is not None:
            for (num_candidates, start, end), layer_shape in init_blocks:
                if (np.prod(layer_shape) * num_candidates) > num_params:
                    continue
                
                for p in params[start:end].view(num_candidates, *layer_shape):
                    init(p)

        self._num_params = num_params
        self._params = nn.Parameter(params)

    def get_params(self, num_candidates, start, end):
        params = self._params[start:end]
        return params.view(num_candidates, -1)

    def __len__(self):
        return self._num_params

class ParameterCombiner(nn.Module):
    def __init__(self, learn_groups, num_candidates, layer_shape, num_params, upsample_type, upsample_window, coefficients, init, layer=None):
        super(ParameterCombiner, self).__init__()
        self._learn_groups = learn_groups
        self._upsample_type = upsample_type
        self._upsample_window = upsample_window
        self._out_param_shape = layer_shape
        self.num_candidates = num_candidates
        self.layer = layer
        self.single_dims = list(np.ones(len(layer_shape), np.int32))
        if coefficients is not None:
            coefficients = nn.Parameter(coefficients)

        self.coefficients = coefficients
        if num_candidates > 1:
            self.mask = nn.Parameter(torch.ones(num_candidates), requires_grad=False)

        self._needs_resize = (np.prod(layer_shape) * num_candidates) > num_params or learn_groups
        if self._needs_resize:
            self._in_param_shape = layer_shape.copy()
            if self._learn_groups:
                in_size = np.prod(layer_shape[1:])
                out_size = layer_shape[0]
                ratio = in_size / float(out_size)
                out_features = int(np.floor(np.sqrt((num_params // num_candidates) // ratio)))
                assert out_features > 0
                self._in_param_shape = [out_features, num_params // (out_features * num_candidates)]
                assert upsample_type == 'inter'
            elif self._upsample_type == 'inter':
                p = 0.95
                while num_params < (round(layer_shape[0]*p))*(round(layer_shape[1]*p) * np.prod(layer_shape[2:]) * num_candidates):
                    p -= 0.05

                assert p > 0
                self._in_param_shape[0] = int(round(self._in_param_shape[0]*p))
                self._in_param_shape[1] = int(round(self._in_param_shape[1]*p))
                assert self._in_param_shape[0] > 0
            else:
                self.upsample_mult = int(np.ceil(np.prod(self._out_param_shape) / float(num_params))) - 1

    def get_candidate_weights(self):
        coefficients = self.coefficients
        if self.layer is not None:
            coefficients = self.layer(coefficients.view(1, -1))

        if self._learn_groups:
            coefficients = F.softmax(coefficients.view(1, -1), 1)

        coefficients = coefficients.view(self.num_candidates, *self.single_dims)
        return coefficients * self.mask.view(self.num_candidates, *self.single_dims)

    def forward(self, x, upsample_layer=None):
        if self._needs_resize:
            if self._upsample_type == 'inter':
                x = x.view(-1)[:(self.num_candidates * np.prod(self._in_param_shape))]
                x = x.view(1, self.num_candidates, self._in_param_shape[0], np.prod(self._in_param_shape[1:]))
                x = F.interpolate(x, [self._out_param_shape[0], np.prod(self._out_param_shape[1:])], mode='bilinear', align_corners=False)
            else:
                assert upsample_layer is not None
                x = upsample_layer(x, self._out_param_shape)

        x = x.view(self.num_candidates, *self._out_param_shape)
        if self.coefficients is not None:
            x = (x * self.get_candidate_weights()).sum(0)
        
        return x

class ParameterUpsampler(nn.Module):
    def __init__(self, upsample_type, upsample_window, upsample_mult):
        super(ParameterUpsampler, self).__init__()
        self._upsample_type = upsample_type
        self._upsample_window = upsample_window
        self.upsample_mult = upsample_mult
        assert upsample_type in ['mask', 'repeat', 'wavg']
        num_params_3by3 = 9
        upsample_params = num_params_3by3 * self._upsample_window
        if self._upsample_type == 'mask':
            upsample_params *= upsample_mult
            self.weights = nn.Parameter(torch.ones(upsample_params) / float(upsample_params))
        elif self._upsample_type == 'wavg':
            N = 2
            params = torch.zeros((upsample_mult + 1) * N, N)
            nn.init.orthogonal_(params)
            self.N = N
            self.weights = nn.Parameter(params.unsqueeze(-1))

    def forward(self, x, out_param_shape):
        x = x.view(-1)

        # we upsample windows based on 3x3 kernels
        if self._upsample_type == 'wavg':
            upsample_window = self.N
        else:
            upsample_window = 9 * self._upsample_window

        num_params = len(x) - (len(x) % upsample_window)
        x = x[:num_params].view(-1, upsample_window)
        if self._upsample_type == 'mask':
            with autocast(enabled=False):
                x = x.unsqueeze(0)
                weights = self.weights.view(-1)[:upsample_window * self.upsample_mult]
                x = torch.cat([x, x * weights.view(self.upsample_mult, 1, upsample_window)], 0)
        elif self._upsample_type == 'repeat':
            x = x.unsqueeze(0).repeat(self.upsample_mult + 1, 1, 1)
        elif self._upsample_type == 'wavg':
            x = (x.view(1, upsample_window, -1) * self.weights).sum(1)

        return x.view(-1)[:int(np.prod(out_param_shape))]

class ParameterGroups(nn.Module):
    def __init__(self, groups, share_type, upsample_type, upsample_window, max_params=0, max_candidates=2):
        super(ParameterGroups, self).__init__()
        self._learn_groups = groups is None
        if self._learn_groups:
            num_groups = 1
            max_candidates = 4
            groups = 0
            assert share_type in ['wavg', 'emb']
        elif isinstance(groups, int):
            if groups <= 0:
                num_groups = 0
                self._layers = []
                self._upsample_type = upsample_type
                self._upsample_window = upsample_window
                if share_type != 'none':
                    share_type += '_slide'

                self._share_type = share_type
                self._max_candidates = max_candidates
            else:
                num_groups = 1
        else:
            num_groups = len(set(groups))

        self._num_groups = num_groups
        self._max_params = max_params
        for group_id in range(num_groups):
            self.add_module('_param_group_' + str(group_id), ParameterBank(share_type, upsample_type, upsample_window, max_params, max_candidates, self._learn_groups))

        self._groups = groups
        self._layer_count = 0

    def add_layer(self, layer):
        layer_id = self._layer_count
        group_id = self._groups
        self._layer_count += 1
        if isinstance(group_id, int):
            if not self._learn_groups and group_id <= 0:
                self._layers.append(layer)
                return None, group_id, layer_id

            group_id = 0
        elif not isinstance(group_id, int):
            group_id = group_id[layer_id]

        params = getattr(self, '_param_group_' + str(group_id))
        params.add_layer(layer, layer_id)
        return params, group_id, layer_id

    def setup_bank(self):
        if isinstance(self._groups, int) and self._groups <= 0 and not self._learn_groups:
            # group together layers of the same size/type
            groups = {}
            for i, layer in enumerate(self._layers):
                layer_type = layer.__class__.__name__
                layer_shape = ' '.join(str(d) for d in layer.shape)
                if layer_type not in groups:
                    groups[layer_type] = {}

                if layer_shape not in groups[layer_type]:
                    groups[layer_type][layer_shape] = []
                    
                groups[layer_type][layer_shape].append(i)

            layer2group = np.zeros(len(self._layers), np.int32)
            group_id = 0
            learn_groups = False
            for _, layer_shapes in groups.items():
                for _, layer_ids in layer_shapes.items():
                    if not hasattr(self, '_param_group_' + str(group_id)):
                        self.add_module('_param_group_' + str(group_id), ParameterBank(self._share_type, self._upsample_type, self._upsample_window, self._max_params, self._max_candidates, learn_groups))

                    params = getattr(self, '_param_group_' + str(group_id))
                    for layer_id in layer_ids:
                        layer2group[layer_id] = group_id

                        # now let's set everything for each layer
                        # that is normally set elsewhere
                        layer = self._layers[layer_id]
                        layer.bank = params
                        layer.group_id = group_id
                        params.add_layer(layer, layer_id)

                    if self._groups < 0:
                        group_id += 1

                if self._groups == 0:
                    group_id += 1

            self._num_groups = group_id
            self._groups = layer2group

        if self._max_params > 0:
            # going to split up parameters based on the relative sizes
            # of the parameter groups
            group_sizes = np.zeros(self._num_groups)
            avail_params = self._max_params
            for group_id in range(self._num_groups):
                param_bank = getattr(self, '_param_group_' + str(group_id))
                if len(param_bank._layers) == 1:
                    avail_params -= np.prod(param_bank._layers[0].shape)
                else:
                    num_params = param_bank.get_num_params()
                    group_sizes[group_id] = num_params

            total_params = np.sum(group_sizes)
            assert total_params > 0
            group_sizes = np.round((group_sizes / total_params) * avail_params).astype(np.int32)
            for group_id, max_params in enumerate(group_sizes):
                param_bank = getattr(self, '_param_group_' + str(group_id))
                if len(param_bank._layers) > 1:
                    assert max_params > 0
                    param_bank.set_num_params(max_params)


        for group_id in range(self._num_groups):
            params = getattr(self, '_param_group_' + str(group_id))
            params.setup_bank()

        if isinstance(self._groups, int):
            self._groups = np.zeros(self._layer_count, np.int32)

    def update_masks(self):
        for group_id in range(self._num_groups):
            params = getattr(self, '_param_group_' + str(group_id))
            params.update_masks()

    def forward(self, layer_id, layer_shape, coeff=None):
        group_id = self._groups[layer_id]
        params = getattr(self, '_param_group_' + str(group_id))
        return params(layer_id, layer_shape, coeff)
        
class ParameterBank(nn.Module):
    def __init__(self, share_type, upsample_type, upsample_window, max_params=0, max_candidates=2, learn_groups=False, init=nn.init.kaiming_normal_, emb_size=24):
        super(ParameterBank, self).__init__()
        self._share_type = share_type
        self._upsample_type = upsample_type
        self._upsample_window = upsample_window

        if max_params > 0:
            # later on we assume we have an even number
            # of params so enfource we get an even number
            remainder = max_params % 9
            if remainder:
                max_params += (9 - remainder)

        self._max_params = max_params
        self._max_candidates = max_candidates
        self._learn_groups = learn_groups
        self._emb_size = emb_size
        self._layers = []
        self._layerid2ind = {}
        self._layer_parms = []
        self._init = init#nn.init.orthogonal_#init
        self._layer_param_blocks = []

    def add_layer(self, layer, layer_id=None):
        if layer_id is None:
            layer_id = len(self._layers)

        self._layerid2ind[layer_id] = len(self._layers)
        self._layers.append(layer)
        return layer_id

    def set_num_params(self, max_params):
        # this should be called before running setup_bank
        assert not hasattr(self, '_params')
        self._max_params = max_params

    def get_num_params(self):
        if hasattr(self, '_params'):
            return len(self._params)

        num_candidates = np.ones(len(self._layers), np.int64)
        params_by_layer, _ = self.params_per_layer(num_candidates)
        return np.sum(params_by_layer)

    def get_num_candidates(self, num_params=None):
        if num_params is None:
            num_params = self.get_num_params()

        num_candidates = np.zeros(len(self._layers), np.int64)
        for i, layer in enumerate(self._layers):
            num_candidates[i] = min(max(1, num_params // np.prod(layer.shape)), self._max_candidates)

        return num_candidates

    def params_per_layer(self, num_candidates=None):
        if num_candidates is None:
            num_candidates = self.get_num_candidates()

        params_by_layer = np.zeros(len(num_candidates), np.int64)
        for i, candidates, layer in zip(range(len(params_by_layer)), num_candidates, self._layers):
            params_by_layer[i] = candidates * np.prod(layer.param_shape())

        return params_by_layer, num_candidates

    def setup_bank(self):
        self.single_layer = len(self._layers) == 1
        if self.single_layer:            
            for layer in self._layers:
                if isinstance(layer, SConv2d):
                    self.add_module('_layer', nn.Conv2d(layer.in_channels, layer.out_channels, kernel_size=layer.kernel_size, stride=layer.stride, padding=layer.padding))
                elif isinstance(layer, SLinear):
                    self.add_module('_layer',  nn.Linear(layer.in_channels, layer.out_channels))
                elif isinstance(layer, SGRU):
                    self.add_module('_layer',  layer.rnn_type(
                        layer.in_features,
                        layer.out_features, layer.num_layers,
                        batch_first=layer.batch_first,
                        bidirectional=layer.bidirectional
                    ))
                else:
                    raise('single shared layer not implemented')

            return

        if self._learn_groups:
            num_params, sharing_blocks = self.even_param_assignments()
        else:
            num_params, sharing_blocks = self.assign_layers_to_params()
    
        self.add_module('_params', SharedParameters(self._upsample_type, self._emb_size, num_params, self._init, sharing_blocks))

    def even_param_assignments(self):
        num_params = self._max_params
        assert num_params > 0
        set_candidate_number = 4
        num_candidates = np.ones(len(self._layers), np.int64) * set_candidate_number
        start_end_sets = {}
        global_start = 0
        layer_shapes = [' '.join(str(d) for d in l.shape) for l in self._layers]
        most_common_shape = [int(d) for d in Counter(layer_shapes).most_common(1)[0][0].split()]
        self._layer_param_blocks = [None for _ in range(len(self._layers))]
        for shape, _ in Counter(layer_shapes).most_common():
            layer_ids = [i for i, layer in enumerate(layer_shapes) if layer == shape]
            for layer_id in layer_ids:
                candidates = num_candidates[layer_id]
                params_start = 0
                params_end = num_params - (num_params % candidates)
                self._layer_param_blocks[layer_id] = [candidates, params_start, params_end]

        sharing_blocks = None
        if self._share_type in ['wavg', 'wavg_slide', 'emb', 'emb_slide']:
            self.set_layer_coeff(num_candidates, num_params)

        return num_params, sharing_blocks

    def param_layer_assignment(self):
        num_params = self._max_params
        if num_params > 0:
            num_candidates = self.get_num_candidates(num_params)
        else:
            num_candidates = np.ones(len(self._layers), np.int64)
            params_by_layer, _ = self.params_per_layer(num_candidates)
            num_params = max(params_by_layer)            
            num_candidates = self.get_num_candidates(num_params)

        params_by_layer, _ = self.params_per_layer(num_candidates)

        # setting coefficients also determines when not enugh params
        # are available for a layer
        if self._share_type in ['wavg', 'wavg_slide', 'emb', 'emb_slide']:
            self.set_layer_coeff(num_candidates, num_params)

        # let's get new layer sizes in case they changed after
        # setting coefficients
        params_by_layer, _ = self.params_per_layer(num_candidates)

        start_end_sets = {}
        global_start = 0
        layer_shapes = [' '.join(str(d) for d in l.shape) for l in self._layers]
        most_common_shape = [int(d) for d in Counter(layer_shapes).most_common(1)[0][0].split()]
        self._layer_param_blocks = [None for _ in range(len(self._layers))]
        for shape, _ in Counter(layer_shapes).most_common():
            layer_ids = [i for i, layer in enumerate(layer_shapes) if layer == shape]
            for layer_id in layer_ids:
                layer_params, candidates = params_by_layer[layer_id], num_candidates[layer_id]
                if layer_params not in start_end_sets:
                    start_end_sets[layer_params] = {}
                    start_end_sets[layer_params]['num_can'] = candidates
                    can_start = int(np.ceil(global_start / float(layer_params)))
                    if can_start >= candidates:
                        global_start = 0
                        can_start = 0
                    else:
                        global_start += layer_params

                    start_end_sets[layer_params]['can'] = can_start                

                params_start = start_end_sets[layer_params]['can'] * layer_params
                params_end = params_start + layer_params
                if params_end > num_params:
                    start_end_sets[layer_params]['can'] = 0
                    params_start = start_end_sets[layer_params]['can'] * layer_params
                    params_end = params_start + layer_params
                    if start_end_sets[layer_params]['can'] < 2:
                        params_end = min(params_end, num_params)


                start_end_sets[layer_params]['can'] += 1
                if self._share_type in ['sliding_window', 'avg_slide', 'wavg_slide', 'emb_slide', 'conv']:
                    if start_end_sets[layer_params]['can'] == start_end_sets[layer_params]['num_can']:
                        start_end_sets[layer_params]['can'] = 0

                self._layer_param_blocks[layer_id] = [candidates, params_start, params_end]

        return num_params, num_candidates

    def assign_layers_to_params(self):
        need_sharing = self._max_params < 1 or self.get_num_params() < self._max_params
        num_params, num_candidates = self.param_layer_assignment()
        layer_shapes = [' '.join(str(d) for d in l.shape) for l in self._layers]
        most_common_shape = [int(d) for d in Counter(layer_shapes).most_common(1)[0][0].split()]

        # sharing blocks give an order for parameter initialization
        # assumption is that some parameters will be initialized
        # mutliple times, but the most common one will be the last
        # and hopefully that is the best
        sharing_blocks = []
        for shape, _ in Counter(layer_shapes).most_common()[::-1]:
            layer_ids = [i for i, layer in enumerate(layer_shapes) if layer == shape]
            for layer_id in layer_ids:
                sharing_blocks.append([self._layer_param_blocks[layer_id], self._layers[layer_id].shape])

        return num_params, sharing_blocks

    def set_layer_coeff(self, num_candidates, num_params):
        candidates_req = Counter(list(num_candidates))
        weight_params = {}
        for candidates, num_instances in candidates_req.items():
            if candidates < 2:
                continue

            if self._share_type in ['wavg', 'wavg_slide']:
                params = torch.zeros((num_instances, candidates))
            else:
                self.add_module('_coeff_proj_%i_candidates' % candidates, nn.Sequential(nn.Linear(self._emb_size, candidates)))

            nn.init.orthogonal_(params)
            weight_params[candidates] = [0, params]

        upsample_mult = 0
        for layer, candidates in zip(self._layers, num_candidates):
            combiner_params, comb_layer = None, None
            if candidates > 1:
                ind, params = weight_params[candidates]
                combiner_params = params[ind]
                weight_params[candidates] = [ind+1, params]
                if self._share_type in ['emb', 'emb_slide']:
                    comb_layer = getattr(self, '_coeff_proj_%i_candidates' % candidates)
    
            comb = ParameterCombiner(self._learn_groups, candidates, layer.shape, num_params, self._upsample_type, self._upsample_window, combiner_params, nn.init.orthogonal_, comb_layer)
            layer.set_coefficients(comb)
            if not self._learn_groups and comb._needs_resize and self._upsample_type != 'inter':
                upsample_mult = comb.upsample_mult
                num_params_3by3 = 9
                upsample_params = num_params_3by3 * self._upsample_window
                if upsample_mult and self._upsample_type != 'inter':
                    self.add_module('_upsample_layer_%i' % layer.layer_id, ParameterUpsampler(self._upsample_type, self._upsample_window, upsample_mult))

    def update_masks(self):
        for layer in self._layers:
            if not hasattr(layer ,'coefficients') or layer.coefficients is None:
                continue

            weights = layer.coefficients.get_candidate_weights().squeeze()
            ind = torch.nonzero(weights)
            if len(ind) > 12:
                ind = ind[torch.min(weights[ind], dim=0)[1]].squeeze()
                layer.coefficients.mask[ind] = 0

    def forward(self, layer_id, coeff=None):
        ind = self._layerid2ind[layer_id]
        num_candidates, blocks_start, blocks_end = self._layer_param_blocks[ind]
        params = self._params.get_params(num_candidates, blocks_start, blocks_end)
        if coeff is not None:
            upsample_layer = None
            if hasattr(self, '_upsample_layer_%i' % layer_id):
                upsample_layer = getattr(self, '_upsample_layer_%i' % layer_id)

            params = coeff(params, upsample_layer)
        elif self._share_type in ['avg', 'avg_slide']:
            params = params.mean(0)

        return params



