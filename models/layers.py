from collections import Counter
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence
# Compatibility with PyTorch v1.5.0 versus earlier.
if hasattr(torch.nn, '_VF'):
    from torch.nn import _VF
else:
    from torch import _VF


def _addindent(s_, numSpaces):
    s = s_.split('\n')
    # don't do anything for single-line stuff
    if len(s) == 1:
        return s_
    first = s.pop(0)
    s = [(numSpaces * ' ') + line for line in s]
    s = '\n'.join(s)
    s = first + '\n' + s
    return s

class SModule(nn.Module):
    def __init__(self, bank, in_features, out_features):
        super(SModule, self).__init__()
        self.in_channels = in_features
        self.out_channels = out_features

    def param_shape(self):
        if hasattr(self, 'coefficients'):
            if self.coefficients is not None and hasattr(self.coefficients, '_in_param_shape'):
                return self.coefficients._in_param_shape
    
        return self.shape

    def __repr__(self):
        # We treat the extra repr like the sub-module, one item per line
        extra_lines = []
        extra_repr = self.extra_repr()
        # empty string will be split into list ['']
        if extra_repr:
            extra_lines = extra_repr.split('\n')

        child_lines = []
        for key, module in self._modules.items():
            if key == 'bank':
                continue
            mod_str = repr(module)
            mod_str = _addindent(mod_str, 2)
            child_lines.append('(' + key + '): ' + mod_str)

        lines = extra_lines + child_lines

        main_str = self._get_name() + '('
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += '\n  ' + '\n  '.join(lines) + '\n'

        main_str += ')'
        return main_str

    def get_params(self):
        return self.bank(self.layer_id, self.coefficients).view(*self.shape)

class SConv2d(SModule):
    def __init__(self, bank, in_features, out_features, kernel_size, stride=1, padding=0, groups=1, bias=None):
        super(SConv2d, self).__init__(bank, in_features, out_features)
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size
        self.groups = groups
        assert in_features % groups == 0
        self.shape = [out_features, in_features // groups, kernel_size, kernel_size]
        self.bank, self.group_id, self.layer_id = bank.add_layer(self)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}, groups={groups}')
        #if self.padding != (0,) * len(self.padding):
        #    s += ', padding={padding}'
        #if self.dilation != (1,) * len(self.dilation):
        #    s += ', dilation={dilation}'
        #if self.output_padding != (0,) * len(self.output_padding):
        #    s += ', output_padding={output_padding}'
        #if self.groups != 1:
        #    s += ', groups={groups}'
        #if self.bias is None:
        #    s += ', bias=False'
        #if self.padding_mode != 'zeros':
        #    s += ', padding_mode={padding_mode}'
        s += ', parameter_group={group_id}'
        return s.format(**self.__dict__)

    def set_coefficients(self, coefficients):
        if coefficients is None:
            self.coefficients = None
        else:
            self.add_module('coefficients', coefficients)

    def forward(self, input):
        if self.bank.single_layer:
            return self.bank._layer(input)

        params = self.get_params()
        return F.conv2d(input, params, stride=self.stride, padding=self.padding, groups=self.groups)

class SLinear(SModule):
    def __init__(self, bank, in_features, out_features):
        super(SLinear, self).__init__(bank, in_features, out_features)
        self.shape = [out_features, in_features]
        self.bank, self.group_id, self.layer_id = bank.add_layer(self)
        self.bias = None

    def set_coefficients(self, coefficients):
        if coefficients is None:
            self.coefficients = None
        else:
            self.add_module('coefficients', coefficients)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}, paramter_group={}'.format(
            self.in_channels, self.out_channels, self.bias is not None, self.group_id
        )

    def forward(self, input):
        if self.bank.single_layer:
            return self.bank._layer(input)

        params = self.get_params()
        return F.linear(input, params)

class SGRU(nn.Module):
    def __init__(self, bank, rnn_type, in_features, out_features, num_layers, batch_first=True, bidirectional=False):
        super(SGRU, self).__init__()
        self.rnn_type = rnn_type
        self.mode = 'RNN_TANH'
        self.batch_first = batch_first
        self.bidirectional = bidirectional

        num_bias_params = 2
        param_size = out_features * 3
        param_in = param_size + param_size * bidirectional
        param_out = in_features + out_features + num_bias_params

        self.in_features = param_in
        self.out_features = param_out

        self.input_size = in_features
        self.latent_size = out_features
        self.num_layers = num_layers

        # not implemented properly for 2 layers, would have to change how parameters
        # are allocated in "set_params"
        assert num_layers == 1
        self.bank = bank
        self.bias = True
        self.dropout = 0
        self.bank, self.group_id, self.layer_id = self.bank.add_layer(self, self.in_features, self.out_features)

    def check_input(self, input, batch_sizes):
        # type: (Tensor, Optional[Tensor]) -> None
        expected_input_dim = 2 if batch_sizes is not None else 3
        if input.dim() != expected_input_dim:
            raise RuntimeError(
                'input must have {} dimensions, got {}'.format(
                    expected_input_dim, input.dim()))
        if self.input_size != input.size(-1):
            raise RuntimeError(
                'input.size(-1) must be equal to input_size. Expected {}, got {}'.format(
                    self.input_size, input.size(-1)))

    def get_expected_hidden_size(self, input, batch_sizes):
        # type: (Tensor, Optional[Tensor]) -> Tuple[int, int, int]
        if batch_sizes is not None:
            mini_batch = batch_sizes[0]
            mini_batch = int(mini_batch)
        else:
            mini_batch = input.size(0) if self.batch_first else input.size(1)
        num_directions = 2 if self.bidirectional else 1
        expected_hidden_size = (self.num_layers * num_directions,
                                mini_batch, self.latent_size)
        return expected_hidden_size

    def check_hidden_size(self, hx, expected_hidden_size, msg='Expected hidden size {}, got {}'):
        # type: (Tensor, Tuple[int, int, int], str) -> None
        if hx.size() != expected_hidden_size:
            raise RuntimeError(msg.format(expected_hidden_size, tuple(hx.size())))

    def check_forward_args(self, input, hidden, batch_sizes):
        # type: (Tensor, Tensor, Optional[Tensor]) -> None
        self.check_input(input, batch_sizes)
        expected_hidden_size = self.get_expected_hidden_size(input, batch_sizes)

        self.check_hidden_size(hidden, expected_hidden_size)

    def set_coefficients(self, coefficients):
        if coefficients is None:
            self.coefficients = None
        else:
            self.add_module('coefficients', coefficients)

    def flatten_parameters(self):
        """Resets parameter data pointer so that they can use faster code paths.

        Right now, this works only if the module is on the GPU and cuDNN is enabled.
        Otherwise, it's a no-op.
        """
        any_param = next(self.parameters()).data
        if not any_param.is_cuda or not torch.backends.cudnn.is_acceptable(any_param):
            return

        # If any parameters alias, we fall back to the slower, copying code path. This is
        # a sufficient check, because overlapping parameter buffers that don't completely
        # alias would break the assumptions of the uniqueness check in
        # Module.named_parameters().
        all_weights = self._flat_weights
        unique_data_ptrs = set(p.data_ptr() for p in all_weights)
        if len(unique_data_ptrs) != len(all_weights):
            return

        with torch.cuda.device_of(any_param):
            import torch.backends.cudnn.rnn as rnn

            # NB: This is a temporary hack while we still don't have Tensor
            # bindings for ATen functions
            with torch.no_grad():
                # NB: this is an INPLACE function on all_weights, that's why the
                # no_grad() is necessary.
                torch._cudnn_rnn_flatten_weight(
                    all_weights, (4 if self.bias else 2),
                    self.input_size, rnn.get_cudnn_mode(self.mode), self.latent_size, self.num_layers,
                    self.batch_first, bool(self.bidirectional))

    def permute_hidden(self, hx, permutation):
        # type: (Tensor, Optional[Tensor]) -> Tensor
        if permutation is None:
            return hx
        return apply_permutation(hx, permutation)

    def set_params(self):
        params = self.bank(self.layer_id, self.in_features, self.out_features, self.coefficients)
        if not isinstance(params, SGRU):
            return params

        params = params.t()

        ih_l0 = params[:, :self.input_size]
        hh_l0 = params[:, self.input_size:(self.input_size+self.latent_size)]
        bias_ih = params[:, -2:-1]
        bias_hh = params[:, -1:]
        if self.bidirectional:
            num_params = self.in_features // 2
            self.weight_hh_l0_reverse = hh_l0[num_params:].contiguous()
            hh_l0 = hh_l0[:num_params]
            self.weight_ih_l0_reverse = ih_l0[:num_params:].contiguous()
            ih_l0 = ih_l0[:num_params]

            self.bias_hh_l0_reverse = bias_hh[num_params:].contiguous()
            bias_hh = bias_hh[:num_params]
            self.bias_ih_l0_reverse = bias_ih[num_params:].contiguous()
            bias_ih = bias_ih[:num_params]

        self.weight_hh_l0 = hh_l0.contiguous()
        self.weight_ih_l0 = ih_l0.contiguous()

        self.bias_hh_l0 = bias_hh.contiguous()
        self.bias_ih_l0 = bias_ih.contiguous()

        self._flat_weights_names = []
        self._all_weights = []
        num_layers = self.num_layers
        num_directions = 2 if self.bidirectional else 1
        for layer in range(num_layers):
            for direction in range(num_directions):
                suffix = '_reverse' if direction == 1 else ''
                param_names = ['weight_ih_l{}{}', 'weight_hh_l{}{}']
                if self.bias:
                    param_names += ['bias_ih_l{}{}', 'bias_hh_l{}{}']
                param_names = [x.format(layer, suffix) for x in param_names]

                self._flat_weights_names.extend(param_names)
                self._all_weights.append(param_names)

        self._flat_weights = [(lambda wn: getattr(self, wn) if hasattr(self, wn) else None)(wn) for wn in self._flat_weights_names]
        #self.flatten_parameters()
        return None

    def forward(self, input, hx=None):
        if self.bank.single_layer:
            return self.bank._layer(input)

        layer = self.set_params()
        orig_input = input
        # xxx: isinstance check needs to be in conditional for TorchScript to compile
        if isinstance(orig_input, PackedSequence):
            input, batch_sizes, sorted_indices, unsorted_indices = input
            max_batch_size = batch_sizes[0]
            max_batch_size = int(max_batch_size)
        else:
            batch_sizes = None
            max_batch_size = input.size(0) if self.batch_first else input.size(1)
            sorted_indices = None
            unsorted_indices = None


        num_directions = 2 if self.bidirectional else 1
        if hx is None:
            num_directions = 2 if self.bidirectional else 1
            hx = torch.zeros(self.num_layers * num_directions,
                             max_batch_size, self.latent_size,
                             dtype=input.dtype, device=input.device)
        else:
            # Each batch of the hidden state should match the input sequence that
            # the user believes he/she is passing in.
            hx = self.permute_hidden(hx, sorted_indices)

        self.check_forward_args(input, hx, batch_sizes)
        if batch_sizes is None:
            result = _VF.gru(input, hx, self._flat_weights, self.bias, self.num_layers,
                             self.dropout, self.training, self.bidirectional, self.batch_first)
        else:
            result = _VF.gru(input, batch_sizes, hx, self._flat_weights, self.bias,
                             self.num_layers, self.dropout, self.training, self.bidirectional)
        output = result[0]
        hidden = result[1]

        # xxx: isinstance check needs to be in conditional for TorchScript to compile
        if isinstance(orig_input, PackedSequence):
            output_packed = PackedSequence(output, batch_sizes, sorted_indices, unsorted_indices)
            return output_packed, self.permute_hidden(hidden, unsorted_indices)
        else:
            return output, self.permute_hidden(hidden, unsorted_indices)


