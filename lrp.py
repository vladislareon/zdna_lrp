import torch
from torch import nn
import torch.nn.functional as F
from copy import deepcopy


class LRP(nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model
        self.model.eval()
        self.eps = 1.0e-05
        self.layers = self.get_layers(self.model)

    def divide_lstm(self, lstm):
        num_layers = lstm.num_layers
        input_size = lstm.input_size
        hidden_size = lstm.hidden_size
        bi = lstm.bidirectional
        params_dict = lstm.state_dict()

        layers = []
        for i in range(num_layers):
            layer = nn.LSTM(input_size, hidden_size, bidirectional=bi)
            layer.load_state_dict(
                {
                    "weight_ih_l0": params_dict["weight_ih_l{}".format(i)],
                    "weight_hh_l0": params_dict["weight_hh_l{}".format(i)],
                    "bias_ih_l0": params_dict["bias_ih_l{}".format(i)],
                    "bias_hh_l0": params_dict["bias_hh_l{}".format(i)],
                }
            )
            layers.append(layer)
            input_size = hidden_size * (bi + 1)
        return layers

    def get_layers(self, model):
        children = list(model.children())
        if len(children) == 0:
            return [model]
        layers = []
        for child in children:
            for layer in self.get_layers(child):
                if layer.__class__.__name__ == "LSTM":
                    layers.append(self.divide_lstm(layer))
                else:
                    layers.append(layer)
        return layers

    def lrp_lstm_back_prop(self, h_in, weight, h_out, r):
        with torch.no_grad():
            z = h_out + self.eps
            s = r / z.cuda()
            c = torch.mm(torch.squeeze(s), torch.squeeze(weight.cuda()).float())
            r = (h_in * c).data
            return r

    def get_features(self, x: torch.tensor):
        layers = deepcopy(self.layers)
        activations = list()
        with torch.no_grad():
            augmentation = []
            activations.append(torch.ones_like(x))
            if type(layers[0]) != list:
                augmentation.append(([0], [0]))
            for layer in layers:
                if type(layer) == list:
                    augmentation.append(
                        (
                            torch.zeros(x.size(0), x.size(1), layer[0].hidden_size),
                            torch.zeros(x.size(0), x.size(1), layer[0].hidden_size),
                        )
                    )
                    x, (h_n, c_n) = layer[0].cuda().forward(x)
                    augmentation.append((h_n, c_n))
                    activations.append(x)

                    for lstm_layer in layer[1:]:
                        x, (h_n, c_n) = lstm_layer.cuda().forward(x, (h_n, c_n))
                        augmentation.append((h_n, c_n))
                        activations.append(x)

                else:
                    augmentation.append(([0], [0]))
                    x = layer.forward(x)
                    activations.append(x)
        augmentation = augmentation[::-1]
        activations = activations[::-1]
        activations = [a.data.requires_grad_(True) for a in activations]
        relevance = activations.pop(0)
        augmentation.pop(0)
        relevance_list = list()
        relevance_list.append(relevance)
        for i, layer in enumerate(layers[::-1]):
            if type(layer) == list:
                layer_back = layer[::-1]
                for k, rnn in enumerate(layer_back):
                    r = relevance_list[-1]
                    a = activations.pop(0)
                    hn, cn = augmentation.pop(0)
                    hn = hn.cuda()
                    cn = cn.cuda()
                    W_ii, W_if, W_ig, W_io = rnn.weight_ih_l0.chunk(4, 0)
                    W_hi, W_hf, W_hg, W_ho = rnn.weight_hh_l0.chunk(4, 0)
                    b_ii, b_if, b_ig, b_io = rnn.bias_ih_l0.chunk(4, 0)
                    b_hi, b_hf, b_hg, b_ho = rnn.bias_hh_l0.chunk(4, 0)
                    f_t = nn.Sigmoid()(a @ W_if.T + hn @ W_hf + b_if + b_hf)
                    f_t = nn.Sigmoid()(a @ W_if.T + hn @ W_hf + b_if + b_hf)
                    i_t = nn.Sigmoid()(a @ W_ii.T + hn @ W_hi + b_ii + b_hi)
                    g_t = torch.tanh(a @ W_ig.T + hn @ W_hg + b_ig + b_hg)
                    eye = torch.eye(cn.shape[-1], dtype=torch.float64)
                    h_n_out, c_n_out = rnn.cuda().forward(a, (h_n, c_n))[1]
                    relevance_channel = self.lrp_lstm_back_prop(
                        f_t * cn, eye, c_n_out, r.cuda()
                    )
                    relevance_new_info = self.lrp_lstm_back_prop(
                        i_t * g_t, eye, c_n_out, r.cuda()
                    )
                    relevance_hidden = self.lrp_lstm_back_prop(
                        hn, W_hg, g_t, relevance_new_info
                    )
                    relevance_input = self.lrp_lstm_back_prop(
                        a, W_ig, g_t, relevance_new_info
                    )
                    if k + 1 == len(layer_back):
                        relevance_list.append(relevance_input)
                    else:
                        relevance_list.append(relevance_channel + relevance_hidden)

            elif layer.__class__.__name__ in ["Linear"]:
                layer.weight = torch.nn.Parameter(layer.weight.clamp(min=0.0))
                layer.bias = torch.nn.Parameter(torch.zeros_like(layer.bias))
                with torch.no_grad():
                    a = activations.pop(0)
                    r = relevance_list[-1]
                    z = layer.forward(a) + self.eps
                    s = r / z
                    c = torch.mm(torch.squeeze(s), torch.squeeze(layer.weight))
                    r = (a * c).data
                    aug = augmentation.pop(0)
                    relevance_list.append(r)
            elif layer.__class__.__name__ in ["Conv2d"]:
                a = activations.pop(0)
                r = relevance_list[-1]
                layer.weight = torch.nn.Parameter(layer.weight.clamp(min=0.0))
                layer.bias = torch.nn.Parameter(torch.zeros_like(layer.bias))
                z = layer.forward(a) + self.eps
                s = (r / z).data
                (z * s).sum().backward()
                c = a.grad
                relevance_list.append((a * c).data)
                aug = augmentation.pop(0)

            elif layer.__class__.__name__ in [
                "Dropout",
                "ReLU",
                "Sigmoid",
                "BatchNorm2d",
            ]:
                activations.pop(0)
                aug = augmentation.pop(0)
            elif layer.__class__.__name__ in ["Flatten"]:
                with torch.no_grad():
                    a = activations.pop(0)
                    r = relevance_list[-1]
                    relevance_list.append(r.view(size=a.shape))
                    aug = augmentation.pop(0)
            elif layer.__class__.__name__ in [
                "AvgPool2d",
                "MaxPool2d",
                "AdaptiveAvgPool2d",
            ]:
                a = activations.pop(0)
                r = relevance_list[-1]
                z = layer.forward(a) + self.eps
                s = (r / z).data
                (z * s).sum().backward()
                c = a.grad
                relevance_list.append((a * c).data)
                aug = augmentation.pop(0)
            else:
                activations.pop(0)
                aug = augmentation.pop(0)
                print(f"{layer.__class__} does not support")
        return relevance_list[-1]
