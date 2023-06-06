import torch
from torch import nn
from copy import deepcopy
import torch.nn.functional as F
import matplotlib.pyplot as plt


class LRP(nn.Module):
    
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model
        self.model.eval()
        
        self.eps = 1.0e-05
        self.layers = self.get_layers(self.model)
    
    def get_layers(self, model):
        children = list(model.children())
        if len(children) == 0: return [model]  
        layers = []
        for child in children:
            for layer in self.get_layers(child):
                if layer.__class__.__name__ == 'LSTM':
                    layers.append(self.divide_lstm(layer))
                else:
                    layers.append(layer)
        return layers

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
            
    
    def lrp_linear(self, h_in, weight, bias, h_out, relevance):
        z = h_out + self.eps
        s = relevance / z
        if len(s.shape) > 2: s = s.squeeze()
        if len(s.shape) == 1: s = s.unsqueeze(0)
        c = torch.mm(s, torch.squeeze(weight).float())
        r = (h_in * c).data
        return r
    
    
    def lrp_conv(self, h_in, weight, bias, h_out, relevance):
        z = h_out + self.eps
        s = (relevance / z).data
        (z * s).sum().backward()
        c = h_in.grad
        r = (h_in * c).data
        return r
    
    
    def get_features(self, x: torch.tensor, visualize=False):
        layers = deepcopy(self.layers)
        with torch.no_grad():  # можем просто torch.inference_mode
            activations = [torch.ones_like(x)]
            lstm_aug = []
            for i, layer in enumerate(layers):
                if type(layer) == list: 
                    h_n, c_n = torch.zeros(1, layer[0].hidden_size), torch.zeros(1, layer[0].hidden_size)
                    augmentations = [(h_n, c_n)]
                    
                    for lstm_layer in layer:
                        x, (h_n, c_n) =  lstm_layer.forward(x, (h_n, c_n))
                        augmentations.append((h_n, c_n))
                        activations.append(x)
                    lstm_aug.append(augmentations[::-1])
                else: 
                    x = layer.forward(x)
                    activations.append(x)
                    lstm_aug.append(None)
                    
        
        activations = [a.data.requires_grad_(True) for a in activations]
        relevance = activations[-1]
        relevance_history = [relevance]
        
        lstm_aug = lstm_aug[::-1]
        activations = activations[::-1][1:]
        act_in = activations.pop(0)
        
        for i, layer in enumerate(layers[::-1]):
            if layer.__class__.__name__ == 'Linear':
                layer.weight = torch.nn.Parameter(layer.weight.clamp(min=0.0))
                layer.bias = torch.nn.Parameter(torch.zeros_like(layer.bias))
                act_out = layer.forward(act_in)
                
                with torch.no_grad():
                    relevance = self.lrp_linear(act_in, layer.weight, layer.bias, act_out, relevance_history[-1])
                    
            elif layer.__class__.__name__ in ['Conv2d', 'AvgPool2d', 'MaxPool2d', 'AdaptiveAvgPool2d']:
                layer.weight = torch.nn.Parameter(layer.weight.clamp(min=0.0))
                layer.bias = torch.nn.Parameter(torch.zeros_like(layer.bias))
                act_out = layer.forward(act_in)
                
                relevance = self.lrp_conv(act_in, layer.weight, layer.bias, act_out, relevance_history[-1])

            elif layer.__class__.__name__  in ['Dropout', 'ReLU', 'Sigmoid', 'BatchNorm2d']:
                relevance = relevance_history[-1]

            elif layer.__class__.__name__ == 'Flatten':
                with torch.no_grad():
                    relevance = relevance_history[-1].view(size=act_in.shape)
                    
            elif type(layer) == list:
                for k, rnn in enumerate(layer[::-1]):
                    relevance = relevance_history[-1]
                    hn, cn = lstm_aug[i][k+1]

                    W_ii, W_if, W_ig, W_io = rnn.weight_ih_l0.chunk(4, 0)
                    W_hi, W_hf, W_hg, W_ho = rnn.weight_hh_l0.chunk(4, 0)
                    b_ii, b_if, b_ig, b_io = rnn.bias_ih_l0.chunk(4, 0)
                    b_hi, b_hf, b_hg, b_ho = rnn.bias_hh_l0.chunk(4, 0)
                    
                    f_t = nn.Sigmoid()(act_in @ W_if.T + hn @ W_hf + b_if + b_hf)
                    i_t = nn.Sigmoid()(act_in @ W_ii.T + hn @ W_hi + b_ii + b_hi)
                    g_t = torch.tanh(act_in @ W_ig.T + hn @ W_hg + b_ig + b_hg)
                    
                    eye = torch.eye(cn.shape[-1], dtype=torch.float64)
                    h_n_out, c_n_out = lstm_aug[i][k]
                    bias = None
                    
                    relevance_channel  = self.lrp_linear(f_t * cn, eye, bias, c_n_out, relevance)
                    relevance_new_info = self.lrp_linear(i_t * g_t, eye, bias, c_n_out, relevance)
                    relevance_hidden   = self.lrp_linear(hn, W_hg, bias, g_t, relevance_new_info)
                    relevance_input    = self.lrp_linear(act_in, W_ig, bias, g_t, relevance_new_info)
                    
                    if k + 1 == len(layer):
                        relevance = relevance_input
                    else:
                        relevance = relevance_channel + relevance_hidden
                        act_in = activations.pop(0)
            else:
                relevance = relevance_history[-1]
                print('Layer {0} was not identified'.format(layer.__class__.__name__))

            if i != len(layers) - 1: act_in = activations.pop(0)
            relevance_history.append(relevance)
        if visualize: self.visualize(relevance)
        return relevance
    
    
    def visualize(self, relevance, xlabel='Omics features', ylabel='Nucleotides', save_id='relevance'):
        plt.rcParams['axes.labelsize'] = 24
        plt.rcParams['axes.titlesize'] = 24
        fig, ax = plt.subplots(figsize=(12, 10))
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(5)
            ax.spines[axis].set_color("black")
        
        relevance = relevance.squeeze()
        if len(relevance.shape) > 2: 
            print('Input relevance tensor has {0} dimensions, \
                  when maximum of two is available').format(len(relevance.shape))
            return None
        if len(relevance.shape) == 1: relevance.unsqueeze(0)
        
        sns.heatmap(relevance.cpu(), cmap="crest", ax=ax)
        ax.set(xlabel=xlabel, ylabel=ylabel, title='Relevance visualization')
        plt.savefig('{0}.pdf'.format(save_id, dpi=100))
        plt.show()
        return None
        
    def aggregate(self, dataset, visualize=False):
        agg_rel = torch.zeros(dataset[0].shape)
        for sample in dataset:
            relevance = self.get_features(sample)
            agg_rel += relevance
        
        mean_rel = agg_rel / len(dataset)
        if visualize: self.visualize(mean_rel, save_id='agg_relevance')
        return mean_rel
