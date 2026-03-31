from statistics import NormalDist
import math
from scipy.stats import laplace
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from collections import defaultdict
import numpy as np
from scipy.stats import halfnorm
import torch
import math

class CNN_network(nn.Module):
    structure = "cumulative"

    def __init__(self, data_size, num_bins,dataset,
                 connection="linear", gain=1,
                 linfunc="avg", mode=0,
                 device=None):
        super().__init__()

        # ---- device ----
        self.device = (
            torch.device(device)
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.data_size, self.num_bins, self.dataset = data_size, num_bins, dataset
        # ---- layers ----
        # --- Conv2D layer (first layer) ---     
       
        self.conv1=nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2=nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=1, bias=False)
        self.linear0 = nn.Linear(4*data_size, num_bins).to(self.device)

        self.nonlin  = nn.ReLU()
        


        if connection == "linear":
            self.linear2 = nn.Linear(num_bins, data_size).to(self.device)
            with torch.no_grad():
                self.linear2.weight.fill_(gain)
                self.linear2.bias.zero_()
        self.connection = connection

        # ---- deterministic init ----
        self.bins = self._get_bins(linfunc)

        with torch.no_grad():
            w0 = self._init_linear_function_first('avg', mode).to(self.device) 
            b0 = self._make_biases().to(self.device)            
            w_conv1=self._init_conv_weights().to(self.device)  
            w_conv2=self._init_conv2_weights().to(self.device) 
            self.linear0.weight.copy_(w0)
            self.linear0.bias.copy_(b0)
            self.conv1.weight.copy_(w_conv1)
            self.conv2.weight.copy_(w_conv2)

    def _init_conv_weights(self, init_type="random"):
        """
        Initialize self.conv.weight.data according to init_type.
        
        Args:
            init_type (str): Type of initialization, e.g., "avg", "pca", "random"
        """
        # Get conv weight shape: (out_channels, in_channels, kernel_H, kernel_W)
        out_ch, in_ch, kH, kW = self.conv1.weight.data.shape
        
        if init_type == "avg":
            # Set all values to 1 / (in_ch * kH * kW)
            val = 1.0 / (in_ch * kH * kW)
            weights = torch.full((out_ch, in_ch, kH, kW), val, dtype=self.conv1.weight.dtype, device=self.conv1.weight.device)

        elif init_type == "random":
            # Random normal init
            weights = torch.randn(out_ch, in_ch, kH, kW, dtype=self.conv1.weight.dtype, device=self.conv1.weight.device) * 0.1
            # Per-filter normalization:
            filter_norms = weights.view(out_ch, -1).norm(p=2, dim=1, keepdim=True)
            weights = weights / filter_norms.view(out_ch, 1, 1, 1)
        elif init_type == "identity":
            if in_ch != out_ch or kH != kW or kH % 2 == 0:
                raise ValueError("Identity init requires square odd kernel and equal in/out channels.")
            weights = torch.zeros(out_ch, in_ch, kH, kW, dtype=self.conv1.weight.dtype, device=self.conv1.weight.device)
            center = kH // 2
            for i in range(out_ch):
                weights[i, i, center, center] = 1.0
        elif init_type=='Kaiming':
            fan_in = in_ch * kH * kW
            std = 1.0 / fan_in**0.5  # standard normal output preservation
            
            weights = torch.randn(out_ch, in_ch, kH, kW,
                                dtype=self.conv1.weight.dtype,
                                device=self.conv1.weight.device) * std
        else:
            raise ValueError(f"Unknown init_type '{init_type}'")

        return weights
    def _init_conv2_weights(self, init_type="identity"):
            """
            Initialize self.conv.weight.data according to init_type.
            
            Args:
                init_type (str): Type of initialization, e.g., "avg", "pca", "random"
            """
            # Get conv weight shape: (out_channels, in_channels, kernel_H, kernel_W)
            out_ch, in_ch, kH, kW = self.conv2.weight.data.shape
            
            if init_type == "avg":
                # Set all values to 1 / (in_ch * kH * kW)
                val = 1.0 / (in_ch * kH * kW)
                weights = torch.full((out_ch, in_ch, kH, kW), val, dtype=self.conv1.weight.dtype, device=self.conv1.weight.device)

            elif init_type == "random":
                # Random normal init
                weights = torch.randn(out_ch, in_ch, kH, kW, dtype=self.conv1.weight.dtype, device=self.conv1.weight.device) * 0.1

            elif init_type == "identity":
                if in_ch != out_ch or kH != kW or kH % 2 == 0:
                    raise ValueError("Identity init requires square odd kernel and equal in/out channels.")
                weights = torch.zeros(out_ch, in_ch, kH, kW, dtype=self.conv1.weight.dtype, device=self.conv1.weight.device)
                center = kH // 2
                for i in range(out_ch):
                    weights[i, i, center, center] = 1.0

            else:
                raise ValueError(f"Unknown init_type '{init_type}'")

            return weights

    def _get_bins(self, linfunc="avg"):
        bins = []
        if linfunc=='avg' or linfunc== 'pca'or linfunc=='gauss':       
            mass_per_bin = 1 / (self.num_bins)
            bins.append(-0.1)  # -Inf is not great here, but NormalDist(mu=0, sigma=1).cdf(10) approx 1
            for i in range(1, self.num_bins):
                if "fourier" in linfunc:
                    bins.append(laplace(loc=0.0, scale=1 / math.sqrt(2)).ppf(i * mass_per_bin))
                else:
                    # p = i * mass_per_bin
                    # value = norm.ppf(p, loc=-3.3478517025287147e-07, scale=6.05e-14)
                    # bins.append(value)
                    
                    bins.append(NormalDist().inv_cdf(i * mass_per_bin))
        # print('first bins:',bins)
        # print('first bins:',len(bins))
    

        return bins
    
 
    def _init_linear_function_first(self, linfunc="avg", mode=0):
        K, N = self.num_bins, self.data_size
        if linfunc == "avg":
            weights = torch.ones_like(self.linear0.weight.data) / N   
        elif linfunc == "fourier":
            weights = torch.cos(math.pi / N * (torch.arange(0, N) + 0.5) * mode).repeat(K, 1) / N * max(mode, 0.33) * 4
        elif linfunc == "randn":
            weights = torch.randn(N).repeat(K, 1)
            std, mu = torch.std_mean(weights[0])  # Enforce mean=0, std=1 with higher precision
            weights = (weights - mu) / std / math.sqrt(N)  # Move to std=1 in output dist
        elif linfunc == "rand":
            weights = torch.rand(N).repeat(K, 1)  # This might be a terrible idea haven't done the math
            std, mu = torch.std_mean(weights[0])  # Enforce mean=0, std=1
            weights = (weights - mu) / std / math.sqrt(N)  # Move to std=1 in output dist
        

        else:
            raise ValueError(f"Invalid linear function choice {linfunc}.")
        
        

        return weights
    
    def _make_biases(self):
        new_biases = torch.zeros_like(self.linear0.bias.data)
        for i in range(new_biases.shape[0]):
            new_biases[i] = -self.bins[i]
        return new_biases
    
    def forward(self, x):
        if self.dataset=='CIFAR10':
            x = x.view(x.size(0), 3, 32, 32)  # (B, C, H, W)
        else:
            x = x.view(x.size(0), 3, 224, 224)
            
        x = self.conv1(x)  
        x = self.nonlin(x)
       
        x = self.conv2(x)            
        x = self.nonlin(x)  

        x = x.view(x.size(0), -1)   
        
        x = self.linear0(x)
        x = self.nonlin(x)

        output = self.linear2(x)
        return output

