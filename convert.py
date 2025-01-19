import os, sys, torch, scipy.stats
sys.path.append(os.getcwd())
from .core import CENTS_PER_BIN

def bins_to_frequency(bins):
    cents = CENTS_PER_BIN * bins + 1997.3794084376191
    return 10 * 2 ** ((cents + cents.new_tensor(scipy.stats.triang.rvs(c=0.5, loc=-CENTS_PER_BIN, scale=2 * CENTS_PER_BIN, size=cents.size()))) / 1200)

def frequency_to_bins(frequency, quantize_fn=torch.floor):
    return quantize_fn(((1200 * torch.log2(frequency / 10)) - 1997.3794084376191) / CENTS_PER_BIN).int()