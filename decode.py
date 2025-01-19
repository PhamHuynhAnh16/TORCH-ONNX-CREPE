import os, sys, torch, librosa, numpy as np

sys.path.append(os.getcwd())

from .convert import bins_to_frequency


def viterbi(logits):
    if not hasattr(viterbi, 'transition'):
        xx, yy = np.meshgrid(range(360), range(360))
        transition = np.maximum(12 - abs(xx - yy), 0)

        viterbi.transition = transition / transition.sum(axis=1, keepdims=True)

    with torch.no_grad():
        probs = torch.nn.functional.softmax(logits, dim=1)

    bins = torch.tensor(np.array([librosa.sequence.viterbi(sequence, viterbi.transition).astype(np.int64) for sequence in probs.cpu().numpy()]), device=probs.device)

    return bins, bins_to_frequency(bins)