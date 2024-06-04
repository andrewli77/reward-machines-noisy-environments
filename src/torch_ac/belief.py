import torch
import itertools
from typing import Dict, Iterable, Tuple

def threshold_rm_beliefs(x):
    """
    Input
        x: rm_belief torch.Tensor with torch.Size([N, num_rm_states]).
    Output
        out: same type and size as the input
    """
    top1 = torch.max(x, dim=1, keepdim=True)[0]
    out = (top1 == x).float()
    return out


def _propositions_probability(s:str, probs:Dict[str, float]) -> float:
    ret = 1.
    for proposition in probs:
        if proposition in s:
            ret *= probs[proposition]
        else:
            ret *= 1 - probs[proposition]
    return ret

def possible_true_propositions(probs: Dict[str, float]) -> \
        Iterable[Tuple[str, float]]:
    """
    Outputs: Truth assignments (str) and its corresponding probability
    """
    full = itertools.product(*(["", p] for p in probs.keys()))
    full = map(lambda xs: str.join('', xs), full)
    full = ((s, _propositions_probability(s, probs)) for s in full)
    return filter(lambda sp: sp[1] > 0., full)

if __name__ == "__main__":
    noisy_props = {
        'a':0.2,
        'b':0.5,
    }
    for s in possible_true_propositions(noisy_props):
        print(s)
