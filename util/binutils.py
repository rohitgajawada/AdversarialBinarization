import torch
import torch.nn as nn
import copy

def binarizeConvParams(convNode, bnnModel):
    """
    Binarizes a given Conv layer
    """
    s = convNode.weight.data.size()
    n = s[1]*s[2]*s[3]
    if bnnModel:
      convNode.weight.data[convNode.weight.data.eq(0)] = -1e-6
      convNode.weight.data = convNode.weight.data.sign()
    else:
      m = convNode.weight.data.norm(1, 3, keepdim=True).sum(2, keepdim=True).sum(1, keepdim=True).div(n)
      orig_alpha = copy.deepcopy(m[0][0][0])

      convNode.weight.data[convNode.weight.data.eq(0)] = -1e-6
      # Used for visualizing a specific set of weight filters
      cur = convNode.weight.data[6].sign()
      cur[cur.lt(0)] = 0

      convNode.weight.data = convNode.weight.data.sign().mul(m.repeat(1, s[1], s[2], s[3]))
      # Return values for visualizations
      return cur.sum(), orig_alpha

def updateBinaryGradWeight(convNode, bnnModel):
    """
    Assign gradients to binarized Conv layer used for weight updates
    """
    s = convNode.weight.data.size()
    n = s[1]*s[2]*s[3]
    m = convNode.weight.data.clone()
    if bnnModel:
        m = convNode.weight.data.clone().fill(1)
        m[convNode.weight.data.le(-1)] = 0
        m[convNode.weight.data.ge(1)] = 0
        m = torch.mul(1 - 1.0/s[1])
    else:
        m = convNode.weight.data.norm(1, 3, keepdim=True).sum(2, keepdim=True).sum(1, keepdim=True).div(n).repeat(1, s[1], s[2], s[3])
        m[convNode.weight.data.le(-1)] = 0
        m[convNode.weight.data.ge(1)] = 0
        m = torch.add(m, 1.0/n)
        m = torch.mul(m, 1.0 - 1.0/s[1])
        m = torch.mul(m, n)
    convNode.weight.grad.data.mul_(m)

def meancenterConvParams(convNode):
    """
    Mean-center the weights of the given Conv Node
    """
    s = convNode.weight.data.size()
    negMean = torch.mul(convNode.weight.data.mean(1,keepdim=True),-1).repeat(1,1,1,1)
    #print(negMean.size())
    negMean = negMean.repeat(1, s[1], 1, 1)
    #print(negMean.size(),convNode.weight.data.size())
    convNode.weight.data.add_(negMean)

def clampConvParams(convNode):
    """
    Clamp the weights of a given Conv layer
    """
    convNode.weight.data.clamp_(-1, 1)
