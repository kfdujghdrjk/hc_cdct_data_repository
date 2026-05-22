import torch
import numpy as np


from models import *

torch.set_printoptions(precision=10)


def Q(s, delta):
    if isinstance(s, np.ndarray) or isinstance(s, np.float64):
        return np.round(s * delta) / delta
    elif isinstance(s, torch.Tensor):
        return torch.round(s * delta) / delta
    print(type(s))
    return torch.round(s * delta) / delta




def emb_Qim(s,d):

    Qs = Q(s-d,500)+d
    return Qs

def extract_Qim(SMME):
    Qf_SMME = Q(SMME,1000)
    d = Qf_SMME - Q(Qf_SMME, 500)
    return d


def emb(s,d,alpha,delta=500):
    Qs = Q(alpha*s-d,delta)+d
    SLCS = (1 - alpha) * s + Qs
    return SLCS,Qs



def emb_MME(s,d,alpha,detal=500):
    k = 1/(2*detal)
    Qs = Q(s - d-k,detal) + d+k
    SMME = (1 - alpha) * s + alpha *Qs
    return SMME,Qs

def extract(SLCS,alpha,delta=500):
    Qf_slcs = Q(alpha * SLCS,2*delta)
    d = Qf_slcs - Q(Qf_slcs,delta)
    return d

def extract_MME(SMME,alpha,detal=500):
    k = 1 / (2 * detal)
    Qf_SMME = Q(SMME-k,detal*2) + k
    d = Qf_SMME - (Q(Qf_SMME-k, detal)+k)
    return d

def restore(SLCS,alpha):
    ss = (SLCS-Q(alpha * SLCS,1000))/(1-alpha)
    return ss

def restore_MME(SMME,alpha):
    ss = (SMME-alpha * Q(SMME,1000))/(1-alpha)
    return ss

# d = extract(torch.tensor([0,0.00204581]),0.66)
# print(d)
# if __name__ == '__main__':
#     device = 'cuda:3' if torch.cuda.is_available() else 'cpu'
#
#     idx2d = torch.tensor([
#         [0.0, 0.0],
#         [0.0, 0.001],
#         [0.001, 0.0],
#         [0.001, 0.001],
#     ]).to(device)
#
#     idx2wm = {
#         0:torch.tensor([0,0]),
#         1:torch.tensor([1,0]),
#         2:torch.tensor([0,1]),
#         3:torch.tensor([1,1])
#     }
#
#
#     net = ResNet18().to(device)
#     checkpoint = torch.load('./checkpoint/ckpt.pth')
#     net.load_state_dict(checkpoint['net'])
#
#     embedding_layer = 'layer4.0.conv1.weight'
#
#     host_signal = net.state_dict()[embedding_layer]
#
#     shape = host_signal.shape
#     print(shape)
#     # host_signal = torch.tensor(host_signal)
#     host_signal = torch.reshape(host_signal,(-1,2))
#
#     watermark = torch.randint(0, 2, (host_signal.shape[0], 2))
#     watermark = watermark.to(device)
#     watermark_id = watermark[:, 0] * 2 + watermark[:, 1]
#     watermark_id = watermark_id.long()
#     d = idx2d[watermark_id]
#
#     watered_signal,_ = emb(host_signal,d,0.66)
#
#
#     d = extract(watered_signal,0.66)
#     water_estimated = torch.round(1000*d)%2
#
#     ber = (watermark!=water_estimated).sum().item()/(host_signal.shape[0]*2)
#     print(ber)
#     restore_signal = restore(watered_signal,0.66)
#
#     count = torch.isclose(restore_signal, host_signal, rtol=1e-6, equal_nan=False).sum().item()
#     print(restore_signal.shape)
#
#     print(count)