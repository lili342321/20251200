import plotly.graph_objects as ply
from plotly.subplots import make_subplots
import numpy as np
from math import pi, sqrt, exp, sin
import torch

from math import sqrt, pi, exp, sin



def gamma_x(z,z1,all_list,condition):

    # Lanczos approximation
    # https://en.wikipedia.org/wiki/Lanczos_approximation
    g = 7
    p = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7
    ]

    if (z<0.5).any():

        if (z == 0).any():
            condition_z_0 = z == 0
            z[condition_z_0] = 0.00000001
        condition_1 = z<0.5
        store_z_condition = condition_1


        z[condition_1] = pi / (torch.sin(pi * z[condition_1]) * gamma_x((1 - z[condition_1]),(1 - z[condition_1]),all_list,condition))

        all_list[condition] = z[store_z_condition]
        return all_list
    else:

        condition_2 = z1 >= 0.5
        store_z1_condition = condition_2
        z1[condition_2]-=1
        x = p[0]
        for i in range(1, len(p)):
            x += p[i] / (z1[condition_2] + i)
        t = z1[condition_2] + g + 0.5
        y = torch.sqrt(2 * torch.tensor(pi)) * t ** (z1[condition_2] + 0.5) * torch.exp(-t) * x

        all_list[condition] = y
        return y



def gamma_D(z,z1,all_list,condition):

    # Lanczos approximation
    # https://en.wikipedia.org/wiki/Lanczos_approximation
    g = 7
    p = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7
    ]



    z-=1
    x = p[0]
    for i in range(1, len(p)):
        x += p[i] / (z + i)
    t = z + g + 0.5
    y = torch.sqrt(2 * torch.tensor(pi)) * t ** (z + 0.5) * torch.exp(-t) * x

    all_list[condition] = y
    return all_list

def gamma_all(input,input1):
    A_input = torch.zeros(input.size(0),input.size(1))
    A_input = A_input.to('cuda')
    B_input = torch.zeros(input.size(0), input.size(1))
    B_input = B_input.to('cuda')
    ALL = B_input
    condition_D = input >= 0.5
    condition_X = input < 0.5
    A_input = gamma_D(input[condition_D], input1[condition_D], A_input, condition_D)
    B_input = gamma_x(input[condition_X], input1[condition_X], B_input, condition_X)
    ALL[condition_D] = A_input[condition_D]
    if len(B_input) == 0:
        return ALL
    else:
        ALL[condition_X] = B_input[condition_X]

        return  ALL







