import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib
from matplotlib import cm
import scipy.io
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
from matplotlib.colors import LogNorm
from random import choice

# setting to cuda

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# generate dataset
def generate_data(m):
    x = torch.rand(m, 1) * 2 - 0.5
    y = torch.rand(m, 1) * 2 - 0.5
    nu = 1.0 /40
    kexi = 1 / 2 / nu - np.sqrt(1 / 4 / nu / nu +4 * np.pi * np.pi)
    z = y - 1.0 / 2 / torch.pi * torch.exp(kexi * x) * torch.sin(2* torch.pi * y)
    return x.requires_grad_(True), y.requires_grad_(True), z

class nn_u(torch.nn.Module):
    '''
    This function is the discriminator and will be the function that will give us the test function. This model can
    intake an arbitrarily long list of inputs but all the lists need to be equally long. The input shape is [L, C, T]
    where L is the number of points, C is the number of dimensions and T is the number of
    time points.

    config: dictionary containing all the hyperparameters ('v_layers' and 'v_hidden_dim' for the
                    discriminator)
    setup: dictionary containing information of the problem
    '''

    def __init__(self):
        super(nn_u, self).__init__()
        self.num_layers = 6
        self.hidden_dim = 30
        self.input = torch.nn.Linear(2, self.hidden_dim)
        self.hidden = torch.nn.Linear(self.hidden_dim, self.hidden_dim)
        self.output = torch.nn.Linear(self.hidden_dim, 1)
        self.net = torch.nn.Sequential(*[
            self.input,
            *[torch.nn.Tanh(), self.hidden] * self.num_layers,
            torch.nn.Tanh(),
            self.output
        ])
#         self.net.double()

    def forward(self, x):
        x = self.net(x)
        return x

    def backward(self, retain_graph=True):
        self.loss.backward(retain_graph=retain_graph)
        return (self.loss)

class nn_v(torch.nn.Module):
    '''
    This function is the discriminator and will be the function that will give us the test function. This model can
    intake an arbitrarily long list of inputs but all the lists need to be equally long. The input shape is [L, C, T]
    where L is the number of points, C is the number of dimensions and T is the number of
    time points.

    config: dictionary containing all the hyperparameters ('v_layers' and 'v_hidden_dim' for the
                    discriminator)
    setup: dictionary containing information of the problem
    '''

    def __init__(self):
        super(nn_v, self).__init__()
        self.num_layers = 6
        self.hidden_dim = 50
        self.input = torch.nn.Linear(2, self.hidden_dim)
        self.hidden = torch.nn.Linear(self.hidden_dim, self.hidden_dim)
        self.output = torch.nn.Linear(self.hidden_dim, 1)
        self.net = torch.nn.Sequential(*[
            self.input,
            *[torch.nn.Tanh(), self.hidden] * self.num_layers,
            torch.nn.Tanh(),
            self.output
        ])
#         self.net.double()

    def forward(self, x):
        x = self.net(x)
        return x

    def backward(self, retain_graph=True):
        self.loss.backward(retain_graph=retain_graph)
        return (self.loss)


if __name__ == '__main__':
    m = 1500
    lr_u1 = 0.001
    lr_u2 = 4e-4
    lr_v1 = 0.001
    lr_v2 = 2e-4
    steps = 20001
    x_d, y_d, z_d = generate_data(m)
    #     u = dde.nn.FNN([2] + 6 * [20] + [1], "tanh", "Glorot normal")
    u = nn_u()
    #     u = nn_u(input_size=2, hidden_size=50, output_size=1)
    v = nn_v()
    w_1, w_2, w_3, w_4 = [1, 1e+4, 1e+4, 1e+4]

    optimizer_u1 = torch.optim.Adam(u.parameters(), lr_u1)
    #     optimizer_u2 = torch.optim.Adam(u.parameters(), lr_u2)
    optimizer_v1 = torch.optim.RMSprop(v.parameters(), lr_v1)
    #     optimizer_v2 = torch.optim.RMSprop(v.parameters(), lr_v2)
    loss = torch.nn.MSELoss()
    loss_function_u = []


    # sample
    def interior(n=m):
        x = torch.rand(n, 1) * 2 - 0.5
        y = torch.rand(n, 1) * 2 - 0.5
        return x.requires_grad_(True), y.requires_grad_(True)


    def gradients(u, x, order=1):
        if order == 1:
            return torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                                       create_graph=True,
                                       only_inputs=True, allow_unused=True)[0]
        else:
            return gradients(gradients(u, x), x, order=order - 1)


    # loss 1
    def l_interior(u):
        x, y = interior()
        nu = 1.0 / 40
        kexi = 1 / 2 / nu - np.sqrt(1 / 4 / nu / nu + 4 * np.pi * np.pi)
        uxy = u(torch.cat([x, y], dim=1))
        vxy = v(torch.cat([x, y], dim=1))
        lu = gradients(uxy, x, 2) + gradients(uxy, y, 2)
        lv = gradients(vxy, x, 2) + gradients(vxy, y, 2)
        ux = gradients(uxy, x, 1)
        uy = gradients(uxy, y, 1)
        vx = gradients(vxy, x, 1)
        vy = gradients(vxy, y, 1)
        int_1 = torch.mean(torch.mul(lu * lv * nu, torch.cat([x, y], dim=1)))
        int_2 = torch.mean(torch.mul(uy * lu * vx, torch.cat([x, y], dim=1)))
        int_3 = torch.mean(torch.mul(ux * lu * vy, torch.cat([x, y], dim=1)))
        #         v_norm = torch.sqrt(vxy ** 2 + vx ** 2+ vy ** 2 + lv **2)
        v_norm = torch.mean(vxy ** 2)
        return ((int_1 + int_2 - int_3) ** 2) / v_norm


    def bound_1(n=100):
        x1 = torch.rand(n, 1) * 2 - 0.5
        x2 = torch.rand(n, 1) * 2 - 0.5
        y1 = torch.ones_like(x1) * (-0.5)
        y2 = torch.ones_like(x2) * 1.5
        y3 = torch.rand(n, 1) * 2 - 0.5
        x3 = torch.ones_like(y3) * (-0.5)
        y4 = torch.rand(n, 1) * 2 - 0.5
        x4 = torch.ones_like(y4) * 1.5
        x = torch.cat([x1, x2, x3, x4], dim=0)
        y = torch.cat([y1, y2, y3, y4], dim=0)
        nu = 1.0 / 40
        kexi = 1 / 2 / nu - np.sqrt(1 / 4 / nu / nu + 4 * np.pi * np.pi)
        cond_psi = y - 1.0 / 2 / torch.pi * torch.exp(kexi * x) * torch.sin(2 * torch.pi * y)
        cond_u = 1 - torch.exp(kexi * x) * torch.cos(2 * torch.pi * y)
        cond_v = kexi / 2 / torch.pi * torch.exp(kexi * x) * torch.sin(2 * torch.pi * y)
        return x.requires_grad_(True), y.requires_grad_(True), cond_psi, cond_u, cond_v


    def right(n=100):
        y = torch.rand(n, 1) * 2 - 0.5
        x = torch.ones_like(y) * 1.5
        nu = 1.0 / 40
        kexi = 1 / 2 / nu - np.sqrt(1 / 4 / nu / nu + 4 * np.pi * np.pi)
        cond = y - 1.0 / 2 / torch.pi * torch.exp(kexi * x) * torch.sin(2 * torch.pi * y)
        return x.requires_grad_(True), y.requires_grad_(True), cond


    # losses 2-7
    def l_bound1(u):
        x, y, cond_psi, cond_u, cond_v = bound_1()
        uxy = u(torch.cat([x, y], dim=1))
        return loss(uxy, cond_psi)


    def l_bound2(u):
        x, y, cond_psi, cond_u, cond_v = bound_1()
        uxy = u(torch.cat([x, y], dim=1))
        bound_u = gradients(uxy, y, 1)
        return loss(bound_u, cond_u)


    def l_bound3(u):
        x, y, cond_psi, cond_u, cond_v = bound_1()
        uxy = u(torch.cat([x, y], dim=1))
        bound_v = - gradients(uxy, x, 1)
        return loss(bound_v, cond_v)


    #     lossu=1000.1
    #     i=0
    #     while lossu>1:
    for i in range(steps):

        for j in range(2):
            optimizer_u1.zero_grad()
            lossu = w_1 * l_interior(u) + w_2 * l_bound1(u) + w_3 * l_bound2(u) + w_4 * l_bound3(u)
            lossu.backward()
            optimizer_u1.step()

        optimizer_v1.zero_grad()
        lossv = - l_interior(u)
        lossv.backward()
        optimizer_v1.step()
        if i % 500 == 0:
            print(f'step: {i}  loss_u = {lossu.item()} loss_v = {lossv.item()}')

    #     for i in range(steps):

    #             for j in range(2):
    #                 optimizer_u2.zero_grad()
    #                 lossu = w_1 *l_interior(u)+w_2 * l_bound1(u) + w_3 * l_bound2(u) + w_4 * l_bound3(u)
    #                 lossu.backward()
    #                 optimizer_u2.step()

    #             optimizer_v2.zero_grad()
    #             lossv = - l_interior(u)
    #             lossv.backward()
    #             optimizer_v2.step()
    #             if i % 500 == 0 or i==steps:
    #                 print(f'step: {i + steps}  loss_u = {lossu.item()} loss_v = {lossv.item()}')
    # #         i+=1

    x_tu = torch.arange(-0.5, 1.5, 1 / 50)
    x_tu = torch.unsqueeze(x_tu, dim=1).requires_grad_(True)
    y_tu = -0.5 * torch.ones(100)
    y_tu = torch.unsqueeze(y_tu, dim=1).requires_grad_(True)
    for i in range(99):
        xi_tu = torch.arange(-0.5, 1.5, 1 / 50)
        xi_tu = torch.unsqueeze(xi_tu, dim=1)
        x_tu = torch.cat((x_tu, xi_tu), dim=0)
    for j in range(99):
        yj_tu = -0.5 * torch.ones(100) + j / 50 * torch.ones(100)
        yj_tu = torch.unsqueeze(yj_tu, dim=1)
        y_tu = torch.cat((y_tu, yj_tu), dim=0)
    nu = 1.0 / 40
    kexi = 1 / 2 / nu - np.sqrt(1 / 4 / nu / nu + 4 * np.pi * np.pi)
    exact_psi = y_tu - 1.0 / 2 / torch.pi * torch.exp(kexi * x_tu) * torch.sin(2 * torch.pi * y_tu)
    exact_u = 1 - torch.exp(kexi * x_tu) * torch.cos(2 * torch.pi * y_tu)
    exact_v = kexi / 2 / torch.pi * torch.exp(kexi * x_tu) * torch.sin(2 * torch.pi * y_tu)
    tu = torch.cat((x_tu, y_tu), dim=1)
    print(tu.shape)

    wan_psi1 = u(tu)
    wan_psi = wan_psi1.detach().numpy()
    wan_u1 = gradients(u(tu), y_tu, 1)
    wan_u = wan_u1.detach().numpy()
    wan_v1 = - gradients(u(tu), x_tu, 1)
    wan_v = wan_v1.detach().numpy()
    X = x_tu.detach().numpy()
    Y = y_tu.detach().numpy()

    S = 5;  # 坐标点的大小 / 尺寸

    wan_psi = plt.scatter(X, Y, S, wan_psi, marker='s')
    plt.ylabel('y')
    plt.xlabel('x')
    plt.colorbar(wan_psi)
    plt.savefig("wan_psi.png")
    plt.show()
    wan_u = plt.scatter(X, Y, S, wan_u, marker='s')
    plt.ylabel('y')
    plt.xlabel('x')
    plt.colorbar(wan_u)
    plt.savefig("wan_u.png")
    plt.show()
    wan_v = plt.scatter(X, Y, S, wan_v, marker='s')
    plt.ylabel('y')
    plt.xlabel('x')
    plt.colorbar(wan_v)
    plt.savefig("wan_v.png")
    plt.show()

    error_psi = ((exact_psi - wan_psi1) ** 2).detach().numpy()
    error_u = ((exact_u - wan_u1) ** 2).detach().numpy()
    error_v = ((exact_v - wan_v1) ** 2).detach().numpy()

    error_psi = plt.scatter(X, Y, S, error_psi, marker='s')
    plt.ylabel('y')
    plt.xlabel('x')
    plt.colorbar(error_psi)
    plt.savefig("error_psi.png")
    plt.show()
    error_u = plt.scatter(X, Y, S, error_u, marker='s')
    plt.ylabel('y')
    plt.xlabel('x')
    plt.colorbar(error_u)
    plt.savefig("error_u.png")
    plt.show()
    error_v = plt.scatter(X, Y, S, error_v, marker='s')
    plt.ylabel('y')
    plt.xlabel('x')
    plt.colorbar(error_v)
    plt.savefig("error_v.png")
    plt.show()

    l2_difference_u = (torch.sqrt((torch.sum((exact_u - wan_u1) ** 2))) / torch.sqrt(
        (torch.sum((exact_u) ** 2)))).detach().numpy()
    l2_difference_v = (torch.sqrt((torch.sum((exact_v - wan_v1) ** 2))) / torch.sqrt(
        (torch.sum((exact_v) ** 2)))).detach().numpy()
    l2_difference_psi = (torch.sqrt((torch.sum((exact_psi - wan_psi1) ** 2))) / torch.sqrt(
        (torch.sum((exact_psi) ** 2)))).detach().numpy()

    print("L2 relative error in u:", l2_difference_u)
    print("L2 relative error in v:", l2_difference_v)
    print("L2 relative error in psi:", l2_difference_psi)

nu = 1.0 /40
kexi = 1/ 2 /nu - np.sqrt(1 / 4 / nu / nu +4 * np.pi * np.pi)
def psi_exact(x,y):
    return  y - 1.0/2/torch.pi * torch.exp(kexi * x) * torch.sin(2* torch.pi * y)
def u_exact(x,y):
    return  1-torch.exp(kexi * x) * torch.cos(2* torch.pi * y)
def v_exact(x,y):
    return  kexi / 2 / torch.pi * torch.exp(kexi * x) * torch.sin(2* torch.pi * y)
x = torch.linspace(-0.5, 1.5, 100)
y = torch.linspace(-0.5, 1.5, 100)
X,Y = torch.meshgrid(x, y)
psi =psi_exact(X,Y)
u = u_exact(X,Y)
v = v_exact(X,Y)
S = 5;  # 坐标点的大小 / 尺寸
exact_psi = plt.scatter(X, Y, S, psi, marker='s')  # filled表示点是实心点，缺省则为空心点
plt.ylabel('y')
plt.xlabel('x')
plt.colorbar(exact_psi)
plt.savefig("exact_psi.png")
plt.show()
exact_u = plt.scatter(X, Y, S, u, marker='s')
plt.ylabel('y')
plt.xlabel('x')   # filled表示点是实心点，缺省则为空心点
plt.colorbar(exact_u)
plt.savefig("exact_u.png")
plt.show()
exact_v = plt.scatter(X, Y, S, v, marker='s')
plt.ylabel('y')
plt.xlabel('x')
plt.colorbar(exact_v)
plt.savefig("exact_v.png")
plt.show()
