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
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# initial values
a1 = 0.1


# Neural Network
class nn_u(torch.nn.Module):
    def __init__(self):
        super(nn_u, self).__init__()
        # hidden layers
        self.net = torch.nn.Sequential(
            torch.nn.Linear(2, 30),
            torch.nn.Tanh(),
            torch.nn.Linear(30, 30),
            torch.nn.Tanh(),
            torch.nn.Linear(30, 30),
            torch.nn.Tanh(),
            torch.nn.Linear(30, 30),
            torch.nn.Tanh(),
            torch.nn.Linear(30, 30),
            torch.nn.Tanh(),
            torch.nn.Linear(30, 30),
            torch.nn.Tanh(),
            torch.nn.Linear(30, 30),
            torch.nn.Tanh(),
            torch.nn.Linear(30, 1)
        )
        #         initialize parameters
        self.a1 = torch.tensor([a1], requires_grad=True).float()
        # register parameters
        self.a1 = torch.nn.Parameter(self.a1)
        self.register_parameter('a1', self.a1)

    def forward(self, x):
        return self.net(x)

# generate dataset
def generate_data(m):
    x = torch.rand(m, 1)
    y = torch.rand(m, 1) - 1/2
    z = 5/6 * y ** 3 - 5/8 * y
    return x.requires_grad_(True), y.requires_grad_(True), z

class nn_v(torch.nn.Module):

    def __init__(self):
        super(nn_v, self).__init__()
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

m = 1000
lr = 0.0001
steps=10001
x_d, y_d, z_d = generate_data(m)
u = nn_u()
v = nn_v()
step_array=[]
a1_array=[]
a2_array=[]
lossu_array = []
lossp_array = []
w_1,w_2,w_3,w_4=[1,2000,2000,2000]
optimizer_v = torch.optim.Adam(v.parameters(), lr)
optimizer_u = torch.optim.Adam(u.parameters(), lr)
loss = torch.nn.MSELoss()
# Problem parameter initialization
a1 = np.array([0])
a1= torch.from_numpy(a1).float().to(device).requires_grad_(True)
a1.grad = torch.ones((1)).to(device)

a2 = np.array([0])
a2= torch.from_numpy(a2).float().to(device).requires_grad_(True)
a2.grad = torch.ones((1)).to(device)

optimizer_u.add_param_group({'params': a1, 'lr': 0.001})


def interior(n=1000):
    x = torch.rand(n, 1)
    y = torch.rand(n, 1) - 1 / 2
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
    nu = u.a1
    uxy = u(torch.cat([x, y], dim=1))
    vxy = v(torch.cat([x, y], dim=1))
    lu = gradients(uxy, x, 2) + gradients(uxy, y, 2)
    lv = gradients(vxy, x, 2) + gradients(vxy, y, 2)
    ux = gradients(uxy, x, 1)
    uy = gradients(uxy, y, 1)
    du = ux + uy
    vx = gradients(vxy, x, 1)
    vy = gradients(vxy, y, 1)
    dv = vx + vy
    vx = gradients(vxy, x, 1)
    vy = gradients(vxy, y, 1)

    int_1 = torch.mean(torch.mul(lu * lv * nu, torch.cat([x, y], dim=1)))
    int_2 = torch.mean(torch.mul(uy * lu * vx, torch.cat([x, y], dim=1)))
    int_3 = torch.mean(torch.mul(ux * lu * vy, torch.cat([x, y], dim=1)))
    dv = vx + vy
    v_norm = torch.mean(torch.sqrt(vxy ** 2))
    return (int_1 + int_2 - int_3)


def bound_1(n=100):
    x1 = torch.rand(n, 1) * 0
    x2 = torch.ones_like(x1)

    y1 = torch.rand(n, 1) - 1 / 2
    y2 = torch.rand(n, 1) - 1 / 2

    x3 = torch.rand(n, 1)
    y3 = torch.ones_like(x1) * 1 / 2

    x4 = torch.rand(n, 1)
    y4 = - torch.ones_like(x1) * 1 / 2

    x = torch.cat([x1, x2, x3, x4], dim=0)
    y = torch.cat([y1, y2, y3, y4], dim=0)

    cond_psi = 5 / 6 * y ** 3 - 5 / 8 * y
    cond_u = - 5 / 2 * (y ** 2 - 1 / 4)
    cond_v = 0 * y
    return x.requires_grad_(True), y.requires_grad_(True), cond_psi, cond_u, cond_v


def l_bound(u):
    x, y, cond_psi, cond_u, cond_v = bound_1()
    uxy = u(torch.cat([x, y], dim=1))
    bound_u = - gradients(uxy, y, 1)
    bound_v = gradients(uxy, x, 1)
    return loss(bound_u, cond_u) + loss(bound_v, cond_v) + loss(uxy, cond_psi)


def l_data1(u):
    x, y = interior()
    uxy = u(torch.cat([x, y], dim=1))
    cond_psi = 5 / 6 * y ** 3 - 5 / 8 * y
    return loss(uxy, cond_psi)


def l_data2(u):
    x, y = interior()
    uxy = u(torch.cat([x, y], dim=1))
    bound_u = - gradients(uxy, y, 1)
    cond_u = - 5 / 2 * (y ** 2 - 1 / 4)
    return loss(bound_u, cond_u)


def l_data3(u):
    x, y = interior()
    uxy = u(torch.cat([x, y], dim=1))
    bound_v = gradients(uxy, x, 1)
    cond_v = 0 * y
    return loss(bound_v, cond_v)


for i in range(steps):
    for j in range(2):
        optimizer_u.zero_grad()
        lossu = w_1 * l_interior(u) + w_2 * (l_data1(u) + l_data2(u) + l_data3(u)) + w_3 * l_bound(u)
        lossu.backward()
        optimizer_u.step()

    optimizer_v.zero_grad()
    lossv = - l_interior(u)
    lossv.backward()
    optimizer_v.step()
    step_array.append(i)
    a1_array.append(float(u.state_dict()['a1'].detach().numpy()))
    lossu_array.append(lossu.item())
    if i % 500 == 0:
        print(f'step: {i}  loss_u = {lossu.item()} loss_v = {lossv.item()}')
        print(u.state_dict()['a1'].detach().numpy())

x_tu = torch.arange(0, 1, 1 / 100)
x_tu = torch.unsqueeze(x_tu, dim=1).requires_grad_(True)
y_tu = 0 * torch.ones(100) - 0.5
y_tu = torch.unsqueeze(y_tu, dim=1).requires_grad_(True)
for i in range(99):
    xi_tu = torch.arange(0, 1, 1 / 100)
    xi_tu = torch.unsqueeze(xi_tu, dim=1)
    x_tu = torch.cat((x_tu, xi_tu), dim=0)
for j in range(99):
    yj_tu = j / 100 * (torch.ones(100)) - 1 / 2
    yj_tu = torch.unsqueeze(yj_tu, dim=1)
    y_tu = torch.cat((y_tu, yj_tu), dim=0)

exact_psi = 5 / 6 * y_tu ** 3 - 5 / 8 * y_tu
exact_u = - 5 / 2 * (y_tu ** 2 - 1 / 4)
exact_v = 0
tu = torch.cat((x_tu, y_tu), dim=1)
print(tu.shape)
wan_psi1 = u(tu)

wan_psi = wan_psi1.detach().numpy()
wan_u1 = - gradients(u(tu), y_tu, 1)
wan_u = wan_u1.detach().numpy()
wan_v1 = gradients(u(tu), x_tu, 1)
wan_v = wan_v1.detach().numpy()
X = x_tu.detach().numpy()
Y = y_tu.detach().numpy()

S = 5;  # 坐标点的大小 / 尺寸
wan_error_psi = ((exact_psi - wan_psi1) ** 2).detach().numpy()
wan_error_u = (torch.abs(exact_u - wan_u1)).detach().numpy()
wan_error_v = ((exact_v - wan_v1) ** 2).detach().numpy()

S = 5
fig, ax = plt.subplots(1,2, figsize=(10.5, 2.8), constrained_layout=True)
ax = ax.flatten()
ax0 = ax[0].scatter(X, Y, S, exact_u.detach().numpy(), marker='s')
ax[0].set_xlabel("x")
ax[0].set_ylabel("y")
ax[0].tick_params(labelsize=11)
ax1 = ax[1].scatter(X, Y, S, wan_u, marker='s')
ax[1].set_xlabel("x")
ax[1].set_ylabel("y")
ax[1].tick_params(labelsize=11)
fig.colorbar(ax0, ax = [ax[0],ax[1]])


plt.plot(step_array,a2_array,color='red',label='WAN')
plt.legend()
plt.ylabel('nu')
plt.xlabel('step')
# plt.title('nu')
plt.show

S = 5
norm1 = matplotlib.colors.Normalize(vmin=0,vmax=0.4)  # 设置colorbar显示的最大最小值
fig, ax = plt.subplots(1,2, figsize=(7.5, 2.8), constrained_layout=True)
ax = ax.flatten()
ax0 = ax[0].scatter(X, Y, S, wan_error_u, marker='s',norm=norm1)
ax[0].set_xlabel("x")
ax[0].set_ylabel("y")
ax[0].tick_params(labelsize=11)
ax1 = ax[1].scatter(X, Y, S, pinn_error_u, marker='s',norm=norm1)
ax[1].set_xlabel("x")
ax[1].set_ylabel("y")
ax[1].tick_params(labelsize=11)
fig.colorbar(ax1, ax = [ax[0],ax[1]])

S = 5
norm2 = matplotlib.colors.Normalize(vmin=0,vmax=0.01)  # 设置colorbar显示的最大最小值
fig, ax = plt.subplots(1,2, figsize=(7.5, 2.8), constrained_layout=True)
ax = ax.flatten()
ax0 = ax[0].scatter(X, Y, S, wan_error_u, marker='s',norm=norm2)
ax[0].set_xlabel("x")
ax[0].set_ylabel("y")
ax1 = ax[1].scatter(X, Y, S, pinn_error_u, marker='s',norm=norm2)
ax[1].set_xlabel("x")
ax[1].set_ylabel("y")
fig.colorbar(ax1, ax = [ax[0],ax[1]])