import torch

x = torch.tensor([[0.1, 0.2]], dtype=torch.float,
                 requires_grad=True)  # size (1,2)
w1 = torch.tensor([[0.2], [0.3]], dtype=torch.float,
                  requires_grad=True)  # size (2,1)
w2 = torch.tensor([[0.4, 0.5], [0.6, 0.7], [0.8, 0.9]],
                  dtype=torch.float, requires_grad=True)  # size (3,2)
y1 = 1 * torch.matmul(w1, x)  # size (2,2)
y2 = 1 * torch.matmul(w2, y1)  # size (3,2)
# y1 = w1 * x
# y2 = w2 * y1
y = y2.mean()  # size (1,)
y.backward(retain_graph=True)
grad = torch.autograd.grad(outputs=y2, inputs=y1,
                           grad_outputs=1 / 6 * torch.ones_like(y2),
                           retain_graph=True,
                           create_graph=True)
grad = torch.autograd.grad(outputs=y2, inputs=w1,
                           grad_outputs=1 / 6 * torch.ones_like(y2),
                           retain_graph=True,
                           create_graph=True)
print(grad)
