import torch

class MyDecisionGate(torch.nn.Module):
    def forward(self, x):
        if x.sum() > 0:
            return x
        return -x

class MyCell(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)
        self.dg = MyDecisionGate()

    def forward(self, x, h):
        # new_h = torch.tanh(self.linear(x) + h)
        new_h = torch.tanh(self.dg(self.linear(x) + h))
        return new_h, new_h

def example():

    my_cell = MyCell()

    x = torch.rand(3,4)
    h = torch.rand(3,4)
    # print(my_cell)

    # print(my_cell(x, h))

    traced_cell = torch.jit.trace(my_cell, (x,h))
    print(traced_cell.code)
    traced_cell(x, h)


# print(torch.nn.Linear(2,2))
example()

