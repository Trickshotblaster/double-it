print("Training a neural net to multiply your inputs by 2")
print("Going though a completely unnecessary number of epochs so you can see the progress bar :,)")
import torch
import progressbar
from time import sleep

num_epochs = 1000
bar = progressbar.ProgressBar(maxval=num_epochs, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
bar.start()


w1 = torch.randn((1, 2), requires_grad=True)
b1 = torch.randn((2), requires_grad=True)
w2 = torch.randn((2, 1), requires_grad=True)
b2 = torch.randn((1), requires_grad=True)

params = [w1, b1, w2, b2]
for epoch in range(num_epochs):
    ins = torch.randn(10000, 1)
    l1 = ins @ w1 + b1
    out = l1 @ w2 + b2
    y = ins * 2
    loss = ((y-out)**2).mean()
    for p in params:
        p.grad = None
    loss.backward()
    for p in params:
        p.data += p.grad * -0.1
    bar.update(epoch+1)

bar.finish()
print("final loss:", loss.item())

while True:
    textin = input("Test the network:")
    if textin == "quit":
        break
    testin = torch.tensor(int(textin)).view(1, 1).float()
    l1 = testin @ w1 + b1
    out = l1 @ w2 + b2
    print("Net prediction:", out.item(), "actual answer:", (testin*2).item())


