import hiddenlayer as hl
import torch
from by19_best import By_3d

model = By_3d().cuda()
hl_graph = hl.build_graph(model, (torch.zeros([1, 3, 352, 480]).cuda()))
hl_graph.theme = hl.graph.THEMES["blue"].copy()  # Two options: basic and blue

hl_graph.save('/home/zjy/what/best.jpg')