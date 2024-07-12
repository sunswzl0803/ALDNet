
from nets.model import PDFnet
import torch
import torch.nn as nn
from torch.nn import functional as F

class PDFNET(nn.Module):
    def __init__(self, num_classes = 21,pretrained = False, model = 'PDFnet'):
        super(PDFNET, self).__init__()
        if model == "PDFnet":
            self.PDFnet = PDFnet()
        self.model = model
    def forward(self, inputs):
        if self.model == "PDFnet":
            logits = self.PDFnet.forward(inputs)
            return logits


