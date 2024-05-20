import torch


class BinarizeSign(torch.autograd.Function):
    @staticmethod
    def forward(self, input):
        self.saved_for_backward = [input]
        return input.sign()

    @staticmethod
    def backward(self, grad_output):
        input = self.saved_for_backward[0]
        grad_input = grad_output.clone()
        grad_input[input < -1] = 0
        grad_input[input > 1] = 0
        return grad_input
