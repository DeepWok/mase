import torch

from chop.tools.onnx_operators import onnx_slice


def test_gather():
    data = torch.Tensor(
        [
            [1.0, 1.2, 1.9],
            [2.3, 3.4, 3.9],
            [4.5, 5.7, 5.9],
        ]
    )

    indices = torch.Tensor(
        [
            [0, 2],
            [0, 2],
            [0, 2],
        ]
    ).to(torch.int64)

    out = torch.gather(data, 1, indices)

    assert torch.equal(out, torch.Tensor([[1.0, 1.9], [2.3, 3.9], [4.5, 5.9]]))


def test_slice():
    data = torch.Tensor(
        [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
        ]
    )

    test1 = onnx_slice(
        data,
        torch.Tensor([1, 0]),
        torch.Tensor([2, 3]),
        steps=torch.Tensor([1, 2]),
        axes=torch.Tensor([0, 1]),
    )
    test2 = onnx_slice(data, torch.Tensor([0, 1]), torch.Tensor([-1, 1000]))

    assert torch.equal(test1, torch.tensor([[5, 7]]))
    assert torch.equal(test2, torch.tensor([[2, 3, 4]]))


if __name__ == "__main__":
    test_gather()
    test_slice()
