import torch

from chop.tools.onnx_operators import onnx_slice, onnx_gather
import sys, traceback, pdb


def excepthook(exc_type, exc_value, exc_traceback):
    traceback.print_exception(exc_type, exc_value, exc_traceback)
    print("\nEntering debugger...")
    pdb.post_mortem(exc_traceback)


sys.excepthook = excepthook


def test_gather():
    data1 = torch.Tensor(
        [
            [1.0, 1.2, 1.9],
            [2.3, 3.4, 3.9],
            [4.5, 5.7, 5.9],
        ]
    )

    data2 = torch.Tensor(
        [
            [1.0, 1.2],
            [2.3, 3.4],
            [4.5, 5.7],
        ]
    )

    indices1 = torch.Tensor(
        [
            [0, 2],
        ]
    ).to(torch.int64)

    indices2 = torch.Tensor(
        [
            [0, 1],
            [1, 2],
        ]
    ).to(torch.int64)

    obs_out1 = onnx_gather(data1, 1, indices1)
    obs_out2 = onnx_gather(data2, 0, indices2)

    exp_out1 = torch.Tensor(
        [
            [[1.0, 1.9]],
            [[2.3, 3.9]],
            [[4.5, 5.9]],
        ]
    )

    exp_out2 = torch.Tensor(
        [
            [
                [1.0, 1.2],
                [2.3, 3.4],
            ],
            [
                [2.3, 3.4],
                [4.5, 5.7],
            ],
        ]
    )

    print(obs_out2)
    print(exp_out2)

    # assert torch.equal(exp_out1, obs_out1)
    # assert torch.equal(exp_out2, obs_out2)


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
