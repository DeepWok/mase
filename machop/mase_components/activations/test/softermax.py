from math import exp


def softmax(l: list[float], pow2=False):

    max_num = max(l)

    norm = 0.0
    for x in l:
        if pow2:
            norm += 2 ** (x - max_num)
        else:
            norm += exp(x - max_num)

    if pow2:
        out = [(2 ** (x - max_num)) / norm for x in l]
    else:
        out = [exp(x - max_num) / norm for x in l]

    assert abs(sum(out) - 1) < 1e-5, f"Sum is {sum(out)}"

    return out


def _softmax_model(l: list[int], parallelism: int, pow2=False):
    """Model used to understand hardware implementation."""

    assert len(l) % parallelism == 0
    iters = len(l) // parallelism

    # Calculate local max & local pow2 values

    local_values_buffer = []
    local_max_buffer = []

    for i in range(iters):
        local_window = l[i * parallelism : (i + 1) * parallelism]
        local_max = max(local_window)
        if pow2:
            local_pow = [2 ** (x - local_max) for x in local_window]
        else:
            local_pow = [exp(x - local_max) for x in local_window]
        local_max_buffer.append(local_max)
        local_values_buffer.append(local_pow)

    # Calculate global max

    global_max = max(local_max_buffer)
    local_max_diff = [global_max - x for x in local_max_buffer]

    adjusted_vals = []
    norm = 0.0

    for diff, vals in zip(local_max_diff, local_values_buffer):
        if pow2:
            adj = [x * (2**-diff) for x in vals]
        else:
            adj = [x * exp(-diff) for x in vals]
        norm += sum(adj)
        adjusted_vals.append(adj)

    out = []
    for i in range(iters):
        vals = adjusted_vals[i]
        out.extend([x / norm for x in vals])

    assert abs(sum(out) - 1) < 1e-5, f"Sum is {sum(out)}"

    return out


def check_lists(l1, l2, eps: float = 1e-5):
    for x, y in zip(l1, l2):
        assert abs(x - y) < eps


if __name__ == "__main__":

    LIST = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    PARALLELISM = 3

    sw_softmax = softmax(LIST, pow2=True)
    hw_softmax = _softmax_model(LIST, PARALLELISM, pow2=True)
    check_lists(sw_softmax, hw_softmax)
    print(hw_softmax)
