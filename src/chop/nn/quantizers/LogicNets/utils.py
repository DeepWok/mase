#  Copyright (C) 2021 Xilinx, Inc
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from functools import reduce

import os
import subprocess

import torch


# Return the indices associated with a '1' value
# TODO: vectorise this function
def fetch_mask_indices(mask: torch.Tensor) -> torch.LongTensor:
    local_mask = mask.detach().clone()
    fan_in = torch.sum(local_mask, dtype=torch.int64)
    indices = [0] * fan_in
    for i in range(fan_in):
        ind = torch.argmax(local_mask)
        indices[i] = ind
        local_mask[ind] = 0
    return tuple(indices)


# Return a matrix which contains all input permutations
# TODO: implement this function
def generate_permutation_matrix(input_state_space) -> torch.Tensor:
    total_permutations = reduce(
        lambda a, b: a * b, map(lambda x: x.nelement(), input_state_space)
    )  # Calculate the total number of permutations
    fan_in = len(input_state_space)
    permutations_matrix = torch.zeros((total_permutations, fan_in))
    # TODO: is there a way to do this that is vectorised?
    for p in range(total_permutations):
        next_perm = p
        for f in range(fan_in):
            div_factor = input_state_space[f].nelement()
            index = next_perm % div_factor
            permutations_matrix[p, f] = input_state_space[f][index]
            next_perm = next_perm // div_factor
    return permutations_matrix


# TODO: Put this inside an abstract base class
def get_int_state_space(bits: int, signed: bool = True, narrow_range: bool = False):
    start = int(
        0 if not signed else (-(2 ** (bits - 1)) + int(narrow_range))
    )  # calculate the minimum value in the range
    end = int(
        start + 2 ** (bits) - int(narrow_range)
    )  # calculate the maximum of the range
    state_space = torch.as_tensor(range(start, end))
    return state_space
