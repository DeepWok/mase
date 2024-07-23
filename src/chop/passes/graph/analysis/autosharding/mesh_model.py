import torch
import numpy as np


class MeshModel:
    def __init__(self, mesh_shape, mesh_alpha=None, mesh_beta=None):
        self.mesh_shape = mesh_shape

        num_devices = np.prod(mesh_shape)
        self.id_mesh = torch.arange(0, num_devices).reshape(mesh_shape)

        # Alpha/beta model is used to estimate communication cost between devices
        self.mesh_alpha = [0] * 2 if mesh_alpha is None else mesh_alpha
        self.mesh_beta = [None] * 2 if mesh_beta is None else mesh_beta

        # For compatibility with torch DeviceMesh when building MeshTopoInfo object
        # for sharding redistribution cost estimation
        self.device_type = "cuda"
        self.ndim = 2

    def __getitem__(self, key):
        return self.mesh_shape[key]

    def size(self, dim=None):
        if dim is None:
            return np.prod(self.mesh_shape)
        else:
            return self.mesh_shape[dim]

    def set_cost_model_parameters(
        self,
        intra_node_bandwidth: int,
        inter_node_bandwidth: int,
        backend: str = "default",
    ):
        # Assign differently depending if backend is NVLink, Infiniband, etc
        if backend == "default":
            # Assuming a setup with ethernet-connected nodes and devices connected through
            # PCIe within each node
            self.mesh_beta = [1 / inter_node_bandwidth, 1 / intra_node_bandwidth]

    def all_gather_cost(self, num_bytes, mesh_dim):
        num_devices = self.id_mesh.shape[mesh_dim]
        return (
            self.mesh_alpha[mesh_dim]
            + self.mesh_beta[mesh_dim] * (num_devices - 1) / num_devices * num_bytes
            + 0.1
        )

    def all_reduce_cost(self, num_bytes, mesh_dim, num_devices=None):
        """
        The term multiplied by beta represents the total number of bytes
        transferred over the full transaction. For the ring implementation
        of all reduce there are 2 rounds of (n-1) transfers, hence 2(n-1).
        In each case num_bytes/num_devices is transferred, where num_bytes
        is the number of bytes for the full tensor on each device.
        """
        if num_devices is None:
            num_devices = self.id_mesh.shape[mesh_dim]
        return (
            self.mesh_alpha[mesh_dim]
            + self.mesh_beta[mesh_dim] * 2 * (num_devices - 1) / num_devices * num_bytes
            + 0.01
        )

    def reduce_scatter_cost(self, num_bytes, mesh_dim):
        num_devices = self.id_mesh.shape[mesh_dim]
        return (
            self.mesh_alpha[mesh_dim]
            + self.mesh_beta[mesh_dim] * (num_devices - 1) / num_devices * num_bytes
            + 0.001
        )

    def all_to_all_cost(self, num_bytes, mesh_dim):
        num_devices = self.id_mesh.shape[mesh_dim]
        penalty_factor = num_devices / 2.0
        return (
            self.mesh_alpha[mesh_dim]
            + self.mesh_beta[mesh_dim]
            * (num_devices - 1)
            / num_devices
            / num_devices
            * num_bytes
            * penalty_factor
            + 0.001
        )
