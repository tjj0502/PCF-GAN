import torch
import torch.nn as nn
from src.PCFGAN.unitary import unitary_projection
from src.PCFGAN.upper_triangular import up_projection, vector_to_matrix, col_idx
from src.PCFGAN.orthogonal_diag import orthogonal_diag_projection
from src.PCFGAN.orthogonal import orthogonal_projection


class development_layer(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        lie_group: str = 'unitary',
        channels: int = 1,
        include_inital: bool = False,
        time_batch=1,
        return_sequence=False,
        init_range=1,
    ):
        """
        Development layer module used for computation of unitary feature on time series.

        Args:
            input_size (int): Input size.
            hidden_size (int): Size of the hidden matrix.
            channels (int, optional): Number of channels. Defaults to 1.
            include_inital (bool, optional): Whether to include the initial value in the input. Defaults to False.
            time_batch (int, optional): Truncation value for batch processing. Defaults to 1.
            return_sequence (bool, optional): Whether to return the entire sequence or just the final output. Defaults to False.
            init_range (int, optional): Range for weight initialization. Defaults to 1.
        """
        super(development_layer, self).__init__()
        self.input_size = input_size
        self.channels = channels
        self.hidden_size = hidden_size
        if lie_group == 'unitary':
            self.projection = unitary_projection(
                input_size, hidden_size, channels, init_range=init_range
            )
            self.complex = True
        elif lie_group == 'upper':
            self.projection = up_projection(input_size, hidden_size, channels, init_range)
            self.complex = False
            self.col_idx = col_idx(self.hidden_size)
        elif lie_group == 'orthogonal_diag':
            self.projection = orthogonal_diag_projection(input_size, hidden_size, channels, init_range)
            self.complex = False
        else:
            raise ValueError("Please provide a valid lie group.")
        self.include_inital = include_inital
        self.truncation = time_batch
        self.return_sequence = return_sequence

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the development layer module.

        Args:
            input (torch.Tensor): Tensor with shape (N, T, input_size).

        Returns:
            torch.Tensor: Tensor with shape (N, T, hidden_size, hidden_size).
        """
        if self.complex:
            input = input.cfloat()

        N, T, C = input.shape
        if self.include_inital:
            input = torch.cat([torch.zeros((N, 1, C)).to(input.device), input], dim=1)

        dX = input[:, 1:] - input[:, :-1]
        # N,T-1,input_size

        M_dX = self.projection(dX.reshape(-1, dX.shape[-1])).reshape(
            N, -1, self.channels, self.hidden_size, self.hidden_size
        )

        return self.dyadic_prod(M_dX)

    @staticmethod
    def dyadic_prod(X: torch.Tensor) -> torch.Tensor:
        """
        Computes the cumulative product on matrix time series with dyadic partitioning.

        Args:
            X (torch.Tensor): Batch of matrix time series of shape (N, T, C, m, m).

        Returns:
            torch.Tensor: Cumulative product on the time dimension of shape (N, T, m, m).
        """
        N, T, C, m, m = X.shape
        max_level = int(torch.ceil(torch.log2(torch.tensor(T))))
        I = (
            torch.eye(m, device=X.device, dtype=X.dtype)
            .reshape(1, 1, 1, m, m)
            .repeat(N, 1, C, 1, 1)
        )
        for i in range(max_level):
            if X.shape[1] % 2 == 1:
                X = torch.cat([X, I], 1)
            X = X.reshape(-1, 2, C, m, m)
            X = torch.einsum("bcij,bcjk->bcik", X[:, 0], X[:, 1])
            X = X.reshape(N, -1, C, m, m)

        return X[:, 0]

class development_layer_v2(nn.Module):
    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            lie_group: str = 'unitary',
            channels: int = 1,
            include_inital: bool = False,
            partition_size=0,
            init_range=1,
    ):
        """
        Development layer module used for computation of unitary feature on time series.

        Args:
            input_size (int): Input size.
            hidden_size (int): Size of the hidden matrix.
            channels (int, optional): Number of channels. Defaults to 1.
            include_inital (bool, optional): Whether to include the initial value in the input. Defaults to False.
            return_sequence (bool, optional): Whether to return the entire sequence or just the final output. Defaults to False.
            init_range (int, optional): Range for weight initialization. Defaults to 1.
        """
        super(development_layer_v2, self).__init__()
        self.input_size = input_size
        self.channels = channels
        self.hidden_size = hidden_size
        if lie_group == 'unitary':
            self.projection = unitary_projection(
                input_size, hidden_size, channels, init_range=init_range
            )
            self.complex = True
        elif lie_group == 'upper':
            self.projection = up_projection(input_size, hidden_size, channels, init_range)
            self.complex = False
            self.col_idx = col_idx(self.hidden_size)
        elif lie_group == 'orthogonal_diag':
            self.projection = orthogonal_diag_projection(input_size, hidden_size, channels, init_range)
            self.complex = False
        elif lie_group == 'orthogonal':
            self.projection = orthogonal_projection(input_size, hidden_size, channels, init_range)
            self.complex = False
        else:
            raise ValueError("Please provide a valid lie group.")
        self.include_inital = include_inital
        self.partition_size = partition_size

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the development layer module.

        Args:
            input (torch.Tensor): Tensor with shape (N, T, input_size).

        Returns:
            torch.Tensor: Tensor with shape (N, T, hidden_size, hidden_size).
        """
        if self.complex:
            input = input.cfloat()

        N, T, C = input.shape
        if self.include_inital:
            input = torch.cat([torch.zeros((N, 1, C)).to(input.device), input], dim=1)

        dX = input[:, 1:] - input[:, :-1]
        # N,T-1,input_size

        M_dX = self.projection(dX.reshape(-1, dX.shape[-1])).reshape(
            N, -1, self.channels, self.hidden_size, self.hidden_size
        )

        # r = M_dX.shape[1] % self.partition_size
        # # Do time partitioning
        # I = (
        #     torch.eye(self.hidden_size, device=M_dX.device, dtype=M_dX.dtype)
        #     .reshape(1, 1, 1, self.hidden_size, self.hidden_size)
        #     .repeat(N, r, self.channels, 1, 1)
        # )
        #
        # M_dX = torch.cat([M_dX, I], 1).reshape([-1, self.partition_size, self.channels, self.hidden_size, self.hidden_size])
        if self.partition_size:
            return self.up_dyadic_prod(M_dX) # [N, 2**n, C, m, m]
        else:
            return self.dyadic_prod(M_dX)

    @staticmethod
    def dyadic_prod(X: torch.Tensor) -> torch.Tensor:
        """
        Computes the cumulative product on matrix time series with dyadic partitioning.

        Args:
            X (torch.Tensor): Batch of matrix time series of shape (N, T, C, m, m).

        Returns:
            torch.Tensor: Cumulative product on the time dimension of shape (N, C, m, m).
        """
        N, T, C, m, m = X.shape
        max_level = int(torch.ceil(torch.log2(torch.tensor(T))))
        I = (
            torch.eye(m, device=X.device, dtype=X.dtype)
            .reshape(1, 1, 1, m, m)
            .repeat(N, 1, C, 1, 1)
        )
        for i in range(max_level):
            if X.shape[1] % 2 == 1:
                X = torch.cat([X, I], 1)
            X = X.reshape(-1, 2, C, m, m)
            X = torch.einsum("bcij,bcjk->bcik", X[:, 0], X[:, 1])
            X = X.reshape(N, -1, C, m, m)

        return X[:, 0]



    def up_dyadic_prod(self, X: torch.Tensor):
        """
        Computes the cumulative product on matrix time series with dyadic partitioning. Specially designed for upper triangular

        Args:
            X (torch.Tensor): Batch of matrix time series of shape (N, T, C, m, m).

        Returns:
            torch.Tensor: Cumulative product on the time dimension of shape (N, 2**n, C, m, m).
        """
        N, T, C, m, m = X.shape
        max_level = int(torch.ceil(torch.log2(torch.tensor(T))))
        # print("MAX level: ", max_level, self.partition_size)
        # If partition_size is provided, then the whole interval is divided into subintervals of length 2**n
        # If partition_size is provided, then the whole interval is divided into subintervals of length 2**n
        # if self.partition_size:
        #     max_level = min(max_level, self.partition_size)
        I = (
            torch.eye(m, device=X.device, dtype=X.dtype)
            .reshape(1, 1, 1, m, m)
            .repeat(N, 1, C, 1, 1)
        )
        dyadic_dev = 0
        for i in range(max_level):
            if X.shape[1] % 2 == 1:
                X = torch.cat([X, I], 1)
            X = X.reshape(-1, 2, C, m, m)
            X = torch.einsum("bcij,bcjk->bcik", X[:, 0], X[:, 1])
            X = X.reshape(N, -1, C, m, m)
            # If partition_size is provided, then the whole interval is divided into subintervals of length 2**n, track the dyadic dev
            if self.partition_size and i == self.partition_size:
                dyadic_dev = X.clone()
        return X, dyadic_dev

    def prod_upper_triangular(self, A, B):

        N, T, m = A.shape

        assert m >= 3, "Matrix size must be larger than 3"

        matrix_A = vector_to_matrix(A, self.hidden_size)

        res = torch.zeros(matrix_A.shape, device=A.device, dtype=A.dtype).to(A.device).to(A.dtype)

        matrix_A = matrix_A[:, :, :self.hidden_size - 2, 1:self.hidden_size-1]

        res += vector_to_matrix(A + B, self.hidden_size)
        init_idx = 1
        for i in range(self.hidden_size - 3, -1, -1):
            #     print(res[0,0,:-(hidden_size-i-1),hidden_size-i-1:])
            #     print(matrix_A[0,0,:i+1,:i+1])
            #     print(row_matrix(Y[:,:,init_idx: init_idx + i+1]))
            res[:, :, :-(self.hidden_size - i - 1), self.hidden_size - i - 1:] += matrix_A[:, :, :i + 1, :i + 1] * row_matrix(
                B[:, :, init_idx: init_idx + i + 1])
            init_idx += i + 2

        return

def row_matrix(vector):
    n, c, d = vector.shape
    matrix = vector.unsqueeze(2)
    return matrix.repeat(1,1,d,1)


