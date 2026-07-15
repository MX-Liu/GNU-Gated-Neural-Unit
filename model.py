
import torch.nn as nn
import torchvision.models as models

import torch
import torch.nn.functional as F
import math

from typing import Literal

class ParametricSiLU(nn.Module):
    """SiLU with a trainable beta parameter.

    Formula:
        y = x * sigmoid(beta * x)

    Parameters can be shared across all channels or learned independently
    for each output feature.
    """

    def __init__(
        self,
        num_features: int,
        channel_wise: bool = True,
        beta_init: float = 1.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()

        if num_features <= 0:
            raise ValueError("num_features must be positive.")
        if beta_init <= 0:
            raise ValueError("beta_init must be positive.")

        parameter_size = num_features if channel_wise else 1
        raw_beta_init = math.log(math.expm1(beta_init))

        self.raw_beta = nn.Parameter(
            torch.full((parameter_size,), raw_beta_init)
        )
        self.eps = eps

    @property
    def beta(self) -> torch.Tensor:
        return F.softplus(self.raw_beta) + self.eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        beta = self.beta

        # Broadcast over all dimensions except the final feature dimension.
        view_shape = [1] * (x.dim() - 1) + [beta.numel()]
        beta = beta.view(*view_shape)

        return x * torch.sigmoid(beta * x)


class Snake(nn.Module):
    """Trainable Snake activation.

    Formula:
        y = x + sin(alpha * x)^2 / alpha

    Alpha can be shared or learned independently for each output feature.
    """

    def __init__(
        self,
        num_features: int,
        channel_wise: bool = True,
        alpha_init: float = 1.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()

        if num_features <= 0:
            raise ValueError("num_features must be positive.")
        if alpha_init <= 0:
            raise ValueError("alpha_init must be positive.")

        parameter_size = num_features if channel_wise else 1
        raw_alpha_init = math.log(math.expm1(alpha_init))

        self.raw_alpha = nn.Parameter(
            torch.full((parameter_size,), raw_alpha_init)
        )
        self.eps = eps

    @property
    def alpha(self) -> torch.Tensor:
        return F.softplus(self.raw_alpha) + self.eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = self.alpha

        # Broadcast over all dimensions except the final feature dimension.
        view_shape = [1] * (x.dim() - 1) + [alpha.numel()]
        alpha = alpha.view(*view_shape)

        return x + torch.sin(alpha * x).square() / alpha


class TrainableActivationLinear(nn.Module):
    """A conventional linear layer followed by a selectable trainable activation.

    Architecture:
        input -> nn.Linear -> trainable activation

    Supported activation types:
        - "prelu": PyTorch nn.PReLU
        - "psilu": Parametric SiLU with trainable beta
        - "snake": Snake with trainable alpha
        - "identity" or "none": no activation, useful as a linear baseline

    The class supports input tensors whose final dimension is `in_features`,
    including:
        (batch, in_features)
        (batch, sequence_length, in_features)
        (..., in_features)
    """

    SUPPORTED_ACTIVATIONS = {
        "prelu",
        "psilu",
        "parametric_silu",
        "snake",
        "identity",
        "none",
    }

    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation_type: Literal[
            "prelu",
            "psilu",
            "parametric_silu",
            "snake",
            "identity",
            "none",
        ] = "prelu",
        bias: bool = True,
        channel_wise: bool = True,
        prelu_init: float = 0.25,
        beta_init: float = 1.0,
        alpha_init: float = 1.0,
    ) -> None:
        super().__init__()

        if in_features <= 0:
            raise ValueError("in_features must be positive.")
        if out_features <= 0:
            raise ValueError("out_features must be positive.")

        activation_type = activation_type.lower()
        if activation_type not in self.SUPPORTED_ACTIVATIONS:
            raise ValueError(
                f"Unsupported activation_type={activation_type!r}. "
                f"Choose from {sorted(self.SUPPORTED_ACTIVATIONS)}."
            )

        self.in_features = in_features
        self.out_features = out_features
        self.activation_type = activation_type
        self.channel_wise = channel_wise

        self.linear = nn.Linear(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
        )

        if activation_type == "prelu":
            # One trainable negative slope per output feature when channel_wise=True.
            num_parameters = out_features if channel_wise else 1
            self.activation = nn.PReLU(
                num_parameters=num_parameters,
                init=prelu_init,
            )

        elif activation_type in {"psilu", "parametric_silu"}:
            self.activation = ParametricSiLU(
                num_features=out_features,
                channel_wise=channel_wise,
                beta_init=beta_init,
            )

        elif activation_type == "snake":
            self.activation = Snake(
                num_features=out_features,
                channel_wise=channel_wise,
                alpha_init=alpha_init,
            )

        else:
            self.activation = nn.Identity()

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset the linear layer and activation parameters."""
        self.linear.reset_parameters()

        if isinstance(self.activation, nn.PReLU):
            nn.init.constant_(self.activation.weight, 0.25)

        elif isinstance(self.activation, ParametricSiLU):
            raw_beta_init = math.log(math.expm1(1.0))
            nn.init.constant_(self.activation.raw_beta, raw_beta_init)

        elif isinstance(self.activation, Snake):
            raw_alpha_init = math.log(math.expm1(1.0))
            nn.init.constant_(self.activation.raw_alpha, raw_alpha_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(-1) != self.in_features:
            raise ValueError(
                f"Expected the final input dimension to be {self.in_features}, "
                f"but received shape {tuple(x.shape)}."
            )

        x = self.linear(x)

        # nn.PReLU interprets dimension 1 as channels. For tensors such as
        # (batch, sequence, features), move the final feature dimension to
        # dimension 1, apply PReLU, and move it back.
        if isinstance(self.activation, nn.PReLU):
            if x.dim() == 2:
                return self.activation(x)

            x = x.movedim(-1, 1)
            x = self.activation(x)
            return x.movedim(1, -1)

        return self.activation(x)

    def activation_parameters(self) -> dict[str, torch.Tensor]:
        """Return the current learned activation parameters."""
        if isinstance(self.activation, nn.PReLU):
            return {
                "negative_slope": self.activation.weight.detach().clone()
            }

        if isinstance(self.activation, ParametricSiLU):
            return {
                "beta": self.activation.beta.detach().clone()
            }

        if isinstance(self.activation, Snake):
            return {
                "alpha": self.activation.alpha.detach().clone()
            }

        return {}

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"activation_type={self.activation_type!r}, "
            f"channel_wise={self.channel_wise}, "
            f"bias={self.linear.bias is not None}"
        )
    

class KANLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                (
                    torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                    - 1 / 2
                )
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                # torch.nn.init.constant_(self.spline_scaler, self.scale_spline)
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        """
        Compute the B-spline bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = (
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).

        Returns:
            torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(
            0, 1
        )  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
        solution = torch.linalg.lstsq(
            A, B
        ).solution  # (in_features, grid_size + spline_order, out_features)
        result = solution.permute(
            2, 0, 1
        )  # (out_features, in_features, grid_size + spline_order)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor):
        assert x.size(-1) == self.in_features
        original_shape = x.shape
        x = x.reshape(-1, self.in_features)

        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        output = base_output + spline_output
        
        output = output.reshape(*original_shape[:-1], self.out_features)
        return output

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        splines = self.b_splines(x)  # (batch, in, coeff)
        splines = splines.permute(1, 0, 2)  # (in, batch, coeff)
        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(
            1, 0, 2
        )  # (batch, in, out)

        # sort each channel individually to collect data distribution
        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(
                self.grid_size + 1, dtype=torch.float32, device=x.device
            ).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        Compute the regularization loss.

        This is a dumb simulation of the original L1 regularization as stated in the
        paper, since the original one requires computing absolutes and entropy from the
        expanded (batch, in_features, out_features) intermediate tensor, which is hidden
        behind the F.linear function if we want an memory efficient implementation.

        The L1 regularization is now computed as mean absolute value of the spline
        weights. The authors implementation also includes this term in addition to the
        sample-based regularization.
        """
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )


class KAN(torch.nn.Module):
    def __init__(
        self,
        layers_hidden,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KAN, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order

        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )

    def forward(self, x: torch.Tensor, update_grid=False):
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )


class GNUNetLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
        alpha1=0.5,
        alpha2=0.5,
        projection = False,
        fixed_alpha=False,
        dyanmic_alpha=False,
        independent_alpha=False,
        trainable_activation = "none"
    ):
        super(GNUNetLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps
        self.fixed_alpha = fixed_alpha
        self.dyanmic_alpha = dyanmic_alpha
        self.independent_alpha = independent_alpha
        self.trainable_activation = trainable_activation

        if self.dyanmic_alpha:
            self.dynamic_layer = nn.Linear(2*out_features, 2)
            print("Dynamic alpha enabled. Using a linear layer to compute alpha values based on the outputs.")
        else:
            self.dynamic_layer = nn.Identity()

        

        if self.fixed_alpha:
            self.alpha1 = torch.nn.Parameter(torch.full((1,), alpha1),requires_grad=False); 
            self.alpha2 = torch.nn.Parameter(torch.full((1,), alpha2),requires_grad=False);
            print("Fixed alpha enabled. Alpha values will not be updated during training.")
        else:
            self.alpha1 = torch.nn.Parameter(torch.full((1,), alpha1),requires_grad=True); 
            self.alpha2 = torch.nn.Parameter(torch.full((1,), alpha2),requires_grad=True); 
        
        if self.trainable_activation != "none":
            self.activation_layer = TrainableActivationLinear(
                in_features=self.in_features,
                out_features=self.out_features,
                activation_type=self.trainable_activation,
                bias=True,
                channel_wise=True,
            )

        self.projection = projection

        if self.projection:
            self.projector_layer = nn.Linear(2*out_features,out_features)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                (
                    torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                    - 1 / 2
                )
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                # torch.nn.init.constant_(self.spline_scaler, self.scale_spline)
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)
    def b_splines(self, x: torch.Tensor):
        """
        Compute the B-spline bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = (
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()
    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).

        Returns:
            torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(
            0, 1
        )  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
        solution = torch.linalg.lstsq(
            A, B
        ).solution  # (in_features, grid_size + spline_order, out_features)
        result = solution.permute(
            2, 0, 1
        )  # (out_features, in_features, grid_size + spline_order)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()
    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )
    @property
    def mixture_weights(self):
        with torch.no_grad():
            gammas = torch.cat([self.alpha1, self.alpha2])
            p = 2 * F.softmax(gammas, dim=0)
            return {
                'alpha1': self.alpha1.item(),
                'alpha2': self.alpha2.item(),
                'p_base': p[0].item(),
                'p_spline': p[1].item()
            }

    def forward(self, x: torch.Tensor):
        assert x.size(-1) == self.in_features
        original_shape = x.shape
        x = x.reshape(-1, self.in_features)

        base_output = F.linear(self.base_activation(x), self.base_weight)

        
        if self.trainable_activation != "none":
            spline_output = self.activation_layer(x)
        else:
            spline_output = F.linear(
                self.b_splines(x).view(x.size(0), -1),
                self.scaled_spline_weight.view(self.out_features, -1),
            )


        gammas = torch.cat([self.alpha1, self.alpha2]); 

        # Shape: [flattened_batch, 2 * out_features]
        if self.dyanmic_alpha:
            gate_input = torch.cat(
                [base_output, spline_output],
                dim=-1,
            )
            gate_logits = self.dynamic_layer(gate_input)
            p = 2.0 * F.softmax(gate_logits, dim=-1)

            p_base = p[:, 0:1]
            p_spline = p[:, 1:2]
        
        elif self.fixed_alpha:
            p = gammas
        else:
            if self.independent_alpha:
                p = gammas
            else:
                p = 2*F.softmax(gammas, dim=0)

        if self.dyanmic_alpha:
            output1 = p_base*base_output 
            output2 = p_spline*spline_output
        else:
            output1 = p[0]*base_output 
            output2 = p[1]*spline_output

        if self.projection:
            output = self.projector_layer(torch.cat((output1, output2), dim=1))
        else:
            output = output1 + output2

        # output = base_output + self.alpha2*spline_output
        output = output.reshape(*original_shape[:-1], self.out_features)
        return output

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        splines = self.b_splines(x)  # (batch, in, coeff)
        splines = splines.permute(1, 0, 2)  # (in, batch, coeff)
        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(
            1, 0, 2
        )  # (batch, in, out)

        # sort each channel individually to collect data distribution
        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(
                self.grid_size + 1, dtype=torch.float32, device=x.device
            ).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        Compute the regularization loss.

        This is a dumb simulation of the original L1 regularization as stated in the
        paper, since the original one requires computing absolutes and entropy from the
        expanded (batch, in_features, out_features) intermediate tensor, which is hidden
        behind the F.linear function if we want an memory efficient implementation.

        The L1 regularization is now computed as mean absolute value of the spline
        weights. The authors implementation also includes this term in addition to the
        sample-based regularization.
        """
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )


class GNUNet(torch.nn.Module):
    def __init__(
        self,
        layers_hidden,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
        alpha1=0.5,
        alpha2=0.5,
        projection=False,
        fixed_alpha=False,
        dyanmic_alpha=False,
        independent_alpha=False,
        trainable_activation = "none"
    ):
        super(GNUNet, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order
        
        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                GNUNetLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                    alpha1 = alpha1,
                    alpha2 = alpha2,
                    projection = projection,
                    fixed_alpha = fixed_alpha,
                    dyanmic_alpha = dyanmic_alpha,
                    independent_alpha = independent_alpha,
                    trainable_activation = trainable_activation
                )
            )

    def forward(self, x: torch.Tensor, update_grid=False):
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )
    
    def get_alphas(self):
        alphas = []
        for i, layer in enumerate(self.layers):
            gammas = torch.cat([layer.alpha1, layer.alpha2])
            p = 2 * F.softmax(gammas, dim=0)
            alphas.append({
                'layer': i,
                'alpha1': layer.alpha1.item(),
                'alpha2': layer.alpha2.item(),
                'p_base': p[0].item(),
                'p_spline': p[1].item()
            })
        return alphas
    

class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

class SimpleCNN(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(SimpleCNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(32 * 8 * 8, num_classes) # Defaulting to CIFAR size
    def forward(self, x):
        out = self.layer1(x); out = self.layer2(x)
        out = out.reshape(out.size(0), -1)
        if self.fc.in_features != out.shape[1]: self.fc = nn.Linear(out.shape[1], self.fc.out_features).to(x.device)
        return self.fc(out)

def get_model(model_name, num_classes, input_size=None, transfer_learning=True, freeze_layers=True,classifier='mlp', activation='silu',grid_size=5, spline_order=3, alpha1=0.5, alpha2=0.5, projection=False,fixed_alpha=False,dyanmic_alpha=False,independent_alpha=False,trainable_activation='none'):
    if model_name == 'mlp':
        return SimpleMLP(input_size=input_size, hidden_size=32, num_classes=num_classes) # hidden_size is a placeholder
    elif model_name == 'simple_cnn':
        return SimpleCNN(in_channels=3, num_classes=num_classes)

    activation_dict = {
        'relu': nn.ReLU,
        'leaky_relu': nn.LeakyReLU,
        'selu': nn.SELU,
        'silu': nn.SiLU,
        'elu': nn.ELU,
        'prelu': nn.PReLU,
        'tanh': nn.Tanh,
        'sigmoid': nn.Sigmoid,
        'softmax': nn.Softmax,
        'identity': nn.Identity
    }

    weights = 'IMAGENET1K_V1' if transfer_learning else None
    if model_name == 'resnet18': model = models.resnet18(weights=weights)
    elif model_name == 'resnet34': model = models.resnet34(weights=weights)
    elif model_name == 'resnet50': model = models.resnet50(weights=weights)
    elif model_name == 'vgg11': model = models.vgg11(weights=weights)
    elif model_name == 'vgg13': model = models.vgg13(weights=weights)
    elif model_name == 'vgg16': model = models.vgg16(weights=weights)
    elif model_name == 'vgg19': model = models.vgg19(weights=weights)
    elif model_name == 'densenet121': model = models.densenet121(weights=weights)
    elif model_name == 'densenet169': model = models.densenet169(weights=weights)
    elif model_name == 'densenet201': model = models.densenet201(weights=weights)
    elif model_name == 'densenet161': model = models.densenet161(weights=weights)
    elif model_name == 'mobilenet_v2': model = models.mobilenet_v2(weights=weights)
    elif model_name == 'googlenet': model = models.googlenet(weights=weights, aux_logits=True)
    else: raise ValueError(f"Model '{model_name}' not supported.")

    if transfer_learning:
        if freeze_layers:
            for param in model.parameters(): param.requires_grad = False
        if 'resnet' in model_name:
            # model.fc = nn.Linear(model.fc.in_features, num_classes)
            if classifier == 'mlp':
                # model.fc = nn.Linear(model.fc.in_features, num_classes)
                # model.fc = SimpleMLP(input_size=model.fc.in_features, hidden_size=num_classes, num_classes=num_classes)
                model.fc = nn.Linear(model.fc.in_features, num_classes)
            elif classifier == 'kan':
                model.fc = KAN(layers_hidden=[model.fc.in_features, num_classes], grid_size=grid_size, spline_order=spline_order, base_activation=torch.nn.SiLU)
            elif classifier == 'gnu':
                model.fc = GNUNet([model.fc.in_features, num_classes], grid_size=grid_size, spline_order=spline_order, base_activation=activation_dict[activation], alpha1=alpha1, alpha2=alpha2, projection=projection,fixed_alpha=fixed_alpha,dyanmic_alpha=dyanmic_alpha, independent_alpha=independent_alpha,trainable_activation=trainable_activation)

        elif 'vgg' in model_name:
            if classifier == 'mlp':
                model.classifier[6] = nn.Linear(model.classifier[6].in_features,num_classes)
                # model.classifier[6] = SimpleMLP(input_size=model.classifier[6].in_features, hidden_size=num_classes, num_classes=num_classes) 
            elif classifier == 'kan':
                model.classifier[6] = KAN([model.classifier[6].in_features, num_classes], grid_size=grid_size, spline_order=spline_order, base_activation=torch.nn.SiLU)
            elif classifier == 'gnu':
                model.classifier[6] = GNUNet([model.classifier[6].in_features, num_classes], grid_size=grid_size, spline_order=spline_order, base_activation=activation_dict[activation], alpha1=alpha1, alpha2=alpha2, projection=projection,fixed_alpha=fixed_alpha,dyanmic_alpha=dyanmic_alpha, independent_alpha=independent_alpha,trainable_activation=trainable_activation)
            else:
                raise ValueError(f"Classifier '{classifier}' not supported for VGG model.")
            
        elif 'densenet' in model_name:
            if classifier == 'mlp':
                model.classifier = nn.Linear(model.classifier.in_features, num_classes)
                # model.classifier = SimpleMLP(input_size=model.classifier.in_features, hidden_size=num_classes, num_classes=num_classes)
            elif classifier == 'kan':
                model.classifier = KAN([model.classifier.in_features, num_classes], grid_size=grid_size, spline_order=spline_order, base_activation=torch.nn.SiLU)
            elif classifier == 'gnu':
                model.classifier = GNUNet([model.classifier.in_features, num_classes], grid_size=grid_size, spline_order=spline_order, base_activation=activation_dict[activation], alpha1=alpha1, alpha2=alpha2, projection=projection,fixed_alpha=fixed_alpha,dyanmic_alpha=dyanmic_alpha, independent_alpha=independent_alpha,trainable_activation=trainable_activation)
            else:
                raise ValueError(f"Classifier '{classifier}' not supported for DenseNet model.")
        elif model_name == 'mobilenet_v2':
            if classifier == 'mlp':
                model.classifier[1] = nn.Linear(model.classifier[1].in_features,num_classes)
                # model.classifier[1] = SimpleMLP(input_size=model.classifier[1].in_features, hidden_size=num_classes, num_classes=num_classes)
            elif classifier == 'kan':
                model.classifier[1] = KAN([model.classifier[1].in_features, num_classes], grid_size=grid_size, spline_order=spline_order, base_activation=torch.nn.SiLU)
            elif classifier == 'gnu':
                model.classifier = GNUNet([model.classifier[1].in_features, num_classes], grid_size=grid_size, spline_order=spline_order, base_activation=activation_dict[activation], alpha1=alpha1, alpha2=alpha2, projection=projection,fixed_alpha=fixed_alpha,dyanmic_alpha=dyanmic_alpha, independent_alpha=independent_alpha,trainable_activation=trainable_activation)
        
        elif model_name == 'googlenet':
            if classifier == 'mlp':
                model.fc = nn.Linear(model.fc.in_features,num_classes)
                # model.fc = SimpleMLP(input_size=model.fc.in_features, hidden_size=num_classes, num_classes=num_classes)
            elif classifier == 'kan':
                model.fc = KAN(layers_hidden=[model.fc.in_features, num_classes], grid_size=grid_size, spline_order=spline_order, base_activation=torch.nn.SiLU)
            elif classifier == 'gnu':
                model.fc = GNUNet([model.fc.in_features, num_classes], grid_size=grid_size, spline_order=spline_order, base_activation=activation_dict[activation], alpha1=alpha1, alpha2=alpha2, projection=projection,fixed_alpha=fixed_alpha,dyanmic_alpha=dyanmic_alpha, independent_alpha=independent_alpha,trainable_activation=trainable_activation)
                
    return model
