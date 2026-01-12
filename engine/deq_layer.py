import torch
import torch.autograd as autograd
from .solver import solve_equilibrium

class DEQLayer(autograd.Function):
    @staticmethod
    def forward(ctx, f, z0, policy):
        # Solve equilibrium WITHOUT tracking path
        with torch.no_grad():
            z_star = solve_equilibrium(f, z0, policy)

        # Re-attach graph at equilibrium
        z_star = z_star.detach().requires_grad_(True)

        ctx.f = f
        ctx.policy = policy
        ctx.save_for_backward(z_star)

        return z_star

    @staticmethod
    def backward(ctx, grad_output):
        f = ctx.f
        policy = ctx.policy
        (z_star,) = ctx.saved_tensors

        # Define Jacobian-vector product
        def Jv(v):
            f_z = f(z_star, policy)
            Jv = autograd.grad(
                f_z, z_star, v,
                retain_graph=True
            )[0]
            return v - Jv

        # Solve (I - J_f)^T v = grad_output
        v = grad_output
        for _ in range(25):
            v = grad_output + Jv(v)

        # Compute gradients wrt parameters
        grads = autograd.grad(
            f(z_star, policy),
            tuple(f.parameters()),
            v,
            retain_graph=False
        )

        # Return gradients in correct structure
        return None, None, None, *grads
