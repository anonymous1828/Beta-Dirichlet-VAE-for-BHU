import torch
import torch.nn as nn


class SADLoss:
    """
        Spectral Angle Distance (SAD) loss function.
        
        Parameters
        ----------
        eps : avoid division by zero
        reduction : reduction method for the loss. 
                    Options are "sum", "mean" or "none" 
                            (the latter returns a tensor of size (..., n_samples)).
        Returns
        -------
        out: shape (n_batch, 
             spectral angle distance 
    """
    def __init__(self, 
                 reduction: str = "sum",
                 eps: float = 1e-8) -> None:

        self.reduction = reduction
        self.eps       = torch.tensor(eps)

    def __repr__(self):
        return f"{self.__class__.__name__}(reduction='{self.reduction}', eps={self.eps})"

    def to(self, device):
        self.eps.to(device)
        return self

    def __call__(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

        if len(target.shape) < 2: target = target.unsqueeze(0)
        if len(input.shape)  < 2: input  = input.unsqueeze(0)

        target_norm = torch.norm(target, p=2, dim=1)
        input_norm  = torch.norm(input,  p=2, dim=1)
        norm_factor = target_norm * input_norm
        
        scalar_product = torch.sum(target * input, dim=1)

        # eps at denominator + 1e-6 in cos for numerical stability
        cos = scalar_product / torch.max(norm_factor, self.eps) 
        cos = torch.clamp(cos, -1 + 1e-6, 1 - 1e-6)

        if   self.reduction == "sum":  loss = torch.acos(cos).sum()
        elif self.reduction == "mean": loss = torch.acos(cos).mean()
        elif self.reduction == "none": loss = torch.acos(cos) 
        else:
            raise ValueError("Invalid reduction type. Must be either'sum', 'none' or 'mean'.")
        
        return loss



class GammaKL:
    """
        KL divergence closed form for 2 Dirichlet distribution.

        Analytical form:
        Let P and Q be 2 multiGamma distributions of parameters
        $\alpha, \beta$ (resp. $\alpha, \beta$) where $\sum_{i=1}^K \alpha_i = 1$
        and $\beta > 0$, (resp. $\sum_{i=1}^K \hat \alpha_i =1 \text{ and } \beta > 0$).
        Based on the paper the analytical form is derived as follows
        $$
            D_{KL} (Q \| P) =   \sum_{i=1}^K log(\Gamma(\hat \alpha_i)) \\
                              - \sum_{i=1}^K log( \Gamma( \alpha_i ) ) 
                              + \sum_{i=1}^K ( \hat \alpha_i - \alpha_i ) \psi(\hat \alpha_i)
        $$ 
        where $\psi(\cdot)$ is the derivative of the gamma function (digamma function). 
    """
    def __init__(self, 
                 alphas: torch.Tensor, 
                 reduction: str = "sum"):
        """
            Provide alphas of target distribution
        """
        self.alphas    = alphas.to(dtype=torch.float32)
        self.reduction = reduction

    def to(self, device):
        self.alphas = self.alphas.to(device)
        return self

    def __repr__(self):
        return f"GammaKL({self.alphas}, beta assumed to be the same as target)"

    def __call__(self, 
                 input: torch.Tensor):
        """
            Args:
                input:  moments of shape (batch_size, n_ems)
            Returns:
                output: shape (batch_size,)
        """
        #alphas shape (1, x) --> shape (batch_size, x)
        batch_size = input.shape[0]
        alphas = self.alphas.expand(batch_size, -1)

        loss  = torch.sum(torch.lgamma(alphas), dim=1)
        loss -= torch.sum(torch.lgamma(input), dim=1)
        loss += torch.sum((input - alphas) * torch.digamma(input), dim=1)

        if   self.reduction   == "sum":  loss = loss.sum()
        elif self.reduction == "mean": loss = loss.mean()
        elif self.reduction == "none": pass
        else:
            raise ValueError("Invalid reduction mode")

        return loss


if __name__ == "__main__":
    target = torch.tensor([[4., 3., 3.], 
                           [2., 4., 4.]])
    input  = torch.tensor([[4., 3., 3.], 
                           [4., 3., 3.]])

    sad_loss = SADLoss(reduction="sum")
    
    print(sad_loss)
    print(sad_loss(input, target))  