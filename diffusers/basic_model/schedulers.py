import torch


def linear_diffusion_scheduler(diffusion_times: torch.Tensor):
    """
    Computes the linear diffusion schedule for a given set of diffusion times.

    Args:
        diffusion_times (torch.Tensor): A tensor
        containing the diffusion times.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - signal_noise (torch.Tensor): The noise component of the signal.
            - signal_rate (torch.Tensor): The rate of the signal.
    """
    min_rate = 1e-4
    max_rate = 0.02

    betas = min_rate + (max_rate - min_rate) * diffusion_times

    alphas = 1 - betas
    signal_rate = torch.cumprod(alphas, dim=0)
    signal_noise = 1 - signal_rate

    return signal_noise, signal_rate


def cosine_diffusion_scheduler(diffusion_times: torch.Tensor):
    """
    Computes the cosine diffusion schedule for given diffusion times.

    Args:
        diffusion_times (torch.Tensor): A tensor
        containing the diffusion times.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing two tensors:
            - signal_noise: The noise component of the signal,
            computed as sin(diffusion_times * 0.5 * pi).
            - signal_rate: The rate component of the signal,
            computed as cos(diffusion_times * 0.5 * pi).
    """
    signal_rate = torch.cos(diffusion_times * 0.5 * torch.pi)
    signal_noise = torch.sin(diffusion_times * 0.5 * torch.pi)

    return signal_noise, signal_rate


def offset_cosine_diffusion_scheduler(diffusion_times: torch.Tensor):
    """
    Computes the signal noise and signal rate for a given tensor of
    diffusion times using an offset cosine schedule.

    Args:
        diffusion_times (torch.Tensor): A tensor of diffusion times,
        typically in the range [0, 1].

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing two tensors:
            - signal_noise (torch.Tensor): The computed signal noise values.
            - signal_rate (torch.Tensor): The computed signal rate values.
    """
    min_signal_rate = torch.tensor(0.02)
    max_signal_rate = torch.tensor(0.95)

    start_angle = torch.acos(max_signal_rate)
    end_angle = torch.acos(min_signal_rate)

    diffusion_angles = start_angle + (end_angle - start_angle) * diffusion_times

    signal_rate = torch.cos(diffusion_angles)
    signal_noise = torch.sin(diffusion_angles)

    return signal_noise, signal_rate
