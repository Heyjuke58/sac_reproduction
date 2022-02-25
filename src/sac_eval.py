from numpy import ndarray
import torch
from src.networks import Policy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SACEvaluator:
    def __init__(self, filename: str, max_action: float) -> None:
        self.policy = torch.load(filename, map_location=device)
        self.max_action = max_action

    def select_action(self, state: ndarray) -> ndarray:
        t_state = torch.tensor(state, device=device, dtype=torch.float32)
        # t_state = torch.from_numpy(state).to(device)
        assert t_state.dim() == 1
        return (
            self.max_action
            * self.policy(t_state.unsqueeze(0), deterministic=True)[0].detach().cpu().numpy()
        )
