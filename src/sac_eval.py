from numpy import ndarray
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SACEvaluator:
    """
    Helper class to render the behaviour of a trained agent
    """

    def __init__(self, filename: str, max_action: float) -> None:
        # load policy from file
        self.policy = torch.load(filename, map_location=device)
        self.max_action = max_action

    def select_action(self, state: ndarray) -> ndarray:
        """
        Select one action by calling the forward function of the policy with the deterministic flag
        """
        t_state = torch.tensor(state, device=device, dtype=torch.float32)
        assert t_state.dim() == 1
        return (
            self.max_action
            * self.policy(t_state.unsqueeze(0), deterministic=True)[0].detach().cpu().numpy()
        )
