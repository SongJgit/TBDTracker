import torch
import numpy as np


class KF:

    def __init__(self, ):
        self.x = None
        self.P = np.zeros((8, 8), dtype=np.float32)
        pass


class BaseLAKF:
    order_by_dim = False

    def __init__(
        self,
        motion_model,
        img_size,
    ):

        self.model = motion_model
        self.device = next(self.model.parameters()).device

        self.kf = KF()

        self.img_size = img_size

        self.model_attr = dict()

    def cv_state_to_xywh(self, x: torch.Tensor):
        """Select the x, y, w, h from the constant velocity state vector \

            [x, x', y, y', w, w', h, h']

        Args:
            x (torch.Tensor | np.ndarray): [x, x', y, y', w, w', h, h']

        Returns:
            _type_: [x, y, w, h, x',y',w',h']
        """
        x = x[[0, 2, 4, 6, 1, 3, 5, 7]]
        return x

    def _norm_box(self, x):
        width = self.img_size[0]
        height = self.img_size[1]
        norm_data = torch.tensor([width, height, width, height]).to(torch.float32)
        return x / norm_data

    def _denorm_box(self, x):
        width = self.img_size[0]
        height = self.img_size[1]
        norm_data = torch.tensor([width, height, width, height, width, height, width, height]).to(torch.float32)
        return x * norm_data

    def preprocess(self, observation: torch.Tensor) -> torch.Tensor:
        """Normalize the observation to [0, 1] and convert to tensor.

        Args:
            observation (torch.tensor | np.ndarray): _description_

        Returns:
            torch.Tensor: _description_
        """
        observation = torch.tensor(observation).to(torch.float32)
        observation = self._norm_box(observation).to(self.device)
        observation = observation.view(1, -1, 1).contiguous()  # b, n, 1
        return observation

    def x2state(self) -> torch.Tensor:
        width = self.img_size[0]
        height = self.img_size[1]
        norm_data = torch.tensor([width, width, height, height, width, width, height,
                                  height]).to(torch.float32).to(self.device)
        x = torch.tensor(self.kf.x[[0, 4, 1, 5, 2, 6, 3, 7]]).to(torch.float32).to(self.device)
        x = x / norm_data

        return x.view(1, -1, 1).contiguous()

    def postprocess(self, state: torch.Tensor) -> torch.Tensor:
        # Convert the state to [x, y, w, h, x', y', w', h'] and denorm.
        state = self.cv_state_to_xywh(state.flatten().detach().cpu())
        state = self._denorm_box(state).numpy().copy()
        return state

    def initialize(self, observation: torch.Tensor) -> None:
        """Initialize the learning-aided model with the first observation.

        Args:
            observation (torch.tensor | np.ndarray): [x, y, w, h]
        """
        self.model.init_beliefs(self.preprocess(observation))
        x = self.postprocess(self.model.state)
        self.kf.x = x
        self.model_attr = self.model._get_attribute()

    def predict(self, is_activated=True):
        self.model._put_attribute(self.model_attr)

        if not is_activated:
            # if not activated, set the velocity of w and h to 0
            # [x, x', y, y', w, w', h, h']
            self.model.state[..., [-1], 0] = 0.0
            self.model.state[..., [-3], 0] = 0.0

        self.model.filter_predict_step(self.model.state)
        x = self.postprocess(self.model.state)
        self.kf.x = x
        self.model_attr = self.model._get_attribute()

    def update(self, observation, **kwargs):
        self.model._put_attribute(self.model_attr)
        self.model.state = self.x2state()
        self.model.filter_update_step(self.preprocess(observation))
        x = self.postprocess(self.model.state)
        self.kf.x = x
        self.model_attr = self.model._get_attribute()

    def get_state(self, ):
        return self.kf.x
