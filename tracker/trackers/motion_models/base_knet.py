import torch


class KF:

    def __init__(self, ):
        self.x = None
        pass


class BaseKNet:

    def __init__(
        self,
        motion_model,
        img_size,
    ):
        # device = "cuda:0"
        # weights_path, config_path = get_path_ckpt_config(motion_model, mode='max')
        # # print(weights_path)
        # weights = torch.load(weights_path)
        # config = Config.fromfile(config_path)
        # motion_model = MODELS.build(dict(type = config.TRAINER.type, cfg = config, save_dir = {}))
        # motion_model = load_state_dict_from_pl(weights['state_dict'], motion_model).model
        # motion_model.eval()

        # motion_model = motion_model.to(device)
        self.model = motion_model['motion_model']
        self.model.load_state_dict(motion_model['weights'], strict=True)
        self.model.to('cuda:0')
        self.model.eval()

        self.device = next(self.model.parameters()).device

        self.kf = KF()
        self.img_size = img_size

    def cv_state_to_xywh(self, x: torch.Tensor):
        """Select the x, y, w, h from the constant velocity state vector \

            [x, x', y, y', w, w', h, h']

        Args:
            x (torch.Tensor | np.ndarray): [x, x', y, y', w, w', h, h']

        Returns:
            _type_: [x, y, w, h]
        """
        x = x[[0, 2, 4, 6]]
        return x

    def _norm_box(self, x):
        width = self.img_size[0]
        height = self.img_size[1]
        norm_data = torch.tensor([width, height, width, height]).to(torch.float32)
        return x / norm_data

    def _denorm_box(self, x):
        width = self.img_size[0]
        height = self.img_size[1]
        norm_data = torch.tensor([width, height, width, height]).to(torch.float32)
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
        observation = observation.view(1, -1, 1).contiguous()
        return observation

    def postprocess(self, state: torch.Tensor) -> torch.Tensor:
        state = self.cv_state_to_xywh(state.flatten().detach().cpu())
        state = self._denorm_box(state).numpy()
        return state

    def initialize(self, observation: torch.Tensor) -> None:
        """Initialize the learning-aided model with the first observation.

        Args:
            observation (torch.tensor | np.ndarray): [x, y, w, h]
        """
        self.model.init_beliefs(self.preprocess(observation))
        x = self.postprocess(self.model.state)
        self.kf.x = x

    def predict(self, is_activated=True):
        if not is_activated:
            # if not activated, set the velocity of h to 0
            # [x, x', y, y', w, w', h, h']
            self.model.state[..., [-1], 0] = 0.0
        self.model.filter_predict_step(self.model.state)
        x = self.postprocess(self.model.state)
        self.kf.x = x

    def update(self, observation, **kwargs):
        self.model.filter_update_step(self.preprocess(observation))
        x = self.postprocess(self.model.state)
        self.kf.x = x

    def get_state(self, ):
        return self.kf.x
