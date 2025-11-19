import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
"""
store all the agents here
"""
from replay_buffer import ReplayBuffer, ReplayBufferNumpy
import numpy as np
import time
import pickle
from collections import deque
import json



class Agent():
    """Base class for all agents
    This class extends to the following classes
    DeepQLearningAgent
    HamiltonianCycleAgent
    BreadthFirstSearchAgent

    Attributes
    ----------
    _board_size : int
        Size of board, keep greater than 6 for useful learning
        should be the same as the env board size
    _n_frames : int
        Total frames to keep in history when making prediction
        should be the same as env board size
    _buffer_size : int
        Size of the buffer, how many examples to keep in memory
        should be large for DQN
    _n_actions : int
        Total actions available in the env, should be same as env
    _gamma : float
        Reward discounting to use for future rewards, useful in policy
        gradient, keep < 1 for convergence
    _use_target_net : bool
        If use a target network to calculate next state Q values,
        necessary to stabilise DQN learning
    _input_shape : tuple
        Tuple to store individual state shapes
    _board_grid : Numpy array
        A square filled with values from 0 to board size **2,
        Useful when converting between row, col and int representation
    _version : str
        model version string
    """
    def __init__(self, board_size=10, frames=2, buffer_size=10000,
                gamma=0.99, n_actions=3, use_target_net=True,
                version=''):
        """ initialize the agent

        Parameters
        ----------
        board_size : int, optional
            The env board size, keep > 6
        frames : int, optional
            The env frame count to keep old frames in state
        buffer_size : int, optional
            Size of the buffer, keep large for DQN
        gamma : float, optional
            Agent's discount factor, keep < 1 for convergence
        n_actions : int, optional
            Count of actions available in env
        use_target_net : bool, optional
            Whether to use target network, necessary for DQN convergence
        version : str, optional except NN based models
            path to the model architecture json
        """
        self._board_size = board_size
        self._n_frames = frames
        self._buffer_size = buffer_size
        self._n_actions = n_actions
        self._gamma = gamma
        self._use_target_net = use_target_net
        self._input_shape = (self._board_size, self._board_size, self._n_frames)
        # reset buffer also initializes the buffer
        self.reset_buffer()
        self._board_grid = np.arange(0, self._board_size**2)\
                            .reshape(self._board_size, -1)
        self._version = version

    def get_gamma(self):
        """Returns the agent's gamma value

        Returns
        -------
        _gamma : float
            Agent's gamma value
        """
        return self._gamma

    def reset_buffer(self, buffer_size=None):
        """Reset current buffer 
        
        Parameters
        ----------
        buffer_size : int, optional
            Initialize the buffer with buffer_size, if not supplied,
            use the original value
        """
        if (buffer_size is not None):
            self._buffer_size = buffer_size
        self._buffer = ReplayBufferNumpy(self._buffer_size, self._board_size, 
                                    self._n_frames, self._n_actions)

    def get_buffer_size(self):
        """Get the current buffer size
        
        Returns
        -------
        buffer size : int
            Current size of the buffer
        """
        return self._buffer.get_current_size()

    def add_to_buffer(self, board, action, reward, next_board, done, legal_moves):
        """Add current game step to the replay buffer

        Parameters
        ----------
        board : Numpy array
            Current state of the board, can contain multiple games
        action : Numpy array or int
            Action that was taken, can contain actions for multiple games
        reward : Numpy array or int
            Reward value(s) for the current action on current states
        next_board : Numpy array
            State obtained after executing action on current state
        done : Numpy array or int
            Binary indicator for game termination
        legal_moves : Numpy array
            Binary indicators for actions which are allowed at next states
        """
        self._buffer.add_to_buffer(board, action, reward, next_board, 
                                done, legal_moves)

    def save_buffer(self, file_path='', iteration=None):
        """Save the buffer to disk

        Parameters
        ----------
        file_path : str, optional
            The location to save the buffer at
        iteration : int, optional
            Iteration number to tag the file name with, if None, iteration is 0
        """
        if (iteration is not None):
            assert isinstance(iteration, int), "iteration should be an integer"
        else:
            iteration = 0
        with open("{}/buffer_{:04d}".format(file_path, iteration), 'wb') as f:
            pickle.dump(self._buffer, f)

    def load_buffer(self, file_path='', iteration=None):
        """Load the buffer from disk
        
        Parameters
        ----------
        file_path : str, optional
            Disk location to fetch the buffer from
        iteration : int, optional
            Iteration number to use in case the file has been tagged
            with one, 0 if iteration is None

        Raises
        ------
        FileNotFoundError
            If the requested file could not be located on the disk
        """
        if(iteration is not None):
            assert isinstance(iteration, int), "iteration should be an integer"
        else:
            iteration = 0
        with open("{}/buffer_{:04d}".format(file_path, iteration), 'rb') as f:
            self._buffer = pickle.load(f)

    def _point_to_row_col(self, point):
        """Covert a point value to row, col value
        point value is the array index when it is flattened

        Parameters
        ----------
        point : int
            The point to convert

        Returns
        -------
        (row, col) : tuple
            Row and column values for the point
        """
        return (point//self._board_size, point%self._board_size)

    def _row_col_to_point(self, row, col):
        """Covert a (row, col) to value
        point value is the array index when it is flattened

        Parameters
        ----------
        row : int
            The row number in array
        col : int
            The column number in array
        Returns
        -------
        point : int
            point value corresponding to the row and col values
        """
        return row*self._board_size + col


# PyTorch DQN Model
class DQNNet(nn.Module):
    """PyTorch implementation of a DQN network constructed from a JSON config.

    This class mimics a Keras-style model definition stored as JSON and converts
    each supported layer into its PyTorch equivalent. It produces Q-values for
    each action from an input state (board frames).

    Parameters
    ----------
    board_size : int
        Size of the board (height/width). Kept for context; actual spatial sizes
        are inferred automatically.
    n_frames : int
        Number of frames (channels) in the input state.
    n_actions : int
        Number of discrete actions; the network outputs one Q-value per action.
    config_path : str
        Path to the JSON config describing the model architecture. The JSON is
        expected to contain a "model" dictionary with Keras-like layer entries.
    """
    def __init__(self, board_size, n_frames, n_actions, config_path):
        super().__init__()
        # Keep frames so forward() can detect NHWC vs NCHW layout
        self._n_frames = n_frames
        
        # Load the Keras-like model specification
        with open(config_path, 'r') as f:
            m = json.load(f)
        layers = []
        
        # Track current number of channels for Conv2d in_channels
        in_channels = n_frames
        
        def _as_tuple(x):
            """Ensure hyperparameters are tuples.

            Keras JSON often stores these as a single int or a 2-list; PyTorch Conv2d
            expects a 2-tuple.
            
            Parameters
            ----------
            x : int or list or tuple
                Hyperparameter to convert.
                
            Returns
            -------
            tuple
                Tuple representation of the hyperparameter.
            """
            
            if isinstance(x, (list, tuple)):
                return tuple(x)
            return (x, x)
        
        def _pad_for_same(kernel_size):
            """
            Compute 'same' padding for given kernel size.
            
            Parameters
            ----------
            kernel_size : int or tuple
                Size of the convolution kernel.
                
            Returns
            -------
            pad : tuple
                Tuple of (pad_height, pad_width) for 'same' padding.
            """

            k = _as_tuple(kernel_size)
            # For stride=1, "same" padding is kernel//2
            return (k[0] // 2, k[1] // 2)
        
        # Build the PyTorch module list from the JSON layer-by-layer
        for layer_name, l in m['model'].items():
            if 'Conv2D' in layer_name:
                # Map Keras Conv2D to PyTorch Conv2d
                filters = l['filters']
                kernel_size = _as_tuple(l['kernel_size'])
                padding_cfg = l.get('padding', 'valid') # only first conv sets "same" in the JSON file
                pad = _pad_for_same(kernel_size) if isinstance(padding_cfg, str) and padding_cfg.lower() == 'same' else (0, 0)
                    
                layers.append(nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=filters,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=pad,
                    bias=l.get('use_bias', True)
                ))
                # Your JSON always uses ReLU; keep it simple
                layers.append(nn.ReLU(inplace=True))
                in_channels = filters
                
            elif 'Flatten' in layer_name:
                layers.append(nn.Flatten())
                
            elif 'Dense' in layer_name:
                # Map Keras Dense to PyTorch Linear; use LazyLinear to infer in_features
                units = l['units']
                layers.append(nn.LazyLinear(units))
                layers.append(nn.ReLU(inplace=True))
            
        # Final output layer: one Q-value per action (linear activation)
        layers.append(nn.LazyLinear(n_actions))
        # Wrap all layers as a single Sequential model
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass that is robust to Keras (NHWC) vs PyTorch (NCHW) layouts.

        Accepts either:
        - NumPy arrays in NHWC (batch, height, width, channels) — common from Keras/envs.
        - Torch tensors in either NHWC or NCHW.

        Steps:
        1) If input is NumPy, convert to torch.Tensor.
        2) If 4D and last dim equals n_frames, assume NHWC and permute to NCHW.
        3) Cast to float32 and move to the model's device.
        4) Run through the assembled nn.Sequential and return Q-values.
        
        Returns
        -------
        q_values : torch.Tensor
            Tensor of shape [batch_size, n_actions] containing Q-values for each action.
        """
        
        # Allow NumPy input for convenience; convert to Torch tensor
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Unsupported input type: {type(x)}")
        
        # If tensor is NHWC (channels-last), permute to NCHW for Conv2d
        if x.ndim == 4 and x.shape[-1] == self._n_frames and (x.shape[1] != self._n_frames):
            x = x.permute(0, 3, 1, 2).contiguous()
            
        # Use float32 compute and ensure the tensor is on the same device as the model
        x = x.float().to(next(self.model.parameters()).device)
        
        # Forward through the model (produces [batch, n_actions] Q-values)
        return self.model(x)

class DeepQLearningAgent(Agent):
    """PyTorch implementation of a Deep Q-Learning agent.

    This agent mirrors the original TensorFlow DQN design:
    - A Q-network that predicts Q-values for each action given a state.
    - An optional target network for stable bootstrapping.
    - Experience replay buffer for training batches.
    - Huber (smooth L1) loss computed only on the chosen action.

    Parameters
    ----------
    board_size : int, optional
        Size of the board; should match the environment's board size.
    frames : int, optional
        Number of frames stacked in the input state.
    buffer_size : int, optional
        Replay buffer capacity; should be large for DQN.
    gamma : float, optional
        Discount factor in [0, 1); controls future reward weighting.
    n_actions : int, optional
        Number of discrete actions in the environment.
    use_target_net : bool, optional
        If True, use a separate target network for the bootstrap term.
    version : str, optional
        Model version string; used to resolve a JSON architecture file.
    use_amp : bool, optional
        If True and a CUDA device is available, enable automatic mixed precision.
    """
    
    def __init__(self, board_size=10, frames=4, buffer_size=10000,
                gamma=0.99, n_actions=3, use_target_net=True,
                version='17.1', use_amp: bool = True):
        
        # Initialize base Agent (buffer, shapes, metadata)
        Agent.__init__(self, board_size=board_size, frames=frames, buffer_size=buffer_size,
                gamma=gamma, n_actions=n_actions, use_target_net=use_target_net,
                version=version)
        
        # Select compute device: 'cuda' when available, else 'cpu'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Resolve model architecture from JSON config
        self.config_path = f'model_config/{version}.json'
        
        # Enable AMP only on CUDA; ROCm builds also use device string "cuda" in PyTorch
        self.use_amp = use_amp and self.device.type == 'cuda'
        
        
        # Create a GradScaler with broad compatibility across torch versions:
        # - First try the newer "device_type" kw
        # - Then try "device"
        # - Finally fall back to default GradScaler(enabled=True)
        if self.use_amp:
            scaler = None
            try:
                scaler = torch.amp.GradScaler(device="cuda", enabled=True)  # PyTorch >= 2.1+
                print("Using torch.amp.GradScaler with device_type")
            except TypeError:
                scaler = torch.cuda.amp.GradScaler(enabled=True)                  # Older/new stubs fallback
                print("Using torch.amp.GradScaler default constructor") 
            self.scaler = scaler
        else:
            self.scaler = None
            
        # Build models, optimizer, and warm up lazy layers
        self.reset_models()

    def reset_models(self):
        
        # Main Q-network
        self._model = DQNNet(self._board_size, self._n_frames, self._n_actions, self.config_path).to(self.device)
        
        # Optional target network (periodically synced from main model)
        if self._use_target_net:
            self._target_net = DQNNet(self._board_size, self._n_frames, self._n_actions, self.config_path).to(self.device)
            self.update_target_net()
            
        # Optimizer: RMSprop to mirror the original TF setup
        self.optimizer = optim.RMSprop(self._model.parameters(), lr=0.0005)
        
        # Warm up LazyLinear layers by running a dummy forward once.
        # This resolves in_features for LazyLinear and avoids lazy-init cost later.
        with torch.inference_mode():
            dummy = torch.zeros(1, self._board_size, self._board_size, self._n_frames, device=self.device)
            _ = self._model(dummy)
            if self._use_target_net:
                _ = self._target_net(dummy)
        
    def _normalize_board(self, board):
        """Normalize board/state values to float32 in [scaled range] for stable training.

        Torch path: in-place float cast and scale to minimize copies.
        Numpy path: return a float32 array for consistency with torch.from_numpy.
        """
        
        # Torch-first normalization to avoid numpy copies
        if isinstance(board, torch.Tensor):
            return board.float().mul_(1.0/4.0)
        return board.astype(np.float32) / 4.0

    def _prepare_input(self, board):
        """Ensure input is batched and normalized.

        - If board is HxWxC, add batch dimension to make 1xHxWxC.
        - Normalize values using _normalize_board.
        - Return the same type (numpy or torch) received.
        """
        
        # Returns NHWC numpy for inference helper, but training uses torch
        if isinstance(board, np.ndarray):
            if board.ndim == 3:
                board = board.reshape((1,) + self._input_shape)
            board = self._normalize_board(board)
            return board
        elif isinstance(board, torch.Tensor):
            if board.ndim == 3:
                board = board.unsqueeze(0)
            return self._normalize_board(board)
        else:
            raise TypeError("Unsupported board type")

    def _to_device_tensor(self, arr, dtype=torch.float32):
        """Convert numpy/torch input to a torch.Tensor on the agent's device.

        - Keeps dtype consistent.
        - Uses non_blocking transfers when possible (beneficial with pinned memory).
        """
        
        if isinstance(arr, np.ndarray):
            t = torch.from_numpy(arr)
        elif isinstance(arr, torch.Tensor):
            t = arr
        else:
            raise TypeError("Expected numpy array or torch tensor")
        if t.dtype != dtype:
            t = t.to(dtype)
        return t.to(self.device, non_blocking=True)

    def _to_chw(self, t: torch.Tensor) -> torch.Tensor:
        """Convert NHWC to NCHW if needed (PyTorch Conv2d expects channels-first)."""
        # Convert NHWC -> NCHW on device; do nothing if already NCHW
        if t.ndim == 4 and t.shape[-1] == self._n_frames:
            return t.permute(0, 3, 1, 2).contiguous()
        return t

    def _get_model_outputs(self, board, model=None):
        """Vectorized inference helper for move/get_action_proba.

        - Prepares and normalizes input.
        - Moves to device, converts to NCHW.
        - Executes model in inference mode.
        - Returns numpy Q-values [batch, n_actions].
        """
        board = self._prepare_input(board)
        if model is None:
            model = self._model
        model.eval()
        with torch.inference_mode():
            x = self._to_device_tensor(board)  # NHWC float
            x = self._to_chw(x)                # NCHW
            q_values = model(x)                # [B, A]
            return q_values.detach().cpu().numpy()

    def move(self, board, legal_moves, value=None):
        """Select action greedily among legal moves (argmax over masked Q-values)."""
        model_outputs = self._get_model_outputs(board, self._model) # numpy [B, A]
        return np.argmax(np.where(legal_moves==1, model_outputs, -np.inf), axis=1)

    def get_action_proba(self, board, values=None):
        """Convert Q-values to a softmax distribution (for diagnostics/visualization)."""
        model_outputs = self._get_model_outputs(board, self._model)
        model_outputs = np.clip(model_outputs, -10, 10)
        model_outputs = model_outputs - model_outputs.max(axis=1).reshape((-1,1))
        model_outputs = np.exp(model_outputs)
        model_outputs = model_outputs/model_outputs.sum(axis=1, keepdims=True)
        return model_outputs

    def save_model(self, file_path='', iteration=None):
        """Save main and target network weights to disk."""
        if iteration is not None:
            assert isinstance(iteration, int), "iteration should be an integer"
        else:
            iteration = 0
        torch.save(self._model.state_dict(), f"{file_path}/model_{iteration:04d}.pth")
        if self._use_target_net:
            torch.save(self._target_net.state_dict(), f"{file_path}/model_{iteration:04d}_target.pth")

    def load_model(self, file_path='', iteration=None):
        """Load main and target network weights from disk."""
        if iteration is not None:
            assert isinstance(iteration, int), "iteration should be an integer"
        else:
            iteration = 0
        self._model.load_state_dict(torch.load(f"{file_path}/model_{iteration:04d}.pth", map_location=self.device))
        if self._use_target_net:
            self._target_net.load_state_dict(torch.load(f"{file_path}/model_{iteration:04d}_target.pth", map_location=self.device))

    def train_agent(self, batch_size=32, num_games=1, reward_clip=False):
        """Sample a batch from the replay buffer, compute the DQN loss, and take an optimizer step.

        Steps:
        1) Sample (s, a(one-hot), r, next_s, done, legal_moves).
        2) Normalize and move tensors to device; convert NHWC->NCHW for model input.
        3) Compute Q(s, .) and target: r + gamma * max_a' Q_target(next_s, a') for legal a'.
        4) Compute Huber loss only on the chosen action via gather.
        5) Backprop with AMP if enabled; otherwise standard fp32.
        """
        
        # 1) Sample batch from buffer (numpy arrays)
        s, a, r, next_s, done, legal_moves = self._buffer.sample(batch_size)
        if reward_clip:
            r = np.sign(r)
        
        # 2) Convert to device tensors and normalize/layout fix
        s_t = self._to_device_tensor(s)            # NHWC
        next_s_t = self._to_device_tensor(next_s)  # NHWC
        r_t = self._to_device_tensor(r).squeeze(-1)         # [B]
        done_t = self._to_device_tensor(done).squeeze(-1)   # [B]
        a_t = self._to_device_tensor(a)            # [B, A] one-hot or probs
        legal_t = self._to_device_tensor(legal_moves)  # [B, A]
        
        # Normalize and convert layout using torch ops (avoid numpy copies)
        s_t = self._normalize_board(s_t)
        next_s_t = self._normalize_board(next_s_t)
        s_t = self._to_chw(s_t)            # [B, C, H, W]
        next_s_t = self._to_chw(next_s_t)  # [B, C, H, W]

        # Boolean mask for legal moves
        legal_mask = legal_t > 0.5  # bool mask [B, A]

        self._model.train()
        
        # 3) Forward and target computation, optionally under AMP
        if self.use_amp:
            # Autocast to mixed precision (float16 by default); on ROCm, consider bfloat16 if more stable.
            try:
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                    q_values = self._model(s_t)  # [B, A]

                    with torch.no_grad():
                        # Choose network used for bootstrapping (target if available)
                        next_q_values = self._target_net(next_s_t) if self._use_target_net else self._model(next_s_t)

                        # Mask out illegal actions with a dtype-safe large negative (avoids fp16 overflow)
                        fill_val = torch.finfo(next_q_values.dtype).min
                        next_q_values = next_q_values.masked_fill(~legal_mask, fill_val)

                        # If any row has no legal moves, treat as terminal for the bootstrap term
                        any_legal = legal_mask.any(dim=1)      # [B] bool
                        max_next_q, _ = next_q_values.max(dim=1)  # [B]
                        max_next_q = torch.where(any_legal, max_next_q, torch.zeros_like(max_next_q))

                        discounted_reward = r_t + self._gamma * max_next_q * (1.0 - done_t)  # [B]
                    
                    # 4) Compute loss on chosen actions only
                    a_idx = a_t.argmax(dim=1, keepdim=True)  # [B, 1]
                    q_sa = q_values.gather(1, a_idx).squeeze(1)  # [B]
                    loss = F.smooth_l1_loss(q_sa, discounted_reward)
                    
            # Fallback for older torch versions without device_type kw
            except TypeError:
                with torch.cuda.amp.autocast(dtype=torch.float16, enabled=True):
                    q_values = self._model(s_t)  # [B, A]

                    with torch.no_grad():
                        # Choose network used for bootstrapping (target if available)
                        next_q_values = self._target_net(next_s_t) if self._use_target_net else self._model(next_s_t)

                        # Mask out illegal actions with a dtype-safe large negative (avoids fp16 overflow)
                        fill_val = torch.finfo(next_q_values.dtype).min
                        next_q_values = next_q_values.masked_fill(~legal_mask, fill_val)

                        # If any row has no legal moves, treat as terminal for the bootstrap term
                        any_legal = legal_mask.any(dim=1)      # [B] bool
                        max_next_q, _ = next_q_values.max(dim=1)  # [B]
                        max_next_q = torch.where(any_legal, max_next_q, torch.zeros_like(max_next_q))

                        discounted_reward = r_t + self._gamma * max_next_q * (1.0 - done_t)  # [B]
                    
                    # 4) Compute loss on chosen actions only
                    a_idx = a_t.argmax(dim=1, keepdim=True)  # [B, 1]
                    q_sa = q_values.gather(1, a_idx).squeeze(1)  # [B]
                    loss = F.smooth_l1_loss(q_sa, discounted_reward)
                
        else:
            q_values = self._model(s_t)  # [B, A]

            with torch.no_grad():
                next_q_values = self._target_net(next_s_t) if self._use_target_net else self._model(next_s_t)

                fill_val = torch.finfo(next_q_values.dtype).min
                next_q_values = next_q_values.masked_fill(~legal_mask, fill_val)

                any_legal = legal_mask.any(dim=1)              # [B] bool
                max_next_q, _ = next_q_values.max(dim=1)       # [B]
                max_next_q = torch.where(any_legal, max_next_q, torch.zeros_like(max_next_q))

                discounted_reward = r_t + self._gamma * max_next_q * (1.0 - done_t)  # [B]

            a_idx = a_t.argmax(dim=1, keepdim=True)  # [B, 1]
            q_sa = q_values.gather(1, a_idx).squeeze(1)  # [B]
            loss = F.smooth_l1_loss(q_sa, discounted_reward)

        # 5) Backpropagation and optimization step (AMP or fp32)
        self.optimizer.zero_grad(set_to_none=True)
        if self.use_amp:
            assert self.scaler is not None
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()

        return float(loss.item())

    def update_target_net(self):
        """Hard update: copy main network weights into the target network."""
        if self._use_target_net:
            self._target_net.load_state_dict(self._model.state_dict())