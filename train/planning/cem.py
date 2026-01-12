from collections import defaultdict
import os
import torch
import numpy as np
from einops import rearrange, repeat
from .base_planner import BasePlanner

def move_to_device(dct, device):
    for key, value in dct.items():
        if isinstance(value, torch.Tensor):
            dct[key] = value.to(device)
    return dct

class CEMPlanner(BasePlanner):
    def __init__(
        self,
        horizon,
        topk,
        num_samples,
        var_scale,
        opt_steps,
        eval_every,
        wm,
        action_dim,
        objective_fn,
        preprocessor,
        evaluator,
        wandb_run,
        logging_prefix="plan_0",
        log_dir="logs/cem",
        metric_keys=["xyz_distance",
                     "visible_xyz_distance",
                     "gt_waypoint-xyz_distance",
                     "no_waypoint-xyz_distance",
                     "start_xyz_distance"],
        **kwargs,
    ):
        """
        Args:
            horizon (int): time horizon / num time steps
            topk: (int): num samples kept per step
            num_samples (int): number of samples per step 
            var_scale (float): initial variance scale sigma
            opt_steps (int): the number of optimization steps (generations)
            eval_every (int): eval_freq, evaluation checks how good mu is
            wm (WorldModel): the world model
            action_dim (int): the dimension of the action
            objective_fn (ObjectiveFunction): the objective function, compares predicted obs, with gt obs
            preprocessor (Preprocessor): the preprocessor, image preprocessing
            evaluator (Evaluator): the evaluator for mu after eval_every steps
            wandb_run (wandb.Run): the wandb run
            logging_prefix (str): the prefix of the logging
            log_dir (str): the directory of the logs
            metric_keys (list): the keys of the metrics to log, the first key is the primary metric
        """
        super().__init__(
            wm,
            action_dim,
            objective_fn,
            preprocessor,
            evaluator,
            wandb_run,
            log_filename="unused",
        )
        self.log_dir = log_dir
        self.horizon = horizon
        self.topk = topk
        self.num_samples = num_samples
        self.var_scale = var_scale
        self.opt_steps = opt_steps
        self.eval_every = eval_every
        self.logging_prefix = logging_prefix
        
        os.makedirs(log_dir, exist_ok=True)
        
        self.metric_keys = metric_keys
        self.accum_metrics = defaultdict(list)
        
    def store_metrics(self, metrics):
        for k, v in metrics.items():
            self.accum_metrics[k].append(v)
    
    def get_average_metrics(self, prefix=""):
        return {f"{prefix}{k}": np.mean(v) for k, v in self.accum_metrics.items()}

    def init_mu_sigma(self, obs_0, actions=None):
        """
        actions: (B, T, action_dim) torch.Tensor, T <= self.horizon
        mu, sigma could depend on current obs, but obs_0 is only used for providing n_evals for now
        """
        n_evals = obs_0["images"].shape[0]
        sigma = self.var_scale * torch.ones([n_evals, self.horizon, self.action_dim])
        if actions is None:
            mu = torch.zeros(n_evals, 0, self.action_dim)
        else:
            mu = actions
        device = mu.device
        t = mu.shape[1]
        remaining_t = self.horizon - t

        if remaining_t > 0:
            new_mu = torch.zeros(n_evals, remaining_t, self.action_dim)
            mu = torch.cat([mu, new_mu.to(device)], dim=1)
        return mu, sigma

    def plan(self, obs_0, obs_g, task_name, actions=None):
        """
        Args:
            actions: normalized
        Returns:
            actions: (B, T, action_dim) torch.Tensor, T <= self.horizon
        """
        trans_obs_0 = move_to_device(
            self.preprocessor.transform_obs(obs_0), self.device
        )
        trans_obs_g = move_to_device(
            self.preprocessor.transform_obs(obs_g), self.device
        )
        z_obs_g = self.wm.encode_obs(trans_obs_g)
        mu, sigma = self.init_mu_sigma(obs_0, actions)
        mu, sigma = mu.to(self.device), sigma.to(self.device)
        assert actions.shape[0] == 1

        best_eval_metrics = {k: float('inf') for k in self.metric_keys}
        
        for i in range(self.opt_steps):
            # optimize individual instances
            losses = []
            curr_state_0 = {
                key: repeat(
                    arr, "1 ... -> n ...", n=self.num_samples
                )
                for key, arr in trans_obs_0.items()
            }
            curr_latent_state_g = {
                key: repeat(
                    arr, "1 ... -> n ...", n=self.num_samples
                )
                for key, arr in z_obs_g.items()
            }
            action = (
                torch.randn(self.num_samples, self.horizon, self.action_dim).to(
                    self.device
                )* sigma + mu
            )
            action[0] = mu  # optional: make the first one mu itself
            with torch.no_grad():
                i_state = self.wm.rollout(
                    state_0=curr_state_0,
                    act=action,
                )

            loss, log_dict = self.objective_fn(i, i_state, curr_state_0, curr_latent_state_g,
                                               save_path=f"{self.log_dir}/{task_name}", topk=self.topk)
            topk_idx = torch.argsort(loss)[: self.topk]
            topk_action = action[topk_idx]
            losses.append(loss[topk_idx[0]].item())
            mu = topk_action.mean(dim=0, keepdim=True)
            sigma = topk_action.std(dim=0, keepdim=True)

            if self.wandb_run is not None:
                self.wandb_run.log(
                    {**log_dict, "avg_sigma": sigma.mean().item(), "step": i + 1}, commit=False
                )
            if self.evaluator is not None and i % self.eval_every == 0:
                metrics, other_vals = self.evaluator.eval_actions(i, mu, trans_obs_0, z_obs_g, save_path=f"{self.log_dir}/{task_name}")
                logs_tagged = {f"{task_name}/{k}": v for k, v in {**metrics, **other_vals}.items()}
                logs_tagged.update({"step": i + 1})
                if self.wandb_run is not None:
                    self.wandb_run.log(logs_tagged, commit=True)
                
                for key in self.metric_keys:
                    if metrics[key] < best_eval_metrics[key]:
                        best_eval_metrics[key] = metrics[key]
        self.store_metrics(best_eval_metrics)
        self.wandb_run.log(self.get_average_metrics(prefix="accum/"), commit=True)
        return mu