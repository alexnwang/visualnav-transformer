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

def get_best_from_dict_of_lists(metric_dict_of_lists: dict, k_steps=None):
    result = {}
    for key, values in metric_dict_of_lists.items():
        if "min-" in key:
            continue
        vals = values[:k_steps]
        best_idx = int(np.argmin(vals))
        result[key] = vals[best_idx]
        min_key = f"min-{key}"
        if min_key in metric_dict_of_lists:
            result[min_key] = metric_dict_of_lists[min_key][best_idx]
    return result

def get_average_from_dict_of_lists(dict_of_lists: dict, k_steps=None):
    return {k: np.mean(v[:k_steps]) for k, v in dict_of_lists.items()}

class CEMPlanner(BasePlanner):
    def __init__(
        self,
        horizon,
        topk,
        num_samples,
        var_scale,
        opt_steps,
        wm,
        action_dim,
        objective_fn,
        preprocessor,
        evaluator,
        wandb_run,
        log_dir="logs/cem",
        **kwargs,
    ):
        """
        Args:
            horizon (int): time horizon / num time steps
            topk: (int): num samples kept per step
            num_samples (int): number of samples per step
            var_scale (float): initial variance scale sigma
            opt_steps (int): the number of optimization steps (generations)
            wm (WorldModel): the world model
            action_dim (int): the dimension of the action
            objective_fn (ObjectiveFunction): the objective function, compares predicted obs, with gt obs
            preprocessor (Preprocessor): the preprocessor, image preprocessing
            evaluator (Evaluator): the evaluator for mu, called every CEM iteration
            wandb_run (wandb.Run): the wandb run
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

        os.makedirs(log_dir, exist_ok=True)
        
        self.accum_metrics = defaultdict(list)

        self.accum_objective_metric_dicts = {}
        self.accum_eval_metric_dicts = {}
        self.accum_eval_other_vals = {}
        self.accum_task_dicts = {}

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
        
    def aggregate_metrics_ddp(self, metrics_dict):
        """
        Aggregate metrics dictionaries across all DDP workers.
        If multiple workers share the same key, the values are average. 
        If only one worker has a key, the value stays as it was found in that worker.
        
        Args:
            metrics_dict: Dictionary with metric keys and values (numbers or scalar tensors)
            
        Returns:
            Combined dictionary with averaged values across all workers
        """
        if not torch.distributed.is_initialized():
            return metrics_dict
        # Gather dictionaries from all workers
        world_size = torch.distributed.get_world_size()
        all_metrics_list = [None for _ in range(world_size)]
        torch.distributed.all_gather_object(all_metrics_list, metrics_dict)
        # Collect all unique keys from all workers
        all_keys = set()
        for worker_metrics in all_metrics_list:
            if worker_metrics is not None:
                all_keys.update(worker_metrics.keys())
        # Combine dictionaries by averaging values
        combined_metrics = {}
        for key in all_keys:
            values = []
            for worker_metrics in all_metrics_list:
                if worker_metrics is not None and key in worker_metrics:
                    value = worker_metrics[key]
                    # Convert to float: handle numbers or scalar tensors
                    if isinstance(value, (int, float, np.number)):
                        values.append(float(value))
                    elif isinstance(value, torch.Tensor):
                        if value.numel() == 1:
                            values.append(value.item())
                        else:
                            raise ValueError(f"Expected scalar tensor for key '{key}', got tensor with shape {value.shape}")
                    else:
                        raise TypeError(f"Expected number or scalar tensor for key '{key}', got {type(value)}")
            # Average the values
            if len(values) > 0:
                combined_metrics[key] = np.mean(values)
            else:
                combined_metrics[key] = metrics_dict.get(key, 0.0)
        return combined_metrics

    def plan(self, obs_0, obs_g, task_name, actions=None, cam_model=None, T_C_pelvis=None):
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

        self.accum_objective_metric_dicts[task_name] = defaultdict(list)
        self.accum_eval_metric_dicts[task_name] = defaultdict(list)
        self.accum_eval_other_vals[task_name] = defaultdict(list)

        if self.evaluator is not None:
            task_metrics, gtwp_metrics, nowp_metrics = self.evaluator.eval_task(trans_obs_0, z_obs_g)
            self.accum_task_dicts[task_name] = {"task": task_metrics, "task_gtwp": gtwp_metrics, "task_nowp": nowp_metrics}
            if self.wandb_run is not None:
                task_log = {f"task/{k}": v for k, v in task_metrics.items()}
                task_log.update({f"task_gtwp/{k}": v for k, v in gtwp_metrics.items()})
                task_log.update({f"task_nowp/{k}": v for k, v in nowp_metrics.items()})
                self.wandb_run.log(task_log, commit=True)

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

            step_dir = f"{self.log_dir}/{task_name}/step{i:03d}"
            loss, log_dict = self.objective_fn(i, i_state, curr_state_0, curr_latent_state_g,
                                               save_path=step_dir, topk=self.topk)
            topk_idx = torch.argsort(loss)[: self.topk]
            topk_action = action[topk_idx]
            losses.append(loss[topk_idx[0]].item())
            mu = topk_action.mean(dim=0, keepdim=True)
            sigma = topk_action.std(dim=0, keepdim=True)

            for k, v in {**log_dict, "avg_sigma": sigma.mean().item(), "step": i + 1}.items():
                self.accum_objective_metric_dicts[task_name][k].append(v)
            log_dict = {**log_dict, "avg_sigma": sigma.mean().item(), "step": i + 1}
            if self.wandb_run is not None:
                self.wandb_run.log(log_dict, commit=False)

            # Per-iteration mu evaluation: extra WM rollout on updated mu
            with torch.no_grad():
                mu_state = self.wm.rollout(state_0=trans_obs_0, act=mu)
            mu_loss, _ = self.objective_fn(i, mu_state, trans_obs_0, z_obs_g, save_path=None, topk=0)
            mu_dreamsim = mu_loss[0].item()

            if self.evaluator is not None:
                metrics, other_vals = self.evaluator.eval_mu_step(
                    i, mu, mu_state, trans_obs_0, z_obs_g,
                    cam_model=cam_model, T_C_pelvis=T_C_pelvis,
                    save_path=step_dir,
                )
                metrics["dreamsim"] = mu_dreamsim
                for k, v in metrics.items():
                    self.accum_eval_metric_dicts[task_name][k].append(v)
                for k, v in other_vals.items():
                    self.accum_eval_other_vals[task_name][k].append(v)

                logs_tagged = {f"eval/{k}": v for k, v in {**metrics, **other_vals}.items()}
                logs_tagged["step"] = i + 1
                if self.wandb_run is not None:
                    self.wandb_run.log(logs_tagged, commit=True)
    
        for k_steps in range(1, self.opt_steps + 1):
            accum_dict = defaultdict(list)
            for task_name in self.accum_eval_metric_dicts.keys():
                for k, v in get_best_from_dict_of_lists(self.accum_eval_metric_dicts[task_name], k_steps).items():
                    accum_dict[k].append(v)
                for k, v in get_average_from_dict_of_lists(self.accum_eval_other_vals[task_name], k_steps).items():
                    accum_dict[k].append(v)
            if self.wandb_run is not None:
                tag = f"accum{k_steps}"
                self.wandb_run.log(
                    {f"{tag}/{k}": np.mean(v) for k, v in accum_dict.items()}, commit=False
                )
        if self.wandb_run is not None:
            for prefix in ["task", "task_gtwp", "task_nowp"]:
                accum_task_dict = defaultdict(list)
                for tn in self.accum_task_dicts.keys():
                    for k, v in self.accum_task_dicts[tn][prefix].items():
                        accum_task_dict[k].append(v)
                if accum_task_dict:
                    self.wandb_run.log(
                        {f"accum_{prefix}/{k}": np.mean(v) for k, v in accum_task_dict.items()}, commit=False
                    )
            self.wandb_run.log({}, commit=True)

        # dump all the accumulated metrics and other values
        torch.save(self.accum_objective_metric_dicts, f"{self.log_dir}/accum_objective_metric_dicts.pth")
        torch.save(self.accum_eval_metric_dicts, f"{self.log_dir}/accum_eval_metric_dicts.pth")
        torch.save(self.accum_eval_other_vals, f"{self.log_dir}/accum_eval_other_vals.pth")
        torch.save(self.accum_task_dicts, f"{self.log_dir}/accum_task_dicts.pth")
        return mu