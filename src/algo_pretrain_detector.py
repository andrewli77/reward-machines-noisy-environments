import numpy
import torch
import torch.nn.functional as F
from utils import *

class Algo():
    """The Proximal Policy Optimization algorithm
    ([Schulman et al., 2015](https://arxiv.org/abs/1707.06347))."""

    def __init__(self, env, test_exps, detector_model, device=None, adam_eps=1e-8,
                 lr=0.0003, rm_update_algo="rm_detector", use_mem=True):

        self.env = env
        self.device = device
        self.rm_update_algo = rm_update_algo
        self.use_mem = use_mem
        self.detector_model = detector_model

        self.optimizer = torch.optim.Adam(self.detector_model.parameters(), lr, eps=adam_eps)
        self.batch_num = 0

        self.test_exps = torch_ac.DictList()
        self.test_obs = torch_ac.DictList({'image': test_exps['obss'].transpose(0,1).to(device)})
        self.test_exps.events = test_exps['events'].transpose(0,1).to(device)
        self.test_exps.rm_states = test_exps['rm_states'].transpose(0,1).to(device)
        self.test_exps.mask = test_exps['masks'].transpose(0,1).to(device)

    def update_parameters(self, exps):
        if not self.detector_model.recurrent:
            return self.update_parameters_flat(exps)

        max_steps, num_episodes = exps.mask.shape[0], exps.mask.shape[1]

        memory = torch.zeros((num_episodes, self.detector_model.memory_size)).to(self.device)
        batch_loss_rm_states = 0
        batch_loss_events = 0
        obs = torch_ac.DictList({'image': exps.obs.to(self.device)}) 


        for step in range(max_steps):
            sb = exps[step]
            sb_obs = obs[step]
            if sb.mask.sum() == 0:
                break

            if self.rm_update_algo in ["tdm"]:
                if step % 16 == 0:
                    out_rm_states, memory = self.detector_model(sb_obs, memory.detach() * sb.mask.to(self.device).unsqueeze(dim=1))
                else:
                    out_rm_states, memory = self.detector_model(sb_obs, memory * sb.mask.to(self.device).unsqueeze(dim=1))

                # This is the actual rm_states
                rm_target = torch.argmax(sb.rm_states, dim=1)
                loss_rm_states = F.cross_entropy(out_rm_states, rm_target, reduction='sum')
                batch_loss_rm_states += (loss_rm_states.sum())


            elif self.rm_update_algo in ["naive", "ibu"]:
                if step % 16 == 0:
                    out_events, memory = self.detector_model(sb_obs, memory.detach() * sb.mask.to(self.device).unsqueeze(dim=1))
                else:
                    out_events, memory = self.detector_model(sb_obs, memory * sb.mask.to(self.device).unsqueeze(dim=1))

                events_targets = sb.events
                loss_events = F.binary_cross_entropy_with_logits(out_events, events_targets, reduction='sum')
                batch_loss_events += (loss_events.sum())


        nnz = exps.mask.sum()
        batch_loss_events /= nnz
        batch_loss_rm_states /= nnz

        self.optimizer.zero_grad()
        (batch_loss_events + batch_loss_rm_states).backward()
        grad_norm = sum(p.grad.data.norm(2).item() ** 2 for p in self.detector_model.parameters() if p.grad is not None) ** 0.5
        self.optimizer.step()

        logs = {"event_loss": batch_loss_events, "rm_state_loss": batch_loss_rm_states, "grad_norm": grad_norm}

        return logs

    # For non-sequential models we can improve the efficiency
    def update_parameters_flat(self, exps):
        assert(self.rm_update_algo in ["naive", "ibu"] and not self.detector_model.recurrent)
        obs = torch_ac.DictList({'image': exps.obs.to(self.device)}) 

        out_events = self.detector_model(obs)
        events_targets = exps.events

        loss_events = F.binary_cross_entropy_with_logits(out_events, events_targets, reduction='sum')
        nnz = exps.mask.sum()
        loss_events /= nnz

        self.optimizer.zero_grad()
        loss_events.backward()
        grad_norm = sum(p.grad.data.norm(2).item() ** 2 for p in self.detector_model.parameters() if p.grad is not None) ** 0.5
        self.optimizer.step()

        logs = {"event_loss": loss_events.item(), "rm_state_loss": 0, "grad_norm": grad_norm}
        return logs


    @torch.no_grad()
    def evaluate(self):
        if self.rm_update_algo in ['naive', 'ibu'] and not self.detector_model.recurrent:
            return self.evaluate_flat()
        max_steps, num_episodes = self.test_exps.mask.shape[0], self.test_exps.mask.shape[1]

        memory = torch.zeros((num_episodes, self.detector_model.memory_size)).to(self.device)
        test_loss_events = 0
        test_loss_rm_states = 0

        for step in range(max_steps):
            sb = self.test_exps[step]
            sb_obs = self.test_obs[step]
            if sb.mask.sum() == 0:
                break

            if self.rm_update_algo in ["tdm"]:
                if step % 16 == 0:
                    out_rm_states, memory = self.detector_model(sb_obs, memory.detach() * sb.mask.to(self.device).unsqueeze(dim=1))
                else:
                    out_rm_states, memory = self.detector_model(sb_obs, memory * sb.mask.to(self.device).unsqueeze(dim=1))

                # This is the actual rm_states
                rm_target = torch.argmax(sb.rm_states, dim=1)
                loss_rm_states = F.cross_entropy(out_rm_states, rm_target, reduction='sum')
                test_loss_rm_states += (loss_rm_states.sum())


            elif self.rm_update_algo in ["naive", "ibu"]:
                if step % 16 == 0:
                    out_events, memory = self.detector_model(sb_obs, memory.detach() * sb.mask.to(self.device).unsqueeze(dim=1))
                else:
                    out_events, memory = self.detector_model(sb_obs, memory * sb.mask.to(self.device).unsqueeze(dim=1))

                events_targets = sb.events
                loss_events = F.binary_cross_entropy_with_logits(out_events, events_targets, reduction='sum')
                test_loss_events += (loss_events.sum())

        test_loss_events /= self.test_exps.mask.sum()
        test_loss_rm_states /= self.test_exps.mask.sum()

        logs = {"event_loss": test_loss_events.item(), "rm_state_loss": 0.} # test_loss_rm_states.item()

        return logs

    @torch.no_grad()
    def evaluate_flat(self):
        assert(self.rm_update_algo in ["naive", "ibu"] and not self.detector_model.recurrent)

        # Compute events loss
        out_events = self.detector_model(self.test_obs)
        events_targets = self.test_exps.events

        loss_events = F.binary_cross_entropy_with_logits(out_events, events_targets, reduction='sum')
        nnz = self.test_exps.mask.sum()
        loss_events /= nnz

        # Compute RM state loss
        test_loss_rm_states = 0.
        max_steps, num_episodes = self.test_exps.mask.shape[0], self.test_exps.mask.shape[1]

        # You can uncomment this to evaluate how well this model predicts RM states given a trajectory. However, this is costly
        # and it is recommended you use accuracy of events/propositions as the evaluation metric.

        # Create dummy envs to simulate RM transitions
        # letter_types = self.env.letter_types
        # dummy_env = [make_env2(DummyEnv(letter_types), self.env.spec.id, self.rm_update_algo, 0) for i in range(num_episodes)]
        # dummy_env.reset()

        # for eps in range(num_episodes):
        #     dummy_env = make_env(self.env.spec.id, self.rm_update_algo, 0)
        #     dummy_env.reset()

        #     if self.detector_model.recurrent:
        #         memory = torch.zeros((self.detector_model.memory_size)).to(self.device)
    
        #     for step in range(max_steps):
        #         sb = self.test_exps[step][eps]
        #         sb_obs = self.test_obs[step][eps]

        #         if sb.mask.sum() == 0:
        #             break

        #         if self.detector_model.recurrent:
        #             out_events, memory = self.detector_model(sb_obs, memory * sb.mask)                
        #         else:
        #             out_events = self.detector_model(sb_obs)

        #         if self.rm_update_algo == "naive":
        #             rm_state_belief = dummy_env.update_rm_beliefs((out_events > 0).cpu().numpy())
        #             rm_state_belief = torch.tensor(np.array(rm_state_belief), device=self.device, dtype=torch.float)
        #             rm_state_belief = torch.log(torch.maximum(rm_state_belief, torch.tensor(0.01)))
        #         if self.rm_update_algo == "ibu":
        #             rm_state_belief = dummy_env.update_rm_beliefs(out_events.cpu().numpy())
        #             rm_state_belief = torch.tensor(np.array(rm_state_belief), device=self.device, dtype=torch.float)
        #             rm_state_belief = torch.log(torch.maximum(rm_state_belief, torch.tensor(0.01)))

        #         rm_target = torch.argmax(sb.rm_states)
        #         loss_rm_states = F.nll_loss(rm_state_belief, rm_target, reduction='sum')
        #         test_loss_rm_states += (loss_rm_states.sum())

        # test_loss_rm_states /= self.test_exps.mask.sum()

        logs = {"event_loss": loss_events.item(), "rm_state_loss": 0} #test_loss_rm_states.item()
        return logs
