import numpy
import torch
import torch.nn.functional as F
from torch_ac.algos.base import BaseAlgo

class PPOAlgo(BaseAlgo):
    """The Proximal Policy Optimization algorithm
    ([Schulman et al., 2015](https://arxiv.org/abs/1707.06347))."""

    def __init__(self, envs, acmodel, detectormodel, device=None, num_frames_per_proc=None, discount=0.99, lr=0.001, gae_lambda=0.95,
                 entropy_coef=0.01, value_loss_coef=0.5, recurrence=4, 
                 detector_epochs=8, detector_batch_size=256, detector_lr=0.0003, detector_recurrence=4,
                 adam_eps=1e-8, clip_eps=0.2, epochs=4, batch_size=256, preprocess_obss=None,
                 reshape_reward=None, rm_update_algo=False):
        num_frames_per_proc = num_frames_per_proc or 128

        super().__init__(envs, acmodel, detectormodel, device, num_frames_per_proc, discount, lr, gae_lambda, entropy_coef,
                         value_loss_coef, recurrence, detector_recurrence, preprocess_obss, reshape_reward, rm_update_algo)
       
        self.clip_eps = clip_eps
        self.epochs = epochs
        self.batch_size = batch_size
        self.detector_epochs = detector_epochs
        self.detector_batch_size = detector_batch_size
        self.detector_lr = detector_lr

        self.act_shape = envs[0].action_space.shape

        assert self.batch_size % self.recurrence == 0
        assert self.batch_size % self.detector_recurrence == 0

        self.optimizer = torch.optim.Adam(self.acmodel.parameters(), lr, eps=adam_eps)
        if not self.detectormodel.params_free:
            self.optimizer_detector = torch.optim.Adam(self.detectormodel.parameters(), self.detector_lr, eps=adam_eps)
        else:
            self.optimizer_detector = None
        self.batch_num = 0

    def update_detector_parameters(self, exps):
        # Collect experiences

        log_losses_train = []
        log_losses_test = []
        log_grad_norms_train = []
        log_grad_norms_test = []
        log_accuracies_train = []
        log_accuracies_test = []


        for epoch in range(self.detector_epochs):
            # Initialize log values
            for inds in self._get_batches_starting_indexes(self.detector_recurrence, self.detector_batch_size):
                # Initialize batch values

                batch_loss = 0
                batch_accuracy = 0

                # Initialize memory

                if self.detectormodel.recurrent:
                    memory = exps.detector_memory[inds]

                for i in range(self.detector_recurrence):
                    # Create a sub-batch of experience

                    sb = exps[inds + i]

                    # Compute loss

                    if self.detectormodel.recurrent:
                        detector_belief, memory = self.detectormodel(sb.obs, memory * sb.mask)
                    else:
                        detector_belief = self.detectormodel(sb.obs)

                    # Train RM state detector
                    if self.rm_update_algo in ["tdm"]:
                        # This is the actual rm_states
                        rm_target = torch.argmax(sb.obs.rm_state, dim=1)
                        loss = F.cross_entropy(detector_belief, rm_target)

                        # Top 1 classification accuracy
                        correct_predictions = torch.sum(torch.argmax(detector_belief, dim=1) == rm_target)
                        accuracy = torch.true_divide(correct_predictions, len(rm_target))

                    # Train event detectors
                    if self.rm_update_algo in ["naive", "ibu"]:
                        # These are the actual events_targets
                        events_targets = sb.obs.events
                        loss = F.binary_cross_entropy_with_logits(detector_belief, events_targets)

                        # Top 1 classification accuracy
                        correct_predictions = torch.sum((detector_belief > 0) == events_targets)
                        accuracy = torch.true_divide(correct_predictions, torch.prod(torch.tensor(events_targets.shape)))

                    if self.rm_update_algo in ["no_rm", "perfect_rm"]:
                        loss = torch.full((1,), float('nan'))
                        accuracy = torch.full((1,), float('nan'))

                    # Update batch values
                    batch_loss += loss
                    batch_accuracy += accuracy

                    # Update memories for next epoch

                    if self.detectormodel.recurrent and i < self.detector_recurrence - 1:
                        exps.detector_memory[inds + i + 1] = memory.detach()

                # Update batch values

                batch_loss /= self.detector_recurrence
                batch_accuracy /= self.detector_recurrence

                # Update detector
                if not self.detectormodel.params_free:
                    self.optimizer_detector.zero_grad()
                    batch_loss.backward()
                    grad_norm = sum(p.grad.data.norm(2).item() ** 2 for p in self.detectormodel.parameters() if p.requires_grad) ** 0.5
                    self.optimizer_detector.step()
                else:
                    grad_norm = float('nan')

                # Update log values
                if epoch == 0:
                    log_losses_test.append(batch_loss.item())
                    log_accuracies_test.append(batch_accuracy.item())
                    log_grad_norms_test.append(grad_norm)
                elif epoch == self.detector_epochs-1:
                    log_losses_train.append(batch_loss.item())
                    log_accuracies_train.append(batch_accuracy.item())
                    log_grad_norms_train.append(grad_norm)
        # Log some values

        logs = {
            "detector_loss_train": numpy.mean(log_losses_train),
            "detector_grad_train": numpy.mean(log_grad_norms_train),
            "detector_top1_accuracy_train": numpy.mean(log_accuracies_train),
            "detector_loss_test": numpy.mean(log_losses_test),
            "detector_grad_test": numpy.mean(log_grad_norms_test),
            "detector_top1_accuracy_test": numpy.mean(log_accuracies_test),
        }

        return logs

    # Relabel the detector beliefs given how the detectors were updated
    def update_detector_beliefs(self, exps):
        if self.rm_update_algo == "rm_detector":
            detector_belief, detector_memory = self.detectormodel(exps.obs, exps.detector_memory * exps.mask)
            exps.obs.rm_belief = detector_belief.detach()

    def update_ac_parameters(self, exps):
        for _ in range(self.epochs):
            # Initialize log values

            log_entropies = []
            log_values = []
            log_policy_losses = []
            log_value_losses = []
            log_grad_norms = []

            for inds in self._get_batches_starting_indexes(self.recurrence, self.batch_size):
                # Initialize batch values

                batch_entropy = 0
                batch_value = 0
                batch_policy_loss = 0
                batch_value_loss = 0
                batch_loss = 0

                # Initialize memory

                if self.acmodel.recurrent:
                    memory = exps.memory[inds]

                for i in range(self.recurrence):
                    # Create a sub-batch of experience

                    sb = exps[inds + i]

                    # Compute loss
                    if self.acmodel.recurrent:
                        dist, value, memory = self.acmodel(sb.obs, memory * sb.mask)
                    else:
                        dist, value = self.acmodel(sb.obs)

                    entropy = dist.entropy().mean()

                    # ratio = torch.exp(dist.log_prob(sb.action) - sb.log_prob)
                    delta_log_prob = dist.log_prob(sb.action) - sb.log_prob
                    if (len(self.act_shape) == 1): # Not scalar actions (multivariate)
                        delta_log_prob = torch.sum(delta_log_prob, dim=1)
                    ratio = torch.exp(delta_log_prob)
                    surr1 = ratio * sb.advantage
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * sb.advantage
                    policy_loss = -torch.min(surr1, surr2).mean()

                    value_clipped = sb.value + torch.clamp(value - sb.value, -self.clip_eps, self.clip_eps)
                    surr1 = (value - sb.returnn).pow(2)
                    surr2 = (value_clipped - sb.returnn).pow(2)
                    value_loss = torch.max(surr1, surr2).mean()

                    loss = policy_loss - self.entropy_coef * entropy + self.value_loss_coef * value_loss

                    # Update batch values

                    batch_entropy += entropy.item()
                    batch_value += value.mean().item()
                    batch_policy_loss += policy_loss.item()
                    batch_value_loss += value_loss.item()
                    batch_loss += loss

                    # Update memories for next epoch

                    if self.acmodel.recurrent and i < self.recurrence - 1:
                        exps.memory[inds + i + 1] = memory.detach()

                # Update batch values

                batch_entropy /= self.recurrence
                batch_value /= self.recurrence
                batch_policy_loss /= self.recurrence
                batch_value_loss /= self.recurrence
                batch_loss /= self.recurrence

                # Update actor-critic

                self.optimizer.zero_grad()
                batch_loss.backward()
                grad_norm = sum(p.grad.data.norm(2).item() ** 2 for p in self.acmodel.parameters() if p.requires_grad) ** 0.5
                self.optimizer.step()

                # Update log values

                log_entropies.append(batch_entropy)
                log_values.append(batch_value)
                log_policy_losses.append(batch_policy_loss)
                log_value_losses.append(batch_value_loss)
                log_grad_norms.append(grad_norm)

        # Log some values

        logs = {
            "entropy": numpy.mean(log_entropies),
            "value": numpy.mean(log_values),
            "policy_loss": numpy.mean(log_policy_losses),
            "value_loss": numpy.mean(log_value_losses),
            "grad_norm": numpy.mean(log_grad_norms)
        }

        return logs

    def _get_batches_starting_indexes(self, recurrence, batch_size):
        """Gives, for each batch, the indexes of the observations given to
        the model and the experiences used to compute the loss at first.

        First, the indexes are the integers from 0 to `self.num_frames` with a step of
        `recurrence`, shifted by `recurrence//2` one time in two for having
        more diverse batches. Then, the indexes are splited into the different batches.

        Returns
        -------
        batches_starting_indexes : list of list of int
            the indexes of the experiences to be used at first for each batch
        """

        indexes = numpy.arange(0, self.num_frames, recurrence)
        indexes = numpy.random.permutation(indexes)

        # Shift starting indexes by recurrence//2 half the time
        if self.batch_num % 2 == 1:
            indexes = indexes[(indexes + recurrence) % self.num_frames_per_proc != 0]
            indexes += recurrence // 2
        self.batch_num += 1

        num_indexes = batch_size // recurrence
        batches_starting_indexes = [indexes[i:i+num_indexes] for i in range(0, len(indexes), num_indexes)]

        return batches_starting_indexes
