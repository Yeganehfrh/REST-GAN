"""Training loop of REST-GAN, with checkpoints that resume a run exactly."""
import copy
import csv
import os
import time

import torch

from .models.FiLMGAN import (FiLMGAN, gradient_penalty, make_masked_batch, make_channel_mask, reconstruction_loss,
                             contrastive_loss)
from .models.spectral import (recording_targets, spectral_distance, spectral_embeddings, spectral_features,
                              spectral_loss)

HPARAMS = dict(
    steps_per_epoch=500,
    lr_g=2e-4,                      # Adam, betas (0, 0.9)
    lr_d=6e-4,
    lr_decay=0.9,                   # staircase decay
    lr_decay_steps=100_000,         # counted in optimizer steps
    n_critic=3,                     # critic updates per generator update
    gp_weight=1.0,
    recon_weight=0.1,               # weight of the reconstruction loss (recon_mask='time')
    contrastive_weight=0.5,         # norm of the contrastive gradient relative to the adversarial one
    contrastive_temperature=0.1,
    recon_balanced_weight=0.5,      # the same for the reconstruction of masked channels (recon_mask='channel')
    spectral_weight=1.0,            # norm of the spectral-loss gradient relative to the adversarial one (0 = off)
    spectral_groups=16,             # recordings per step in the spectral loss
    spectral_group_size=8,          # generated segments per recording in the spectral loss
    ema_decay=0.0,                  # running average of the generator's weights for evaluation (0 = off)
    embedding_init='spectral',      # 'spectral' (start from the spectral features of the participants) or 'random'
)
SPECTRAL_REFERENCE_BATCHES = 20  # real batches that set the reference level of the spectral loss

# Checkpoints saved before an option existed do not have it: they were made with these values
LEGACY_CONFIG = dict(critic_cond='film', recon_mask='time', gen_arch='latent', spectral_head=False, head_bins=65)
LEGACY_HPARAMS = dict(spectral_weight=0.0, embedding_init='random')
REMOVED_HPARAMS = ('warmup_epochs', 'n_critic_after_warmup')  # fewer critic updates after epoch 300 (never reached)


class Trainer:
    """Holds the model, the two optimizers and the state that a run needs to continue."""

    def __init__(self, model, device='cpu', amp=None, seed=0, **hparams):
        unknown = set(hparams) - set(HPARAMS)
        if unknown:
            raise TypeError(f'Unknown training options: {sorted(unknown)}')
        self.hparams = {**HPARAMS, **hparams}
        self.device = torch.device(device)
        self.amp = self.device.type == 'cuda' if amp is None else bool(amp)  # mixed precision: on for CUDA by default
        self.model = model.to(self.device)
        self.rng = torch.Generator(device=self.device).manual_seed(seed)   # noise, interpolation, masking
        self.seed = seed
        self.epoch = 0
        self.global_step = 0
        self._sampler_rng_state = None
        self._nonfinite = {}  # losses found non-finite since the last check
        self._spectral_targets = None  # set by `prepare`
        self._spectral_reference = None

        hp = self.hparams
        if hp['embedding_init'] not in ('random', 'spectral'):
            raise ValueError(f"embedding_init must be 'random' or 'spectral', got {hp['embedding_init']!r}")
        if (hp['spectral_weight'] > 0 or hp['embedding_init'] == 'spectral') and model.subject_cond != 'table':
            raise ValueError("The spectral loss and the spectral start of the embeddings need subject_cond='table'.")

        # running average of the generator's weights
        self.ema_generator = None
        if hp['ema_decay'] > 0:
            self.ema_generator = copy.deepcopy(model.generator).requires_grad_(False)

        # optimizers and learning-rate schedules
        self.d_optimizer = torch.optim.Adam(model.critic.parameters(), lr=hp['lr_d'], betas=(0.0, 0.9), eps=1e-7)
        self.g_optimizer = torch.optim.Adam(model.generator.parameters(), lr=hp['lr_g'], betas=(0.0, 0.9), eps=1e-7)

        def staircase(step):
            return hp['lr_decay'] ** (step // hp['lr_decay_steps'])
        self.d_scheduler = torch.optim.lr_scheduler.LambdaLR(self.d_optimizer, staircase)
        self.g_scheduler = torch.optim.lr_scheduler.LambdaLR(self.g_optimizer, staircase)

    def _autocast(self):
        dtype = torch.float16 if self.device.type == 'cuda' else torch.bfloat16
        return torch.autocast(self.device.type, dtype=dtype, enabled=self.amp)

    def _randn(self, *shape):
        return torch.randn(*shape, device=self.device, generator=self.rng)

    def prepare(self, sampler):
        """Compute what the spectral loss needs from the training data, and set the spectral start of the embeddings."""
        hp = self.hparams
        spectral_start = hp['embedding_init'] == 'spectral'
        if (hp['spectral_weight'] <= 0 and not spectral_start) or self._spectral_targets is not None:
            return
        if hp['spectral_weight'] > 0 and sampler.n_groups != hp['spectral_groups']:
            raise ValueError(f"The spectral loss needs a sampler with n_groups={hp['spectral_groups']}.")

        # spectral features of every recording: the targets of the spectral loss
        self._spectral_targets = recording_targets(sampler.x, sampler.seg_len, sampler.block_len)

        # value of the two terms for groups of real segments: the reference level of the loss
        if self._spectral_reference is None:  # a resumed run has it from the checkpoint
            g = torch.Generator(device=self.device).manual_seed(self.seed + 1)  # its own random stream
            total = torch.zeros(2, device=self.device)
            for _ in range(SPECTRAL_REFERENCE_BATCHES):
                rec = torch.randint(0, sampler.n_recordings, (hp['spectral_groups'],), generator=g, device=self.device)
                real = sampler._segments(rec.repeat_interleave(hp['spectral_group_size']), generator=g)
                real = real.view(len(rec), hp['spectral_group_size'], *real.shape[1:])
                power, coherency = spectral_distance(spectral_features(real), [t[rec] for t in self._spectral_targets])
                total += torch.stack([power.mean(), coherency.mean()])
            self._spectral_reference = (total / SPECTRAL_REFERENCE_BATCHES).tolist()

        # spectral start of the participant embeddings (generator and critic)
        if spectral_start and self.global_step == 0:  # a resumed run has its embeddings from the checkpoint
            scores = spectral_embeddings(self._spectral_targets, self._spectral_reference, sampler.sub, sampler.state,
                                         self.model.n_subjects, self.model.generator.sub_emb.embedding_dim)
            with torch.no_grad():
                for table in (self.model.generator.sub_emb, self.model.critic.sub_emb):
                    table.weight[:, :scores.shape[1]] = scores.to(table.weight.dtype)
                if self.ema_generator is not None:
                    self.ema_generator.load_state_dict(self.model.generator.state_dict())

    def train_step(self, batch, sync=True):
        """One training step: the critic updates, then one generator update. Returns the diagnostics of the step.
        sync=False keeps them on the device (0-d tensors) and only records non-finite losses (see `fit`).
        """
        model, hp = self.model, self.hparams
        generator, critic = model.generator, model.critic
        if model.use_contrastive and 'x2' not in batch:
            raise ValueError("The contrastive loss needs a second segment per recording: batch['x2'].")
        encoder = model.subject_cond == 'encoder'
        if encoder and 'refs' not in batch:
            raise ValueError("The participant encoder needs reference segments: batch['refs'].")
        real = batch['x'].to(self.device)
        sub, state = batch['sub'].to(self.device).view(-1), batch['state'].to(self.device).view(-1)
        real_2 = batch['x2'].to(self.device) if model.use_contrastive else None
        refs = batch['refs'].to(self.device) if encoder else None
        n = len(real)
        logs = {}

        # train critic
        for _ in range(hp['n_critic']):
            # generated segments get the participants and eye states of a shuffled copy of the batch
            perm = torch.randperm(n, device=self.device, generator=self.rng)
            fake_sub, fake_state = sub[perm], state[perm]
            with torch.no_grad(), self._autocast():
                g_emb = generator.embed_refs(refs[perm]) if encoder else None
                fake = generator(generator.sample_latent(n, self._randn), fake_sub, fake_state, subj_emb=g_emb)
            with self._autocast():
                # with the encoder, the critic infers the embedding of each participant once
                c_emb = critic.embed_refs(refs) if encoder else None
                real_pred = critic(real, sub, state, subj_emb=c_emb)
                fake_pred = critic(fake, fake_sub, fake_state, subj_emb=c_emb[perm] if encoder else None)
                eps = torch.rand(n, 1, 1, device=self.device, generator=self.rng)
                gp = gradient_penalty(critic, real, fake, sub, state, eps, subj_emb=c_emb)
                d_loss = fake_pred.mean() - real_pred.mean() + hp['gp_weight'] * gp
                # reconstruction of masked time spans (fixed weight)
                if model.use_recon and model.recon_mask == 'time':
                    _, mask = make_masked_batch(real, generator=self.rng)
                    recon = reconstruction_loss(critic, real, sub, state, mask)
                    d_loss = d_loss + hp['recon_weight'] * recon
                    logs['recon_loss'] = recon.detach()
            self._watch(sync, d_loss=d_loss)
            self.d_optimizer.zero_grad(set_to_none=True)
            d_loss.backward()
            # auxiliary losses with a balanced gradient: reconstruction of masked channels, contrastive loss
            recording = batch['recording'].to(self.device) if 'recording' in batch else None
            if model.use_recon and model.recon_mask == 'channel':
                logs.update(self._add_balanced_gradients(real, real_2, sub, state, sync, recording))
            elif model.use_contrastive:
                logs['con_loss'], logs['con_acc'] = self._add_contrastive_gradients(real, real_2, sub, state, sync,
                                                                                    recording)
            grad_norm = torch.sqrt(sum((p.grad.float() ** 2).sum() for p in critic.parameters() if p.grad is not None))
            self.d_optimizer.step()
            self.d_scheduler.step()

        # train generator
        with self._autocast():
            g_emb = generator.embed_refs(refs[perm]) if encoder else None
            generated = generator(generator.sample_latent(n, self._randn), fake_sub, fake_state, subj_emb=g_emb)
            if encoder:
                with torch.no_grad():
                    c_emb = critic.embed_refs(refs[perm])
            g_loss = -critic(generated, fake_sub, fake_state, subj_emb=c_emb if encoder else None).mean()
        self._watch(sync, g_loss=g_loss)
        self.g_optimizer.zero_grad(set_to_none=True)
        g_loss.backward(inputs=list(generator.parameters()))
        # spectral loss
        if hp['spectral_weight'] > 0:
            logs.update(self._add_spectral_gradients(batch, sync))
        self.g_optimizer.step()
        self.g_scheduler.step()
        self.global_step += 1

        # running average of the generator's weights (follows them closely at first)
        if self.ema_generator is not None:
            decay = min(hp['ema_decay'], (1 + self.global_step) / (10 + self.global_step))
            with torch.no_grad():
                for average, weight in zip(self.ema_generator.parameters(), generator.parameters()):
                    average.lerp_(weight, 1 - decay)

        if encoder:  # spread of the inferred embeddings across the batch (near 0: collapse)
            logs['g_emb_spread'] = g_emb.detach().float().std(0).mean()
            logs['c_emb_spread'] = c_emb.detach().float().std(0).mean()
        # diagnostics
        logs.update(d_loss=d_loss.detach(), g_loss=g_loss.detach(), gp=gp.detach(), critic_grad_norm=grad_norm.detach(),
                    real_pred=real_pred.detach().mean(), fake_pred=fake_pred.detach().mean(),
                    real_pred_std=real_pred.detach().std(), fake_pred_std=fake_pred.detach().std())
        return {k: v.item() for k, v in logs.items()} if sync else logs

    def _add_spectral_gradients(self, batch, sync=True):
        """Add the gradient of the spectral loss to the adversarial gradient that is stored on the generator.
        It is rescaled to `spectral_weight` times the norm of the adversarial gradient, and reduced to zero as the
        loss approaches its value for real data.
        """
        if self._spectral_targets is None:
            raise RuntimeError('Call `prepare(sampler)` before training with the spectral loss.')
        if 'g_recording' not in batch:
            raise ValueError("The spectral loss needs group recordings: batch['g_recording'].")
        hp, generator = self.hparams, self.model.generator
        size = hp['spectral_group_size']
        rec = batch['g_recording'].to(self.device)
        sub = batch['g_sub'].to(self.device).repeat_interleave(size)
        state = batch['g_state'].to(self.device).repeat_interleave(size)
        # keep the adversarial gradient
        params = list(generator.parameters())
        adversarial = [None if p.grad is None else p.grad.detach().clone() for p in params]
        self.g_optimizer.zero_grad(set_to_none=True)
        # generate a group of segments for each drawn recording and compare it with the recording
        with self._autocast():
            fake = generator(generator.sample_latent(len(sub), self._randn), sub, state)
        loss, power, coherency = spectral_loss(spectral_features(fake.view(len(rec), size, *fake.shape[1:])),
                                               [t[rec] for t in self._spectral_targets], self._spectral_reference)
        self._watch(sync, spec_loss=loss)
        loss.backward()
        # rescale to the norm of the adversarial gradient
        shared = [(a, p.grad) for a, p in zip(adversarial, params) if a is not None and p.grad is not None]
        adv_norm = torch.sqrt(sum((a.float() ** 2).sum() for a, _ in shared))
        spec_norm = torch.sqrt(sum((g.float() ** 2).sum() for _, g in shared))
        gate = ((loss.detach() - 2.0) / 0.5).clamp(0.0, 1.0)  # gate: 1 for a loss of 2.5 or more, 0 at the level of real data (2)
        scale = hp['spectral_weight'] * gate * adv_norm / (spec_norm + 1e-12)
        # add the two gradients
        for a, p in zip(adversarial, params):
            if p.grad is None:
                p.grad = a
            else:
                p.grad.mul_(scale)
                if a is not None:
                    p.grad.add_(a)
        return {'spec_loss': loss.detach(), 'spec_power': power.detach(), 'spec_coherency': coherency.detach(),
                'spec_gate': gate}

    def _add_contrastive_gradients(self, x, x2, sub, state, sync=True, recording=None):
        """Add the contrastive gradient to the adversarial gradient that is stored on the critic.
        It is rescaled to `contrastive_weight` times the norm of the adversarial gradient on the weights that the two
        losses share (an adaptive weight, as in Esser et al., 2021).
        """
        params = list(self.model.critic.parameters())
        adversarial = [None if p.grad is None else p.grad.detach().clone() for p in params]
        self.d_optimizer.zero_grad(set_to_none=True)
        with self._autocast():
            loss, accuracy = contrastive_loss(self.model.critic, x, x2, sub, state,
                                              self.hparams['contrastive_temperature'], recording)
        self._watch(sync, con_loss=loss)
        loss.backward()
        shared = [(a, p.grad) for a, p in zip(adversarial, params) if a is not None and p.grad is not None]
        adv_norm = torch.sqrt(sum((a.float() ** 2).sum() for a, _ in shared))
        con_norm = torch.sqrt(sum((c.float() ** 2).sum() for _, c in shared))
        scale = self.hparams['contrastive_weight'] * adv_norm / (con_norm + 1e-12)
        for a, p in zip(adversarial, params):
            if p.grad is None:
                p.grad = a
            else:
                p.grad.mul_(scale)
                if a is not None:
                    p.grad.add_(a)
        return loss.detach(), accuracy

    def _add_balanced_gradients(self, x, x2, sub, state, sync=True, recording=None):
        """Add the gradients of the auxiliary losses (contrastive loss, reconstruction of masked channels) to the
        adversarial gradient that is stored on the critic. Each is rescaled to its weight times the norm of the
        adversarial gradient on the shared weights (an adaptive weight, as in Esser et al., 2021).
        """
        critic, hp = self.model.critic, self.hparams
        losses = []
        if self.model.use_contrastive:
            losses.append(('con', hp['contrastive_weight'],
                           lambda: contrastive_loss(critic, x, x2, sub, state, hp['contrastive_temperature'], recording)))
        mask = make_channel_mask(x, generator=self.rng)
        losses.append(('recon', hp['recon_balanced_weight'], lambda: (reconstruction_loss(critic, x, sub, state, mask), None)))

        params = list(critic.parameters())
        adversarial = [None if p.grad is None else p.grad.detach().clone() for p in params]
        total = [None if a is None else a.clone() for a in adversarial]
        logs = {}
        for name, weight, compute in losses:
            self.d_optimizer.zero_grad(set_to_none=True)
            with self._autocast():
                loss, accuracy = compute()
            self._watch(sync, **{f'{name}_loss': loss})
            loss.backward()
            shared = [(a, p.grad) for a, p in zip(adversarial, params) if a is not None and p.grad is not None]
            adv_norm = torch.sqrt(sum((a.float() ** 2).sum() for a, _ in shared))
            aux_norm = torch.sqrt(sum((g.float() ** 2).sum() for _, g in shared))
            scale = weight * adv_norm / (aux_norm + 1e-12)
            for i, p in enumerate(params):
                if p.grad is not None:
                    total[i] = scale * p.grad if total[i] is None else total[i] + scale * p.grad
            logs[f'{name}_loss'] = loss.detach()
            if accuracy is not None:
                logs[f'{name}_acc'] = accuracy
        for p, g in zip(params, total):
            p.grad = g
        return logs

    def _watch(self, sync, **values):
        """Stop on a non-finite loss: at once (sync=True), or record it for `_raise_if_nonfinite`."""
        for name, value in values.items():
            bad = ~torch.isfinite(value.detach()).all()
            if sync:
                if bad:
                    raise FloatingPointError(f'{name} is not finite: training stopped.')
            else:
                self._nonfinite[name] = self._nonfinite[name] | bad if name in self._nonfinite else bad

    def _raise_if_nonfinite(self):
        bad = [name for name, flag in self._nonfinite.items() if bool(flag)]
        self._nonfinite = {}
        if bad:
            raise FloatingPointError(f"{', '.join(bad)} became non-finite during epoch {self.epoch + 1}: "
                                     'training stopped (nothing was saved for this epoch).')

    def fit(self, sampler, max_epochs, out_prefix, save_freq=10, log=print):
        """Train up to `max_epochs`. Writes `<out_prefix>.csv` (one row of epoch means per epoch) and
        `<out_prefix>_epoch_<n>.pt` every `save_freq` epochs and at the end.
        """
        if self._sampler_rng_state is not None:
            sampler.generator.set_state(self._sampler_rng_state)
            self._sampler_rng_state = None
        csv_path = f'{out_prefix}.csv'
        rows = []
        if self.epoch > 0 and os.path.exists(csv_path):  # resumed run: keep the epochs before the checkpoint
            with open(csv_path, newline='') as f:
                rows = [r for r in csv.DictReader(f) if int(r['epoch']) <= self.epoch]

        self.prepare(sampler)
        self.model.train()
        hp = self.hparams
        while self.epoch < max_epochs:
            start, sums = time.time(), {}
            for _ in range(hp['steps_per_epoch']):  # diagnostics are summed on the device and read once per epoch
                for key, value in self.train_step(sampler.batch(), sync=False).items():
                    sums[key] = sums[key] + value.float() if key in sums else value.float().clone()
            self._raise_if_nonfinite()
            self.epoch += 1
            row = {'epoch': self.epoch, **{k: v.item() / hp['steps_per_epoch'] for k, v in sums.items()},
                   'lr_d': self.d_scheduler.get_last_lr()[0], 'lr_g': self.g_scheduler.get_last_lr()[0],
                   'seconds': time.time() - start}
            rows.append(row)
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(row))
                writer.writeheader()
                writer.writerows(rows)
            log(f"epoch {self.epoch}/{max_epochs}  " + '  '.join(
                f'{k} {v:.4g}' for k, v in row.items() if k not in ('epoch', 'lr_d', 'lr_g')), flush=True)
            if self.epoch % save_freq == 0 or self.epoch == max_epochs:
                self.save(f'{out_prefix}_epoch_{self.epoch}.pt', sampler)
        return self

    def save(self, path, sampler=None):
        """Save everything needed to continue the run: weights, optimizers, schedules, counters, random state."""
        checkpoint = {
            'config': self.model.config, 'model': self.model.state_dict(),
            'hparams': self.hparams, 'amp': self.amp, 'epoch': self.epoch, 'global_step': self.global_step,
            'd_optimizer': self.d_optimizer.state_dict(), 'g_optimizer': self.g_optimizer.state_dict(),
            'd_scheduler': self.d_scheduler.state_dict(), 'g_scheduler': self.g_scheduler.state_dict(),
            'rng': self.rng.get_state(),
            'sampler_rng': None if sampler is None else sampler.generator.get_state(),
            'torch_version': str(torch.__version__),
        }
        if self.ema_generator is not None:
            checkpoint['ema_generator'] = self.ema_generator.state_dict()
        if self._spectral_reference is not None:
            checkpoint['spectral_reference'] = self._spectral_reference
        torch.save(checkpoint, f'{path}.tmp')
        os.replace(f'{path}.tmp', path)  # a job killed while saving cannot leave a truncated checkpoint

    @classmethod
    def load(cls, path, device='cpu', amp=None):
        """Restore a trainer from a checkpoint written by `save`. Training continues exactly where it stopped."""
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        hparams = {**LEGACY_HPARAMS, **checkpoint['hparams']}
        for name in REMOVED_HPARAMS:
            hparams.pop(name, None)
        model = build_model(checkpoint)
        trainer = cls(model, device=device, amp=checkpoint['amp'] if amp is None else amp, **hparams)
        for name in ('d_optimizer', 'g_optimizer', 'd_scheduler', 'g_scheduler'):
            getattr(trainer, name).load_state_dict(checkpoint[name])
        try:
            trainer.rng.set_state(checkpoint['rng'])
        except RuntimeError:  # saved on another kind of device (e.g. CUDA, loaded on a CPU): the random streams restart
            print(f'!! the random state in {path} is for another device; training continues, but not bit-identically')
        trainer._sampler_rng_state = checkpoint['sampler_rng']
        trainer.epoch, trainer.global_step = checkpoint['epoch'], checkpoint['global_step']
        trainer.seed = None  # the random streams come from the checkpoint
        if trainer.ema_generator is not None:
            trainer.ema_generator.load_state_dict(checkpoint['ema_generator'])
        trainer._spectral_reference = checkpoint.get('spectral_reference')
        return trainer


def build_model(checkpoint):
    """Build the model of a checkpoint and load its weights."""
    model = FiLMGAN(**{**LEGACY_CONFIG, **checkpoint['config']})
    model.load_state_dict(checkpoint['model'])
    return model


def load_model(path, ema=True):
    """Load the model of a checkpoint (`.pt`, or `.model.keras` of the Keras implementation), on the CPU.
    ema=True uses the running average of the generator's weights, if the run kept one.
    """
    if path.endswith('.keras'):
        from .models.keras_import import load_keras_checkpoint
        return load_keras_checkpoint(path)
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    model = build_model(checkpoint)
    if ema and 'ema_generator' in checkpoint:
        model.generator.load_state_dict(checkpoint['ema_generator'])
    return model
