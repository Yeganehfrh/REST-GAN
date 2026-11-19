"""Random segments from continuous recordings, one batch at a time."""
import torch


class RecordingSampler:
    """Draws training batches: a random recording, then a random segment of it, for every item."""

    def __init__(self,
                 x,                 # (recordings, channels, time)
                 sub,               # participant of each recording
                 state,             # eye state of each recording
                 seg_len=512,
                 batch_size=128,
                 paired=False,      # also return 'x2', a second segment of the same recording (contrastive loss)
                 n_refs=0,          # also return 'refs', further segments of the same recording (participant encoder)
                 seed=0,
                 device='cpu',      # where the recordings are kept and the batches are cut
                 block_len=None,    # segments are cut within blocks of this many samples
                 n_groups=0):       # also return recordings drawn for the spectral loss ('g_recording', 'g_sub', 'g_state')
        self.device = torch.device(device)
        self.x = torch.as_tensor(x).to(self.device)
        self.sub = torch.as_tensor(sub).view(-1).to(self.device)
        self.state = torch.as_tensor(state).view(-1).to(self.device)
        self.n_recordings, self.n_channels, self.length = self.x.shape
        if self.length < seg_len:
            raise ValueError(f'Recordings ({self.length} samples) are shorter than a segment ({seg_len}).')
        if block_len is not None and (self.length % block_len or block_len < seg_len):
            raise ValueError(f'Recordings of {self.length} samples cannot be split into blocks of {block_len} '
                             f'that hold {seg_len}-sample segments.')
        self.seg_len = seg_len
        self.batch_size = batch_size
        self.paired = paired
        self.n_refs = n_refs
        self.block_len = block_len
        self.n_groups = n_groups
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        self._offsets = torch.arange(seg_len, device=self.device)

    def _randint(self, high, n, generator=None):
        return torch.randint(0, high, (n,), generator=self.generator if generator is None else generator,
                             device=self.device)

    def _segments(self, recording, generator=None):
        """A random segment of each recording in `recording`."""
        n = len(recording)
        if self.block_len is None:
            start = self._randint(self.length - self.seg_len + 1, n, generator)
        else:  # a random block, then a random start within it
            start = self._randint(self.length // self.block_len, n, generator) * self.block_len \
                + self._randint(self.block_len - self.seg_len + 1, n, generator)
        time = start[:, None] + self._offsets[None, :]
        return self.x[recording[:, None], :, time]  # (batch, seg_len, channels)

    def batch(self):
        # randomly pick a recording for every item
        recording = torch.randint(0, self.n_recordings, (self.batch_size,), generator=self.generator,
                                  device=self.device)
        out = {'x': self._segments(recording), 'sub': self.sub[recording], 'state': self.state[recording],
               'recording': recording}
        if self.paired:
            out['x2'] = self._segments(recording)
        if self.n_refs:
            out['refs'] = torch.stack([self._segments(recording) for _ in range(self.n_refs)], dim=1)
        if self.n_groups:
            group = self._randint(self.n_recordings, self.n_groups)
            out.update(g_recording=group, g_sub=self.sub[group], g_state=self.state[group])
        return out
