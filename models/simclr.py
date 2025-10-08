import jax.numpy as jnp
from flax import linen as nn
import flaxmodels
import optax

class SimCLR(nn.Module):
    hidden_dim: int
    temperature: float

    def setup(self):
        self.convnet = flaxmodels.ResNet18(
            output='activations', pretrained=False,
            normalize=False, num_classes=4*self.hidden_dim
        )
        self.head = nn.Sequential([
            nn.relu,
            nn.Dense(self.hidden_dim)
        ])

    def __call__(self, imgs, train=True):
        # Encode all images
        model_feats = self.convnet(imgs, train=train)
        feats = self.head(model_feats['fc'])

        # Calculate cosine similarity between all images
        cos_sim = optax.cosine_similarity(feats[:, None, :], feats[None, :, :])
        cos_sim /= self.temperature
        # Masking cosine similarities to itself
        diag_range = jnp.arange(feats.shape[0], dtype=jnp.int32)
        cos_sim = cos_sim.at[diag_range, diag_range].set(-9e15)
        # Find positive example -> batch_size//2 away from the original example
        shifted_diag = jnp.roll(diag_range, imgs.shape[0] // 2)
        pos_logits = cos_sim[diag_range, shifted_diag]
        # InfoNCE loss
        nll = - pos_logits + nn.logsumexp(cos_sim, axis=-1)
        nll = nll.mean()

        # Logging
        metrics = {'loss': nll}
        # Determine ranking position of positive example
        comb_sim = jnp.concatenate([pos_logits[:, None],
                                    cos_sim.at[shifted_diag, diag_range].set(-9e15)],
                                   axis=-1)
        sim_argsort = (-comb_sim).argsort(axis=-1).argmin(axis=-1)
        # Logging of ranking position
        metrics['acc_top1'] = (sim_argsort == 0).mean()
        metrics['acc_top5'] = (sim_argsort < 5).mean()
        metrics['acc_mean_pos'] = (sim_argsort + 1).mean()

        return nll, metrics

    def encode(self, imgs, train=False):
        model_feats = self.convnet(imgs, train=train)
        return model_feats['block4_1'].mean(axis=(1,2))

    def calculate_features(self, imgs, train: bool = False):
        """Return projection head output (logits) directly."""
        model_feats = self.convnet(imgs, train=train)
        return model_feats


