import einops
import equinox as eqx
import jax
import math

from typing import Optional, Tuple

from jax import numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray


def scaled_dot_product(
    q: Float[Array, "... dims seq_len"],
    k: Float[Array, "... dims seq_len"],
    v: Float[Array, "... dims seq_len"],
    mask: Optional[Array] = None,
) -> Tuple[Float[Array, "... seq_q dims"], Float[Array, "... seq_q seq_k"]]:
    d_k = q.shape[-1]
    logits = einops.einsum(q, k, "... seq_q dims, ... seq_k dims -> ... seq_q seq_k")
    attn_logits = logits / math.sqrt(d_k)
    if mask is not None:
        logits = jnp.where(mask == 0, jnp.finfo(logits.dtype).min, attn_logits)
    else:
        logits = attn_logits
    attention = jax.nn.softmax(logits, axis=-1)
    values = jnp.matmul(attention, v)
    return values, attention


def expand_mask(mask: Array) -> Array:
    assert mask.ndim >= 2, "mask must be atleast 2 dim"
    if mask.ndim == 3:
        mask = jnp.expand_dims(mask, axis=1)
    while mask.ndim < 4:
        mask = jnp.expand_dims(mask, axis=0)
    return mask


class MultiHeadAttention(eqx.Module):
    """Given initial embeddings, get q k v, apply attention, and output projection"""

    n_embed: int
    n_heads: int
    qkv_proj: eqx.nn.Linear
    output_proj: eqx.nn.Linear

    def __init__(self, key: PRNGKeyArray, n_embed: int, n_heads: int) -> None:
        self.n_embed = n_embed
        self.n_heads = n_heads

        key_qkv, key_proj = jax.random.split(key, 2)

        # TODO: is bias initialization and kernel init with xavier required?
        qkv_out_size = 3 * self.n_embed
        self.qkv_proj = eqx.nn.Linear(
            in_features=self.n_embed,
            out_features=qkv_out_size,
            key=key_qkv,
            use_bias=True,
        )
        self.output_proj = eqx.nn.Linear(
            in_features=self.n_embed,
            out_features=self.n_embed,
            use_bias=True,
            key=key_proj,
        )

    def __call__(
        self, x: Float[Array, "seq_len n_embed"], mask: Optional[Array] = None
    ) -> Tuple[
        Float[Array, "seq_len n_embed"], Float[Array, "n_heads seq_len seq_len"]
    ]:
        seq_len, n_embed = x.shape
        # a single projection layer, given the input produces Q, K, V matrices
        qkv = jax.vmap(self.qkv_proj)(x)

        # The scaled dot product attention allows a network to attend over a sequence.
        # However, often there are multiple different aspects a sequence element
        # wants to attend to, and a single weighted average is not a good option for it.
        # This is why we extend the attention mechanisms to multiple heads,
        # i.e. multiple different query-key-value triplets on the same features.
        # Specifically, given a query, key, and value matrix, we transform those into sub-queries,
        # sub-keys, and sub-values, which we pass through the scaled dot product attention independently.
        # Afterward, we concatenate the heads and combine them with a final weight matrix

        # split the embeding_dim into multiple heads
        # dim here is different from embed_dim, it's 3 * embed_dims
        reshaped_qkv = einops.rearrange(
            qkv,
            "seq_len (n_heads d) -> n_heads seq_len d",
            seq_len=seq_len,
            n_heads=self.n_heads,
        )
        # embedding dims contains all of qkv, so split
        q, k, v = jnp.array_split(reshaped_qkv, 3, axis=-1)
        values, attention = scaled_dot_product(q, k, v, mask=mask)
        values = einops.rearrange(
            values,
            "n_heads seq_len d -> seq_len (n_heads d)",
            n_heads=self.n_heads,
            seq_len=seq_len,
        )
        output_embeddings = jax.vmap(self.output_proj)(values)
        return output_embeddings, attention
