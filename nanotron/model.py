import equinox as eqx
import jax

from equinox import nn
from dataclasses import dataclass
from jax import numpy as jnp
from jaxtyping import Integer, Float, Array, PRNGKeyArray
from typing import List, Tuple, Optional

from . import attention
from .config import GPTConfig


class SwiGLU(eqx.Module):
    """
    https://azizbelaweid.substack.com/p/what-is-swiglu-how-to-implement-it
    """

    W: Float[Array, "in_features out_features"]
    V: Float[Array, "in_features out_features"]
    b: Float[Array, "out_features"]
    c: Float[Array, "out_features"]

    def __init__(self, key: PRNGKeyArray, in_features: int, out_features: int) -> None:
        k1, k2, k3, k4 = jax.random.split(key, 4)
        self.W = jax.random.normal(k1, (in_features, out_features))
        self.V = jax.random.normal(k2, (in_features, out_features))
        self.b = jax.random.normal(k3, (out_features,))
        self.c = jax.random.normal(k4, (out_features,))

    def __call__(
        self, x: Float[Array, "... in_features"]
    ) -> Float[Array, "... out_features"]:
        return jax.nn.swish(jnp.dot(x, self.W) + self.b) * (jnp.dot(x, self.V) + self.c)


class MLP(eqx.Module):
    c_fc: nn.Linear
    swiglu: SwiGLU
    c_proj: nn.Linear
    dropout: nn.Dropout

    def __init__(self, key: PRNGKeyArray, model_config: GPTConfig) -> None:
        key_fc, key_swiglu, key_proj, key_dropout = jax.random.split(key, 4)

        self.c_fc = nn.Linear(
            key=key_fc,
            in_features=model_config.n_embed,
            out_features=4 * model_config.n_embed,
            use_bias=model_config.bias,
        )

        self.swiglu = SwiGLU(
            key=key_swiglu,
            in_features=4 * model_config.n_embed,
            out_features=4 * model_config.n_embed,
        )

        self.c_proj = nn.Linear(
            key=key_proj,
            in_features=4 * model_config.n_embed,
            out_features=model_config.n_embed,
            use_bias=model_config.bias,
        )

        self.dropout = nn.Dropout(model_config.dropout)

    def __call__(
        self,
        key: PRNGKeyArray,
        x: Float[Array, "n_tokens n_embed"],
        inference: bool = False,
    ) -> Float[Array, "n_tokens n_embed"]:
        x = self.c_fc(x)
        x = self.swiglu(x)
        x = self.c_proj(x)
        x = self.dropout(x, key=key, inference=inference)
        return x


class CasualSelfAttention(eqx.Module):
    mha: attention.MultiHeadAttention

    def __init__(self, key: PRNGKeyArray, model_config: GPTConfig) -> None:
        self.mha = attention.MultiHeadAttention(
            key=key, n_embed=model_config.n_embed, n_heads=model_config.n_head
        )

    def __call__(
        self, x: Float[Array, "n_tokens n_embed"]
    ) -> Tuple[Float[Array, "n_tokens n_embed"], Float[Array, "n_tokens n_tokens"]]:
        """
        Args:
            x: Input embeddings of shape (n_tokens, n_embed)
        Returns:
            Tuple containing:
                - Output embeddings of shape (n_tokens, n_embed)
                - Attention weights of shape (n_tokens, n_tokens)
        """
        n_tokens = x.shape[0]
        mask = jnp.tril(jnp.ones((n_tokens, n_tokens)))
        return self.mha(x, mask=mask)


class Block(eqx.Module):
    ln_1: nn.LayerNorm
    attn: CasualSelfAttention
    ln_2: nn.LayerNorm
    mlp: MLP

    def __init__(self, key: PRNGKeyArray, model_config: GPTConfig) -> None:
        key_attn, key_mlp = jax.random.split(key, 2)

        self.ln_1 = nn.LayerNorm(model_config.n_embed, use_bias=model_config.bias)
        self.attn = CasualSelfAttention(key=key_attn, model_config=model_config)
        self.ln_2 = nn.LayerNorm(model_config.n_embed, use_bias=model_config.bias)
        self.mlp = MLP(key=key_mlp, model_config=model_config)

    def __call__(
        self, key: PRNGKeyArray, x: Float[Array, "n_tokens n_embed"]
    ) -> Float[Array, "n_tokens n_embed"]:
        mlp_keys = jax.random.split(key, x.shape[0])  # Create a key for each token
        x = jax.vmap(self.ln_1)(x)
        output_embeddings, attn = self.attn(x)
        x = x + output_embeddings
        x = jax.vmap(self.ln_2)(x)
        x = x + jax.vmap(self.mlp)(mlp_keys, x)
        return x


class Transformer(eqx.Module):
    wte: nn.Embedding
    wpe: nn.Embedding
    drop: nn.Dropout
    h: List[Block]
    ln_f: nn.LayerNorm

    def __init__(self, key: PRNGKeyArray, model_config: GPTConfig) -> None:
        te_key, pe_key, h_key = jax.random.split(key, 3)

        # token embeddings
        self.wte = nn.Embedding(
            key=te_key,
            num_embeddings=model_config.vocab_size,
            embedding_size=model_config.n_embed,
        )
        # positional embeddings
        self.wpe = nn.Embedding(
            key=pe_key,
            num_embeddings=model_config.block_size,
            embedding_size=model_config.n_embed,
        )
        self.drop = nn.Dropout(model_config.dropout)
        block_keys = jax.random.split(h_key, model_config.n_layers)
        self.h = [
            Block(key=block_keys[i], model_config=model_config)
            for i in range(model_config.n_layers)
        ]
        self.ln_f = nn.LayerNorm(model_config.n_embed, use_bias=model_config.bias)

    def __call__(
        self,
        key: PRNGKeyArray,
        tokens: Integer[Array, "n_tokens"],
        inference: bool = False,
    ) -> Float[Array, "n_tokens n_embed"]:
        pos = jnp.arange(0, len(tokens), dtype=jnp.int32)
        t_embed = jax.vmap(self.wte)(tokens)  # token embeddings
        p_embed = jax.vmap(self.wpe)(pos)  # positional embedidngs
        x = self.drop(
            t_embed + p_embed, inference=inference, key=key
        )  # TODO: confirm why is key optional in params
        for block in self.h:
            x = block(key, x)
        x = jax.vmap(self.ln_f)(x)
        return x


class GPT(eqx.Module):
    transformer: Transformer
    lm_head: nn.Linear

    def __init__(self, key: PRNGKeyArray, model_config: GPTConfig) -> None:
        key_transformer, key_lm_head = jax.random.split(key, 2)

        self.transformer = Transformer(key=key_transformer, model_config=model_config)
        self.lm_head = nn.Linear(
            key=key_lm_head,
            in_features=model_config.n_embed,
            out_features=model_config.vocab_size,
            use_bias=True,
        )

    def __call__(
        self,
        key: PRNGKeyArray,
        tokens: Integer[Array, "n_tokens"],
        inference: bool = False,
    ) -> Float[Array, "n_tokens vocab_size"]:
        x = self.transformer(key, tokens, inference=inference)
        if not inference:
            logits = jax.vmap(self.lm_head)(x)  # (n_tokens, vocab_size)
        else:
            last_token_embedding = x[-1]
            # during inference we only care about the last token
            # vmap is not needed here, because it's only single token
            logits = self.lm_head(last_token_embedding)
            logits = jnp.expand_dims(logits, axis=0)
        return logits

    def decode(
        self,
        key: PRNGKeyArray,
        initial_tokens: Integer[Array, "n_tokens"],
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[Integer] = None,
    ) -> Integer[Array, "n_tokens + max_new_tokens"]:
        """Generate text tokens given an initial sequence.

        Args:
            key: Random key for sampling
            initial_tokens: Initial sequence of tokens to continue from
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature (1.0 = no change, <1.0 = more conservative, >1.0 = more random)
            top_k: If set, only sample from the top k most likely tokens

        Returns:
            Array of generated tokens including the initial sequence
        """
        # Start with the initial tokens
        tokens = initial_tokens

        for i in range(max_new_tokens):
            # Get key for this iteration
            key, subkey = jax.random.split(key)
            # during inference, we only get last token logits
            logits = self(subkey, tokens, inference=True)  # (1, vocab_size)
            logits = logits / temperature

            if top_k is not None:
                v, _ = jax.lax.top_k(logits, top_k)
                min_value = v[0, -1]
                logits = jnp.where(logits < min_value, -jnp.inf, logits)

            # jax.random.categorical expects log-probabilities. The logits are
            # already unnormalized log-probabilities, so we pass them directly
            # after applying temperature scaling and optional top-k filtering.
            next_token = jax.random.categorical(subkey, logits[0])
            print(f"Generated token {i+1}/{max_new_tokens}: {next_token}")
            tokens = jnp.append(tokens, next_token)

        return tokens
