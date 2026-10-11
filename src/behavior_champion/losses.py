"""Supervised heldout metrics with augmentation disabled, not train-mode eval.

The pinned author's compute_detailed_loss(train=False) omits stage CE entirely.
Compute that diagnostic explicitly from the SAME causally masked base task
token; never switch train=True just to obtain it or report a missing loss as 0.
"""


def stage_metrics(logits, task_ids, targets, num_stages):
    import jax
    import jax.numpy as jnp
    counts = jnp.asarray(num_stages, dtype=jnp.int32)[task_ids]
    masked = jnp.where(jnp.arange(logits.shape[-1])[None] < counts[:, None], logits, -jnp.inf)
    ce = -jax.nn.log_softmax(masked)[jnp.arange(targets.shape[0]), targets]
    accuracy = (jnp.argmax(masked, axis=-1) == targets).astype(jnp.float32)
    return ce, accuracy


def compute_eval_losses(model, rng, observation, actions, *, num_flow_samples=4):
    import jax
    import jax.numpy as jnp
    from b1k.models.observation import preprocess_observation
    from openpi.models.pi0 import make_attn_mask
    model.eval()
    losses = dict(model.compute_detailed_loss(rng, observation, actions,
                                              train=False, num_flow_samples=num_flow_samples))
    preprocess_rng, _ = jax.random.split(rng)
    observation = preprocess_observation(preprocess_rng, observation, train=False)
    tokens, valid, autoregressive = model.embed_prefix(observation)
    (prefix, _), _ = model.PaliGemma.llm([tokens, None],
        mask=make_attn_mask(valid, autoregressive), positions=jnp.cumsum(valid, axis=1) - 1)
    base_task_index = jnp.argmax(autoregressive) - 1
    logits = model.stage_pred_from_vlm(prefix[:, base_task_index, :])
    ce, accuracy = stage_metrics(logits, observation.tokenized_prompt[:, 0],
                                observation.tokenized_prompt[:, 1], model.task_num_stages)
    losses['subtask_loss'] = ce
    losses['subtask_accuracy'] = accuracy
    # Report an actual per-example total so old/new task and task-equal cohort
    # aggregation do not broadcast the other samples' auxiliary losses.
    fast = losses.get('fast_loss', jnp.zeros_like(ce))
    if model.config.use_fast_auxiliary and 'fast_loss' not in losses:
        raise ValueError('FAST target metrics missing from the heldout observation')
    losses['total_loss'] = losses['action_loss'] + model.config.fast_loss_weight * fast + model.config.subtask_loss_weight * ce
    return losses
