"""Scale *Adam updates* on newly appended embedding rows, never raw gradients."""
import dataclasses


@dataclasses.dataclass(frozen=True)
class RowScaledAdamW:
    b1: float = .9
    b2: float = .95
    eps: float = 1e-8
    weight_decay: float = 1e-10
    clip_gradient_norm: float = 1.
    new_row_scale: float = 4.
    task_old_rows: int = 50
    stage_old_rows: int = 596

    def create(self, lr, weight_decay_mask=None):
        import jax
        import jax.numpy as jnp
        import optax
        if self.new_row_scale <= 0 or min(self.task_old_rows, self.stage_old_rows) < 1:
            raise ValueError('Invalid registered row optimizer')
        tables = {'task_embeddings': self.task_old_rows, 'task_stage_embeddings': self.stage_old_rows}

        def path_keys(path):
            return tuple(getattr(key, 'key', getattr(key, 'name', None)) for key in path)

        def init_rows(params):
            found = set()
            for path, value in jax.tree_util.tree_flatten_with_path(params)[0]:
                keys = path_keys(path)
                if len(keys) >= 2 and keys[0] in tables and keys[1] == 'embedding':
                    if value.ndim != 2 or value.shape[0] <= tables[keys[0]]:
                        raise ValueError('Expected expanded embedding table')
                    found.add(keys[0])
            if found != set(tables):
                raise ValueError('Missing new-row parameter group')
            return optax.EmptyState()

        def scale(path, value):
            keys = path_keys(path)
            if len(keys) >= 2 and keys[0] in tables and keys[1] == 'embedding':
                factor = jnp.where(jnp.arange(value.shape[0]) >= tables[keys[0]], self.new_row_scale, 1)
                return value * factor[:, None].astype(value.dtype)
            return value

        def update_rows(updates, state, params=None):
            del params
            return jax.tree_util.tree_map_with_path(scale, updates), state

        return optax.chain(
            optax.clip_by_global_norm(self.clip_gradient_norm),
            optax.adamw(lr, b1=self.b1, b2=self.b2, eps=self.eps,
                        weight_decay=self.weight_decay, mask=weight_decay_mask),
            optax.GradientTransformation(init_rows, update_rows))
