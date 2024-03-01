# Coding guidelines

## Module `forward` signature

In modules that do not implement the `Module` or ModuleT` traits, we use the
following arguments order:

- The primary input argument (e.g. hidden representations or piece
  identifiers).
- Other input arguments in alphabetical order.
- Option arguments in alphabetical order.

If there is no primary input argument, all input arguments are ordered
alphabetically.

## Module creation pattern

In Curated Transformers, many models and layers accepted layers as arguments to
build models through composition. For instance (simplified):

```python
class DecoderLayer(nn.Module):
    def __init__(self, *, attention: SelfAttention, feed_forward: PointwiseFeedForward):
        super().__init__()
        self.attention = attention
        self.feed_forward = feed_forward
```

This works well with PyTorch, because parameter variable naming is based on
attribute names. So, in this case, the prefixes `attention` and `feed_forward`
are used for variables in the `SelfAttention` and `PointwiseFeedForward`
modules.

However, this does not translate well to Candle because variable names are
constructed by pushing prefixes to a `VarBuilder`. Since `SelfAttention` is
created separately from `DecoderLayer`, it does not its prefix from
`DecoderLayer`. We could push a prefix like `layer_1.attention` to the
`VarBuilder` that is passed to `SelfAttention`, but this breaks separation of
concerns since `DecoderLayer` should be responsible for its internal naming.

Summarized, we want `DecoderLayer` to construct its layers such as
`SelfAttention` with the correct prefixes. Our solution is similar to passing a
closure to the module of the form `fn(VarBuilder) -> SelfAttention`. This allows
the module to push a prefix to the `VarBuilder` and then pass the `VarBuilder`
to the closure to construct a `SelfAttention` module.

We formalize this idea by instead defining a trait such as:

```rust
pub trait BuildAttention {
    fn build(&self, vb: VarBuilder) -> Result<Box<dyn Attention>, BoxedError>;
}
```

Implementations of this trait can be passed to `DecoderLayer` to construct an
attention module with the right prefix by accepting (for instance) a `Box<dyn
BuildAttention>`. A pattern that naturally emerges from this approach this type
of building traits for a configuration structs. For instance, for self-attention
we could have a struct `SelfAttentionConfig` that contains all the
hyperparameters for self-attention and its `BuildAttention` implementation
builds the corresponding module. With this approach, models can be constructed
completely from nested configuration structs.
