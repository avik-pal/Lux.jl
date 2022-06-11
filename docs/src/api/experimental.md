```@meta
CurrentModule = Lux
```

## Experimental Features

!!! warning
    These features are relatively new additions to `Lux`, and as such haven't been pressure tested enough. We recommend users to carefully go through these docs, since they highlight some of the common gotchas when using these features. As always, if something doesn't work and the documentation doesn't explicitly mention that it shouldn't work, please file an issue

## `Lux.@layerdef` API

```@docs
Lux.@layerdef
```

### Showcase

* [`SimpleRNN`](/examples/SimpleRNN/main.jl) example rewitten using `@layerdef`

```@example
using Lux, Random, NNlib

Lux.@layerdef function SpiralClassifier(x::AbstractArray{T, 3}) where {T}
    lstm_cell ← LSTMCell(2 => 8)
    h, c = lstm_cell(view(x, :, 1, :))
    for i in 1:size(x, 2)
        h, c = lstm_cell(view(x, :, i, :), h, c)
    end
    y = Dense(8, 2, sigmoid)(h)
    return vec(y)
end

model = SpiralClassifier()

x = randn(Float32, 2, 4, 1)
ps, st = Lux.setup(Random.default_rng(), model)

model(x, ps, st)
```

## Index

```@index
Pages = ["experimental.md"]
```
