# BruLux

An **extremely experimental research project** making an attempt to use another [**experimental research project: Brutus**](https://github.com/JuliaLabs/brutus) for using MLIR with Lux.jl

## Example

### Dense Layer

```julia
using BruLux, Functors, Lux, Random

rng = Random.default_rng()
Random.seed!(rng, 0)

# Dense without bias
d = Dense(2 => 4; bias = false)
ps, st = Lux.setup(rng, d);
ps_b, st_b = fmap(BruLuxArray, ps), fmap(BruLuxArray, st);
x = randn(Float32, 2, 3);
x_b = BruLuxArray(x);

d(x, ps, st)[1]
d(x_b, ps_b, st_b)[1]

# Dense with bias
d = Dense(2 => 4)
ps, st = Lux.setup(rng, d);
ps_b, st_b = fmap(BruLuxArray, ps), fmap(BruLuxArray, st);
x = randn(Float32, 2, 3);
x_b = BruLuxArray(x);

d(x, ps, st)[1]
d(x_b, ps_b, st_b)[1]
```

### Convolutions

```julia
using BruLux, Functors, Lux, Random

rng = Random.default_rng()
Random.seed!(rng, 0)

c = Conv((3, 3), 3 => 16)
ps, st = Lux.setup(rng, c);
ps_b, st_b = fmap(BruLuxArray, ps), fmap(BruLuxArray, st);
x = randn(Float32, 3, 3, 3, 1);
x_b = BruLuxArray(x);

c(x, ps, st)[1]
c(x_b, ps_b, st_b)[1]
```

### Vision Transformer

```julia
using Boltz, BruLux, Functors, Lux, Random

rng = Random.default_rng()
Random.seed!(rng, 0)

make_brulux_array(x::AbstractArray) = BruLuxArray(x)
make_brulux_array(x) = x

vit = Boltz.vision_transformer_from_config(:tiny);
ps, st = Lux.setup(rng, vit);
ps_b, st_b = fmap(make_brulux_array, ps), fmap(make_brulux_array, st);
x = randn(Float32, 256, 256, 3, 2);
x_b = BruLuxArray(x);

r = vit(x, ps, st)[1]
r_b = vit(x_b, ps_b, st_b)[1]

r ≈ r_b
```