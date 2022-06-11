# Introduction

Welcome to the documentation of Lux!

# What is Lux?

`Lux` is a julia deep learning framework which decouples models and parameterization using deeply nested named tuples.

- Functional Design -- Pure Functions and Deterministic Function Calls.
- No more implicit parameterization.
- Compiler and AD-friendly Neural Networks

# Installation Guide

Install [julia v1.7 or above](https://julialang.org/downloads/).

```julia
using Pkg
Pkg.add("Lux")
```

# Resources to Get Started

* Go through the [Quickstart Example](#quickstart).
* Read the introductory tutorials on [julia](https://jump.dev/JuMP.jl/stable/tutorials/getting_started/getting_started_with_julia/#Getting-started-with-Julia) and [Lux](introduction/overview.md)
* Go through the examples sorted based on their complexity in the documentation

!!! tip
    For usage related questions, please use [Github Discussions](https://github.com/avik-pal/Lux.jl/discussions) or [JuliaLang Discourse (machine learning domain)](https://discourse.julialang.org/c/domain/ml/) which allows questions and answers to be indexed. To report bugs use [github issues](https://github.com/avik-pal/Lux.jl/issues) or even better send in a [pull request](https://github.com/avik-pal/Lux.jl/pulls).

# Quickstart

```julia
using Lux, Random, Optimisers, Zygote
```

We take randomness very seriously

```julia
# Seeding
rng = Random.default_rng()
Random.seed!(rng, 0)
```

Build the model

```julia
# Construct the layer
model = Chain(BatchNorm(128), Dense(128, 256, tanh), BatchNorm(256),
              Chain(Dense(256, 1, tanh), Dense(1, 10)))
```

Models don't hold parameters and states so initialize them. From there on, we just use our standard AD and Optimisers API.

```julia
# Parameter and State Variables
ps, st = Lux.setup(rng, model) .|> gpu

# Dummy Input
x = rand(rng, Float32, 128, 2) |> gpu

# Run the model
y, st = Lux.apply(model, x, ps, st)

# Gradients
## Pullback API to capture change in state
(l, st_), pb = pullback(p -> Lux.apply(model, x, p, st), ps)
gs = pb((one.(l), nothing))

# Optimization
st_opt = Optimisers.setup(Optimisers.ADAM(0.0001), ps)
st_opt, ps = Optimisers.update(st_opt, ps, gs)
```

# How the documentation is structured

Having a high-level overview of how this documentation is structured will help you know where to look for certain things.

* `Introduction` -- Talks about why we wrote Lux and has pointers to frameworks in the extended julia ecosystem which might help users to get started with deep learning
* `Examples` -- Contain tutorials of varying complexity. These contain worked examples of solving problems with Lux. Start here if you are new to Lux, or you have a particular problem class you want to model.
* `API` -- Contains a complete list of the functions you can use in Lux. Look here if you want to know how to use a particular function.
* `Design Docs` -- Contains information for people contributing to Lux development or writing Lux extensions. Don't worry about this section if you are using Lux to formulate and solve problems as a user.

# Citation

If you found this library to be useful in academic work, then please cite:

```bibtex
@misc{pal2022lux,
    author = {Pal, Avik},
    title = {Lux: Explicit Parameterization of Deep Neural Networks in Julia},
    year = {2022},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/avik-pal/Lux.jl/}}
}
```

Also consider starring [our github repo](https://github.com/avik-pal/Lux.jl/)