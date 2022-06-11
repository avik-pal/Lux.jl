# Why we wrote Lux?

Julia already has quite a few well established Neural Network Frameworks -- [Flux](https://fluxml.ai/) & [KNet](https://denizyuret.github.io/Knet.jl/latest/). However, certain design elements -- **Coupled Model and Parameters** & **Internal Mutations** -- associated with these frameworks make them less compiler and user friendly. Making changes to address these problems in the respective frameworks would be too disruptive for users. Here comes in `Lux` a neural network framework built completely using pure functions to make it both compiler and autodiff friendly.

# Design Principles

* **Layers must be immutable** -- cannot store any parameter/state but rather store the information to construct them
* **Layers are pure functions**
* **Layers return a Tuple containing the result and the updated state**
* **Given same inputs the outputs must be same** -- yes this must hold true even for stochastic functions. Randomness must be controlled using `rng`s passed in the state.
* **Easily extendible**

# `AbstractExplicitLayer` API

We provide 2 abstract layers:

1. `AbstractExplicitLayer`: Useful for Base Layers and needs to define the following functions
    * `initialparameters(rng::AbstractRNG, layer::CustomAbstractExplicitLayer)` -- This returns a `ComponentArray`/`NamedTuple` containing the trainable parameters for the layer.
    * `initialstates(rng::AbstractRNG, layer::CustomAbstractExplicitLayer)` -- This returns a NamedTuple containing the current state for the layer. For most layers this is typically empty. Layers that would potentially contain this include `BatchNorm`, `LSTM`, `GRU` etc.
    * `parameterlength(layer::CustomAbstractExplicitLayer)` -- These can be automatically calculated, but it is recommended that the user defines these.
    * `statelength(layer::CustomAbstractExplicitLayer)` -- These can be automatically calculated, but it is recommended that the user defines these.

2. `AbstractExplicitContainerLayer`: Used when the layer is storing other `AbstractExplicitLayer`s or `AbstractExplicitContainerLayer`s. This allows good defaults of the dispatches for functions mentioned in the previous point.

!!! note
    We recommend users to subtype their custom layers using `AbstractExplicitLayer` or `AbstractExplicitContainerLayer`. However, this is **not mandatory**.

# Why use Lux over Flux?

* **Large Neural Networks**
  * For small neural networks we recommend [SimpleChains.jl](https://github.com/PumasAI/SimpleChains.jl).
  * For SciML Applications (Neural ODEs, Deep Equilibrium Models) solvers typically expect a monolithic parameter vector. Flux enables this via its `destructure` mechanism, however, it often leads to [weird bugs](https://github.com/FluxML/Flux.jl/issues?q=is%3Aissue+destructure). Lux forces users to make an explicit distinction between state variables and parameter variables to avoid these issues.
  * Comes battery-included for distributed training using [FluxMPI.jl](https://github.com/avik-pal/FluxMPI.jl)
* **Sensible display of Custom Layers** -- Ever wanted to see Pytorch like Network printouts or wondered how to extend the pretty printing of Flux's layers. Lux handles all of that by default.
* **Less Bug-ridden Code**
  * *No arbitrary internal mutations* -- all layers are implemented as pure functions.
  * *All layers are deterministic* given the parameter and state -- if the layer is supposed to be stochastic (say `Dropout`), the state must contain a seed which is then updated after the function call.
* **Easy Parameter Manipulation** -- Wondering why Flux doesn't have `WeightNorm`, `SpectralNorm`, etc. The implicit parameter handling makes it extremely hard to pass parameters around without mutations which AD systems don't like. With Lux implementing them is outright simple.

# When not to use Lux?


