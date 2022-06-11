# Base Type
## API:
### initialparameters(rng, l) --> Must return a NamedTuple/ComponentArray
### initialstates(rng, l)     --> Must return a NamedTuple
### parameterlength(l)        --> Integer
### statelength(l)            --> Integer
### l(x, ps, st)

abstract type AbstractExplicitLayer end

"""
    initialparameters(rng::AbstractRNG, l)

Generate the initial parameters of the layer `l`.
"""
initialparameters(::AbstractRNG, ::Any) = NamedTuple()
function initialparameters(rng::AbstractRNG, l::NamedTuple)
    map(Base.Fix1(initialparameters, rng), l)
end

"""
    initialstates(rng::AbstractRNG, l)

Generate the initial states of the layer `l`.
"""
initialstates(::AbstractRNG, ::Any) = NamedTuple()
initialstates(rng::AbstractRNG, l::NamedTuple) = map(Base.Fix1(initialstates, rng), l)

"""
    parameterlength(l)

Return the total number of parameters of the layer `l`.
"""
function parameterlength(l::AbstractExplicitLayer)
    parameterlength(initialparameters(Random.default_rng(), l))
end
function parameterlength(nt::Union{NamedTuple, Tuple})
    length(nt) == 0 ? 0 : sum(parameterlength, nt)
end
parameterlength(a::AbstractArray) = length(a)
parameterlength(x) = 0

"""
    statelength(l)

Return the total number of states of the layer `l`.
"""
statelength(l::AbstractExplicitLayer) = statelength(initialstates(Random.default_rng(), l))
statelength(nt::Union{NamedTuple, Tuple}) = length(nt) == 0 ? 0 : sum(statelength, nt)
statelength(a::AbstractArray) = length(a)
statelength(x::Union{Number, Symbol}) = 1
statelength(x) = 0

"""
    setup(rng::AbstractRNG, l::AbstractExplicitLayer)

Shorthand for getting the parameters and states of the layer `l`. Is equivalent to `(initialparameters(rng, l), initialstates(rng, l))`.
"""
function setup(rng::AbstractRNG, l::AbstractExplicitLayer)
    (initialparameters(rng, l), initialstates(rng, l))
end

"""
    apply(model::AbstractExplicitLayer, x, ps::Union{ComponentArray,NamedTuple}, st::NamedTuple)

Simply calls `model(x, ps, st)`
"""
function apply(model::AbstractExplicitLayer, x, ps::VALID_PARAMETER_TYPES,
               st::NamedTuple)
    model(x, ps, st)
end

function Base.show(io::IO, x::AbstractExplicitLayer)
    __t = rsplit(string(get_typename(x)), "."; limit=2)
    T = length(__t) == 2 ? __t[2] : __t[1]
    print(io, "$T()")
end

# Abstract Container Layers
abstract type AbstractExplicitContainerLayer{layers} <: AbstractExplicitLayer end

function initialparameters(rng::AbstractRNG,
                           l::AbstractExplicitContainerLayer{layers}) where {layers}
    return NamedTuple{layers}(initialparameters.(rng, getfield.((l,), layers)))
end

function initialstates(rng::AbstractRNG,
                       l::AbstractExplicitContainerLayer{layers}) where {layers}
    return NamedTuple{layers}(initialstates.(rng, getfield.((l,), layers)))
end

function parameterlength(l::AbstractExplicitContainerLayer{layers}) where {layers}
    sum(parameterlength, getfield.((l,), layers))
end

function statelength(l::AbstractExplicitContainerLayer{layers}) where {layers}
    sum(statelength, getfield.((l,), layers))
end

# Test Mode
"""
    testmode(st::NamedTuple, mode::Bool=true)

Make all occurances of `training` in state `st` `!mode`
"""
testmode(st::NamedTuple, mode::Bool=true) = update_state(st, :training, Val(!mode))

"""
    trainmode(x::Any, mode::Bool=true)

Make all occurances of `training` in state `st` `mode`
"""
trainmode(x::Any, mode::Bool=true) = testmode(x, !mode)

"""
    update_state(st::NamedTuple, key::Symbol, value; layer_check=_default_layer_check(key))

Recursively update all occurances of the `key` in the state `st` with the `value`.
"""
function update_state(st::NamedTuple, key::Symbol, value;
                      layer_check=_default_layer_check(key))
    function _update_state(st, key::Symbol, value)
        return Setfield.set(st, Setfield.PropertyLens{key}(), value)
    end
    return fmap(_st -> _update_state(_st, key, value), st; exclude=layer_check)
end

function _default_layer_check(key)
    _default_layer_check_closure(x) = hasmethod(keys, (typeof(x),)) ? key ∈ keys(x) : false
    return _default_layer_check_closure
end
