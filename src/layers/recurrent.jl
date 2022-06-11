"""
    RNNCell(in_dims => out_dims, activation=tanh; bias::Bool=true, init_bias=zeros32, init_weight=glorot_uniform, init_state=ones32)

An Elman RNNCell cell with `activation` (typically set to `tanh` or `relu`).

``h_{new} = activation(weight_{ih} \\times x + weight_{hh} \\times h_{prev} + bias)``

## Arguments

* `in_dims`: Input Dimension
* `out_dims`: Output (Hidden State) Dimension
* `activation`: Activation function
* `bias`: Set to false to deactivate bias
* `init_bias`: Initializer for bias
* `init_weight`: Initializer for weight
* `init_state`: Initializer for hidden state

## Inputs

* Case 1: Only a single input `x` of shape `(in_dims, batch_size)` - Creates a hidden state using `init_state` and proceeds to Case 2.
* Case 2: Tuple (`x`, `h`) is provided, then the updated hidden state is returned.

## Returns

* New hidden state ``h_{new}`` of shape `(out_dims, batch_size)`
* Updated model state

## Parameters

* `weight_ih`: Maps the input to the hidden state.
* `weight_hh`: Maps the hidden state to the hidden state.
* `bias`: Bias vector (not present if `bias=false`)

## States

* `rng`: Controls the randomness (if any) in the initial state generation
"""
struct RNNCell{bias, A, B, W, S} <: AbstractExplicitLayer
    activation::A
    in_dims::Int
    out_dims::Int
    init_bias::B
    init_weight::W
    init_state::S
end

function RNNCell((in_dims, out_dims)::Pair{<:Int, <:Int},
                 activation=tanh;
                 bias::Bool=true,
                 init_bias=zeros32,
                 init_weight=glorot_uniform,
                 init_state=ones32)
    return RNNCell{bias, typeof(activation), typeof(init_bias), typeof(init_weight),
                   typeof(init_state)}(activation, in_dims, out_dims, init_bias,
                                       init_weight, init_state)
end

function initialparameters(rng::AbstractRNG, rnn::RNNCell{bias}) where {bias}
    ps = (weight_ih=rnn.init_weight(rng, rnn.out_dims, rnn.in_dims),
          weight_hh=rnn.init_weight(rng, rnn.out_dims, rnn.out_dims))
    if bias
        ps = merge(ps, (bias=rnn.init_bias(rng, rnn.out_dims),))
    end
    return ps
end

function initialstates(rng::AbstractRNG, ::RNNCell)
    # FIXME: Take PRNGs seriously
    randn(rng, 1)
    return (rng=replicate(rng),)
end

function (rnn::RNNCell)(x::AbstractMatrix, ps::VALID_PARAMETER_TYPES,
                        st::NamedTuple)
    rng = replicate(st.rng)
    @set! st.rng = rng
    hidden_state = rnn.init_state(rng, rnn.out_dims, size(x, 2))
    return rnn((x, hidden_state), ps, st)
end

function (rnn::RNNCell{true})((x, hidden_state)::Tuple{<:AbstractMatrix, <:AbstractMatrix},
                              ps::VALID_PARAMETER_TYPES, st::NamedTuple)
    h_new = rnn.activation.(ps.weight_ih * x .+ ps.weight_hh * hidden_state .+ ps.bias)
    return h_new, st
end

function (rnn::RNNCell{true, typeof(identity)})((x,
                                                 hidden_state)::Tuple{<:AbstractMatrix,
                                                                      <:AbstractMatrix},
                                                ps::VALID_PARAMETER_TYPES,
                                                st::NamedTuple)
    h_new = ps.weight_ih * x .+ ps.weight_hh * hidden_state .+ ps.bias
    return h_new, st
end

function (rnn::RNNCell{false})((x, hidden_state)::Tuple{<:AbstractMatrix, <:AbstractMatrix},
                               ps::VALID_PARAMETER_TYPES, st::NamedTuple)
    h_new = rnn.activation.(ps.weight_ih * x .+ ps.weight_hh * hidden_state)
    return h_new, st
end

function (rnn::RNNCell{false, typeof(identity)})((x,
                                                  hidden_state)::Tuple{<:AbstractMatrix,
                                                                       <:AbstractMatrix},
                                                 ps::VALID_PARAMETER_TYPES,
                                                 st::NamedTuple)
    h_new = ps.weight_ih * x .+ ps.weight_hh * hidden_state
    return h_new, st
end

function Base.show(io::IO, r::RNNCell{bias}) where {bias}
    print(io, "RNNCell($(r.in_dims) => $(r.out_dims)")
    (r.activation == identity) || print(io, ", $(r.activation)")
    bias || print(io, ", bias=false")
    return print(io, ")")
end

"""
    LSTMCell(in_dims => out_dims; init_weight=(glorot_uniform, glorot_uniform, glorot_uniform, glorot_uniform), init_bias=(zeros32, zeros32, ones32, zeros32), init_state=zeros32)

Long Short-Term (LSTM) Cell

``i = \\sigma(W_{ii} \\times x + W_{hi} \\times h_{prev} + b_{i})``

``f = \\sigma(W_{if} \\times x + W_{hf} \\times h_{prev} + b_{f})``

``g = tanh(W_{ig} \\times x + W_{hg} \\times h_{prev} + b_{g})``

``o = \\sigma(W_{io} \\times x + W_{ho} \\times h_{prev} + b_{o})``

``c_{new} = f \\cdot c_{prev} + i \\cdot g``

``h_{new} = o \\cdot tanh(c_{new})``

## Arguments

* `in_dims`: Input Dimension
* `out_dims`: Output (Hidden State & Memory) Dimension
* `init_bias`: Initializer for bias. Must be a tuple containing 4 functions
* `init_weight`: Initializer for weight. Must be a tuple containing 4 functions
* `init_state`: Initializer for hidden state and memory

## Inputs

* Case 1: Only a single input `x` of shape `(in_dims, batch_size)` - Creates a hidden state and memory using `init_state` and proceeds to Case 2.
* Case 2: Tuple (`x`, `h`, `c`) is provided, then the updated hidden state and memory is returned.

## Returns

* Tuple Containing
    * New hidden state ``h_{new}`` of shape `(out_dims, batch_size)`
    * Updated Memory ``c_{new}`` of shape `(out_dims, batch_size)`
* Updated model state

## Parameters

* `weight_i`: Concatenated Weights to map from input space ``\\left\\{ W_{ii}, W_{if}, W_{ig}, W_{io} \\right\\}``.
* `weight_h`: Concatenated Weights to map from hidden space ``\\left\\{ W_{hi}, W_{hf}, W_{hg}, W_{ho} \\right\\}``
* `bias`: Bias vector

## States

* `rng`: Controls the randomness (if any) in the initial state generation
"""
struct LSTMCell{B, W, S} <: AbstractExplicitLayer
    in_dims::Int
    out_dims::Int
    init_bias::B
    init_weight::W
    init_state::S
end

function LSTMCell((in_dims, out_dims)::Pair{<:Int, <:Int};
                  init_weight::Tuple{Function, Function, Function, Function}=(glorot_uniform,
                                                                              glorot_uniform,
                                                                              glorot_uniform,
                                                                              glorot_uniform),
                  init_bias::Tuple{Function, Function, Function, Function}=(zeros32,
                                                                            zeros32,
                                                                            ones32,
                                                                            zeros32),
                  init_state::Function=zeros32)
    return LSTMCell(in_dims, out_dims, init_bias, init_weight, init_state)
end

function initialparameters(rng::AbstractRNG, lstm::LSTMCell)
    weight_i = vcat([init_weight(rng, lstm.out_dims, lstm.in_dims)
                     for init_weight in lstm.init_weight]...)
    weight_h = vcat([init_weight(rng, lstm.out_dims, lstm.out_dims)
                     for init_weight in lstm.init_weight]...)
    bias = vcat([init_bias(rng, lstm.out_dims, 1) for init_bias in lstm.init_bias]...)
    return (weight_i=weight_i, weight_h=weight_h, bias=bias)
end

function initialstates(rng::AbstractRNG, ::LSTMCell)
    # FIXME: Take PRNGs seriously
    randn(rng, 1)
    return (rng=replicate(rng),)
end

function (lstm::LSTMCell)(x::AbstractMatrix, ps::VALID_PARAMETER_TYPES,
                          st::NamedTuple)
    rng = replicate(st.rng)
    @set! st.rng = rng
    hidden_state = lstm.init_state(rng, lstm.out_dims, size(x, 2))
    memory = lstm.init_state(rng, lstm.out_dims, size(x, 2))
    return lstm((x, hidden_state, memory), ps, st)
end

function (lstm::LSTMCell)((x, hidden_state,
                           memory)::Tuple{<:AbstractMatrix, <:AbstractMatrix,
                                          <:AbstractMatrix},
                          ps::VALID_PARAMETER_TYPES,
                          st::NamedTuple)
    g = ps.weight_i * x .+ ps.weight_h * hidden_state .+ ps.bias
    input, forget, cell, output = multigate(g, Val(4))
    memory_new = @. sigmoid_fast(forget) * memory + sigmoid_fast(input) * tanh_fast(cell)
    hidden_state_new = @. sigmoid_fast(output) * tanh_fast(memory_new)
    return (hidden_state_new, memory_new), st
end

Base.show(io::IO, l::LSTMCell) = print(io, "LSTMCell($(l.in_dims) => $(l.out_dims))")

"""
    GRUCell((in_dims, out_dims)::Pair{<:Int,<:Int}; init_weight::Tuple{Function,Function,Function}=(glorot_uniform, glorot_uniform, glorot_uniform), init_bias::Tuple{Function,Function,Function}=(zeros32, zeros32, zeros32), init_state::Function=zeros32)

Gated Recurrent Unit (GRU) Cell

``r = \\sigma(W_{ir} \\times x + W_{hr} \\times h_{prev} + b_{hr})``

``z = \\sigma(W_{iz} \\times x + W_{hz} \\times h_{prev} + b_{hz})``

``n = \\sigma(W_{in} \\times x + b_{in} + r \\cdot (W_{hn} \\times h_{prev} + b_{hn}))``

``h_{new} = (1 - z) \\cdot n + z \\cdot h_{prev}``

## Arguments

* `in_dims`: Input Dimension
* `out_dims`: Output (Hidden State) Dimension
* `init_bias`: Initializer for bias. Must be a tuple containing 3 functions
* `init_weight`: Initializer for weight. Must be a tuple containing 3 functions
* `init_state`: Initializer for hidden state

## Inputs

* Case 1: Only a single input `x` of shape `(in_dims, batch_size)` - Creates a hidden state using `init_state` and proceeds to Case 2.
* Case 2: Tuple (`x`, `h`) is provided, then the updated hidden state is returned.

## Returns

* New hidden state ``h_{new}`` of shape `(out_dims, batch_size)`
* Updated model state

## Parameters

* `weight_i`: Concatenated Weights to map from input space ``\\left\\{ W_{ir}, W_{iz}, W_{in} \\right\\}``.
* `weight_h`: Concatenated Weights to map from hidden space ``\\left\\{ W_{hr}, W_{hz}, W_{hn} \\right\\}``
* `bias_i`: Bias vector (``b_{in}``)
* `bias_h`: Concatenated Bias vector for the hidden space ``\\left\\{ b_{hr}, b_{hz}, b_{hn} \\right\\}``

## States

* `rng`: Controls the randomness (if any) in the initial state generation
"""
struct GRUCell{W, B, S} <: AbstractExplicitLayer
    in_dims::Int
    out_dims::Int
    init_weight::W
    init_bias::B
    init_state::S
end

function GRUCell((in_dims, out_dims)::Pair{<:Int, <:Int};
                 init_weight::Tuple{Function, Function, Function}=(glorot_uniform,
                                                                   glorot_uniform,
                                                                   glorot_uniform),
                 init_bias::Tuple{Function, Function, Function}=(zeros32, zeros32,
                                                                 zeros32),
                 init_state::Function=zeros32)
    return GRUCell(in_dims, out_dims, init_weight, init_bias, init_state)
end

function initialparameters(rng::AbstractRNG, gru::GRUCell)
    weight_i = vcat([init_weight(rng, gru.out_dims, gru.in_dims)
                     for init_weight in gru.init_weight]...)
    weight_h = vcat([init_weight(rng, gru.out_dims, gru.out_dims)
                     for init_weight in gru.init_weight]...)
    bias_i = gru.init_bias[1](rng, gru.out_dims, 1)
    bias_h = vcat([init_bias(rng, gru.out_dims, 1) for init_bias in gru.init_bias]...)
    return (weight_i=weight_i, weight_h=weight_h, bias_i=bias_i, bias_h=bias_h)
end

function initialstates(rng::AbstractRNG, ::GRUCell)
    # FIXME: Take PRNGs seriously
    randn(rng, 1)
    return (rng=replicate(rng),)
end

function (gru::GRUCell)(x::AbstractMatrix, ps::VALID_PARAMETER_TYPES,
                        st::NamedTuple)
    rng = replicate(st.rng)
    @set! st.rng = rng
    hidden_state = gru.init_state(rng, gru.out_dims, size(x, 2))
    return gru((x, hidden_state), ps, st)
end

function (gru::GRUCell)((x, hidden_state)::Tuple{<:AbstractMatrix, <:AbstractMatrix},
                        ps::VALID_PARAMETER_TYPES, st::NamedTuple)
    gxs = multigate(ps.weight_i * x, Val(3))
    ghbs = multigate(ps.weight_h * hidden_state .+ ps.bias_h, Val(3))

    r = @. sigmoid_fast(gxs[1] + ghbs[1])
    z = @. sigmoid_fast(gxs[2] + ghbs[2])
    n = @. tanh_fast(gxs[3] + ps.bias_i + r * ghbs[3])
    hidden_state_new = @. (1 - z) * n + z * hidden_state

    return hidden_state_new, st
end

Base.show(io::IO, g::GRUCell) = print(io, "GRUCell($(g.in_dims) => $(g.out_dims))")
