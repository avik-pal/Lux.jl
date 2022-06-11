using MacroTools

"""
An Experimental API to make it easier to write Lux models.

## Example

```julia
using Lux, Random, Zygote, NNlib

Lux.@layerdef function MyCustomModel((x, y)::T) where {T}
    @assert size(x, 1) == 2 && size(y, 1) == 2
    a = Dense(3, 3, relu)(Dense(2, 3)(x) .+ Dense(2, 3, relu)(x)) .- Dense(2, 3, gelu)(x)
    b = Dense(3, 2, sigmoid)(a)
    c = Dense(2, 2, relu)(y)
    return b .+ c
end

model = MyCustomModel()

ps, st = Lux.setup(Random.default_rng(), model)
x, y = randn(Float32, 2, 1), randn(Float32, 2, 1)

model((x, y), ps, st)
```

## Verifying Generated Code

!!! tip
    Since this is an highly experimental feature, we recommend always manually verifying the generated model. This can be done simply using `@macroexpand Lux.@layerdef ...`

```julia-repl
julia> @macroexpand Lux.@layerdef function MyCustomModel((x, y)::T) where {T}
           @assert size(x, 1) == 2 && size(y, 1) == 2
           a = Dense(3, 3, relu)(Dense(2, 3)(x) .+ Dense(2, 3, relu)(x)) .- Dense(2, 3, gelu)(x)
           b = Dense(3, 2, sigmoid)(a)
           c = Dense(2, 2, relu)(y)
           return b .+ c
       end
quote
    struct MyCustomModel{__dense_1_type, __dense_2_type, __dense_3_type, __dense_4_type, __dense_5_type, __dense_6_type} <: Lux.AbstractExplicitContainerLayer{(:dense_1, :dense_2, :dense_3, :dense_4, :dense_5, :dense_6)}
        dense_1::__dense_1_type
        dense_2::__dense_2_type
        dense_3::__dense_3_type
        dense_4::__dense_4_type
        dense_5::__dense_5_type
        dense_6::__dense_6_type
    end
    function MyCustomModel(; )
        return MyCustomModel(Dense(3, 3, relu), Dense(2, 3), Dense(2, 3, relu), Dense(2, 3, gelu), Dense(3, 2, sigmoid), Dense(2, 2, relu))
    end
    function (mandrill::MyCustomModel)((x, y)::T, ps::Union{Lux.ComponentArray, NamedTuple}, st::NamedTuple; ) where T
        if size(x, 1) == 2 && size(y, 1) == 2
            nothing
        else
            Base.throw(Base.AssertionError("size(x, 1) == 2 && size(y, 1) == 2"))
        end
        (wildebeest, camel) = mandrill.dense_4(x, ps.dense_4, st.dense_4)
        (antelope, gaur) = mandrill.dense_3(x, ps.dense_3, st.dense_3)
        (rhinoceros, ibis) = mandrill.dense_2(x, ps.dense_2, st.dense_2)
        (mallard, jellyfish) = mandrill.dense_1(rhinoceros .+ antelope, ps.dense_1, st.dense_1)
        a = mallard .- wildebeest
        (hamster, monkey) = mandrill.dense_5(a, ps.dense_5, st.dense_5)
        b = hamster
        (wasp, sanddollar) = mandrill.dense_6(y, ps.dense_6, st.dense_6)
        c = wasp
        return (b .+ c, NamedTuple{(:dense_1, :dense_2, :dense_3, :dense_4, :dense_5, :dense_6)}((jellyfish, ibis, gaur, camel, monkey, sanddollar)))
    end
end
```

## What not to do while writing `layerdef`ed layers?

Here are some common cases where `@layerdef` will silently fail to generate the correct code. (Encountered some additional cases, open an issue so that we can solve it or expand this list)

1. **Imcomplete Layer Calls**: All layer calls must be complete, i.e. the following is not allowed:

```julia
Lux.@layerdef function ThisModelDoesNotWork(x)
    @assert size(x, 1) == 2 && size(y, 1) == 2
    shared = Dense(2, 3, gelu)
    a = Dense(3, 3, relu)(shared(x) .+ Dense(2, 3, relu)(x)) .- shared(x)
    b = Dense(3, 2, sigmoid)(a)
    return b
end
```

Instead write it as:

```julia
Lux.@layerdef function ThisModelWorks(x)
    @assert size(x, 1) == 2 && size(y, 1) == 2
    shared = Dense(2, 3, gelu)(x)
    a = Dense(3, 3, relu)(shared .+ Dense(2, 3, relu)(x)) .- shared
    b = Dense(3, 2, sigmoid)(a)
    return b
end
```

2. **Creating layers without `=`**: All layer definitions must be preceeded by `=`. Anything else be it `return`, `.+=`. `.=`, etc. is not allowed.

```julia
# Invalid
Lux.@layerdef function WrongReturn(x)
    return Dense(2, 2)(x) .+ Dense(2, 2)(x)
end

Lux.@layerdef WrongOneLiner(x) = Dense(2, 2)(x) .+ Dense(2, 2)(x)

# Valid
Lux.@layerdef function CorrectReturn(x)
    x = Dense(2, 2)(x) .+ Dense(2, 2)(x)
    return x
end
```

## Deviations from `flax.linen.compact`

This is heavily inspired by [flax.linen.compact](https://flax.readthedocs.io/en/latest/design_notes/module_lifecycle.html?highlight=compact#compact-modules), though the way this works is entirely different from flax.

* Unlike Flax, in Lux.@layerdef we still need to use the complete API to define the model. We don't perform automatic shape inference at this point. (This is a planned feature for future releases).
* We don't allow specifying names for each layer.
* Conditionals are not an issue

## Shared Parameters / Layer Definition API

Use `←` (type `\\leftarrow` followed by a tab) to declare a new layer. For Example:

```julia
Lux.@layerdef function MySharedLayers((x, y))
    s1 ← Dense(2, 2)
    s2 ← Dense(2, 2)
    x = s1(x) .+ s2(x) .- s1(y)
    return x
end
```

## Known Issues & Workarounds

* If definining a `@layerdef` containing only 1 layer, there is a syntax error `ERROR: syntax: function argument and static parameter names must be distinct around <line>`. To workaround this, use `@macroexpand` and copy-paste the generated code

!!! warning
    This is an experimental API. It is likely to have bugs and untested edge cases. If you find the generated code to be incorrect, please file an issue.
"""
macro layerdef(expr)
    return esc(layerdef(__module__, __source__, expr))
end

function layerdef(mod, linenumbernode, expr)
    # Verify expr is a function
    @assert expr.head==:function AssertionError("`Lux.@layerdef` should be applied to a function")

    # Split the function call
    modeldef = splitdef(expr)
    layer_name, args, arg_syms, body = check_valid_layerdef_expr(modeldef)

    # Setup
    layer_declaration_variables = Dict()
    layers = Dict()
    layer_counters = Dict()
    layer_state_variables = Dict()
    first_encounters = Dict()

    # Descend into the expressions
    lname = gensym("layer")
    rewritten_body = __rewrite_body_expression(mod, lname, body,
                                               (layer_declaration_variables=layer_declaration_variables,
                                                layers=layers,
                                                layer_counters=layer_counters,
                                                layer_state_variables=layer_state_variables,
                                                first_encounters=first_encounters))

    which_layers = Tuple(sort(collect(keys(layers))))
    tnames = [Symbol("__$(w)_type") for w in which_layers]
    struct_fields = [:($w::$t) for (w, t) in zip(which_layers, tnames)]

    structdef = :(struct $layer_name{$(tnames...)} <:
                         Lux.AbstractExplicitContainerLayer{$(which_layers)}
                      $(struct_fields...)
                  end)

    layerdefs = [layers[w] for w in which_layers]

    structconstructor = combinedef(Dict(:name => layer_name, :args => [], :kwargs => [],
                                        :whereparams => [],
                                        :body => :(return $layer_name($(layerdefs...)))))

    fcall = combinedef(Dict(:name => :($lname::$layer_name), :body => rewritten_body,
                            :kwargs => [],
                            :args => append!(modeldef[:args],
                                             [
                                                 :(ps::Union{Lux.ComponentArray,
                                                             NamedTuple}),
                                                 :(st::NamedTuple),
                                             ]),
                            :whereparams => modeldef[:whereparams]))

    return prettify(Expr(:block, structdef, structconstructor, fcall))
end

function check_valid_layerdef_expr(modeldef)
    if !haskey(modeldef, :name)
        throw(ArgumentError("Anonymous Functions without name are not supported"))
    end
    layer_name = modeldef[:name]

    args = modeldef[:args]
    if length(args) != 1
        throw(ArgumentError("Model must take exactly 1 input!!!"))
    elseif :ps in args || :st in args
        throw(ArgumentError("Model must not take `ps` or `st` as inputs!!!"))
    end
    # Extract the names of the arguments.
    args_syms = map(args) do arg
        MacroTools.@match arg begin
            (name_::_) => name
            x_ => x
        end
    end

    kwargs = modeldef[:kwargs]
    if length(kwargs) != 0
        throw(ArgumentError("Model cannot take kwargs!!!"))
    end

    body = modeldef[:body]
    if all(l -> isa(l, LineNumberNode), body.args)
        @warn("Model definition seems empty, still continue.")
    end

    return (layer_name, args, args_syms, body)
end

# Not `Expr` then just return `x`
__rewrite_body_expression(mod, layer_name, x, storage) = x

function __rewrite_body_expression(mod, layer_name, expr::Expr, storage)
    # Do not touch interpolated expressions
    expr.head === :$ && return expr.args[1]

    # Do we don't want escaped expressions because we unfortunately escape the entire body afterwards.
    if Meta.isexpr(expr, :escape)
        return __rewrite_body_expression(mod, layer_name, expr.args[1], storage)
    end

    # If it's a macro, we expand it
    if Meta.isexpr(expr, :macrocall)
        return __rewrite_body_expression(mod, layer_name,
                                         macroexpand(mod, expr; recursive=true),
                                         storage)
    end

    if expr.head == :call && first(expr.args) == :←
        # Direct Layer Definition
        if length(expr.args) != 3
            throw(Meta.ParseError("Expected Syntax L ← AbstractExplicitLayer or ←(L, AbstractExplicitLayer)"))
        end
        varname = expr.args[2]
        if varname in keys(storage.layer_declaration_variables)
            throw(Meta.ParseError("$varname -- variable already in use for another layer definition"))
        end
        if !__check_is_abstractexplicitlayer(expr.args[3])
            throw(Meta.ParseError("$(expr.args[3]) is not a valid Definition for an AbstractExplicitLayer"))
        end
        lname, updated_expr = __get_layer_name(expr.args[3], storage.layer_counters,
                                               storage.layers,
                                               storage.layer_state_variables, layer_name)
        storage.layer_declaration_variables[varname] = lname

        return :()
    end

    if expr.head == :(=)
        rewritten_expr, _, lifted_exprs = __update_layer_expressions(mod, layer_name,
                                                                     expr.args[2],
                                                                     expr.args[1],
                                                                     storage)
        expr.args[2] = rewritten_expr
        return Expr(:block, lifted_exprs..., expr)
    end

    if expr.head == :return
        field_names = Tuple(sort(collect(keys(storage.layer_state_variables))))
        state_vars = [storage.layer_state_variables[n] for n in field_names]
        state_expr = :(NamedTuple{$field_names}(tuple($(state_vars...))))
        return Expr(:return, :($(expr.args...), $state_expr))
    end

    # If no special case, recurse into the arguments
    return Expr(expr.head,
                map(x -> __rewrite_body_expression(mod, layer_name, x, storage),
                    expr.args)...)
end

function __check_is_abstractexplicitlayer(expr::Expr)
    try
        if expr.head == :call && expr.args[1] isa Symbol &&
           eval(expr.args[1]) <: AbstractExplicitLayer
            return true
        end
        return false
    catch
        return false
    end
end

function __get_layer_name(expr::Expr, counters::Dict, layers::Dict,
                          layer_state_variables::Dict, model_name::Symbol)
    layer_name = lowercase(string(expr.args[1]))
    count = get(counters, Symbol(layer_name), 0)
    counters[Symbol(layer_name)] = count + 1
    layer_name = layer_name * "_" * string(count + 1)
    layer_state = layer_name * "___state"
    layer_state_variables[Symbol(layer_name)] = gensym(layer_state)
    layers[Symbol(layer_name)] = expr
    return Symbol(layer_name), :($model_name.$(Symbol(layer_name)))
end

# Fallback
__update_layer_expressions(mod, layer_name, x, store_in, storage) = (x, nothing, [])

function __update_layer_expressions(mod, layer_name, x::Symbol, store_in, storage)
    if x in keys(storage.layer_declaration_variables)
        lname = storage.layer_declaration_variables[x]
        lifted_exprs = if x in keys(storage.first_encounters)
            []
        else
            storage.first_encounters[x] = true
            [:($(storage.layer_state_variables[lname]) = st.$lname)]
        end
        return (:($layer_name.$lname), lname, lifted_exprs)
    end
    return (x, nothing, [])
end

function __update_layer_expressions(mod, layer_name, expr::Expr, store_in, storage)
    # Try to see if the `expr` is a layer definition
    try
        if expr.head === :call && expr.args[1] isa Symbol
            if eval(expr.args[1]) <: AbstractExplicitLayer
                lname, updated_expr = __get_layer_name(expr, storage.layer_counters,
                                                       storage.layers,
                                                       storage.layer_state_variables,
                                                       layer_name)
                return updated_expr, lname,
                       [:($(storage.layer_state_variables[lname]) = st.$lname)]
            end
        end
    catch e
        if e isa Meta.ParseError
            rethrow(e)
        end
    end
    expr_calls, lifted_exprs, i = [], [], 1
    while i <= length(expr.args)
        arg = expr.args[i]
        updated_expr, lname, lifted_priors = __update_layer_expressions(mod, layer_name,
                                                                        arg, store_in,
                                                                        storage)
        lifted_exprs = append!(lifted_priors, lifted_exprs)
        if lname !== nothing
            store_in_var = gensym()

            func_args = []
            for arg in expr.args[(i + 1):end]
                arg_expr, _, lifted_priors = __update_layer_expressions(mod, layer_name,
                                                                        arg, store_in_var,
                                                                        storage)
                lifted_exprs = append!(lifted_priors, lifted_exprs)
                push!(func_args, arg_expr)
            end

            if length(func_args) > 1
                push!(lifted_exprs,
                      :(($(store_in_var),
                        $(storage.layer_state_variables[lname])) = $updated_expr(tuple($(func_args...)),
                                                                                 ps.$lname,
                                                                                 $(storage.layer_state_variables[lname]))))
            else
                push!(lifted_exprs,
                      :(($(store_in_var),
                        $(storage.layer_state_variables[lname])) = $updated_expr($(func_args[1]),
                                                                                 ps.$lname,
                                                                                 $(storage.layer_state_variables[lname]))))
            end
            push!(expr_calls, :__do_collapse__)
            push!(expr_calls, :($store_in_var))
            break
        else
            push!(expr_calls, updated_expr)
        end
        i += 1
    end
    if :__do_collapse__ in expr_calls
        return expr_calls[2], nothing, lifted_exprs
    else
        return Expr(expr.head, expr_calls...), nothing, lifted_exprs
    end
end
