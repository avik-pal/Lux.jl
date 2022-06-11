# Adding New Functionality/Layers

For Style we try to follow [SciMLStyle](https://github.com/SciML/SciMLStyle). The only reason we don't have a badge yet, is we haven't yet updated the package to followed all the guidelines. Here, I am documenting some additional guidelines we enforce:

## Mutability

See [SciMLStyle](https://github.com/SciML/SciMLStyle#out-of-place-and-immutability-is-preferred-when-sufficient-performant) for reference. This is strictly enforced, i.e. all layers/functions provided as part of the external API must be pure functions, even if they come with a performance penalty.

## Branching -- Generated Functions

Zygote doesn't like branches in code. Like it or not, we are stuck with it for the near future. Even if julia is able to optimize branches away, Zygote will most certainly throw away those optimizations (these can be tested via `Zygote.@code_ir`).

### Writing efficient non-branching code to make Zygote happy

* Rely on `@generated` functions to remove **most** runtime branching. Certain examples:
  * Layers behaving differently during training and inference -- we know at compile-time whether a layer is being run in training/inference mode via `istraining(st)`.
  * Composite Layers relying on a variable number of internal layers -- Again we know the length of the number of internal layers at compile time. Hence we can manually unroll the loops. See [`Parallel`](@ref), [`Chain`](@ref), etc.
* Pass around `Val` in state. `Flux.jl` sets `training` to be `(:auto, true, false)`. Hence, which branch will be evaluated, will have to be determined at runtime time (*bad*). Instead if we pass `Val(true)`, we will be able to specialize functions directly based on `true`, `false`, etc. ensuring there is no runtime cost for these operations. See [`BatchNorm`](@ref), [`Dropout`](@ref), etc.
