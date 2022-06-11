# v0.5

## v0.5.0

* **{BREAKING CHANGE]**: Even if an AbstractExplicitContainerLayer has a single layer inside it, the returned parameters and state needs to be indexed with it. For example, `Chain` used to be indexed with `layer_1`, `layer_2`, etc. From now it will have to be indexed as `layers.layer_*`.
* Introduces an experimental macro `Lux.@compact` for simpler custom model definitions

# v0.4

## v0.4.4

* Updated to support julia v1.6 (test time dependency issues)

## v0.4.3

* Extending Scale to allow for multiple dimension inputs (https://github.com/avik-pal/Lux.jl/pull/40)

## v0.4.2

* `SelectDim` is no longer type unstable -- Internal storage for the Layer has been changed
* `Dropout` & `VariationalDropout` return `NoOpLayer` if the probability of dropout is `0`
* Code Formatting -- SciMLStyle (https://github.com/avik-pal/Lux.jl/pull/31)

## v0.4.1

* Fix math rendering in docs
* Add Setfield compat for v1.0