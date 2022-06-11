using CUDA, Lux, Random, Test

include("utils.jl")

@testset "Elementwise Operation Dispatches" begin
    rng = Random.default_rng()
    Random.seed!(rng, 0)
    custom_activation(x) = abs(x)

    for T in [Float64, Float32, ComplexF64, ComplexF32]
        x = randn(rng, T, 10, 5, 2)
        y = randn(rng, T, 10, 1, 2)

        # On CPU the fallback should always work
        @test Lux.elementwise_add(x, y) == x .+ y
        @test Lux.elementwise_mul(x, y) == x .* y
        @test Lux.applyactivation(tanh, x) == tanh.(x)
        @test Lux.applyactivation(custom_activation, x) == custom_activation.(x)

        if T <: Real
            # Gradient for complex outputs are not defined
            test_gradient_correctness_fdm(sum ∘ Lux.elementwise_add, x, y)
        end

        # On GPU try to use CUDNN
        if CUDA.functional()
            x_g = x |> gpu
            y_g = y |> gpu

            @test Lux.elementwise_add(x_g, y_g) == x_g .+ y_g
            @test Lux.elementwise_mul(x_g, y_g) == x_g .* y_g
            @test Lux.applyactivation(tanh, x_g) == tanh.(x_g)
            # Custom Activation test
            @test Lux.applyactivation(custom_activation, x_g) == custom_activation.(x_g)
        end
    end
end
