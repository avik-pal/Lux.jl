# # MNIST Classification using Neural ODEs

# ## Package Imports
using Lux
using Pkg #hide
Pkg.activate(joinpath(dirname(pathof(Lux)), "..", "examples")) #hide
using ComponentArrays, CUDA, DiffEqSensitivity, NNlib, Optimisers, OrdinaryDiffEq, Random,
      Statistics, Zygote
import MLDatasets: MNIST
import MLDataUtils: convertlabel, LabelEnc
import MLUtils: DataLoader, splitobs
CUDA.allowscalar(false)

# ## Loading MNIST
## Use MLDataUtils LabelEnc for natural onehot conversion
function onehot(labels_raw)
    convertlabel(LabelEnc.OneOfK, labels_raw, LabelEnc.NativeLabels(collect(0:9)))
end

function loadmnist(batchsize, train_split)
    ## Load MNIST: Only 1500 for demonstration purposes
    N = 1500
    imgs = MNIST.traintensor(1:N)
    labels_raw = MNIST.trainlabels(1:N)

    ## Process images into (H,W,C,BS) batches
    x_data = Float32.(reshape(imgs, size(imgs, 1), size(imgs, 2), 1, size(imgs, 3)))
    y_data = onehot(labels_raw)
    (x_train, y_train), (x_test, y_test) = splitobs((x_data, y_data); at=train_split)

    return (
            ## Use DataLoader to automatically minibatch and shuffle the data
            DataLoader(collect.((x_train, y_train)); batchsize=batchsize, shuffle=true),
            ## Don't shuffle the test data
            DataLoader(collect.((x_test, y_test)); batchsize=batchsize, shuffle=false))
end

# ## Define the Neural ODE Layer
#
# The NeuralODE is a ContainerLayer. It stores a `model` and the parameters and states of the NeuralODE is
# same as that of the underlying model.
struct NeuralODE{M <: Lux.AbstractExplicitLayer, So, Se, T, K} <:
       Lux.AbstractExplicitContainerLayer{(:model,)}
    model::M
    solver::So
    sensealg::Se
    tspan::T
    kwargs::K
end

function NeuralODE(model::Lux.AbstractExplicitLayer;
                   solver=Tsit5(),
                   sensealg=InterpolatingAdjoint(; autojacvec=ZygoteVJP()),
                   tspan=(0.0f0, 1.0f0),
                   kwargs...)
    return NeuralODE(model, solver, sensealg, tspan, kwargs)
end

function (n::NeuralODE)(x, ps, st)
    st_ = st.model
    function dudt(u, p, t)
        u_, st_ = n.model(u, p, st_)
        return u_
    end
    prob = ODEProblem{false}(ODEFunction{false}(dudt), x, n.tspan, ps.model)
    return solve(prob, n.solver; sensealg=n.sensealg, n.kwargs...), (model=st_,)
end

function diffeqsol_to_array(x::ODESolution{T, N, <:AbstractVector{<:CuArray}}) where {T, N}
    dropdims(gpu(x); dims=3)
end
diffeqsol_to_array(x::ODESolution) = dropdims(Array(x); dims=3)

# ## Create and Initialize the Neural ODE Layer
function create_model()
    ## Construct the Neural ODE Model
    model = Chain(FlattenLayer(),
                  Dense(784, 20, tanh),
                  NeuralODE(Chain(Dense(20, 10, tanh), Dense(10, 10, tanh),
                                  Dense(10, 20, tanh));
                            save_everystep=false,
                            reltol=1.0f-3,
                            abstol=1.0f-3,
                            save_start=false),
                  diffeqsol_to_array,
                  Dense(20, 10))

    rng = Random.default_rng()
    Random.seed!(rng, 0)

    ps, st = Lux.setup(rng, model)
    ps = ComponentArray(ps) |> gpu
    st = st |> gpu

    return model, ps, st
end

# ## Define Utility Functions
get_class(x) = argmax.(eachcol(x))

logitcrossentropy(ŷ, y) = mean(-sum(y .* logsoftmax(ŷ); dims=1))

function loss(x, y, model, ps, st)
    ŷ, st = model(x, ps, st)
    return logitcrossentropy(ŷ, y), st
end

function accuracy(model, ps, st, dataloader)
    total_correct, total = 0, 0
    st = Lux.testmode(st)
    iterator = CUDA.functional() ? CuIterator(dataloader) : dataloader
    for (x, y) in iterator
        target_class = get_class(cpu(y))
        predicted_class = get_class(cpu(model(x, ps, st)[1]))
        total_correct += sum(target_class .== predicted_class)
        total += length(target_class)
    end
    return total_correct / total
end

# ## Training
function train()
    model, ps, st = create_model()

    ## Training
    train_dataloader, test_dataloader = loadmnist(128, 0.9)

    opt = Optimisers.ADAM(0.001f0)
    st_opt = Optimisers.setup(opt, ps)

    ### Warmup the Model
    img, lab = gpu(train_dataloader.data[1][:, :, :, 1:1]),
               gpu(train_dataloader.data[2][:, 1:1])
    loss(img, lab, model, ps, st)
    (l, _), back = pullback(p -> loss(img, lab, model, p, st), ps)
    back((one(l), nothing))

    ### Lets train the model
    nepochs = 9
    for epoch in 1:nepochs
        stime = time()
        iterator = CUDA.functional() ? CuIterator(train_dataloader) : train_dataloader
        for (x, y) in iterator
            (l, _), back = pullback(p -> loss(x, y, model, p, st), ps)
            gs = back((one(l), nothing))[1]
            st_opt, ps = Optimisers.update(st_opt, ps, gs)
        end
        ttime = time() - stime

        println("[$epoch/$nepochs] \t Time $(round(ttime; digits=2))s \t Training Accuracy: " *
                "$(round(accuracy(model, ps, st, train_dataloader) * 100; digits=2))% \t " *
                "Test Accuracy: $(round(accuracy(model, ps, st, test_dataloader) * 100; digits=2))%")
    end
end

train()
