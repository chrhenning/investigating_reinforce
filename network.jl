#!/usr/bin/env julia
# Copyright 2017 Christian Henning
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

using Distributions

include("./lib/Lists/src/Lists.jl")
using Lists

"""
    sigmoid(z::Float64)

Compute the sigmoid.
"""
function sigmoid(z::Float64)
    return 1.0 / (1.0 + exp(-z))
end

"""
The dataset considered for training and testing. It is always expected, that
the last dimension defines the batch size (cmp.
[MLDatasets.jl](https://github.com/JuliaML/MLDatasets.jl)). If no targets are
defined, then a self-supervised training is expected.
Inputs should be values between 0 and .
"""
type Dataset
    "Training input."
    train_in::Array{Float64, 2}
    "Training output/target."
    train_out::Any
    "Test input."
    test_in::Array{Float64, 2}
    "Test output/target."
    test_out::Any

    """
        Dataset(train_in::Array{Float64, 2}, test_in::Array{Float64, 2};
                train_out::Any = nothing, test_out::Any = nothing)

    Set fields of this type.
    """
    function Dataset(train_in::Array{Float64, 2}, test_in::Array{Float64, 2};
                     train_out::Any = nothing, test_out::Any = nothing)
        x = new()
        x.train_in = train_in
        x.train_out = train_out
        x.test_in = test_in
        x.test_out = test_out
        x
    end
end

"Type containing the hyperparameters of the learning process."
type Hyperparameters
    "Learning Rate."
    λ::Float64
    "Gamma parameter for exponential averaging."
    γ::Float64
    "Loss trade-off for sparsity regularizer."
    αₛ::Float64
    "Loss trade-off for participation regularizer."
    αₚ::Float64
    "Reward baseline."
    rᵦ::Float64
    "Momentum Learning: Influence of recent weight update."
    η::Float64
    """
    Number of epochs for training the network. Each epoch, a mini-batch will be
    presented to the network. Afterwards, the network weights will be updated.
    """
    num_epochs::Int
    "Batch size. Number of samples to process per epoch."
    batch_size::Int
    "Sample size. How often a single batch shall be presented during an epoch"
    sample_size::Int
    "Whether or not gradient clipping should be applied."
    clip_gradients::Bool
    """
    The value to which large gradient values should be clipped. This value
    applies to positive and negative gradients.
    """
    clipping_value::Float64
    "One can enable exponential rate decay."
    rate_decay::Bool
    "After how many epochs the learning rate should be halfed."
    rate_decay_interval::Int

    """
        Hyperparameters(;λ::Float64 = 1.0, γ::Float64 =0.99,
                        αₛ::Float64 = 0.0, αₚ::Float64 = 0.0
                        rᵦ::Float64 = 0.0, η::Float64 = 0.0,
                        num_epochs::Int = 1000, batch_size::Int = 1,
                        sample_size::Int = 1, clip_gradients::Bool = true,
                        clipping_value::Float64 = 1.0, rate_decay::Bool = true,
                        rate_decay_interval::Int = 1000)

    Set default values of type fields.
    """
    function Hyperparameters(;λ::Float64 = 1.0, γ::Float64 =0.99,
                             αₛ::Float64 = 0.0, αₚ::Float64 = 0.0,
                             rᵦ::Float64 = 0.0, η::Float64 = 0.0,
                             num_epochs::Int = 1000, batch_size::Int = 1,
                             sample_size::Int = 1, clip_gradients::Bool = true,
                             clipping_value::Float64 = 1.0,
                             rate_decay::Bool = true,
                             rate_decay_interval::Int = 1000)

        new(λ, γ, αₛ, αₚ, rᵦ, η, num_epochs, batch_size, sample_size,
            clip_gradients, clipping_value, rate_decay, rate_decay_interval)
    end
end

"""
Parameters of loss terms (including regularizers) that may differ within
layers.
"""
type LossParameters
    "Sparsity constraint."
    sparsity::Float64

    """
        LossParameters(;sparsity::Float64 = 0.5)

    Set default values of type fields.
    """
    function LossParameters(;sparsity::Float64 = 0.5)
        new(sparsity)
    end
end

"Struct that represents a single layer."
type Layer
    "The name of the layer."
    name::String
    "Size of layer (number of neurons)."
    size::Int
    "Function handle to forward function (computes forward pass)."
    forward_fcn::Function
    "Function handle to update function (computes weight update)."
    update_fcn::Function
    "Weights of layer."
    W::Array{Float64,2}
    "Recent update history. Used for momentum learning."
    ΔW::Array{Float64,2}
    "Struct of loss parameters defined for this layer."
    loss_params::LossParameters
    """
    Some loss terms might not only depend on the current forward computation
    but also on a state, that needs to be maintained. This is typically the
    case for a participation regularizer, that enforces a cell to be active
    every once in a while.
    """
    loss_memory::Dict{String,Any}
    "The regularizers added to this layer."
    regularizers::List{Function}

    """
        Layer(name::String, size::Int;
              loss_params::LossParameters = LossParameters())

    Set default values of type fields.
    """
    function Layer(name::String, size::Int;
                   loss_params::LossParameters = LossParameters())
        x = new()
        x.name = name
        x.size = size
        x.loss_params = loss_params
        x.loss_memory = Dict{String,Any}()
        x.regularizers = List{Function}()
        x
    end
end

"""
A `Network` struct shall contain the full architecture and state of a network.
"""
type Network
    "List of layers."
    layer::List{Layer}
    "Hyperparameters of the network."
    hparams::Hyperparameters
    """
    The global reward function used to compare output with target.
    Important, the actual reward is computed from this together with the
    specified regularizers.
    """
    global_reward_fcn::Function
    """
        Network(;hparams::Hyperparameters = Hyperparameters,
                global_reward_fcn::Function = mse_reward)

    Set default values of type fields.
    """
    function Network(;hparams::Hyperparameters = Hyperparameters(),
                     global_reward_fcn::Function = mse_reward)
        x = new()
        x.hparams = hparams
        x.layer = List{Layer}()
        x.global_reward_fcn = global_reward_fcn
        x
    end
end

"""
    mse_reward{T,S}(output::AbstractArray{T, 2},
                        target::AbstractArray{S, 2})

Compute the negative mean-squared error between `output` and `target`.

```math
- \\frac{1}{|output|} \\sum_i (output_i - target_i)^2
```
"""
function mse_reward{T,S}(output::AbstractArray{T, 2},
                         target::AbstractArray{S, 2})
    squeeze(-sum((output .- target).^2, 1) / size(output, 1), 1)
end

"""
    regularizer(x::Array{Float64, 2}, y::Array{Float64, 2},
                p::Array{Float64, 2}, layer::Layer, hparams::Hyperparameters)

This is the interface of a typical regularizer that may be implemented within
this framework. We assume, that all interesting regularizers can be expressed
on a layer-wise level.

# Arguments
- `x::Array{Float64, 2}`: the presynaptic output.
- `y::Array{Float64, 2}`: the postsynaptic output (current layer).
- `p::Array{Float64, 2}`: the postsynaptic p-values.
- `layer:Layer`: the postsynaptic layer.
- `hparams::Hyperparameters`: the training hyperparameters.
"""
function regularizer(x::AbstractArray{Bool, 2}, y::AbstractArray{Bool, 2},
                     p::Array{Float64, 2}, layer::Layer,
                     hparams::Hyperparameters)
    error("Not callable! Only abstract function interface.");
end

"""
    sparsity_regularizer(::Any, y::AbstractArray{Bool, 2}, ::Any, layer::Layer,
                         hparams::Hyperparameters)

A sparsity regularizer, that enforces the sparsity level defined in the
`LossParLossParameters` in this layer.

Across a batch the mean loss is returned.

```math
    - αₛ ((1-sparsity) - \\frac{1}{|y|} \\sum_i y_i)^2
```
"""
function sparsity_regularizer(::Any, y::AbstractArray{Bool, 2},
                              ::Any, layer::Layer, hparams::Hyperparameters)
    αₛ = hparams.αₛ
    part = 1.0 - layer.loss_params.sparsity
    # Note, that julia autmatically converts divisions into floats.
    squeeze(-αₛ * (part .- sum(y, 1) ./ size(y, 1)).^2, 1)
end

"""
    participation_regularizer(::Any, y::AbstractArray{Bool, 2}, ::Any,
                              layer::Layer, hparams::Hyperparameters)

A participation regularizer, that enforces neurons to participate on the
computation every once in a while (level of participation is defined by
``1-sparsity``). Therefore, this regularizer has to maintain a state of past
actactivation, which is kept as an exponentially smoothed trace.

```math
 - αₚ \\sum_i ((1-sparsity) - y_{trace,i})^2
```
"""
function participation_regularizer(::Any, y::AbstractArray{Bool, 2}, ::Any,
                                   layer::Layer, hparams::Hyperparameters)
    γ = hparams.γ
    αₚ = hparams.αₚ
    part = 1.0 - layer.loss_params.sparsity

    # Initialize trace, if not existing.
    if get(layer.loss_memory, "y_trace", -1) == -1
        layer.loss_memory["y_trace"] = part .* ones(size(y, 1))
    end
    y_trace = layer.loss_memory["y_trace"]

    y_trace_batch = γ .* y_trace .+ (1-γ) .* y
    # Update trace.
    y_trace .= squeeze(mean(y_trace_batch, 2), 2)

    squeeze(-αₚ * sum((part .- y_trace_batch).^2, 1) ./ length(y_trace), 1)
end

"""
    fully_connected_forward{T}(x::AbstractArray{T, 2},
                               W::Array{Float64, 2})

Computes the forward propagation of an input `x`  through a fully-connected
layer in a binomial network.

# Returns
Returns the p-values ``p = σ(W x)`` of the layer and the layer activation
``y = Bin(p)`` as an tuple:

    p::Array{Float64, 2}, y::BitArray{2}
"""
function fully_connected_forward{T}(x::AbstractArray{T, 2},
                                    W::Array{Float64, 2})
    p = sigmoid.(W * x)
    y = rand(size(p)) .< p

    p, y
end

"""
    fully_connected_update(x::Array{Float64, 2}, y::Array{Float64, 2},
                           p::Array{Float64, 2}, r::Array{Float64, 1},
                           hparams::Hyperparameters)

Compute the weight update `ΔW` of the current, fully-connected layer.

```math
ΔW = (r - rᵦ) * (y - p) * x^T
```

This method returns a 3-rank tensor ΔW, where the last dimension specifies the
sample from the batch that defines the weight update.
"""
function fully_connected_update(x::Array{Float64, 2}, y::Array{Float64, 2},
                                p::Array{Float64, 2}, r::Array{Float64, 1},
                                hparams::Hyperparameters)
    rᵦ = hparams.rᵦ
    ΔW = zeros(size(y, 1), size(x, 1), size(y, 2))
    for b = 1:size(y, 2)
        ΔW[:,:,b] = (r[b] - rᵦ) .* (y[:,b] .- p[:,b]) * x[:,b]'
    end

    ΔW
end

"""
    fully_connected_layer(name::String, in_size::Int, out_size::Int;
                          loss_params::LossParameters = LossParameters(),
                          initializer::Function = rand)

Create a fully-connected layer. The method returns a `layer` type.

# Arguments
- `name::String`: the name of the layer.
- `in_layer`: input layer to this one.
- `out_size`: output size.
- `loss_params`: loss parameters that shall apply to this layer.
- `initializer`: a function handle that is used to initialize the layer
    weights.

# Examples

```jldoctest
julia> init(dims...) = 0.1 * rand(dims)
init (generic function with 1 method)

julia> layer = fully_connected_layer("fc1", 784, 100, initializer=init)
Layer("fc1", ...
```
"""
function fully_connected_layer(name::String, in_layer::Layer, out_size::Int;
                               loss_params::LossParameters = LossParameters(),
                               initializer::Function = rand)
    layer = Layer(name, out_size, loss_params = loss_params)
    layer.W = initializer(out_size, in_layer.size)
    layer.ΔW = zeros(out_size, in_layer.size)
    forward_fcn(x) = fully_connected_forward(x, layer.W)
    layer.forward_fcn = forward_fcn
    layer.update_fcn = fully_connected_update

    layer
end

"""
    add_layer(network::Network, layer::Layer)

Add `layer` to the current end of the `network`.
"""
function add_layer(network::Network, layer::Layer)
    push!(network.layer, layer)
end

"""
    add_regularizer(layer::Layer, reg_fun::Function)

Add a regularizer function to a given layer. The `reg_fun` must have the same
interface as the function `regularizer(...)`.
"""
function add_regularizer(layer::Layer, reg_fun::Function)
    push!(layer.regularizers, reg_fun)
end

"""
    input_layer(name::String, size::Int)

Create an input layer. Every network has to start with an input layer. The
forward function of this layer is simply the identity function. An update
function does not exist for such a layer.
"""
function input_layer(name::String, size::Int)
    layer = Layer(name, size)
    layer.forward_fcn = identity
    update_fcn(args...) = error("Input layer has no weights to update")
    layer.update_fcn = update_fcn

    layer
end

"""
    train_network(network::Network, train_in::Array{Float64, 2};
                  train_out::Any = nothing, eval_interval::Int = 100,
                  test_in::Any = nothing, test_out::Any = nothing)

Train a `network` with the given training set. If no training output
`train_out` is provided, then self-supervision is assumed.

If you want the test accuracy to be printed, then please provide a test set.

# Returns

It returns a tuple containing two arrays. The first one maintains the training
loss development (one value each epoch). The second array contains the
evaluation of the test set (e.g., the test accuracy).
"""
function train_network(network::Network, train_in::Array{Float64, 2};
                       train_out::Any = nothing, eval_interval::Int = 100,
                       eval_accuracy::Bool = true, test_in::Any = nothing,
                       test_out::Any = nothing)
    num_samples = size(train_in, ndims(train_in))
    batch_indices = randperm(num_samples)
    batch_size = network.hparams.batch_size

    # Keep History of train loss and test evaluation values for later
    # assessment of the training progress.
    train_loss_dev = zeros(network.hparams.num_epochs)
    test_eval_dev = []

    hparams = network.hparams

    for epoch = 1:network.hparams.num_epochs
        # Apply exponential rate decay.
        if hparams.rate_decay && epoch % hparams.rate_decay_interval == 0
            hparams.λ /= 2
            @printf("Epoch %d, learning rate decayed to: %f\n",
                    epoch, hparams.λ)
        end

        # Determine current batch.
        bs_start = ((epoch-1) * batch_size) % num_samples + 1
        bs_end = bs_start + batch_size -1

        if bs_end > num_samples
            tmp_batch_inds = batch_indices[bs_start:num_samples]
            batch_indices = randperm(num_samples)
            batch_inds = vcat(tmp_batch_inds,
                              batch_indices[1:bs_end-num_samples])
        else
            batch_inds = batch_indices[bs_start:bs_end]
        end
        batch_in = train_in[:, batch_inds]
        if train_out == nothing
            batch_out = batch_in
        else
            batch_out = train_out[:, batch_inds]
        end

        # Process batch and update weights.
        train_loss = train_step(network, batch_in, batch_out)
        train_loss_dev[epoch] = train_loss

        # Report progress.
        if epoch % eval_interval == 0
            if test_in == nothing
                @printf("Epoch %d, loss of current train batch: %f\n",
                        epoch, train_loss)
            else
                if test_out == nothing
                    test_out = test_in
                end

                if eval_accuracy
                    test_acc = accuracy(network, test_in, test_out)
                    @printf("Epoch %d, current test accuracy: %f\n", epoch,
                            test_acc)
                    push!(test_eval_dev, test_acc)
                else
                    test_pred = predict(network, test_in)
                    test_loss = -mean(network.global_reward_fcn(test_pred,
                                                                test_out))
                    @printf("Epoch %d, global test loss: %f\n", epoch,
                            test_loss)
                    push!(test_eval_dev, test_loss)
                end
            end
        end
    end

    train_loss_dev, test_eval_dev
end

"""
    one_hot(labels::Array{Int, 1}; num_labels::Int = -1)

Encode a given vector of `labels` into an one hot array.

# Return
An array of type `Array{Bool, 2}`, where the last dimension has the same size
as `labels`.
"""
function one_hot(labels::Array{Int, 1}; num_labels::Int = -1)
    num_labels = (num_labels == -1 ? maximum(labels) : num_labels)
    encoded = zeros(Bool, num_labels, length(labels))
    for i = 1:length(labels)
        encoded[labels[i], i] = 1
    end
    encoded
end

"""
    train_step{T}(network::Network, batch_in::Array{Float64, 2},
                  batch_out::Array{T, 2})

Train the network according to the current batch. This includes the sampling of
the batch (i.e., a sufficient amount of forward sweeps (acc. to the
hyperparamers) until we have good estimates of the network activity.). The
weight update is the average update ``ΔW`` that would have been applied to the
each sample in the batch.
"""
function train_step{T}(network::Network, batch_in::Array{Float64, 2},
                       batch_out::Array{T, 2})
    layers = network.layer

    hparams = network.hparams
    sample_size = hparams.sample_size

    # During a forward sweep, we need to memorize the layer activities, as we
    # cannot compute ΔW without knowing the final reward.
    X = Array{Float64, 2}[]
    P = Array{Float64, 2}[]
    Y = Array{Float64, 2}[]
    # Reward is global (one number per sample in batch).
    R = zeros(size(batch_in, 2))

    # For each layer, keep a list of update matrices (one for each sample in
    # the batch).
    ΔW_batch = Array{Float64, 3}[]

    x = nothing
    p = nothing
    y = nothing

    # Sweep through the network `sample_size` times to get an average weight
    # update ΔW for each sample in the batch.
    for ss in 1:sample_size
        r = zeros(size(batch_in, 2))

        # Compute the activities an the final reward.
        for (lind, curr_layer) in enumerate(network.layer)
            # First layer is always input layer.
            if lind == 1
                y = curr_layer.forward_fcn(batch_in)
                continue
            end

            x = copy(y)

            p, y = curr_layer.forward_fcn(x)

            for reg_fcn in curr_layer.regularizers
                r .+= reg_fcn(x, y, p, curr_layer, hparams)
            end

            if ss == 1
                push!(X, x)
                push!(P, p)
                push!(Y, y)
            else
                X[lind-1] = x
                Y[lind-1] = y
                P[lind-1] = p
            end
        end
        r .+= network.global_reward_fcn(y, batch_out)
        R .+= r

        # Compute ΔW for each layer.
        for (lind, curr_layer) in enumerate(network.layer)
            # Input layers don't have weights.
            if lind == 1
                continue
            end

            x = X[lind-1]
            p = P[lind-1]
            y = Y[lind-1]

            ΔW = curr_layer.update_fcn(x, y, p, r, hparams)

            if ss == 1
                push!(ΔW_batch, ΔW)
            else
                ΔW_batch[lind-1] += ΔW
            end
        end
    end

    ΔW_batch /= sample_size

    ### FIXME dirty insertion
    for (lind, curr_layer) in enumerate(network.layer)
        # Input layers don't have weights.
        if lind == 1
            continue
        end

        y_trace = curr_layer.loss_memory["y_trace"]
        norm_unit = (y_trace .* (1 .- y_trace)) .< 1e-2
        norm_updates = norm_unit .* (0.5 .- y_trace)
        ΔW_batch[lind-1] .+= 1 .* norm_updates

        if lind == 4 && sum(norm_updates) > 0
            println("layer $(lind)")
            println(y_trace)
            println(norm_updates)
        end
    end
    ###

    ΔW_batch *= hparams.λ
    R /= sample_size

    # Go once through the network and update all weights, given the samples
    # weight update.
    for (lind, curr_layer) in enumerate(network.layer)
        # Input layers don't have weights.
        if lind == 1
            continue
        end

        # Use the mean update matrix ΔW across a batch.
        ΔW = squeeze(mean(ΔW_batch[lind-1], 3), 3)

        # If we are using momentum learning.
        if hparams.η > 0
            curr_layer.ΔW = hparams.η * curr_layer.ΔW + ΔW
            ΔW = curr_layer.ΔW
        end

        # Clip gradients.
        if hparams.clip_gradients
            ΔW[ΔW .> hparams.clipping_value] = hparams.clipping_value
            ΔW[ΔW .< -hparams.clipping_value] = -hparams.clipping_value

            if hparams.η > 0
                curr_layer.ΔW .= ΔW
            end
        end

        curr_layer.W .+= ΔW
    end

    train_loss = -mean(R)
end

"""
    accuracy{T}(network::Network, batch_in::Array{Float64, 2},
                batch_out::Array{T, 2})

This method calls the `predict` method to process the given input batch in the
network and subsequently compute an accuracy based on the output vector and the
one-hot encoded given output batch.
"""
function accuracy{T}(network::Network, batch_in::Array{Float64, 2},
                     batch_out::Array{T, 2})
    Y = predict(network, batch_in)

    correct_ind = mapslices(indmax, batch_out, 1) # argmax function
    predicted_ind = mapslices(indmax, Y, 1)

    sum(correct_ind .== predicted_ind) / length(correct_ind)
end

"""
    predict(network::Network, batch_in::Array{Float64, 2})

Process a batch with the given network and return the output activity for each
sample in the batch.
"""
function predict(network::Network, batch_in::Array{Float64, 2})
    layers = network.layer

    hparams = network.hparams
    sample_size = hparams.sample_size

    out_size = network.layer[end].size

    # We need the average output activity.
    Y = zeros(out_size, size(batch_in, 2))

    x = nothing
    p = nothing
    y = nothing

    for ss in 1:sample_size
        for (lind, curr_layer) in enumerate(network.layer)
            # First layer is always input layer.
            if lind == 1
                y = curr_layer.forward_fcn(batch_in)
                continue
            end

            x = copy(y)

            p, y = curr_layer.forward_fcn(x)
        end
        Y .+= y
    end

    Y /= sample_size
end

"""
    xavier_initializer(n_in::Int, n_out::Int)

This method returns a function that can be used as a weight initializer for a
layer. The weights sampled by this function follow the Xavier initialization
for a layer with `n_in` input neurons and `n_out` output neurons.

The distribution is a normal distribution with mean ``μ = 0`` and varianve
``σ = \\sqrt{\\frac{2}{n_{in} + n_{out}}}``.
"""
function xavier_initializer(n_in::Int, n_out::Int)
    #dist = Normal(0, 1/n_in) # As implemented in Caffe.
    dist = Normal(0, sqrt(2/(n_in+n_out))) # As implemented in Tensorflow.

    init(dims...) = rand(dist, dims...)
end

"""
    neural_bias_data_transform(data::Array{Float64, 2})

The neuron model implemented here does not consider neural biases. As a trick,
one can add an additional constant dimension to the input data, whose learned
corresponding weight is then considered as the bias.

# Return
A copy of the given data with the size of the first dimension being increased
by one, such that a 1 is added for all samples at the end of the first
dimension.
"""
function neural_bias_data_transform(data::Array{Float64, 2})
    batch_size = size(data)[2]
    tdata = vcat(data, ones(1, batch_size))

    tdata
end
