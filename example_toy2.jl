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

workspace() # In case, this code block is copied to a notebook.

using PyPlot

include("network.jl")

# Specify dataset.
function linear_labels(p_x::Array{Float64, 2}, out_size::Int)
    w = ones(size(p_x, 1))
    w /= norm(w)

    #b = norm(ones(size(p_x, 1)) * 0.5)
    b = 0

    y = zeros(Bool, out_size, size(p_x, 2))
    for i = 1:size(p_x, 2)
        if (dot(w, p_x[:,i]) - b) > 0
            y[:,i] = 1
        end
    end

    y
end

N_train = 100 # Size of training set.
N_test = 100 # Size of test set.
n_in = 10 # Number of input units.
n_out = 20 # Number of output units.

train_in = rand(n_in, N_train) .- 0.5
test_in = rand(n_in, N_test) .- 0.5

# Uncomment, to work with static sample.
#sample = rand(n_in)
#for i = 1:n_in
#    train_in[i,:] = sample[i]
#    test_in[i,:] = sample[i]
#end

train_out = linear_labels(train_in, n_out)
test_out = linear_labels(test_in, n_out)

in_size = size(train_in)[1]
out_size = size(train_out)[1]

# Define hyperparameters.
hparams = Hyperparameters()
hparams.λ = 1
hparams.γ = 0.99
hparams.αₛ = 0.5
hparams.αₚ = 0.5
hparams.rᵦ = 0.0
hparams.η = 0.9
hparams.num_epochs = 1000
hparams.batch_size = 1
hparams.sample_size = 100
hparams.clip_gradients = false
hparams.clipping_value = 0.1

# Weight initializer.
initializer(dims...) = 1e-1 * (2*rand(dims) - 1)

# Build network.
network = Network(hparams = hparams)

loss_params = LossParameters()
loss_params.sparsity = 0.5

layer0 = input_layer("in", in_size)
add_layer(network, layer0)

layer1 = fully_connected_layer("fc1", layer0, 4, loss_params=loss_params,
                               initializer=initializer)
add_regularizer(layer1, sparsity_regularizer)
add_regularizer(layer1, participation_regularizer)
add_layer(network, layer1)

layer2 = fully_connected_layer("out", layer1, out_size, loss_params=loss_params,
                               initializer=initializer)
add_layer(network, layer2)

# Train Network.
T, E = train_network(network, train_in, train_out = train_out,
                    test_in = test_in,
                    test_out = test_out,
                    eval_interval = 10,
                    eval_accuracy = false)

# Plot training development.
title("Development of training loss")
xlabel("Epoch")
ylabel("Train Loss")
plot(1:length(T), T, label = "Train Loss")
legend()
show()

title("Development of global test loss (without regularizers) during training")
xlabel("Epoch")
ylabel("Test Eval")
plot(10:10:length(T), E, label="Test Evaluation")
axhline(y=0.0, color="k", linestyle="--")
legend()
ax = gca()
ax[:set_xlim]([0, length(T)])
show()
