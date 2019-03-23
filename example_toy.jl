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
N_train = 100 # Size of training set.
N_test = 100 # Size of test set.
n_in = 3 # Number of input units.
n_out = 2 # Number of output units.

train_out = rand(1:n_out, N_train)
test_out = rand(1:n_out, N_test)

train_in = ones(n_in, N_train) * 0.2
test_in = ones(n_in, N_test) * 0.2

train_in[:,train_out .== 2] = 0.8
test_in[:,test_out .== 2] = 0.8

train_out = one_hot(train_out)
test_out = one_hot(test_out)

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
                    eval_interval = 10)

# Plot training development.
title("Development of training loss")
xlabel("Epoch")
ylabel("Train Loss")
plot(1:length(T), T, label = "Train Loss")
legend()
show()

title("Development of test evaluation during training")
xlabel("Epoch")
ylabel("Test Eval")
plot(10:10:length(T), E, label="Test Evaluation")
axhline(y=1.0, color="k", linestyle="--")
legend()
ax = gca()
ax[:set_xlim]([0, length(T)])
show()
