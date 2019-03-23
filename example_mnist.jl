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

using MNIST
using PyPlot

include("network.jl")

# Specify dataset.
train_in, train_out = MNIST.traindata()
test_in, test_out = MNIST.testdata()
train_in /= 255
test_in /= 255
train_out = one_hot(Array{Int, 1}(train_out) + 1)
test_out = one_hot(Array{Int, 1}(test_out) + 1)

# Downsample data from 28x28 to 14x14.
if true
    train_in = reshape(train_in, 28, 28, size(train_in, 2))
    test_in = reshape(test_in, 28, 28, size(test_in, 2))

    train_in = train_in[2:2:end, 2:2:end, :]
    test_in = test_in[2:2:end, 2:2:end, :]

    train_in = reshape(train_in, 14*14, size(train_in, 3))
    test_in = reshape(test_in, 14*14, size(test_in, 3))
end

# Add weights for neural biases.
train_in = neural_bias_data_transform(train_in)
test_in = neural_bias_data_transform(test_in)

in_size = size(train_in)[1]
out_size = 10

# Define hyperparameters.
hparams = Hyperparameters()
hparams.λ = 1
hparams.γ = 0.99
hparams.αₛ = 2
hparams.αₚ = 2
hparams.rᵦ = 0.0
hparams.η = 0.9
hparams.num_epochs = 100
hparams.batch_size = 10
hparams.sample_size = 100
hparams.clip_gradients = true
hparams.clipping_value = 0.1
hparams.rate_decay = true
hparams.rate_decay_interval = 1000

# Weight initializer.
#initializer(dims...) = 1e-3 * (2*rand(dims) - 1) / prod(dims)
# We use Xavier initialization instead.

# Build network.
network = Network(hparams = hparams)

loss_params = LossParameters()
loss_params.sparsity = 0.5

layer0 = input_layer("in", in_size)
add_layer(network, layer0)

initializer = xavier_initializer(layer0.size, 300)
layer1 = fully_connected_layer("fc1", layer0, 300, loss_params=loss_params,
                               initializer=initializer)
add_regularizer(layer1, sparsity_regularizer)
add_regularizer(layer1, participation_regularizer)
add_layer(network, layer1)

initializer = xavier_initializer(layer1.size, 100)
layer2 = fully_connected_layer("fc2", layer1, 100, loss_params=loss_params,
                               initializer=initializer)
add_regularizer(layer2, sparsity_regularizer)
add_regularizer(layer2, participation_regularizer)
add_layer(network, layer2)

loss_params = LossParameters()
loss_params.sparsity = 0.9
initializer = xavier_initializer(layer2.size, out_size)
layer3 = fully_connected_layer("out", layer2, out_size, loss_params=loss_params,
                               initializer=initializer)
add_regularizer(layer3, sparsity_regularizer)
add_regularizer(layer3, participation_regularizer)
add_layer(network, layer3)

println("Predictions in the untrained network (to check initialization):")
println(predict(network, test_in[:,1:1]))
println(predict(network, test_in[:,2:2]))
println(predict(network, test_in[:,3:3]))
println()

# Train Network.
T, E = train_network(network, train_in, train_out = train_out,
                    test_in = test_in[:, 1:200],
                    test_out = test_out[:, 1:200],
                    eval_interval = 10)

#println("\nThe final participation trace in the output layer:")
#println(layer3.loss_memory["y_trace"])
#println()

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
plot(10:10:length(T), E, label="Test Accuracy")
axhline(y=0.0, color="k", linestyle="--")
axhline(y=1.0, color="k", linestyle="--")
legend()
ax = gca()
ax[:set_xlim]([0, length(T)])
show()

# Plot downsampling examples.
#subplot(221)
#axis("off")
#title("Original")
#imshow(reshape(test_in[1:end-1,1], 28, 28), vmin=-1.0, vmax=1.0)
#subplot(222)
#axis("off")
#title("Downsampled")
#imshow(reshape(test_in[1:end-1,1], 28, 28)[2:2:end, 2:2:end], vmin=-1.0,
#   vmax=1.0)
#
#subplot(223)
#axis("off")
#imshow(reshape(test_in[1:end-1,2], 28, 28), vmin=-1.0, vmax=1.0)
#subplot(224)
#axis("off")
#imshow(reshape(test_in[1:end-1,2], 28, 28)[2:2:end, 2:2:end], vmin=-1.0,
#   vmax=1.0)
#
#suptitle("Downsampling example")
#show()

for ii = 1:16
    pred = predict(network, test_in[:,ii:ii])

    subplot(4,8,(ii-1)*2 + 1)
    imshow(reshape(test_in[1:end-1,ii], 14, 14), vmin=-1.0, vmax=1.0)
    axis("off")

    subplot(4,8,(ii-1)*2 + 2)
    bar(1:10, squeeze(pred, 2), align="center")
    ax = gca()
    ax[:xaxis][:set_ticks](1:10)
end

title("Example of input (left) and output response (right)")
show()
