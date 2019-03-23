# Investigating the REINFORCE learning rule for stochastic networks with Bernoulli units

For a network with stochastic Bernoulli units, the REINFORCE learning rule simplifies to a Hebbian term, which is modulated by a global reward. Moreover, the learning rule is proven to be an unbiased estimate of the gradient that maximizes the global reward. In that sense, the REINFORCE learning rule is very powerful and biological plausible.

In the [notebook](expected_weight_update.ipynb) I derive the exact expectation of the weight update (hence, the correct gradient to maximize reward) for simple network structures. Therefore, the notebook may provide an intuition for how efficient learning can occur in the REINFORCE framework (which is often deemed to suffer from high variance) compared to learning with *exact gradients*.

On the other hand, the [julia framework](network.jl) allows to empirically test the REINFORCE framework in large-scale problem. I already provide two toy examples and an MNIST implementation. 

## Background

The idea behind this project has been to find out how reliable the global reward has to be in order for REINFORCE to work. If the global reward can be noisy or even random except for its sign, then the learning rule would be a perfect candidate for a biological-plausible mechanism implemented in the brain (Hebbian learning plus global neuromodulatory signal). 

In short, the goal of the project is to marginalize the influence of a global reward by choosing proper regularizers that can essentially be computed through biological plausible and local mechanisms. In the current implementation, we explore two kinds of regularizers: **sparsity** and **participation**.

The sparsity constraints will induce competition. If chosen properly, they might lead to a desired clustering of inputs, such that this process only needs mild (global) supervision. A sparsity constraint alone is not sufficient, as it can be satisfied by pruning neurons away. Therefore, we postulate an additional participation loss. 

### Sanity Check

To test whether the regularizers proposed above make sense, we test them in a backprop setting (see [here](sanity_check)). Note, this comparison should not be taken too serious, as the networks considered when applying backprop are usually not stochastic (though, there are solutions that allow one to backprop through stochastic nodes, e.g., the reparametrization trick or score functions). The goal of this sanity check was simply to get an intuition on whether networks can be learned that enforce the regularized constraints and how this influences their performance.

## References

Ronald J. Williams. [Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning](https://link.springer.com/article/10.1007/BF00992696). *Machine Learning*, 1992
