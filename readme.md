# JAX Nano Model

A mini LLM built from scratch using JAX and the broader JAX AI stack (Flax/NNX, Optax, Grain, and Orbax). Using most JAX features including automatic differentiation, JIT compilation via XLA, and vectorized/parallelized execution across CPU, GPU, and TPU. 
This follows a functional programming paradigm that helps train and serve a transformer based language model with scalable data loading, optimisation, and checkpointing.


### Numpy Style with a functional API

The JAX core library essentially lets you write code that looks pretty much like NumPy.

### Automatic differentiation
JAX automatically computes gradients of functions, allowing you to easily implement gradient-based optimization algorithms.

### Just-in-time compilation (JIT)
jit() compiles Python functions to optimized machine code at runtime.

### Vectorization
vmap() applies a function to multiple inputs in parallel and pmap() applies it across multiple devices.

### GPU/TPU Support
JAX supports GPU and TPU accelerators, allowing us to leverage hardware acceleration for faster computations.
