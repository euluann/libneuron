# libneuron
A lightweight neural network library written in C, implementing forward propagation and backpropagation from scratch.

## Features
- Fully connected neural networks
- Configurable network architecture
- Forward propagation
- Backpropagation
- Gradient descent optimization
- ReLU activation function
- Configurable learning rate
- Written in pure C

## Compile your program
```bash
gcc -lm -I./include src/* main.c -o main
```

## Usage
```C
#include "neuron.h"

int main(void){
  // Configure the network architecture
  int architecture[] = {2,4,4,2};

  ForwardMemory memory = {
    .model = malloc(4 * sizeof(int)),
    .model_length = 4,
    .ReLU_enabled = 1,
    .learning_rate = 0.02
  }

  memcpy(memory.model, architecture, sizeof(architecture));

  initialize_forward_memory(&memory);

  // Your traning and inference code here
}
```

## Building
Requires GCC or another compatible C compiler

## License
MIT License
