// encoding: utf-8
// Copyright (c) 2026 Luan Pestana
// SPDX-License-Identifier: MIT

#ifndef NEURON_H
#define NEURON_H

#include <stddef.h>
#include <stdbool.h>


// Estrutura onde ficara as memorias que o algoritmo reutilizara
typedef struct {
	// Obrigatorio informar
	int *model; // Modelo de camadas da rede neuron, lista indicando a quantidade de neuronios em cada camada
	size_t model_length; // Quantidade de camadas
	
	// Nao necessario
	size_t memory_size; // A quantidade de doubles que a memoria podera guardar, deve ser o numero de neuronios do layer com maior numero de neuronios

	size_t weights_length; // Quantidade de pesos
	double max_weight; // Peso maximo, se for 5, o peso podera variar de -5 a 5
	double max_weight_mutation; // Mutacao maxima do peso, se for 2, a mutacao aplicada no peso podera variar de -2 a 2
	double *weights; // Endereco para a array de pesos
	double *weights_gradients; // Endereco para a array dos gradiente dos pesos

	size_t biases_length; // Quantidade de biases
	double max_bias; // Bias maximo, se for 5, o bias podera variar de -5 a 5
	double max_bias_mutation; // Mutacao maxima do bias, se for 2, a mutacao aplicada no bias podera variar de -2 a 2
	double *biases; // Endereco para a array de biases
	double *biases_gradients; // Endereco para a array dos gradiente dos biases

	// Layers usados durante o calculos
	double *layer_input; // Entrada (valores iniciais)
	double *layer_output; // Saida (valores finais)
	bool ReLU_enabled; // Aplica ReLU nas camadas internas
	bool sigmoid_enabled; // Aplica sigmoid nas camadas internas
	// Funcoes de ativacoes na saida
	bool tanh_output_enabled;
	bool softmax_output_enabled;
	double learning_rate; // Taxa de aprendizagem durante o backward
} ForwardMemory;


// Funcoes de ativacao e suas derivadas
double ReLU(double value);
double ReLU_derivative(double value);
double sigmoid(double value);
double sigmoid_derivative(double value);
void softmax(double *input, size_t size);

// Escolhe o maior valor
double max(double a, double b);
// Escolhe o menor valor
double min(double a, double b);

// Calcula a loss (erro) do modelo, usando o Cross-Entropy
double compute_loss(ForwardMemory *forward_memory, const double *response, const double *desired);

// Inferencia //
// Funcao que calcula a saida do algoritmo Forward para Inteligencias Artificiais
double *forward(ForwardMemory *forward_memory, const double *neuron_input);

// Funcao que calcula a saida do algoritmo Forward para Inteligencias Artificiais, porem armazenando e retornando os valores de cada neuronio
double *forward_with_cache(ForwardMemory *forward_memory, const double *neuron_input);

// Calcula os gradientes de todos os pesos e bias, e os soma aos gradientes ja existentes, acumulando gradientes
// Um gradiente eh a influencia que determinada variavel tem na Loss (erro)
bool backward(ForwardMemory *forward_memory, const double *desired, const double *neuron_input);

// Aplica os gradientes calculados e acumulados com o uso sequencial do backward()
bool apply_gradients(ForwardMemory *forward_memory);

// Persistencia //
// Funcao para salvar os pesos em um arquivo binario
bool save(ForwardMemory *forward_memory, const char *file_name);
// Funcao para carregar pesos de um arquivo de save
bool load(ForwardMemory *forward_memory, const char *file_name);

// Manipulacao de parametros //
// Gera pesos aleatorios dentro do limite
bool weights_randomize(ForwardMemory *forward_memory);
// Gera biases aleatorios dentro do limite
bool biases_randomize(ForwardMemory *forward_memory);
// Gera mutacoes aleatorias (dentro do limite) nos pesos
bool weights_mutate(ForwardMemory *forward_memory);
// Gera mutacoes aleatorias (dentro do limite) nos biases
bool biases_mutate(ForwardMemory *forward_memory);

// Inicializacao //
// Funcao que inicializa a memoria que sera reutilizada pela funcao forward
bool initialize_forward_memory(ForwardMemory *forward_memory);

// Liberacao de memoria //
// Funcao que libera a memoria utilizada pela rede neuron
bool destroy_forward_memory(ForwardMemory *forward_memory);


#endif