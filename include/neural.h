// encoding: utf-8
// Copyright (c) 2026 Luan Pestana
// SPDX-License-Identifier: MIT

#ifndef NEURAL_H
#define NEURAL_H

#include <stddef.h>
#include <stdbool.h>


// Estrutura onde ficara as memorias que o algoritmo reutilizara
typedef struct {
	// Obrigatorio informar
	int *model; // Modelo de camadas da rede neural, lista indicando a quantidade de neuronios em cada camada
	size_t model_length; // Quantidade de camadas
	
	// Nao necessario
	size_t memory_size; // A quantidade de doubles que a memoria podera guardar, deve ser o numero de neuronios do layer com maior numero de neuronios

	size_t weights_length; // Quantidade de pesos
	double max_weight; // Peso maximo, se for 5, o peso podera variar de -5 a 5
	double max_weight_mutation; // Mutacao maxima do peso, se for 2, a mutacao aplicada no peso podera variar de -2 a 2
	double *weights; // Endereco para a array de pesos

	size_t biases_length; // Quantidade de biases
	double max_bias; // Bias maximo, se for 5, o bias podera variar de -5 a 5
	double max_bias_mutation; // Mutacao maxima do bias, se for 2, a mutacao aplicada no bias podera variar de -2 a 2
	double *biases; // Endereco para a array de biases

	// Layers usados durante o calculos
	double *layer_input; // Entrada (valores iniciais)
	double *layer_output; // Saida (valores finais)
	bool ReLU;
} ForwardMemory;


// Inferencia //
// Funcao que calcula a saida do algoritmo Forward para Inteligencias Artificiais
double *forward(ForwardMemory *foward_memory, const double *neural_input);

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
// Funcao que libera a memoria utilizada pela rede neural
bool destroy_forward_memory(ForwardMemory *forward_memory);


#endif