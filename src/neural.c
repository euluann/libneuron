// encoding: utf-8
// Copyright (c) 2026 Luan Pestana
// SPDX-License-Identifier: MIT

#include "neural.h"

#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <stdbool.h>
#include <string.h>
#include <time.h>

// Funcao que calcula a saida do algoritmo Forward para Inteligencias Artificiais
double *forward(ForwardMemory *foward_memory, const double *neural_input){
	
	// Obtem os enderecos de forward_memory
	// restrict, afirma que os nao ha sobreposicao de enderecos, permitindo vetorizacao apos a compilacao
	double * restrict layer_input = foward_memory->layer_input;
	double * restrict layer_output = foward_memory->layer_output;
	int * restrict model = foward_memory->model;
	int model_length = foward_memory->model_length;
	const double * restrict weights = foward_memory->weights;
	const double * restrict biases = foward_memory->biases;
	bool ReLU = foward_memory->ReLU;
	bool make_biases = false;
	
	// Copia os valores iniciais recebidos para a memoria reutilizada
	memcpy(layer_input, neural_input, model[0] * sizeof(double));
	
	// Percorre as layers (camadas) da rede neural (menos a primeira, pois eh a entrada)
	for (int layer_i = 0; layer_i+1 < model_length; layer_i++){
		// Obtem o tamanho do layer atual e do layer seguinte
		int current_layer = model[layer_i];
		int next_layer = model[layer_i+1];

		// Verifica se nao eh a ultima camada
		bool isnt_last_layer = layer_i+2 < model_length;
		// Se uma lista de biases foi informada
		if (biases != NULL){
			// Se nao for a ultima camada e nem a primeira
			make_biases = isnt_last_layer;
		}
		
		// Percorre todos os neuronios do layer seguinte
		for (int neural_i = 0; neural_i < next_layer; neural_i++){
			// Variavel que guarra a soma dos neuronios anteriores multiplicados pelo peso
			double sum = 0;
			
			// Percorre todos os neuronios do layer atual
			for (int value_i = 0; value_i < current_layer; value_i++){
				// Multiplica o valor do neuronio atual da camama atual pelo peso atual
				sum += layer_input[value_i] * weights[value_i];
			}
			
			// Acrescenta no endereco dos pesos, para pular os pesos ja usados
			weights += current_layer;
			
			// Verifica se deve aplicar o bias
			if (make_biases){
				// Soma o bias do endereco atual em biases
				sum += *biases;
				// Incrementa no endereco da lista de biases, indo para o proximo valor
				biases++;
			}
			
			// Se ReLU estiver ativado e nao for a ultima camada
			if (ReLU && isnt_last_layer){
				// Zera o valor do neuronio se for menor que 0
				sum = sum < 0 ? 0 : sum;
			}
			
			// Por o resultado da soma no respectivo neuronio da layer seguinte, acrescentando o bias do endereco atual da lista de biases
			layer_output[neural_i] = sum;
			
			// Descomentar isso ira printar os valores dos neuronios de cada camada
			//printf("Neuronio %d, Layer %d: %f\n", neural_i, layer_i+1, layer_output[neural_i]);
		}
		
		// Troca os enderecos de layer_input e layer_output, fazendo os neuronios da camada seguinte irem pra camada atual //
		// Guarda o endereco do layer_input numa variavel temporaria
		double *temp = layer_input;
		// Poe o endereco de layer_output em layer_input
		layer_input = layer_output;
		// Poe o endereco de layer_input em layer_output
		layer_output = temp;
		
	}
	
	// Retorna o endereco da ultima camada
	return layer_input;
}

// Funcao para salvar os pesos em um arquivo binario
bool save(ForwardMemory *forward_memory, const char *file_name){
	// Cria um arquivo em modo escrita binaria (Write-Binary)
	FILE *file = fopen(file_name, "wb");
	
	// Se retornou NULL, nao foi possivel criar o arquivo
	if (file == NULL){
		// Retorna 1 para erro
		return 1;
	}
	
	// Escreve os pesos no arquivo
	size_t written = fwrite(forward_memory->weights, sizeof(double), forward_memory->weights_length, file);
	
	// Se quantidade de pesos escritos eh diferente da quantidade de pesos necessario para o modelo
	bool error = written != forward_memory->weights_length;

	// Escreve os biases no arquivo
	written = fwrite(forward_memory->biases, sizeof(double), forward_memory->biases_length, file);

	// Se quantidade de biases escritos eh diferente da quantidade de biases necessario para o modelo
	error = error && written != forward_memory->biases_length;
	
	// Fecha o arquivo
	fclose(file);
	
	// Retorna a variavel error, que sera 1 se ocorreu um erro
	return error;
}

// Funcao para carregar pesos de um arquivo de save
bool load(ForwardMemory *forward_memory, const char *file_name){
	// Abre o arquivo
	FILE *file = fopen(file_name, "rb");
	
	// Se NULL, nao conseguiu abrir o arquivo
	if (file == NULL){
		// Retorna 1 para erro
		return 1;
	}
	
	// Le todos os pesos e armazena em forward_memory
	size_t read = fread(forward_memory->weights, sizeof(double), forward_memory->weights_length, file);
	
	// Se quantidade de pesos lidos eh diferente da quantidade de pesos necessario para modelo
	bool error = read != forward_memory->weights_length;
	
	// Le todos os biases e armazena em forward_memory
	read = fread(forward_memory->biases, sizeof(double), forward_memory->biases_length, file);
	
	// Se quantidade de biases lidos eh diferente da quantidade de biases necessario para o modelo
	error = error && read != forward_memory->biases_length;
	
	// Fecha o arquivo
	fclose(file);
	
	// Retorna a variavel error, que sera 1 se ocorreu um erro
	return error;
}

// Gera pesos aleatorios dentro do limite
bool weights_randomize(ForwardMemory *forward_memory){
	// Obtem os enderecos de forward_memory
	// restrict, afirma que os nao ha sobreposicao de enderecos, permitindo vetorizacao apos a compilacao
	double * restrict weights = forward_memory->weights;
	const size_t weights_length = forward_memory->weights_length;
	const double max_weight = forward_memory->max_weight;

	// Percorre todos os pesos
	for (int i = 0; i < weights_length; i++){
		// Gera um numeto aleatorio dentro de -max_weight a max_weight
		weights[i] = (double)rand() / RAND_MAX * max_weight * 2 - max_weight;
	}

	return 0;
}

// Gera biases aleatorios dentro do limite
bool biases_randomize(ForwardMemory *forward_memory){
	// Obtem os enderecos de forward_memory
	// restrict, afirma que os nao ha sobreposicao de enderecos, permitindo vetorizacao apos a compilacao
	double * restrict biases = forward_memory->biases;
	const size_t biases_length = forward_memory->biases_length;
	const double max_bias = forward_memory->max_bias;

	// Percorre todos os biases
	for (int i = 0; i < biases_length; i++){
		// Gera um numeto aleatorio dentro de -max_biases a max_biases
		biases[i] = (double)rand() / RAND_MAX * max_bias * 2 - max_bias;
	}

	return 0;
}


// Gera mutacoes aleatorias (dentro do limite) nos pesos
bool weights_mutate(ForwardMemory *forward_memory){
	// Obtem os enderecos de forward_memory
	// restrict, afirma que os nao ha sobreposicao de enderecos, permitindo vetorizacao apos a compilacao
	double * restrict weights = forward_memory->weights;
	const size_t weights_length = forward_memory->weights_length;
	const double max_weight_mutation = forward_memory->max_weight_mutation;
	const double max_weight = forward_memory->max_weight;

	// Percorre todos os pesos
	for (int i = 0; i < weights_length; i++){
		// Gera um numeto aleatorio dentro de -max_weight a max_weight
		weights[i] += (double)rand() / RAND_MAX * max_weight_mutation * 2 - max_weight_mutation;
		weights[i] = weights[i] > max_weight ? max_weight : (weights[i] < -max_weight ? -max_weight : weights[i]);
	}

	return 0;
}

// Gera mutacoes aleatorias (dentro do limite) nos biases
bool biases_mutate(ForwardMemory *forward_memory) {
	// Obtem os enderecos de forward_memory
	// restrict, afirma que os nao ha sobreposicao de enderecos, permitindo vetorizacao apos a compilacao
	double * restrict biases = forward_memory->biases;
	const size_t biases_length = forward_memory->biases_length;
	const double max_bias_mutation = forward_memory->max_bias_mutation;
	const double max_bias = forward_memory->max_bias;

	// Percorre todos os biases
	for (int i = 0; i < biases_length; i++){
		// Gera um numeto aleatorio dentro de -max_biases a max_biases
		biases[i] += (double)rand() / RAND_MAX * max_bias_mutation * 2 - max_bias_mutation;
		biases[i] = biases[i] > max_bias ? max_bias : (biases[i] < -max_bias ? -max_bias : biases[i]);
	}

	return 0;
}

// Funcao que inicializa a memoria e os valores que seram utilizados pela rede neural
bool initialize_forward_memory(ForwardMemory *forward_memory){
	// Se um modelo nao foi informado
	if (forward_memory->model == NULL){
		printf("Neural Network model not specified.");
		return 1;
	}
	// Se o tamanho do modelo nao foi informado
	if (forward_memory->model_length <= 0){
		printf("Neural Network model length not specified or is ilegal.");
		return 1;
	}
	
	// Se ja houver memoria alocada, a libera antes de alocar mais
	if (forward_memory->layer_input != NULL){
		free(forward_memory->layer_input);
	}
	if (forward_memory->layer_output != NULL){
		free(forward_memory->layer_output);
	}
	if (forward_memory->weights != NULL){
		free(forward_memory->weights);
	}
	if (forward_memory->biases != NULL){
		free(forward_memory->biases);
	}
	
	// Inicializa os tamanhos que serao contados como 0
	forward_memory->memory_size = 0;
	forward_memory->biases_length = 0;
	forward_memory->weights_length = 0;

	// Define valores maximos se ja nao foram definidos
	if (forward_memory->max_weight <= 0){
		forward_memory->max_weight = 1;
	}
	if (forward_memory->max_bias <= 0){
		forward_memory->max_bias = 1;
	}
	if (forward_memory->max_weight_mutation <= 0){
		forward_memory->max_weight_mutation = 0.01;
	}
	if (forward_memory->max_bias_mutation <= 0){
		forward_memory->max_bias_mutation = 0.01;
	}
	
	// Calcula a quantidade de pesos que o modelo usara e procura a maior camada para alocar memoria com o tamanho da maior camada
	for (int i = 0; i < forward_memory->model_length-1; i++){
		// Verifica se a camada atual ou a proxima eh maior que o tamanho da memoria
		if (forward_memory->model[i] > forward_memory->memory_size){
			forward_memory->memory_size = forward_memory->model[i];
		}
		if (forward_memory->model[i+1] > forward_memory->memory_size){
			forward_memory->memory_size = forward_memory->model[i+1];
		}
		
		// Verifica se o modelo eh invalido
		if (forward_memory->model[i] <= 0 && forward_memory->model[i+1] <= 0){
			printf("Invalid Neural Network model.");
			// Retorna 1 para erro
			return 1;
		}
		
		// Multiplica a camada atual com a proxima, resultando na quantidade de pesos que essa camada usara com a proxima para somar ao total
		forward_memory->weights_length += forward_memory->model[i] * forward_memory->model[i+1];
		
		// Se nao for a ultima camada
		if (i+1 < forward_memory->model_length-1){
			// Soma a quantidade de neuronios da camada atual na quantidade de biases
			forward_memory->biases_length += forward_memory->model[i+1];
		}
	}
	
	// Aloca memoria suficiente para todos os pesos que o modelo usara
	forward_memory->weights = malloc(forward_memory->weights_length * sizeof(double));
	
	// Aloca memoria suficiente para todos os biases que o modelo usara
	forward_memory->biases = malloc(forward_memory->biases_length * sizeof(double));
	
	// Aloca memoria suficiente para a maior layer
	forward_memory->layer_input = malloc(forward_memory->memory_size * sizeof(double));
	forward_memory->layer_output = malloc(forward_memory->memory_size * sizeof(double));
	
	// Verifica se foi possivel alocar a memoria suficiente para todos
	if (forward_memory->weights == NULL || forward_memory->biases == NULL || forward_memory->layer_input == NULL || forward_memory->layer_output == NULL){
		printf("Memory error.");
		return 1;
	}

	// Gera aleatorios limitados para os pesos e biases
	weights_randomize(forward_memory);
	biases_randomize(forward_memory);

	return 0;
}


// Funcao que libera a memoria utilizada pela rede neural
bool destroy_forward_memory(ForwardMemory *forward_memory){
	// Se houver memoria alocada, a libera
	if (forward_memory->layer_input != NULL){
		free(forward_memory->layer_input);
	}
	if (forward_memory->layer_output != NULL){
		free(forward_memory->layer_output);
	}
	if (forward_memory->weights != NULL){
		free(forward_memory->weights);
	}
	if (forward_memory->biases != NULL){
		free(forward_memory->biases);
	}

	return 0;
}