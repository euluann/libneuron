// encoding: utf-8
// Copyright (c) 2026 Luan Pestana
// SPDX-License-Identifier: MIT

#include "neuron.h"

#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <stdbool.h>
#include <string.h>
#include <time.h>
#include <math.h>


// Funcao de ativacao (AF) ReLU
double ReLU(double value){
	return value > 0 ? value : 0;
}
// Derivada da AF ReLU 
double ReLU_derivative(double value){
	return value > 0 ? 1 : 0;
}
// AF Sigmoid
double sigmoid(double value){
	return 1.0 / (1.0 + exp(-value));
}
// Derivada da AF Sigmoid
double sigmoid_derivative(double value){
	double sigmoid_outpout = sigmoid(value);
	return sigmoid_outpout * (1 - sigmoid_outpout);
}
// Derivada da AF Sigmoid apartir da sigmoid de um valor
double _sigmoid_derivative_after_sigmoid(double value){
	return value * (1 - value);
}
// AF Softmax
void softmax(double *input, size_t size){
	// Soma de todos os expoentes
	double exp_sum = 0;
	// O maior valor da lista
	double logits_max = 0;
	// Obtem o maior valor
	for (int i = 0; i < size; i++){
		if (input[i] > logits_max){
			logits_max = input[i];
		}
	}
	// Obtem a soma de todos os expoentes
	for (int i = 0; i < size; i++){
		exp_sum += exp(input[i]-logits_max); // Subtrai pelo maior numero para evitar overflow, mas ainda mantem a matematica correta
	}
	// Obtem o resultado do expoente de cada numero da lista dividido pela soma de todos os expoentes
	for (int i = 0; i < size; i++){
		input[i] = exp(input[i]-logits_max)/exp_sum;
	}
}


// Escolhe o maior valor
double max(double a, double b){
	return a > b ? a : b;
}
// Escolhe o menor valor
double min(double a, double b){
	return a < b ? a : b;
}

// Calcula a loss (erro) do modelo, usando Cross-Entropy
double compute_loss(ForwardMemory *forward_memory, const double *response, const double *desired){
	int response_length = forward_memory->model[ forward_memory->model_length - 1 ];
	double epsilon = 1e-15; // Valor minimo para evitar log(0)
	double loss = 0;
	for (int i = 0; i < response_length; i++){
		loss -= desired[i] * log(fmax(response[i], epsilon));
	}
	return loss;
}

// Funcao que calcula a saida do algoritmo Forward para Inteligencias Artificiais
double *forward(ForwardMemory *forward_memory, const double *neuron_input){
	
	// Obtem os enderecos de forward_memory
	// restrict, afirma que os nao ha sobreposicao de enderecos, permitindo vetorizacao apos a compilacao
	double * restrict layer_input = forward_memory->layer_input;
	double * restrict layer_output = forward_memory->layer_output;
	int * restrict model = forward_memory->model;
	int model_length = forward_memory->model_length;
	const double * restrict weights = forward_memory->weights;
	const double * restrict biases = forward_memory->biases;
	bool ReLU_enabled = forward_memory->ReLU_enabled;
	bool sigmoid_enabled = forward_memory->sigmoid_enabled;
	bool tanh_output_enabled = forward_memory->tanh_output_enabled;
	bool softmax_output_enabled = forward_memory->softmax_output_enabled;
	bool make_biases = false;
	
	// Copia os valores iniciais recebidos para a memoria reutilizada
	memcpy(layer_input, neuron_input, model[0] * sizeof(double));
	
	// Percorre as layers (camadas) da rede neuron (menos a primeira, pois eh a entrada)
	for (int layer_i = 0; layer_i+1 < model_length; layer_i++){
		// Obtem o tamanho do layer atual e do layer seguinte
		int current_layer = model[layer_i];
		int next_layer = model[layer_i+1];

		// Verifica se nao eh a ultima camada
		bool isnt_last_layer = layer_i+2 < model_length;
		// Se uma lista de biases foi informada
		if (biases != NULL){
			// Se nao for a ultima camada e nem a primeira
			make_biases = 1;
		}
		
		// Percorre todos os neuronios do layer seguinte
		for (int neuron_i = 0; neuron_i < next_layer; neuron_i++){
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
			
			// Funcao de ativacao
			// Se ReLU estiver ativado e nao for a ultima camada
			if (ReLU_enabled && isnt_last_layer){
				// Zera o valor do neuronio se for menor que 0
				sum = ReLU(sum);
			}
			// Se sigmoid estiver ativado e nao for a ultima camada
			if (sigmoid_enabled && isnt_last_layer){
				// Zera o valor do neuronio se for menor que 0
				sum = sigmoid(sum);
			}
			
			// Por o resultado da soma no respectivo neuronio da layer seguinte, acrescentando o bias do endereco atual da lista de biases
			layer_output[neuron_i] = sum;
			
			// Descomentar isso ira printar os valores dos neuronios de cada camada
			//printf("Neuronio %d, Layer %d: %f\n", neuron_i, layer_i+1, layer_output[neuron_i]);
		}
		
		// Troca os enderecos de layer_input e layer_output, fazendo os neuronios da camada seguinte irem pra camada atual //
		// Guarda o endereco do layer_input numa variavel temporaria
		double *temp = layer_input;
		// Poe o endereco de layer_output em layer_input
		layer_input = layer_output;
		// Poe o endereco de layer_input em layer_output
		layer_output = temp;
		
	}

	// Aplica tanh na saida se ativado
	if (tanh_output_enabled){
		for (int i = 0; i < model[model_length-1]; i++){
			layer_input[i] = tanh(layer_input[i]);
		}
	}

	// Aplica softmax na saida se ativado
	if (softmax_output_enabled){
		softmax(layer_input, model[model_length-1]);
	}
	
	// Retorna o endereco da ultima camada
	return layer_input;
}

// Funcao que calcula a saida do algoritmo Forward para Inteligencias Artificiais, porem armazenando e retornando os valores de cada neuronio
double *forward_with_cache(ForwardMemory *forward_memory, const double *neuron_input){
	// Obtem os enderecos de forward_memory
	// restrict, afirma que os nao ha sobreposicao de enderecos, permitindo vetorizacao apos a compilacao
	int * restrict model = forward_memory->model;
	int model_length = forward_memory->model_length;
	int biases_length = forward_memory-> biases_length;
	const double * restrict weights = forward_memory->weights;
	const double * restrict biases = forward_memory->biases;
	bool ReLU_enabled = forward_memory->ReLU_enabled;
	bool sigmoid_enabled = forward_memory->sigmoid_enabled;
	bool tanh_output_enabled = forward_memory->tanh_output_enabled;
	bool softmax_output_enabled = forward_memory->softmax_output_enabled;
	bool make_biases = false;

	// Obtem o tamanho do cache, quantidade de neuronios total; quantidade de biases mais a quantidade de neuronios na segunda camada
	int neuron_cache_length = biases_length+model[0];
	
	// Alloca memoria suficiente para todos os neuronios da rede mais um para o tamanho do cache
	double *neuron_cache = malloc(neuron_cache_length * sizeof(double));
	
	int cache_i = 0;
	
	// Copia os valores iniciais recebidos para o cache
	memcpy(neuron_cache+cache_i, neuron_input, model[0] * sizeof(double));
	
	// Percorre as layers (camadas) da rede neuron (menos a primeira, pois eh a entrada)
	for (int layer_i = 0; layer_i+1 < model_length; layer_i++){
		// Obtem o tamanho do layer atual e do layer seguinte
		int current_layer = model[layer_i];
		int next_layer = model[layer_i+1];

		// Verifica se nao eh a ultima camada
		bool isnt_last_layer = layer_i+2 < model_length;
		// Se uma lista de biases foi informada
		if (biases != NULL){
			// Se nao for a ultima camada e nem a primeira
			make_biases = 1;
		}
		
		// Percorre todos os neuronios do layer seguinte
		for (int neuron_i = 0; neuron_i < next_layer; neuron_i++){
			// Variavel que guarra a soma dos neuronios anteriores multiplicados pelo peso
			double sum = 0;
			
			// Percorre todos os neuronios do layer atual
			for (int value_i = 0; value_i < current_layer; value_i++){
				// Multiplica o valor do neuronio atual da camama atual pelo peso atual
				sum += neuron_cache[value_i+cache_i] * weights[value_i];
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
			
			// Funcao de ativacao
			// Se ReLU estiver ativado e nao for a ultima camada
			if (ReLU_enabled && isnt_last_layer){
				// Zera o valor do neuronio se for menor que 0
				sum = ReLU(sum);
			}
			// Se sigmoid estiver ativado e nao for a ultima camada
			if (sigmoid_enabled && isnt_last_layer){
				// Zera o valor do neuronio se for menor que 0
				sum = sigmoid(sum);
			}
			
			// Por o resultado da soma no respectivo neuronio da layer seguinte, acrescentando o bias do endereco atual da lista de biases
			neuron_cache[neuron_i+cache_i+current_layer] = sum;
			
			// Descomentar isso ira printar os valores dos neuronios de cada camada
			//printf("Neuronio %d, Layer %d: %f\n", neuron_i, layer_i+1, layer_output[neuron_i]);
		}
		
		cache_i += current_layer;
		
	}

	// Aplica tanh na saida se ativado
	if (tanh_output_enabled){
		for (int i = neuron_cache_length-model[model_length-1]; i < neuron_cache_length; i++){
			neuron_cache[i] = tanh(neuron_cache[i]);
		}
	}

	// Aplica softmax na saida se ativado
	if (softmax_output_enabled){
		softmax(neuron_cache+neuron_cache_length-model[model_length-1], model[model_length-1]);
	}
	
	// Retorna o endereco da ultima camada
	return neuron_cache;
}

// Calcula os gradientes de todos os pesos e bias, e os soma aos gradientes ja existentes, acumulando gradientes
// Um gradiente eh a influencia que determinada variavel tem na Loss (erro)
bool backward(ForwardMemory *forward_memory, const double *desired, const double *neuron_input){

	// Obtem os enderecos de forward_memory
	// restrict, afirma que os nao ha sobreposicao de enderecos, permitindo vetorizacao apos a compilacao
	int * restrict model = forward_memory->model;
	int model_length = forward_memory->model_length;
	int biases_length = forward_memory-> biases_length;
	int weights_length = forward_memory->weights_length;
	double * restrict weights = forward_memory->weights;
	double * restrict biases = forward_memory->biases;
	bool ReLU_enabled = forward_memory->ReLU_enabled;
	bool sigmoid_enabled = forward_memory->sigmoid_enabled;
	bool make_biases = false;
	double learning_rate = forward_memory->learning_rate;

	// Se uma lista de biases foi informada
	if (biases != NULL){
		// Se nao for a ultima camada e nem a primeira
		make_biases = 1;
	}
	
	// Obtem o cache da rede neuron, os valores de todos os neuronios apos o forward
	double *neuron_cache = forward_with_cache(forward_memory, neuron_input);
	
	// Obtem o tamanho do cache, quantidade de neuronios total; quantidade de biases mais a quantidade de neuronios na segunda camada
	int neuron_cache_length = biases_length+model[0];
	
	//printf("Neuron: ");
	for (int i = 0; i < neuron_cache_length; i++){
		//printf("%f, ", neuron_cache[i]);
	}
	//printf("\n");


	// Aloca memoria suficiente pada armazenar os gradientes de todos os neuronios e pesos
	double *neuron_gradients = calloc(neuron_cache_length, sizeof(double));
	double *weights_gradients = forward_memory->weights_gradients;
	double *biases_gradients = forward_memory->biases_gradients;


	// Camadas a frente; camadas ja percorridas no backpropagation que devem ser consideradas para acessar os valores corretos
	int front_neurons = 0;
	int front_weights = 0;
	// Percorre as camadas de frente pra tras, exceto pela primeira camada (camada de entrada)
	for (int layer_i = model_length-2; layer_i >= 0; layer_i--){

		// Duas variaveis que guardam o tamanho das duas camadas que os pesos que estao sendo analisados conectam
		// Camada atual
		int current_layer = model[layer_i];
		// Camada seguinte
		int next_layer = model[layer_i+1];

		// Verifica se nao eh a ultima camada
		bool isnt_last_layer = layer_i+2 < model_length;

		// Primeiro loop, para calcular os gradientes dos neuronios
		// Percorre todos os neuronios da camada seguinte
		for (int next_layer_neuron_i = 0; next_layer_neuron_i < next_layer; next_layer_neuron_i++){
			// Se for a ultima camada
			if (!isnt_last_layer){
				// Calcula os gradientes dos neuronios da ultima camada; derivada do Cross-Entropy dL/dZ; (A - Y)
				neuron_gradients[
					neuron_cache_length -
					next_layer +
					next_layer_neuron_i -
					front_neurons
				] = neuron_cache[
					neuron_cache_length -
					next_layer +
					next_layer_neuron_i -
					front_neurons
				] -
				desired[next_layer_neuron_i];
				
			}
			
			// Percorre todos os neuronios da camada atual
			for (int current_layer_neuron_i = 0; current_layer_neuron_i < current_layer; current_layer_neuron_i++){
				// Obtendo os gradientes dos neuronios //
				// Soma as derivadas L/A[L]i * A[L]i/A[L-1]i formando a derivada L/A[L-1]i no final do loop
				neuron_gradients
				// Quebrando as expressoes em linha
				[
					neuron_cache_length -
					current_layer -
					next_layer +
					current_layer_neuron_i -
					front_neurons
				] += neuron_gradients[
					neuron_cache_length -
					next_layer +
					next_layer_neuron_i -
					front_neurons
				] * ( // equivalente: if (ReLU_enabled){ReLU_derivative(...)} else {1}
					(ReLU_enabled ? ReLU_derivative(
						neuron_cache[
						neuron_cache_length -
						next_layer +
						next_layer_neuron_i -
						front_neurons
					]) : 1) * // equivalente: if (sigmoid_enabled){_sigmoid_derivative_after_sigmoid(...)} else {1}
					(sigmoid_enabled ? _sigmoid_derivative_after_sigmoid(
						neuron_cache[
						neuron_cache_length -
						next_layer +
						next_layer_neuron_i -
						front_neurons
					]) : 1) *
					weights // Obtendo o peso
					[
						weights_length -
						current_layer *
						next_layer +
						current_layer_neuron_i +
						next_layer_neuron_i * 
						current_layer -
						front_weights
					]
				);
				
			}
		}

		// Novamente percorrer, porem o loop anterior calculou os gradientes dos neuronios, e agora com os gradientes dos neuronios, sera calculado os gradientes dos pesos
		// Percorre todos os neuronios da camada seguinte
		for (int next_layer_neuron_i = 0; next_layer_neuron_i < next_layer; next_layer_neuron_i++){
			// Percorre todos os neuronios da camada atual
			for (int current_layer_neuron_i = 0; current_layer_neuron_i < current_layer; current_layer_neuron_i++){
				weights_gradients
				// Quebrando as expressoes em linha
				[
					weights_length -
					current_layer *
					next_layer +
					current_layer_neuron_i +
					next_layer_neuron_i * 
					current_layer -
					front_weights
				] += neuron_gradients
				[
					neuron_cache_length -
					next_layer +
					next_layer_neuron_i -
					front_neurons
				] * ( // equivalente: if (ReLU_enabled){ReLU_derivative(...)} else {1}
					ReLU_enabled ? ReLU_derivative(
						neuron_cache[
						neuron_cache_length -
						next_layer +
						next_layer_neuron_i -
						front_neurons
					]) : 1) * // equivalente: if (sigmoid_enabled){_sigmoid_derivative_after_sigmoid(...)} else {1}
					(sigmoid_enabled ? _sigmoid_derivative_after_sigmoid(
						neuron_cache[
						neuron_cache_length -
						next_layer +
						next_layer_neuron_i -
						front_neurons
					]) : 1) * neuron_cache[
					neuron_cache_length -
					current_layer -
					next_layer +
					current_layer_neuron_i -
					front_neurons
				];
			}
		}
		// Soma os neuronios da camada seguinte aos neuronios que estao a frente, para nao reutilizar os mesmos neuronios (a camada atual nao eh somada pois ela ira virar a camada seguinte na proxima iteracao)
		front_neurons += next_layer;
		// Soma os pesos utilizados aos pesos a frente, para nao reutilizar os mesmos pesos
		front_weights += current_layer*next_layer;
	}

	// Calcula os gradientes dos biases
	for (int i = 0; i < biases_length; i++){
		// Multiplica a derivada da AF
		biases_gradients[i] += neuron_gradients[i+model[0]] * ( // equivalente: if (ReLU_enabled){ReLU_derivative(...)} else {1}
		ReLU_enabled ? ReLU_derivative(
			neuron_cache[i+model[0]]) :
			1
		 *
		sigmoid_enabled ? _sigmoid_derivative_after_sigmoid(
			neuron_cache[i+model[0]]) :
			1
		);
	}

	
	
	// Libera a memoria utilizada
	free(neuron_cache);
	free(neuron_gradients);
	return 0;
}

// Aplica os gradientes calculados e acumulados com o uso sequencial do backward()
bool apply_gradients(ForwardMemory *forward_memory){
	// Obtem os enderecos de forward_memory
	// restrict, afirma que os nao ha sobreposicao de enderecos, permitindo vetorizacao apos a compilacao
	int biases_length = forward_memory-> biases_length;
	int weights_length = forward_memory->weights_length;
	double * restrict weights = forward_memory->weights;
	double * restrict biases = forward_memory->biases;
	double learning_rate = forward_memory->learning_rate;

	double *weights_gradients = forward_memory->weights_gradients;
	double *biases_gradients = forward_memory->biases_gradients;

	// Aplica os gradientes aos pesos usando o learning rate
	for (int i = 0; i < weights_length; i++){
		//printf("%f, ", weights_gradients[i]);
		weights[i] -= learning_rate * weights_gradients[i];
		// Zera o gradiente ja aplicado
		weights_gradients[i] = 0;
	}
	// Aplica os gradientes dos biases usando o learning rate
	for (int i = 0; i < biases_length; i++){
		// Multiplica a derivada da AF
		biases[i] -= learning_rate * biases_gradients[i];
		// Zera o gradiente ja aplicado
		biases_gradients[i] = 0;
	}
	return 0;
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
		weights[i] = fmax(fmin(weights[i], max_weight), -max_weight);
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
		biases[i] = fmax(fmin(biases[i], max_bias), -max_bias);
	}

	return 0;
}

// Funcao que inicializa a memoria e os valores que seram utilizados pela rede neuron
bool initialize_forward_memory(ForwardMemory *forward_memory){
	// Se um modelo nao foi informado
	if (forward_memory->model == NULL){
		printf("neuron Network model not specified.");
		return 1;
	}
	// Se o tamanho do modelo nao foi informado
	if (forward_memory->model_length <= 0){
		printf("neuron Network model length not specified or is ilegal.");
		return 1;
	}
	
	// Se ja houver memoria alocada, a libera antes-0.907064 -0.450509  de alocar mais
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
	if (forward_memory->learning_rate == 0){
		forward_memory->learning_rate = 0.01;
	}

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
			printf("Invalid neuron Network model.");
			// Retorna 1 para erro
			return 1;
		}
		
		// Multiplica a camada atual com a proxima, resultando na quantidade de pesos que essa camada usara com a proxima para somar ao total
		forward_memory->weights_length += forward_memory->model[i] * forward_memory->model[i+1];
		
		// Soma a quantidade de neuronios da camada atual na quantidade de biases
		forward_memory->biases_length += forward_memory->model[i+1];
	}
	
	// Aloca memoria suficiente para todos os pesos que o modelo usara
	forward_memory->weights = malloc(forward_memory->weights_length * sizeof(double));
	
	// Aloca memoria suficiente para todos os biases que o modelo usara
	forward_memory->biases = malloc(forward_memory->biases_length * sizeof(double));

	// Aloca memoria suficiente para os gradientes de todos os pesos que o modelo usara
	forward_memory->weights_gradients = calloc(forward_memory->weights_length, sizeof(double));
	
	// Aloca memoria suficiente para os gradientes de todos os biases que o modelo usara
	forward_memory->biases_gradients = calloc(forward_memory->biases_length, sizeof(double));
	
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


// Funcao que libera a memoria utilizada pela rede neuron
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