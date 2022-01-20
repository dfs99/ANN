#include <iostream>
#include <array>
#include <cmath>
#include <chrono>
#include <vector>
#include <fstream>
#include <sstream>
#include <bits/stdc++.h>    // para la funcion min.
#include "../headers/CSVReaderANN.hpp"


using namespace std;

struct Neuron{
    double delta;
    double output;
};

double activation_function(double value);
double delta_activation_function(double value);

double activation_function(double value, string func){
    double result;
    const string relu = "RELU";
    const string hiperbolic_tan = "TANH";
    const string sigmoid = "SIGMOID";
    if (func == relu){
        (value < 0) ? result = 0 : result = value;
    }else if (func == hiperbolic_tan){
        result = tanh(value);
    }else if (func == sigmoid){
        result = ( 1 / (1 + exp(-value)));
    }else{
        cerr << "ERROR: Use any of these Activation Functions {RELU, TANH, SIGMOID}.\n";
        return -1000;
    }
    return result;
}

double delta_activation_function(double value, string func){
    double result;
    const string relu = "RELU";
    const string hiperbolic_tan = "TANH";
    const string sigmoid = "SIGMOID";
    if (func == relu){
        result = 1;
    }else if (func == hiperbolic_tan){
        result = (1 - pow(value, 2));
    }else if (func == sigmoid){
        result = ( value * (1 - value));
    }else{
        cerr << "ERROR: Use any of these Activation Functions {RELU, TANH, SIGMOID}.\n";
        return -1000;
    }
    return result;
}

/*
    Vars to handle arguments. 
*/

const int MIN_ARGS = 9;
const int MIN_LAYERS = 3;
const int NUM_PARAMETERS_TILL_LAYERS = 6;
const string USAGE_MESSAGE = "\n\n\t\tHow to use Backpropagation implementation:\n\n\tOnce it has been compiled."
    "You must pass the following parameters in order:\n\t\t1-.)\tcsv file containing the data to be trained.\n"
    "\t\t2-.)\tLearning rate [0-1]\n\t\t3-.)\tCycles, number of epochs to be performed.\n\t\t4-.)\tActivation "
    "function for all neurons, choose one of these {RELU, SIGMOID, TANH}\n\t\t5-.)\t"
    "Number of Layers, the minimum number of layers are three. At least there will be a hidden layer.\n\t\t"    
    "*-.)\tFor each layer, introduce its number of neurons. For example, if there are 3 layers, and it's introduced: 4 5 1"
    " the program will assign first layer (input) 4 neuron/s, second layer (hidden) 5 neuron/s and third layer (output) 1 neuron/s.\n\n\n";


int main(int argc, char *argv[]){

    /*
       Values to set up the Multilayer Perceptron.
    */

    string filename;
    double lr;
    int cycles;
    string activation_func;
    int num_layers;
    vector<int>neurons_per_layer;

    /*
        Getting those values from stdin.
    */

    if (argc >= MIN_ARGS){
        filename = argv[1];
        lr = atof(argv[2]);
        cycles = atoi(argv[3]);
        activation_func = argv[4];
        num_layers = atoi(argv[5]);
        if ((num_layers < MIN_LAYERS) || (num_layers > MIN_LAYERS && (num_layers != (argc - NUM_PARAMETERS_TILL_LAYERS)))){
            cerr << USAGE_MESSAGE;
            return -1;
        }
        for (int i = 0; i < num_layers; i++){
            neurons_per_layer.push_back(atoi(argv[NUM_PARAMETERS_TILL_LAYERS+i]));
        }
        
        cout << "filename: " << filename << "\nlearning rate: " << lr << "\n";
        cout << "cycles: " << cycles << "\nactivation_func: " << activation_func << "\n";
        cout << "num layers: " << num_layers << "\n";
        for (size_t i = 0; i < neurons_per_layer.size(); ++i){
            cout << "layer[" << i << "] has <" << neurons_per_layer[i] << "> neurons\n";
        }
    }else{
        cerr << USAGE_MESSAGE;
        return -1;
    }

    /*

        Get data from the csv file.

    */
   /*try{
        CSVReaderANN parser1 {filename,  neurons_per_layer[0], neurons_per_layer[neurons_per_layer.size()-1]};
   }catch(CSVReaderANNException & e){
        cerr << e.what() << "\n";
        return -2;
   }*/

    CSVReaderANN parser {filename,  neurons_per_layer[0], neurons_per_layer[neurons_per_layer.size()-1]};
    cout << "tamaño entrada: " << parser.get_input_dimension() << " tamaño salida: " << parser.get_output_dimension() << "\n";
    
    for (size_t i = 0; i < parser.ref_to_examples_.get()->size(); ++i){
        cout << "patron: " << i << " ";
        for (size_t j = 0; j < parser.ref_to_examples_.get()->operator[](i).inputs.size(); ++j){
            std::cout << parser.ref_to_examples_.get()->operator[](i).inputs[j] << " ";
        }
        for (size_t j = 0; j < parser.ref_to_examples_.get()->operator[](i).desired.size(); ++j){
            std::cout << parser.ref_to_examples_.get()->operator[](i).desired[j] << " ";
        }
        std::cout << "\n";
    }

    // todo: algo de aqui peta!
    /*
    
        Represent the net.

    */
    /*array<int, num_layers> topology;
    for (size_t i = 0; i < neurons_per_layer.size(); ++i){
        topology[i] = neurons_per_layer[i];
    }*/
    vector<Neuron*> net;
    vector<double**> weight_matrices;
    vector<double> errors;  // almacena todos los errores cometidos por los patrones para 1 ciclo.
    vector<double> mse_per_epoch; // almacena el mse de cada ciclo / epoch.

    /*
    
        Setting the net up: neurons and weights (with thresholds).

    */
    for (int i = 0; i < num_layers-1; ++i){
        net.push_back(new Neuron [neurons_per_layer[i+1]]);
        for (int j = 0; j < neurons_per_layer[i+1]; ++j){
            net[i][j].delta = 0;
            net[i][j].output = 0;
        }
    }

    for (int i = 0; i < num_layers-1; ++i){
        // Note: topology[i]+1 to take thresholds into account.
        weight_matrices.push_back(new double* [neurons_per_layer[i]+1]);
        //weight_matrices[i] = new double* [neurons_per_layer[i]+1];
        for (int j = 0; j < neurons_per_layer[i]+1; ++j){
            weight_matrices[i][j] = new double [neurons_per_layer[i+1]];
        }
    }

    for (int i = 0; i < num_layers-1; ++i){
        for (int j = 0; j < neurons_per_layer[i]+1; ++j){
            for (int k = 0; k < neurons_per_layer[i+1]; ++k){
                weight_matrices[i][j][k] = (double) rand() / RAND_MAX;
            }
        }
    }

    // Imprimir valores de la funcion de pesos y células.
    for (int i = 0; i < num_layers-1; ++i){
        cout << "weight Matrix: " << endl;
        for (int j = 0; j < neurons_per_layer[i]+1; ++j){
            for (int k = 0; k < neurons_per_layer[i+1]; ++k){
                cout << weight_matrices[i][j][k] << " ";
            }
            cout << endl;
        }
    }

    for (int i = 0; i < num_layers-1; ++i){
        cout << "Layer <" << neurons_per_layer[i+1] << ">" << endl;
        for (int j = 0; j < neurons_per_layer[i+1]; ++j){
            cout << "cell <"<< i << "," << j <<">: " << net[i][j].delta << " : " << net[i][j].output << endl;
        }
    }
    
    //cout << "Comenzar el ciclo...\n";
    auto start = std::chrono::system_clock::now();
    // For each cycle...
    for (int c = 0; c < cycles; ++c){
        // For each labeled example...
        for (size_t p = 0; p < parser.ref_to_examples_.get()->size(); ++p){
            
            //cout << "Paso1\n";
            // STEP 1: Propagate the entry through the net.
            for (int i = 0; i < num_layers-1; ++i){
                for (int j = 0; j < neurons_per_layer[i+1]; ++j){
                    if (i == 0){
                        // tomar el tamaño de la entrada si primera capa.
                        for (size_t k = 0; k < (parser.ref_to_examples_.get()->operator[](p).inputs.size()+1); ++k){
                            //cout << "valor" << k << "comprobacion puntero\n";
                            //cout << "tamaño patron entrada: " << parser.ref_to_examples_.get()->operator[](p).inputs.size() << "\n";
                            if (k == parser.ref_to_examples_.get()->operator[](p).inputs.size()){
                                //cout << "input<" << 1 << "> * w<" << weight_matrices[i][k][j] << "> ";
                                net[i][j].output += weight_matrices[i][k][j];
                            }else{
                                //cout << "input<" << parser.ref_to_examples_.get()->operator[](p).inputs[k] << "> * w<" << weight_matrices[i][k][j] << "> ";
                                net[i][j].output += (parser.ref_to_examples_.get()->operator[](p).inputs[k] * weight_matrices[i][k][j]);
                            }
                        }
                    }else{
                        // tomar el tamaño de la capa anterior si capa != primera.
                        for(int k = 0; k < neurons_per_layer[i]+1; ++k){
                            // threshold.
                            if (k == neurons_per_layer[i]){
                                //cout << "input<" << 1 << "> * w<" << weight_matrices[i][k][j] << "> ";
                                net[i][j].output += weight_matrices[i][k][j];
                            }else{
                                //cout << "input<" << net[i-1][k].output << "> * w<" << weight_matrices[i][k][j] << "> ";
                                net[i][j].output += (net[i-1][k].output * weight_matrices[i][k][j]);
                            }
                        }
                    }
                    // todo: aqui decimos si se activan todas las neuronas o no.
                    // todo: En este caso, si LAYERS-2, solo las hidden; si LAYERS-1 todas.
                    if (i != num_layers-1){
                        //cout << "valor cell sin activ <" << i << "," << j << "> : " << net[i][j].output << "\n";
                        net[i][j].output = activation_function(net[i][j].output, activation_func);
                        //cout << "valor cell <" << i << "," << j << "> : " << net[i][j].output << "\n";
                    }
                }
            }

            //cout << "Paso2\n";
            // STEP 2: Figuring deltas out.
            for (int i = num_layers-1; i > 0; --i){
                for (int j = 0; j < neurons_per_layer[i]; ++j){
                    if (i == neurons_per_layer.size()-1){
                        //cout << "Paso2-1\n";
                        // deltas de la salida.
                        //cout << "obteniendo deltas capa salida.\n";
                        //cout << "deseado: " << desired_output[j] << " salida: " << net[i-1][j].output << " derivada, func act: " << delta_activation_function(net[i-1][j].output, activation_func) << "\n";
                        //cout << "deseado: " << parser.ref_to_examples_.get()->operator[](p).desired[j]  << "\n";
                        net[i-1][j].delta = ((parser.ref_to_examples_.get()->operator[](p).desired[j] - net[i-1][j].output) * delta_activation_function(net[i-1][j].output, activation_func));
                        //cout << "valor final delta: " << net[i-1][j].delta << "\n";
                    }else{
                        //cout << "Paso2-2\n";
                        double sum = 0;
                        //cout << "obteniendo delta capas ocultas:"<< i << ".\n";
                        // tomar las dimensiones de la capa anterior.
                        for (int z = 0; z < neurons_per_layer[i+1]; ++z){
                            //cout << "w: " << weight_matrices[i][j][z] << " delta anterior: " << net[i][z].delta << "\n";
                            sum += (weight_matrices[i][j][z] * net[i][z].delta);
                        }
                        net[i-1][j].delta = (delta_activation_function(net[i-1][j].output, activation_func) * sum);
                        //cout << "salida cell<" << i-1 << "," << j << " : " << net[i-1][j].output << " func. act: " << delta_activation_function(net[i-1][j].output, activation_func) << " valor final delta: " << net[i-1][j].delta << "\n";
                    }
                }
            }
            //cout << "Paso3\n";
            // STEP 3: update weights
            for(size_t i = 0; i < weight_matrices.size(); ++i){
                for (int j = 0; j < neurons_per_layer[i]+1; ++j){
                    for (int k = 0; k < neurons_per_layer[i+1]; ++k){
                        // update thresholds.
                        if (j == neurons_per_layer[i]){
                            //cout << "peso: " << weight_matrices[i][j][k]; 
                            //cout << " lr(" << lr << ") + delta(" << net[i][k].delta << "): " << (lr * net[i][k].delta);
                            weight_matrices[i][j][k] += (lr * net[i][k].delta);
                            //cout << " resultado: " << weight_matrices[i][j][k] << endl;
                        }else{
                            // update weights.
                            if (i == 0){
                                //cout << "peso: " << weight_matrices[i][j][k]; 
                                //cout << " lr ("<< lr << ") + entrada("<< input[j] << ") + delta capa destino("<< net[i][k].delta <<") : " << (lr * input[j] * net[i][k].delta);
                                weight_matrices[i][j][k] += (lr * parser.ref_to_examples_.get()->operator[](p).inputs[j] * net[i][k].delta);
                                //cout << " resultado: " << weight_matrices[i][j][k] << endl;
                            }else{
                                //cout << "peso: " << weight_matrices[i][j][k]; 
                                //cout << " lr ("<< lr << ") + salida capa anterior("<< net[i-1][j].output << ") + delta capa destino("<< net[i][k].delta <<") : " << (lr * net[i-1][j].output * net[i][k].delta);
                                weight_matrices[i][j][k] += (lr * net[i-1][j].output * net[i][k].delta);
                                //cout << " resultado: " << weight_matrices[i][j][k] << endl;
                            }
                        }
                    }
                }
            }
            //cout << "error individual\n";
            // calculate an entry error and store it.
            double error_i = 0;
            for (int i = 0; i < neurons_per_layer[num_layers-1] ; ++i){
                //cout << "desired: " << desired_output[i] << " output: " <<  net[num_layers-2][i].output << "\n";
                //cout << "error: " << pow(( desired_output[i] - net[num_layers-2][i].output), 2) << "\n";
                error_i += pow(( parser.ref_to_examples_.get()->operator[](p).desired[i] - net[num_layers-2][i].output), 2);
                //cout << "aumentando error: " << error_i << "\n";
            }
            errors.push_back(error_i*1/2);
            //cout << "error almacenado: " << errors.at(0) << "\n";
            //cout << "error: " << error_i << endl;
        }
        /*cout << "Evaluar error final\n";
        cout << "dimensiones: " << parser.ref_to_examples_.get()->size() << "\n";
        cout << "dimensiones error: " << errors.size() << "\n";
        cout << "prueba: " << parser.ref_to_examples_.get()->operator[]((parser.ref_to_examples_.get()->size() - 1)).desired[0] << "\n";
        cout << "prueba errores: " << errors[errors.size() - 1] << "\n";*/
        // Evaluar el error total.
        double total_sum = 0;
        for (size_t q = 0; q < errors.size(); q++){
            //cout << q << " \n";
            //cout << "error patron <" << q << "> : " << errors[q] << "\n"; 
            //cout << "tamaño: " << errors.size() << "\n";
            total_sum += errors[q];
            //cout << "total acumulado:" << total_sum << "\n";
        }
        //cout << "aqui";
        mse_per_epoch[c] = total_sum / errors.size();
        // vaciar los errores almacenados.
        errors.clear();
        cout << "MSE at epoch <"<< c << "> :" << mse_per_epoch[c] << "\n";
    }

    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double>duration = end - start;
    auto time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count();
    auto time_s = std::chrono::duration_cast<std::chrono::seconds>(duration).count();
    std::cout << "Elapsed time training the net:\t" << time_ns << " ns " << std::endl;
    std::cout << "Elapsed time training the net:\t" << time_s << " s " << std::endl;

    double min = *min_element(mse_per_epoch.begin(), mse_per_epoch.end());
    cout << "Minimum error generated:\t" << min << endl;

    /*
    // Print values: 
    // print weights
    for (int i = 0; i < num_layers-1; ++i){
        cout << "weight Matrix: " << endl;
        for (int j = 0; j < neurons_per_layer[i]+1; ++j){
            for (int k = 0; k < neurons_per_layer[i+1]; ++k){
                cout << weight_matrices[i][j][k] << " ";
            }
            cout << endl;
        }
    }

    // print net
    for (int i = 0; i < num_layers-1; ++i){
        cout << "Layer <" << neurons_per_layer[i+1] << ">" << endl;
        for (int j = 0; j < neurons_per_layer[i+1]; ++j){
            cout << "cell <"<< i << "," << j <<">: " << net[i][j].delta << " : " << net[i][j].output << endl;
        }
    }*/

    /*
    
        Deallocate Memory.

    */
    for (int i = 0; i < num_layers-1; ++i){
        for (int j = 0; j < neurons_per_layer[i]+1; ++j){
            delete [] weight_matrices[i][j];
        }
        delete [] weight_matrices[i];
        delete [] net[i];
    }

    return 0;
}


