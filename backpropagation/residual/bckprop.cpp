/*
    Unfixed version of backpropagation using vector and unique ptrs instead
    of new and delete to avoid memory leaks.
    todo: arreglar propagación de patrones por la red. En backpropagation.cpp
    si se encuentra solucionado. SIn embargo, me sale distinto a keras en Python.
*/
#include <iostream>
#include <array>
#include <cmath>
#include <chrono>
#include <vector>
#include <fstream>
#include <sstream>
#include <memory>

#include <bits/stdc++.h>


#define LAYERS 3
#define LEARNING_RATE 0.001
#define CYCLES 1
#define ACTIVATION_FUNCTION "SIGMOID"
#define ENTRY_DIMENSION 6
#define OUTPUT_DIMENSION 3

using namespace std;

struct Neuron{
    double delta;
    double output;
};

double activation_function(double value);
double delta_activation_function(double value);

double activation_function(double value){
    double result;
    const string relu = "RELU";
    const string hiperbolic_tan = "TANH";
    const string sigmoid = "SIGMOID";
    if (ACTIVATION_FUNCTION == relu){
        (value < 0) ? result = 0 : result = value;
    }else if (ACTIVATION_FUNCTION == hiperbolic_tan){
        result = tanh(value);
    }else if (ACTIVATION_FUNCTION == sigmoid){
        result = ( 1 / (1 + exp(-value)));
    }else{
        cerr << "ERROR: Use any of these Activation Functions {RELU, TANH, SIGMOID}.\n";
        return -1000;
    }
    return result;
}

double delta_activation_function(double value){
    double result;
    const string relu = "RELU";
    const string hiperbolic_tan = "TANH";
    const string sigmoid = "SIGMOID";
    if (ACTIVATION_FUNCTION == relu){
        result = 1;
    }else if (ACTIVATION_FUNCTION == hiperbolic_tan){
        result = (1 - pow(value, 2));
    }else if (ACTIVATION_FUNCTION == sigmoid){
        result = ( value * (1 - value));
    }else{
        cerr << "ERROR: Use any of these Activation Functions {RELU, TANH, SIGMOID}.\n";
        return -1000;
    }
    return result;
}



int main(){
    /**
     * Usando robots.csv, tiene 6 entradas y 3 salidas. 
     * 
     * 
     * 
     */

    // read data
    cout << "Fetching data from csv..." << endl;
    /*
    Labels are stored apart from data to maintain the order.
    Data contains all values extracted from csv.
    */

    vector<vector<double>>data;
    vector<string>labels;

    /* Aux variables to extract data.*/
    string line, word;
    int number_lines = 0;    
    ifstream ifs;
    ifs.open("robots.csv");
    //todo: falla si dejo un espacio vacio al final de data.csv
    while(!ifs.eof()){
        getline(ifs, line);
        stringstream ss(line);
        if (number_lines == 0){
            while(getline(ss, word, ',')){
                labels.push_back(word);
            }
        }else{
            std::vector<double> current_entry;
            while(std::getline(ss, word, ',')){
                current_entry.push_back(stod(word));
            }
            data.push_back(current_entry);
        }
        number_lines++;
    }
    ifs.close();

    std::cout << "Data successfully fetched." << std::endl;
   


    // topology
    array<int, LAYERS> topology = {6, 6, 3};
    array<vector<Neuron>, LAYERS-1> net;
    array<vector<unique_ptr<vector<double>>>, LAYERS-1> weight_matrices;

    vector<double> errors;
    array<double, CYCLES> mce_per_epoch;

    
    // NUEVA
    for (int i = 0; i < LAYERS-1; ++i){
        vector<Neuron> layer_i;
        for (int j = 0; j < topology[i+1]; ++j){
            Neuron neuron_j {0, 0};
            layer_i.push_back(neuron_j);
        }
        net[i] = layer_i;
    }

    // nueva
    for (int i = 0; i < LAYERS-1; ++i){
        vector<unique_ptr<vector<double>>> matrix_i;
        for (int j = 0; j < topology[i]+1; ++j){
            unique_ptr<vector<double>> col_j = make_unique<vector<double>>();
            for (int k = 0; k < topology[i+1]; ++k){
                col_j->push_back( (double) rand() / RAND_MAX );
            }
            // move pointer cause it cannot be copied.
            matrix_i.push_back(move(col_j));
        }
        // using move due to unique ptr.
        weight_matrices[i] = move(matrix_i);
    }


    auto start = std::chrono::system_clock::now();
    // iterate C cycles the whole train set.
    for (int c = 0; c < CYCLES; ++c){
        // introduce each entry.
        for (size_t p = 0; p < data.size(); ++p){

            // extraer del patrón de entrada
            // TODO: mejorar, se puede hacer mejor...
            array<double, ENTRY_DIMENSION> input;
            array<double, OUTPUT_DIMENSION> desired_output;
            for (size_t e = 0; e < data[p].size(); ++e){
                if (e < ENTRY_DIMENSION){
                    input[e] = data[p][e];
                }else{
                    desired_output[e-ENTRY_DIMENSION] = data[p][e];
                }
            }
 
            // propagate the entry through the net.
            for (int i = 0; i < LAYERS-1; ++i){
                for (int j = 0; j < topology[i+1]; ++j){
                    // loop over input 
                    for (size_t k = 0; k < input.size(); ++k){
                        if (i == 0){
                            // first, get data from input, otherwise from other cells.
                            cout << "input<" << input[k] << "> * w<" << weight_matrices[i][k]->operator[](j) << "> ";
                            net[i][j].output += (input[k] * weight_matrices[i][k]->operator[](j));
                        }else{
                            cout << "input<" << input[k] << "> * w<" << weight_matrices[i][k]->operator[](j) << "> ";
                            net[i][j].output += (net[i-1][k].output * weight_matrices[i][k]->operator[](j));
                        }
                    }
                    cout << "\n";
                    // todo: aqui decimos si se activan todas las neuronas o no.
                    // todo: En este caso, si LAYERS-2, solo las hidden; si LAYERS-1 todas.
                    if (i != LAYERS-1){
                        net[i][j].output = activation_function(net[i][j].output);
                    }
                }
            }

            // figuring deltas out.
            for (int i = LAYERS-1; i > 0; --i){
                for (int j = 0; j < topology[i]; ++j){
                    if (i == topology.size()-1){
                        net[i-1][j].delta = ((desired_output[j] - net[i-1][j].output) * delta_activation_function(net[i-1][j].output));
                    }else{
                        double sum = 0;
                        // tomar las dimensiones de la capa anterior.
                        for (int z = 0; z < topology[i+1]; ++z){
                            sum += (weight_matrices[i][j]->operator[](z) * net[i][z].delta);
                        }
                        net[i-1][j].delta = (delta_activation_function(net[i-1][j].output) * sum);
                    }
                }
            }

            // update weights
            for(size_t i = 0; i < weight_matrices.size(); ++i){
                for (int j = 0; j < topology[i]+1; ++j){
                    for (int k = 0; k < topology[i+1]; ++k){
                        // update thresholds.
                        if (j == topology[i]){
                            weight_matrices[i][j]->operator[](k) += (LEARNING_RATE * net[i][k].delta);
                        }else{
                            // update weights.
                            if (i == 0){
                                weight_matrices[i][j]->operator[](k) += (LEARNING_RATE * input[j] * net[i][k].delta);
                            }else{
                                weight_matrices[i][j]->operator[](k) += (LEARNING_RATE * net[i-1][j].output * net[i][k].delta);
                            }
                        }
                    }
                }
            }
            // calculate an entry error and store it.
            double error_i = 0;
            for (int i = 0; i < topology[LAYERS-1] ; ++i){
                error_i += pow(( desired_output[i] - net[LAYERS-2][i].output), 2); 
            }
            errors.push_back((0.5 * error_i));
            //cout << "error: " << error_i << endl;
        }
        
        // Evaluar el error total.
        double total_sum = 0;
        for (size_t q = 0; q < errors.size(); ++q){
            total_sum += errors[q];
        }
        mce_per_epoch[c] = total_sum / errors.size();
        // vaciar los errores almacenados.
        errors.clear();
        //cout << "MCE at epoch <"<< c << "> :" << mce_per_epoch[c] << "\n";
    }

    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double>duration = end - start;
    auto time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count();
    auto time_s = std::chrono::duration_cast<std::chrono::seconds>(duration).count();
    std::cout << "Elapsed time training the net:\t" << time_ns << " ns " << std::endl;
    std::cout << "Elapsed time training the net:\t" << time_s << " s " << std::endl;

    double min = *min_element(mce_per_epoch.begin(), mce_per_epoch.end());
    cout << "Minimum error generated:\t" << min << endl;


    // Print values: 
    // IMprimir valores de la nueva aproximación.
    for (int i = 0; i < LAYERS-1; ++i){
        cout << "weight Matrix using unique_ptr: " << endl;
        for (int j = 0; j < topology[i]+1; ++j){
            for (int k = 0; k < topology[i+1]; ++k){
                cout << weight_matrices[i][j]->operator[](k) << " ";
            }
            cout << endl;
        }
    }

    // print net
    for (int i = 0; i < LAYERS-1; ++i){
        cout << "Layer <" << topology[i+1] << ">" << endl;
        for (int j = 0; j < topology[i+1]; ++j){
            cout << "cell <"<< i << "," << j <<">: " << net[i][j].delta << " : " << net[i][j].output << endl;
        }
    }

    return 0;
}


