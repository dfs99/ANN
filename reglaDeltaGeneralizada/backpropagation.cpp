#include <iostream>
#include <array>
#include <cmath>
#include <chrono>
#include <vector>
#include <fstream>
#include <sstream>
#include <bits/stdc++.h>    // para la funcion min.


#define LAYERS 3
#define LEARNING_RATE 0.2
#define CYCLES 1000
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


    // Values to represent the net.
    array<int, LAYERS> topology = {6, 4, 3};
    array<Neuron*, LAYERS-1> net;
    array<double**, LAYERS-1> weight_matrices;
    vector<double> errors;  // almacena todos los errores cometidos por los patrones para 1 ciclo.
    array<double, CYCLES> mse_per_epoch;    // almacena el mse de cada ciclo / epoch.

    // initializate net.
    for (int i = 0; i < LAYERS-1; ++i){
        net[i] = new Neuron [topology[i+1]];
        for (int j = 0; j < topology[i+1]; ++j){
            net[i][j].delta = 0;
            net[i][j].output = 0;
        }
    }

    // generate weight matrices with thresholds.
    for (int i = 0; i < LAYERS-1; ++i){
        // el +1 de topology[i] es para reservar memoria para los umbrales.
        weight_matrices[i] = new double* [topology[i]+1];
        for (int j = 0; j < topology[i]+1; ++j){
            // coges la dimensión de la siguiente capa para la matriz de pesos.
            weight_matrices[i][j] = new double [topology[i+1]];
        }
    }
    // generate random weights
    for (int i = 0; i < LAYERS-1; ++i){
        for (int j = 0; j < topology[i]+1; ++j){
            for (int k = 0; k < topology[i+1]; ++k){
                weight_matrices[i][j][k] = (double) rand() / RAND_MAX;
            }
        }
    }

    // Imprimir valores de la funcion de pesos y células.
    /*for (int i = 0; i < LAYERS-1; ++i){
        cout << "weight Matrix: " << endl;
        for (int j = 0; j < topology[i]+1; ++j){
            for (int k = 0; k < topology[i+1]; ++k){
                cout << weight_matrices[i][j][k] << " ";
            }
            cout << endl;
        }
    }

    for (int i = 0; i < LAYERS-1; ++i){
        cout << "Layer <" << topology[i+1] << ">" << endl;
        for (int j = 0; j < topology[i+1]; ++j){
            cout << "cell <"<< i << "," << j <<">: " << net[i][j].delta << " : " << net[i][j].output << endl;
        }
    }*/
    

    auto start = std::chrono::system_clock::now();
    // iterate C cycles the whole train set.
    for (int c = 0; c < CYCLES; ++c){
        // introduce each entry.
        for (size_t p = 0; p < data.size(); ++p){
            // extraer del patrón de entrada
            // TODO: mejorar, se puede hacer mejor...
            array<double, ENTRY_DIMENSION> input;
            array<double, OUTPUT_DIMENSION> desired_output;
            // cout << "patron de entrada:\n"; 
            for (size_t e = 0; e < data[p].size(); ++e){
                if (e < ENTRY_DIMENSION){
                    input[e] = data[p][e];
                    //cout << input[e] << "\n";
                }else{
                    desired_output[e-ENTRY_DIMENSION] = data[p][e];
                   //cout << desired_output[e-ENTRY_DIMENSION] << "\n";
                }
            }
 
            // STEP 1: Propagate the entry through the net.
            for (int i = 0; i < LAYERS-1; ++i){
                for (int j = 0; j < topology[i+1]; ++j){
                    if (i == 0){
                        // tomar el tamaño de la entrada si primera capa.
                        for (size_t k = 0; k < input.size()+1; ++k){
                            if (k == input.size()){
                                //cout << "input<" << 1 << "> * w<" << weight_matrices[i][k][j] << "> ";
                                net[i][j].output += weight_matrices[i][k][j];
                            }else{
                                //cout << "input<" << input[k] << "> * w<" << weight_matrices[i][k][j] << "> ";
                                net[i][j].output += (input[k] * weight_matrices[i][k][j]);
                            }
                        }
                    }else{
                        // tomar el tamaño de la capa anterior si capa != primera.
                        for(int k = 0; k < topology[i]+1; ++k){
                            // threshold.
                            if (k == topology[i]){
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
                    if (i != LAYERS-1){
                        //cout << "valor cell sin activ <" << i << "," << j << "> : " << net[i][j].output << "\n";
                        net[i][j].output = activation_function(net[i][j].output);
                        //cout << "valor cell <" << i << "," << j << "> : " << net[i][j].output << "\n";
                    }
                }
            }

            // STEP 2: Figuring deltas out.
            for (int i = LAYERS-1; i > 0; --i){
                for (int j = 0; j < topology[i]; ++j){
                    if (i == topology.size()-1){
                        // deltas de la salida.
                        //cout << "obteniendo deltas capa salida.\n";
                        //cout << "deseado: " << desired_output[j] << " salida: " << net[i-1][j].output << " derivada, func act: " << delta_activation_function(net[i-1][j].output) << "\n";
                        net[i-1][j].delta = ((desired_output[j] - net[i-1][j].output) * delta_activation_function(net[i-1][j].output));
                        //cout << "valor final delta: " << net[i-1][j].delta << "\n";
                    }else{
                        double sum = 0;
                        //cout << "obteniendo delta capas ocultas:"<< i << ".\n";
                        // tomar las dimensiones de la capa anterior.
                        for (int z = 0; z < topology[i+1]; ++z){
                            //cout << "w: " << weight_matrices[i][j][z] << " delta anterior: " << net[i][z].delta << "\n";
                            sum += (weight_matrices[i][j][z] * net[i][z].delta);
                        }
                        net[i-1][j].delta = (delta_activation_function(net[i-1][j].output) * sum);
                        //cout << "salida cell<" << i-1 << "," << j << " : " << net[i-1][j].output << " func. act: " << delta_activation_function(net[i-1][j].output) << " valor final delta: " << net[i-1][j].delta << "\n";
                    }
                }
            }

            // STEP 3: update weights
            for(size_t i = 0; i < weight_matrices.size(); ++i){
                for (int j = 0; j < topology[i]+1; ++j){
                    for (int k = 0; k < topology[i+1]; ++k){
                        // update thresholds.
                        if (j == topology[i]){
                            //cout << "peso: " << weight_matrices[i][j][k]; 
                            //cout << " lr(" << LEARNING_RATE << ") + delta(" << net[i][k].delta << "): " << (LEARNING_RATE * net[i][k].delta);
                            weight_matrices[i][j][k] += (LEARNING_RATE * net[i][k].delta);
                            //cout << " resultado: " << weight_matrices[i][j][k] << endl;
                        }else{
                            // update weights.
                            if (i == 0){
                                //cout << "peso: " << weight_matrices[i][j][k]; 
                                //cout << " lr ("<< LEARNING_RATE << ") + entrada("<< input[j] << ") + delta capa destino("<< net[i][k].delta <<") : " << (LEARNING_RATE * input[j] * net[i][k].delta);
                                weight_matrices[i][j][k] += (LEARNING_RATE * input[j] * net[i][k].delta);
                                //cout << " resultado: " << weight_matrices[i][j][k] << endl;
                            }else{
                                //cout << "peso: " << weight_matrices[i][j][k]; 
                                //cout << " lr ("<< LEARNING_RATE << ") + salida capa anterior("<< net[i-1][j].output << ") + delta capa destino("<< net[i][k].delta <<") : " << (LEARNING_RATE * net[i-1][j].output * net[i][k].delta);
                                weight_matrices[i][j][k] += (LEARNING_RATE * net[i-1][j].output * net[i][k].delta);
                                //cout << " resultado: " << weight_matrices[i][j][k] << endl;
                            }
                        }
                    }
                }
            }
            // calculate an entry error and store it.
            double error_i = 0;
            for (int i = 0; i < topology[LAYERS-1] ; ++i){
                //cout << "desired: " << desired_output[i] << " output: " <<  net[LAYERS-2][i].output << "\n";
                //cout << "error: " << pow(( desired_output[i] - net[LAYERS-2][i].output), 2) << "\n";
                error_i += pow(( desired_output[i] - net[LAYERS-2][i].output), 2);
                //cout << "aumentando error: " << error_i << "\n";
            }
            errors.push_back(error_i*1/2);
            //cout << "error almacenado: " << errors.at(0) << "\n";
            //cout << "error: " << error_i << endl;
        }
        
        // Evaluar el error total.
        double total_sum = 0;
        for (size_t q = 0; q < errors.size(); ++q){
            //cout << "error patron <" << q << "> : " << errors[q] << "\n"; 
            //cout << "tamaño: " << errors.size() << "\n";
            total_sum += errors[q];
        }
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
    for (int i = 0; i < LAYERS-1; ++i){
        cout << "weight Matrix: " << endl;
        for (int j = 0; j < topology[i]+1; ++j){
            for (int k = 0; k < topology[i+1]; ++k){
                cout << weight_matrices[i][j][k] << " ";
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
    }*/

    // free memory
    for (int i = 0; i < LAYERS-1; ++i){
        for (int j = 0; j < topology[i]+1; ++j){
            delete [] weight_matrices[i][j];
        }
        delete [] weight_matrices[i];
        delete [] net[i];
    }

    return 0;
}


