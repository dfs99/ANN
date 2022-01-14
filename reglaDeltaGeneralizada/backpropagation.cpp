#include <iostream>
#include <array>
#include <cmath>
#include <chrono>
#include <vector>
#include <fstream>
#include <sstream>


#define LAYERS 3
#define LEARNING_RATE 0.1
#define CYCLES 50
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
    if (ACTIVATION_FUNCTION == "RELU"){
        (value < 0) ? result = 0 : result = value;
    }else if (ACTIVATION_FUNCTION == "TANH"){
        result = tanh(value);
    }else if (ACTIVATION_FUNCTION == "SIGMOID"){
        result = ( 1 / (1 + exp(-value)));
    }
    return result;
}

double delta_activation_function(double value){
    double result;
    if (ACTIVATION_FUNCTION == "RELU"){
        result = 1;
    }else if (ACTIVATION_FUNCTION == "TANH"){
        result = (1 - pow(value, 2));
    }else if (ACTIVATION_FUNCTION == "SIGMOID"){
        result = ( value * (1 - value));
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
    /*for (int i = 0; i < 5; ++i){
        for (int j = 0; j < data[i].size(); ++j){
            cout << data[i][j] << " ";
        }
        cout << endl;
    }*/


    // topology
    array<int, LAYERS> topology = {6, 6, 3};
    array<Neuron*, LAYERS-1> net;
    array<double**, LAYERS-1> weight_matrices;

    vector<double> errors;
    array<double, CYCLES> mce_per_epoch;

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

    // place weights
    // generate random weights
    for (int i = 0; i < LAYERS-1; ++i){
        for (int j = 0; j < topology[i]+1; ++j){
            for (int k = 0; k < topology[i+1]; ++k){
                weight_matrices[i][j][k] = (double) rand() / RAND_MAX;
            }
        }
    }
    
    /*array<double, 3> input = {0.5, 0.5, 0.5};
    array<double, 3> desired_output = {1, 0, 0};*/

    auto start = std::chrono::system_clock::now();
    // iterate C cycles the whole train set.
    for (int c = 0; c < CYCLES; ++c){
        // introduce each entry.
        for (int p = 0; p < data.size(); ++p){

            // extraer del patrón de entrada
            // TODO: mejorar, se puede hacer mejor...
            array<double, ENTRY_DIMENSION> input;
            array<double, OUTPUT_DIMENSION> desired_output;
            for (int e = 0; e < data[p].size(); ++e){
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
                    for (int k = 0; k < input.size(); ++k){
                        if (i == 0){
                            // first, get data from input, otherwise from other cells.
                            net[i][j].output += (input[k] * weight_matrices[i][k][j]);
                        }else{
                            net[i][j].output += (net[i-1][k].output * weight_matrices[i][k][j]);
                        }
                    }
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
                            sum += (weight_matrices[i][j][z] * net[i][z].delta);
                        }
                        net[i-1][j].delta = (delta_activation_function(net[i-1][j].output) * sum);
                    }
                }
            }

            // update weights
            for(int i = 0; i < weight_matrices.size(); ++i){
                for (int j = 0; j < topology[i]+1; ++j){
                    for (int k = 0; k < topology[i+1]; ++k){
                        // update thresholds.
                        if (j == topology[i]){
                            weight_matrices[i][j][k] += (LEARNING_RATE * net[i][k].delta);
                        }else{
                            // update weights.
                            if (i == 0){
                                weight_matrices[i][j][k] += (LEARNING_RATE * input[j] * net[i][k].delta);
                            }else{
                                weight_matrices[i][j][k] += (LEARNING_RATE * net[i-1][j].output * net[i][k].delta);
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
        for (int q = 0; q < errors.size(); ++q){
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
    }

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


