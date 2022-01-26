#include <iostream>
#include <array>
#include <cmath>
#include <chrono>
#include <vector>
#include <fstream>
#include <sstream>
#include "../headers/CSVReaderANN.hpp"

#define SIGMOID_FUNCTION "SIGMOID"
#define RELU_FUNCTION "RELU"
#define TANH_FUNCTION "TANH"

using namespace std;

struct Neuron{
    double delta;
    double output;
};

double activation_function(double value, string func, bool derivative);

double activation_function(double value, string func, bool derivative){
    double result;
    if (!derivative){
        if (func == RELU_FUNCTION){
            (value < 0) ? result = 0 : result = value;
        }else if (func == TANH_FUNCTION){
            result = tanh(value);
        }else if (func == SIGMOID_FUNCTION){
            result = ( 1 / (1 + exp(-value)));
        }
    }else{
        if (func == RELU_FUNCTION){
            result = 1;
        }else if (func == TANH_FUNCTION){
            result = (1 - pow(value, 2));
        }else if (func == SIGMOID_FUNCTION){
            result = ( value * (1 - value));
        }
    }
    return result;
}

/*
    Vars & Messages to handle arguments. 
*/

const int MIN_ARGS = 9;
const int MIN_LAYERS = 3;
const int NUM_PARAMETERS_TILL_LAYERS = 6;
const string USAGE_MESSAGE = "\n\n\t\tHow to use Backpropagation implementation:\n\n\tOnce it has been compiled."
    "You must pass the following parameters in order:\n\t\t1-.)\tcsv file containing the data to be trained.\n"
    "\t\t2-.)\tLearning rate [0,1]\n\t\t3-.)\tCycles, number of epochs to be performed.\n\t\t4-.)\tActivation "
    "function for all neurons, choose one of these {RELU, SIGMOID, TANH}\n\t\t5-.)\t"
    "Number of Layers, the minimum number of layers are three. At least there will be a hidden layer.\n\t\t"    
    "*-.)\tFor each layer, introduce its number of neurons. For example, if there are 3 layers, and it's introduced: 4 5 1"
    " the program will assign first layer (input) 4 neuron/s, second layer (hidden) 5 neuron/s and third layer (output) 1 neuron/s.\n\n\n";
const string ACT_FUNCTION_MESSAGE = "Invalid Activation function. Choose one of these {RELU, SIGMOID, TANH}.\n";
const string LR_MESSAGE = "Invalid learning rate. Take a learning rate that belongs this range [0,1]\n";

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
        if (lr > 1 || lr < 0){
            cerr << LR_MESSAGE;
            return -1;
        }
        cycles = atoi(argv[3]);
        activation_func = argv[4];
        if (activation_func != RELU_FUNCTION && activation_func != SIGMOID_FUNCTION && activation_func != TANH_FUNCTION){
            cerr << ACT_FUNCTION_MESSAGE;
            return -1;
        }
        num_layers = atoi(argv[5]);
        if ((num_layers < MIN_LAYERS) || (num_layers > MIN_LAYERS && (num_layers != (argc - NUM_PARAMETERS_TILL_LAYERS)))){
            cerr << USAGE_MESSAGE;
            return -1;
        }
        for (int i = 0; i < num_layers; i++){
            neurons_per_layer.push_back(atoi(argv[NUM_PARAMETERS_TILL_LAYERS+i]));
        }
        
        // Printout traces
        /*
        cout << "filename: " << filename << "\nlearning rate: " << lr << "\n";
        cout << "cycles: " << cycles << "\nactivation_func: " << activation_func << "\n";
        cout << "num layers: " << num_layers << "\n";
        for (size_t i = 0; i < neurons_per_layer.size(); ++i){
            cout << "layer[" << i << "] has <" << neurons_per_layer[i] << "> neurons\n";
        }*/
    }else{
        cerr << USAGE_MESSAGE;
        return -1;
    }

    /*

        Get data from the csv file.

    */

    CSVReaderANN parser {filename,  neurons_per_layer[0], neurons_per_layer[neurons_per_layer.size()-1]};
    
    /*
    
        Represent the net.

    */
    
    vector<Neuron*> net;
    vector<double**> weight_matrices;
    vector<double> mse_per_epoch;  // stores the MSE for all examples in one cycle.
    //vector<double> mse_epochs; // stores the MSE for all epochs.

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
        for (int j = 0; j < neurons_per_layer[i]+1; ++j){
            weight_matrices[i][j] = new double [neurons_per_layer[i+1]];
        }
    }

    // Pseudorandom generator.
    srand((unsigned) time (NULL));
    for (int i = 0; i < num_layers-1; ++i){
        for (int j = 0; j < neurons_per_layer[i]+1; ++j){
            for (int k = 0; k < neurons_per_layer[i+1]; ++k){
                weight_matrices[i][j][k] = (double) rand() / RAND_MAX;
            }
        }
    }
    ofstream ofs;
    ofs.open("../results/output2.csv");
    ofs << "Iter" << "," << "MSE" << "\n";
    //auto start = std::chrono::system_clock::now();
    // For each cycle...
    for (int c = 0; c < cycles; ++c){
        // First clear last epoch MSE's
        mse_per_epoch.clear();
        // For each labeled example...
        for (size_t p = 0; p < parser.ref_to_examples_.get()->size(); ++p){
            
            // STEP 0: Reinitiate all cells values that were previously stored.
            for (int i = 0; i < num_layers-1; ++i){
                for (int j = 0; j < neurons_per_layer[i+1]; ++j){
                    net[i][j].output = 0;
                }
            }

            // STEP 1: Propagate the entry through the net.
            for (int i = 0; i < num_layers-1; ++i){
                for (int j = 0; j < neurons_per_layer[i+1]; ++j){
                    if (i == 0){
                        // Take entry size if it's the entry layer.
                        for (size_t k = 0; k < (parser.ref_to_examples_.get()->operator[](p).inputs.size()+1); ++k){
                            if (k == parser.ref_to_examples_.get()->operator[](p).inputs.size()){
                                // threshold
                                net[i][j].output += weight_matrices[i][k][j];
                            }else{
                                net[i][j].output += (parser.ref_to_examples_.get()->operator[](p).inputs[k] * weight_matrices[i][k][j]);
                            }
                        }
                    }else{
                        // Take the previous layer size for the rest of the layers.
                        for(int k = 0; k < neurons_per_layer[i]+1; ++k){
                            // threshold.
                            if (k == neurons_per_layer[i]){
                                net[i][j].output += weight_matrices[i][k][j];
                            }else{
                                net[i][j].output += (net[i-1][k].output * weight_matrices[i][k][j]);
                            }
                        }
                    }
                    /*  IMPORTANT!: Here you can decide which activation function you 
                        want to apply to each cell. It could be implemented a softmax 
                        as well. Note that: num_layers-2 just hidden cells;
                        num_layers-1 takes all cells.
                    */
                    if (i != num_layers-1){
                        net[i][j].output = activation_function(net[i][j].output, activation_func, false);
                    }
                }
            }

            // STEP 2: Figuring deltas out.
            for (int i = num_layers-1; i > 0; --i){
                for (int j = 0; j < neurons_per_layer[i]; ++j){
                    if (i == neurons_per_layer.size()-1){                        
                        net[i-1][j].delta = ((net[i-1][j].output - parser.ref_to_examples_.get()->operator[](p).desired[j]) * activation_function(net[i-1][j].output, activation_func, true));
                    }else{
                        double sum = 0;
                        for (int z = 0; z < neurons_per_layer[i+1]; ++z){
                            sum += (weight_matrices[i][j][z] * net[i][z].delta);
                        }
                        net[i-1][j].delta = (activation_function(net[i-1][j].output, activation_func, true) * sum);
                    }
                }
            }
            // STEP 3: update weights
            for(size_t i = 0; i < weight_matrices.size(); ++i){
                for (int j = 0; j < neurons_per_layer[i]+1; ++j){
                    for (int k = 0; k < neurons_per_layer[i+1]; ++k){
                        // update thresholds.
                        if (j == neurons_per_layer[i]){
                            weight_matrices[i][j][k] -= (lr * net[i][k].delta);
                        }else{
                            // update weights.
                            if (i == 0){
                                weight_matrices[i][j][k] -= (lr * parser.ref_to_examples_.get()->operator[](p).inputs[j] * net[i][k].delta);
                            }else{
                                weight_matrices[i][j][k] -= (lr * net[i-1][j].output * net[i][k].delta);
                            }
                        }
                    }
                }
            }

            // STEP 4: Figure MSE out.
            double example_error = 0;
            for (int i = 0; i < neurons_per_layer[num_layers-1] ; ++i){
                example_error += pow(( parser.ref_to_examples_.get()->operator[](p).desired[i] - net[num_layers-2][i].output), 2);
            }
            mse_per_epoch.push_back(example_error * 1 / neurons_per_layer[num_layers-1]);
        }
        // STEP 5: Figure total MSE out. It takes into account every single error.
        double total_Err = 0;
        for(size_t i = 0; i < mse_per_epoch.size(); ++i){
            total_Err += mse_per_epoch[i];
        }
        ofs << c << "," << total_Err / mse_per_epoch.size() << "\n";
        //cout << "Total MSE error: " << total_Err / mse_per_epoch.size() << " at epoch<" << c << ">." << "\n";
        //mse_epochs.push_back(total_Err / mse_per_epoch.size());
    }

    // In order to measure the performance.
    /*auto end = std::chrono::system_clock::now();
    std::chrono::duration<double>duration = end - start;
    auto time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count();
    auto time_s = std::chrono::duration_cast<std::chrono::seconds>(duration).count();
    std::cout << "Elapsed time training the net:\t" << time_ns << " ns " << std::endl;
    std::cout << "Elapsed time training the net:\t" << time_s << " s " << std::endl;*/

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


