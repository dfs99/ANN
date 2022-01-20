#include <iostream>
#include <vector>

using namespace std;

/*
    Vars to handle errors. 
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


int main(int argc, char **argv){
    
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

    return 0;
}