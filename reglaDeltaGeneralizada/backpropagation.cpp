#include <iostream>
#include <array>

#define LAYERS 3
#define LEARNING_RATE 0.1
#define ACTIVATION_FUNCTION "RELU"

using namespace std;

struct Neuron{
    double delta;
    double output;
};


int main(){

    // topology
    array<int, LAYERS> topology = {3, 2, 3};


    array<Neuron*, LAYERS-1> net;
    // initializate net.
    for (int i = 0; i < LAYERS-1; ++i){
        net[i] = new Neuron [topology[i+1]];
        for (int j = 0; j < topology[i+1]; ++j){
            net[i][j].delta = 0;
            net[i][j].output = 0;
        }
    }

    // generate weight matrices with thresholds.
    array<double**, LAYERS-1> weigh_matrices;
    for (int i = 0; i < LAYERS-1; ++i){
        weigh_matrices[i] = new double* [topology[i]+1];
        for (int j = 0; j < topology[i]+1; ++j){
            weigh_matrices[i][j] = new double [topology[i+1]];
        }
    }

    // 


    // place weights
    // generate random weights
    for (int i = 0; i < LAYERS-1; ++i){
        for (int j = 0; j < topology[i]+1; ++j){
            for (int k = 0; k < topology[i+1]; ++k){
                weigh_matrices[i][j][k] = (double) rand() / RAND_MAX;
            }
        }
    }
    // place fixed weights.
    /*weigh_matrices[0][0][0] = 0.1;
    weigh_matrices[0][0][1] = -0.1;
    weigh_matrices[0][1][0] = 0;
    weigh_matrices[0][1][1] = 0.2;
    weigh_matrices[0][2][0] = -0.2;
    weigh_matrices[0][2][1] = 0.1;
    // and thresholds.
    weigh_matrices[0][3][0] = 0;
    weigh_matrices[0][3][1] = 0;

    weigh_matrices[1][0][0] = 0.5;
    weigh_matrices[1][0][1] = 0;
    weigh_matrices[1][0][2] = 0.1;
    weigh_matrices[1][1][0] = -0.3;
    weigh_matrices[1][1][1] = 0.1;
    weigh_matrices[1][1][2] = -0.1; 
    // and thresholds.
    weigh_matrices[1][2][0] = 0;
    weigh_matrices[1][2][1] = 0;
    weigh_matrices[1][2][2] = 0;*/


    // given an input, propagate it through the net.
    array<double, 3> input = {0.5, 0.5, 0.5};
    for (int i = 0; i < LAYERS-1; ++i){
        for (int j = 0; j < topology[i+1]; ++j){
            // loop over input 
            for (int k = 0; k < 3; ++k){
                if (i == 0){
                    // first, get data from input, otherwise from other cells.
                    net[i][j].output += (input[k] * weigh_matrices[i][k][j]);
                }else{
                    net[i][j].output += (net[i-1][k].output * weigh_matrices[i][k][j]);
                }
            }
            // applying RELU only in hidden
            if (net[i][j].output < 0.0 && i != (LAYERS-2)){ 
                net[i][j].output = 0;
            }
        }
    }

    // print weights
    for (int i = 0; i < LAYERS-1; ++i){
        cout << "Matrix: " << endl;
        for (int j = 0; j < topology[i]+1; ++j){
            for (int k = 0; k < topology[i+1]; ++k){
                cout << weigh_matrices[i][j][k] << " ";
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
        delete net[i];
        for (int j = 0; j < topology[i]; ++j){
            delete [] weigh_matrices[i][j];
        }
        delete weigh_matrices[i];
    }
 

    return 0;
}


