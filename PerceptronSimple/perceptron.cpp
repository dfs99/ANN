#include <iostream>
#include <chrono>
#include <array>
#include <vector>

#define DIMENSION 2
#define LEARNING_RATE 0.7


int main(){
    /* weight vector has DIMENSION + 1, the last element is the threshold's weight. */
    std::array<double, (DIMENSION + 1) > weights;
    /*  entries vector has DIMENSION + 2. 
        The last element is the desided output. 
        The second last element is the value given to the threshold.
    */
    std::vector<std::array<double, (DIMENSION + 2) >> entries = {   {-1, -1, 1, -1}, 
                                                                    {1, -1, 1, -1},
                                                                    {-1, 1, 1, -1},
                                                                    {1, 1, 1, 1}
                                                                };

    for (int i = 0; i < weights.size(); ++i){       /* Initialize randomly weight values  */
        weights[i] =  (double) rand() / RAND_MAX;   /* between 0 - 1. Last element is the */
    }                                               /* threshold.                         */
    std::cout << "Initial weights: " << std::endl;
    for (auto it = weights.begin(); it != weights.end(); ++it){
        std::cout << *it << " ";
    }
    std::cout << std::endl;

    auto start = std::chrono::system_clock::now();
    for (int i = 0; i < entries.size(); ++i) {
        double y = 0.0;                              /* Initialize 'y' to store values    */
        for (int j = 0; j < entries[i].size()-1; ++j) {
            y += (weights[j] * entries[i][j]);
        }
        y = (y > 0.0) ? 1 : -1;                      /* Apply activation function to 'y'  */
        if (y != entries[i].back()){                 /* Whether weight update or not.     */
            for (int k = 0; k < weights.size(); ++k){
                weights[k] += (LEARNING_RATE * entries[i].back() * entries[i][k]); 
            }
        }
    }

    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double>duration = end - start;
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count();
    std::cout << "Elapsed time generating solution:\t" << time << " ns " << std::endl;

    std::cout << "Hyperplane that satisfies the solution:\t";
    for (int i  = 0; i < weights.size(); ++i){
        if (i == weights.size()-1) {
            std::cout << weights[i];
        } else {
            std::cout << weights[i] << "*x_" << weights.size()-i-1 << " + ";
        }
    }
    std::cout << std::endl;
    if (weights.size() == 3){
        std::cout << "Ecuation for x,y coordinates:" << std::endl;
        std:: cout << "y = " << (-weights[0] / weights[1]) << "x " << (-weights[2] / weights[1]) << std::endl;
    }

    return 0;

}



