#include <iostream>
#include <chrono>
# include <array>

#define DIMENSION 20


std::array<double, DIMENSION> weights;

// initialize randomly weight values between 0 - 1. 
for (int i = 0; i < weights.size(); ++i){
    weights[i] = (double) ( rand() / RAND_MAX );
}

std::cout << "Printing weight elements..." << std::endl;
for (auto it = weights.begin(); it != weights.end(); ++it){
    std::cout << *it << " " << std::endl;
}

