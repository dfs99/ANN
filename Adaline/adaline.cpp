#include <iostream>
#include <chrono>
#include <array>
#include <vector>
#include <fstream>
#include <sstream>


#define DIMENSION 8
#define LEARNING_RATE 0.0001
#define NUM_CYCLES 200000


int main(){
    std::cout << "Fetching data from csv..." << std::endl;
    /*
    Labels are stored apart from data to maintain the order.
    Data contains all values extracted from csv.
    */

    std::vector<std::vector<double>>data;
    std::vector<std::string>labels;

    /* Aux variables to extract data.*/
    std::string line, word;
    int number_lines = 0;    
    std::ifstream ifs;
    ifs.open("data.csv");
    //todo: falla si dejo un espacio vacio al final de data.csv
    while(!ifs.eof()){
        getline(ifs, line);
        std::stringstream ss(line);
        if (number_lines == 0){
            while(std::getline(ss, word, ',')){
                labels.push_back(word);
            }
            labels.insert(labels.end() - 1, "Threshold");
        }else{
            std::vector<double> current_entry;
            while(std::getline(ss, word, ',')){
                current_entry.push_back(std::stod(word));
            }
            current_entry.insert(current_entry.end() - 1, 1.0);
            data.push_back(current_entry);
        }
        number_lines++;
    }
    ifs.close();
    std::cout << "Data successfully fetched." << std::endl;

    /* weight vector has DIMENSION + 1, the last element is the threshold's weight. */
    std::array<double, (DIMENSION + 1) > weights;
    for (int i = 0; i < weights.size(); ++i){       /* Initialize randomly weight values  */
        weights[i] =  (double) rand() / RAND_MAX;   /* between 0 - 1. Last element is the */
    }                                               /* threshold.                         */
    
    std::cout << "Initial weights: " << std::endl;
    for (auto it = weights.begin(); it != weights.end(); ++it){
        std::cout << *it << " ";
    }
    std::cout << std::endl;

    auto start = std::chrono::system_clock::now();
    for (int z = 0; z < NUM_CYCLES; ++z){
        for (int i = 0; i < data.size(); ++i) {
            double y = 0.0;                              
            for (int j = 0; j < data[i].size()-1; ++j) {
                y += (weights[j] * data[i][j]);
            }
            /* Applying Delta Rule */
            if (y != data[i].back()){                     
                double difference =  (data[i].back() - y);
                for (int k = 0; k < weights.size(); ++k){
                    weights[k] += (LEARNING_RATE * difference * data[i][k]); 
                }   
            }
        }
    }
    auto end = std::chrono::system_clock::now();

    std::chrono::duration<double>duration = end - start;
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count();
    std::cout << "Elapsed time generating solution using " << data.size() << " instances and "<< NUM_CYCLES << " cycles:\t" << time << " ns " << std::endl;
    for (int k = 0; k < weights.size(); ++k){
        std::cout << weights[k] << " "; 
    }
    std::cout << "\n";
    return 0;

}



