#include "../headers/CSVReaderANN.hpp"
#include <memory>
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>

const char* NON_EXISTING_FILE = "Error: csv does not exists in the directory.";

CSVReaderANN::CSVReaderANN(std::string filename, int input_dimension, int output_dimension){
    std::cout << "valor devuelto funcion: " << CSVReaderANN::exists(filename) << "\n";
    filename_ = CSVReaderANN::exists(filename) == true ? filename : throw CSVReaderANNException(NON_EXISTING_FILE);
    input_dimension_ = input_dimension;
    output_dimension_ = output_dimension;
    std::string line, word;    
    std::ifstream ifs;
    ifs.open(filename_);
    // get first line to remove data labels.
    getline(ifs, line);
    size_t counter;
    // todo: solucionar el eof
    while(!ifs.eof()){
        getline(ifs, line);
        std::stringstream ss(line);
        ExampleLabeled<double> current_example;
        counter = 0;
        while(std::getline(ss, word, ',')){
            if (counter < CSVReaderANN::get_input_dimension()){
                current_example.inputs.push_back(stod(word));
            }else{
                current_example.desired.push_back(stod(word));
            }
            counter++;
        }
        examples_.push_back(current_example);
    }
    ifs.close();
    ref_to_examples_ = std::make_unique<std::vector<ExampleLabeled<double>>>(examples_);
}

int CSVReaderANN::get_input_dimension() const noexcept { return CSVReaderANN::input_dimension_; }
int CSVReaderANN::get_output_dimension() const noexcept { return CSVReaderANN::output_dimension_; }
std::string CSVReaderANN::get_filename() const noexcept { return CSVReaderANN::filename_; }


bool CSVReaderANN::exists(std::string filename){
    std::ifstream f(filename);
    return f.good();
}

CSVReaderANNException::CSVReaderANNException(const char *message): message_{message} {}
char const* CSVReaderANNException::what() const throw(){ return CSVReaderANNException::message_; }
