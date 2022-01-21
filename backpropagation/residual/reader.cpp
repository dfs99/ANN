#include <iostream>
#include <vector>
#include <array>
#include <memory>
#include "../headers/CSVReaderANN.hpp"

int main(){
    std::cout << "Funciona????\n";
    try{
        CSVReaderANN parser {"../robots_copy_data.csv", 6 ,3};
        //std::cout << "dimensiones: " << parser.get_input_dimension() << " " << parser.get_output_dimension() << "\n";
        for (size_t i = 0; i < parser.ref_to_examples_.get()->size(); ++i){
                std::cout << "patron: " << i << ": ";
            for (size_t j = 0; j < parser.ref_to_examples_.get()->operator[](i).inputs.size(); ++j){
                std::cout << parser.ref_to_examples_.get()->operator[](i).inputs[j] << " ";
            }
            for (size_t j = 0; j < parser.ref_to_examples_.get()->operator[](i).desired.size(); ++j){
                std::cout << parser.ref_to_examples_.get()->operator[](i).desired[j] << " ";
            }
            std::cout << "\n";
        }
    }catch(const CSVReaderANNException & e){
        std::cerr << e.what() << "\n";
    }
    
    return 0;
}







/*#define A 2
const int SIZE = 1;

using namespace std;

template <typename T, int N, int M>
struct ExampleLabeled{
    std::array<T, N>inputs;
    std::array<T, M>desired;
};

template <int N, int M>
class CSVReaderANN{
    public:
        explicit CSVReaderANN(const std::string filename);
        ~CSVReaderANN();
        int get_input_dimension() const noexcept;
        int get_output_dimension() const noexcept;
        string get_filename() const noexcept;
        std::unique_ptr<std::vector<ExampleLabeled<double, N, M>>> get_examples();
    private:
        int input_dimension_ = N;
        int output_dimension_ = M;
        std::string filename_;
        std::unique_ptr<std::vector<ExampleLabeled<double, N, M>>> examples_;
        //std::vector<ExampleLabeled<double, N, M>> examples_;
};

int main(){
    ExampleLabeled<double, 1, 2> e1 = {{1.0}, {2, 4}};
    cout << sizeof(e1.inputs) << " " << sizeof(e1.desired) << "\n";
    return 0;
}

    // read data
    cout << "Fetching data from csv..." << endl;


    vector<vector<double>>data;
    vector<string>labels;

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
*/
