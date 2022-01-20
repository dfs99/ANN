#pragma once

#include <vector>
#include <memory>
#include <exception>


template <typename T>
struct ExampleLabeled{
    std::vector<T>inputs;
    std::vector<T>desired;
};

class CSVReaderANN{
    public:
        explicit CSVReaderANN(std::string filename, int input_dimension, int output_dimension);

        std::unique_ptr<std::vector<ExampleLabeled<double>>> ref_to_examples_;
        int get_input_dimension() const noexcept;
        int get_output_dimension() const noexcept;
        std::string get_filename() const noexcept;
    private:
        int input_dimension_;
        int output_dimension_;
        std::string filename_;
        std::vector<ExampleLabeled<double>> examples_;
        bool exists(std::string filename);
};

class CSVReaderANNException : public std::exception {
    const char *message_;
public:
    explicit CSVReaderANNException(const char* msg);
    virtual char const* what() const throw();
};