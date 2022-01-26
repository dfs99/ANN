# ANN
it contains a few Artificial Neural Networks models implemented in C++.
The models it contains are:
    => Adaline
    => Single Perceptron
    => Multilayer Perceptron - MP

The Multilayer Perceptron follows the backpropagation algorithm in order
to train a model. In addition, the SDG is implemented with the MSE error
function.

SDG: Stocastic Gradient Decent
MSE: Mean Squared Error

In order to use the MP, you have to store the data you want to train into
data directory inside backpropagation directory. Once you have write 'make'
down in the terminal you may modify the execute.sh script in order to 
generate the model you want. Finally, the results will be given in csv
format at results directory.
