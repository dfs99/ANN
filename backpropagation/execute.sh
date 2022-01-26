#bin/./main "../data/data.csv" 0.15 1000 SIGMOID 3 8 4 1
bin/./main "../data/robots_copy_data.csv" 0.1 100 SIGMOID 3 6 2 3
#valgrind --tool=memcheck --leak-check=full bin/./main "robots_copy_data.csv" 0.01 10 SIGMOID 3 6 3 3