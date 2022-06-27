

CC=	g++
CFLAGS=	-Wall -std=c++20

all: neuralnetwork

%.o: %.cpp %h
	$(CC) $(CFLAGS) -c $^

neuralnetwork: MLP.o NeuralNetwork.cpp
	$(CC) $(CFLAGS) -o $@ $^


clean:
	rm *.o neuralnetwork