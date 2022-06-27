#pragma once
#include <algorithm>
#include<iterator>
#include <vector>
#include <iostream>
#include <random>
#include <numeric>
#include <cmath>
#include <ctime>
#include<functional>


    //Class at repersenting a multilayer perceptron

    class Perceptron{
        
    

        public:
        double bias_;
        std::vector<double> weights_; //weights for inputs
        Perceptron(int inputs, double bias=1.0);//specified number of inputs
        double run(std::vector<double> x);
        void set_weights(std::vector<double> w_init);
        double sigmoid(double x);
    };

    class MultiLayerPerceptron{
        
        public:
        MultiLayerPerceptron(std::vector<int> layers, double bias=1.0,double eta=0.5);
        void set_weights(std::vector<std::vector<std::vector<double> > > w_init);
        void print_weights();
        std::vector<double> run(std::vector<double> x);
        double bp(std::vector<double> x, std::vector<double> y);


        std::vector<int> layers_; //number of neurons per layer remeber input layer has no neurons
        double bias_;
        double eta_;//used for better learning later on
        std::vector<std::vector<Perceptron> > network_;
        std::vector<std::vector<double> > values_;
        std::vector<std::vector<double> > d_; //will contain the error terms for the neurons


    };

