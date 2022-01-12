#include "MLP.h"


        // generates values from 0 to 1 and applies a scaling factor 
        // of 2 and a shift of -1
    double frand(){
        return (2.0*(double)rand()/RAND_MAX)-1.0;
    }

    //reuturn a new Perceptron with a specified number of inputs (+1 for bias)
    Perceptron::Perceptron(int inputs, double bias){
            bias_=bias;
            weights_.resize(inputs+1);
            std::generate(weights_.begin(),weights_.end(),frand);// provide vector with random weights

    }
    //run the perceptron. x is a vector with the input values.
    double Perceptron::run(std::vector<double> x){ // one liner but written in 3
        x.push_back(bias_);
        double sum = std::inner_product(x.begin(),x.end(),weights_.begin(),0.0);
        return sigmoid(sum);
    }

    void Perceptron::set_weights(std::vector<double> w_init){
        //set the weights of the perceptron
        weights_=w_init;
    }

    double Perceptron::sigmoid(double x){
        return 1.0/(1+std::exp(-x));
    }

    MultiLayerPerceptron::MultiLayerPerceptron(std::vector<int> layers,double bias, double eta){
        layers_=layers;
        bias_=bias;
        eta_=eta;
        
        for(int i=0;i<layers_.size();i++){//create neurons layer by layer
            values_.push_back(std::vector<double>(layers_[i],0.0));
            network_.push_back(std::vector<Perceptron>());
            if(i>0)// ignore first layer cause we dont have neurons
            for(int j=0;j<layers_[i];j++)
                network_[i].push_back(Perceptron(layers_[i-1],bias_));// set up each perceptron in the network
        }
    }

    void MultiLayerPerceptron::print_weights(){
        std::cout<<std::endl;
        for(int i=1;i<network_.size();i++){
            for(int j=0;j<layers_[i];j++){
                std::cout<<"Layer "<<i+1<< " Neuron "<< j << ": ";
                std::copy(std::begin(network_[i][j].weights_),std::end(network_[i][j].weights_),
                std::ostream_iterator<double>(std::cout," "));
                std::cout<<std::endl;
            }
            

        }

    }
    void MultiLayerPerceptron::set_weights(std::vector<std::vector<std::vector<double>>> w_init){
            //writes all weights into neural network
            //w_init is a vector of vector of vector of doubles that are the weights
            for(int i=0;i<network_.size();i++){
                for(int j=0;j<layers_[i];j++)
                    network_[i+1][j].set_weights(w_init[i][j]);//not specifiying input layer
            }

    }

    std::vector<double> MultiLayerPerceptron::run(std::vector<double> x){
        //run an input forward through the neural network
        //x is a vector with input values 
        values_[0]=x;
        for(int i=1;i<network_.size();i++){
            for(int j=0;j<layers_[i];j++){
                values_[i][j]=network_[i][j].run(values_[i-1]);// the current neuron is fed the values 
                //from previous layer
            }
        }
        return values_.back();
    }