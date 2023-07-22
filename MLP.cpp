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
            d_.assign(layers_.size(),std::vector<double>(layers_[i],0.0));
            network_.push_back(std::vector<Perceptron>());
            if(i>0)// ignore first layer cause we dont have neurons
                for(int j=0;j<layers_[i];j++)
                    network_[i].push_back(Perceptron(layers_[i-1],bias_));
                    // set up each perceptron in the network
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
    void MultiLayerPerceptron::set_weights(std::vector<std::vector<std::vector<double> > > w_init){
            //writes all weights into neural network
            //w_init is a vector of vector of vector of doubles that are the weights
            for(int i=0;i<w_init.size();i++){
                for(int j=0;j<w_init[i].size();j++)
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

    double MultiLayerPerceptron::bp(std::vector<double> x, std::vector<double> y){
        double MSE=0.0;
        /*
        Feed sample to network 

        Calculate the mean squared error. 

        Calculate the error term of each output neuron 

        Iteratively calculate the error terms in the hidden layers 

        Apply the delta rule 

        Adjust the weights
        */
       //Feed sample to network 
       x=this->run(x);

       //Calculate the mean squared error. 
       double err=0.0;
       for(int i=0;i<x.size();i++){
            err=x[i]-y[i];
            MSE+=err*err;
       }
       /*MSE=std::transform_reduce(x.cbegin(),x.cend(),
                                y.cbegin(),
                                0.0,
                                std::plus<double>(),
                                [](double observed, double expected){
                                    double err=observed-expected;
                                    return err*err;
                                });*/
       MSE/=layers_.size();//ignoring degrees of freedom

       // Calculate the error term of each output neuron 
       for(int i=0;i<x.size();i++){
           d_.back()[i]=x[i]*(1-x[i])*(y[i]-x[i]);
       }
       //Iteratively calculate the error terms in the hidden layers 
        for(int i=network_.size()-2;i>0;i--){
            for(int j=0;j<network_[i].size();j++){
                double fwd_err=0.0;
                for(int k=0;k<layers_[i+1];k++)
                    fwd_err+=network_[i+1][k].weights_[j]*d_[i+1][k];
                d_[i][j]=values_[i][j]*(1-values_[i][j])*fwd_err;
            }
        }

        //Delta rule plus adjust weights
        for(int i=1;i<layers_.size();i++){
            for(int j=0;j<layers_[i];j++){
                for(int k=0;k<layers_[i-1]+1;k++){
                    double delta;
                    if(k==layers_[i-1])
                        delta=eta_*d_[i][j]*bias_;
                    else
                        delta=eta_*d_[i][j]*values_[i-1][k];
                    network_[i][j].weights_[k]+=delta;//maybe should be multiplying by i,j
                }
            }
        }
        
       return MSE;//returns mean squared error cause well need for training
    }

    /*
    Back Propagation 
    
Here are the steps 



 

 

 Calculate output error terms 

Sk=ok*(1-ok)*(yk-ok) 

 Intermediate error metric used for guessing how bad each neuron is doing 

Were paying attention to the output layer.  

Well use these error terms to calculate the error terms moving backward through the hidden layer. 

Well know the error terms for all of the nodes in the network and well apply the delta rule to calculate the deltas and adjust the weights. 

 

Sk is related to the partial derivative of the error terms in the network with respect to each weight in that neuron 

Ok*(1-Ok) is the derivative of the sigmoid function 

 

Calculate the hidden layer input terms 

 

Sh=Oh(1-oh)*sigma(wkhSk,k contained within outs(kEouts)) 

This iterates form the last layer to the first to find the error term per neuron. 

In the hidden layer we have no idea about the error because we do not know what to expect from the previous layer. 

So what we do is take the sum a product that uses the error connected to this neurons output. 

We must multiply the weight of the input connected to the output 

 

Lets do h=1 

 

S1=o1*(1-o1)*sigma(wk1*Sk,kEouts) 

Were multiplying the weight 1 by Sk  

 

Were scaling the error terms with the weights 

 

This means errors with higher weights will get more of the blame then weights that are lower. 

When were talking about the index of the neuron 

S12 is the third neuron in the first layer  
K index of weights that’s outputed 

H is layer 

Apply the delta rule 

d/dx(wij)=n*Si*xij  

Si the error of the layer per neuron 

Adjust the weights 

Wij+=d/dx(wij) 
    */