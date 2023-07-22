//seven segment display reader neural network
#include "MLP.h"
#include <iostream>
#include<vector>
class SDR{

    MultiLayerPerceptron mlpReader_;
    public: 
        SDR();
        void trainNetwork(std::ostream&,int,const std::vector<std::vector<double>>&,
 const std::vector<std::vector<double>>&);


};