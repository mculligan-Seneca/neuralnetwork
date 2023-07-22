#include "SDR.h"
//seven segment display guesser using neural network


SDR::SDR(): mlpReader_{{7,7,9}}
{

}

//validation and trainiing set look up which is which
void SDR::trainNetwork(std::ostream& os, int epochs,const std::vector<std::vector<double>>& x,
 const std::vector<std::vector<double>>& y){

     double MSE;
     os<<"Epochs, MSE\n";
     for(int i=0;i<epochs;i++){
        MSE=0.0;
        for(int j=0;j<x.size();j++)
            MSE+=mlpReader_.bp(x[j],y[j]);
        MSE/=x.size();
        os<<i<<", "<<MSE<<"\n";
    }

}

