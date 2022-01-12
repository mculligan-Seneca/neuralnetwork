#include <iostream>
#include "MLP.h"
#define INPUTS 2

void runPerceptron(Perceptron* p){
     for(double i=0;i<INPUTS;i++){
        for(double j=0;j<INPUTS;j++){
            std::cout<<"{"<<i<<","
            <<j<<"} : "
            <<p->run({i,j})<<std::endl;//Accepts different inputs for Gates
        }
    }
}
int main(){
    srand(time(NULL));
    rand();
    std::cout<<"\n\n- - - - - Logic Gate Example - - - - - - -\n"<<std::endl;
    Perceptron *p=new Perceptron(INPUTS);// create perceptron with 2 inputs
    p->set_weights({10,10,-15});//AND gate; weights are set because we want a negative sum when we want output=0
    std::cout<<"AND Gate: "<<std::endl;
    runPerceptron(p);
     //sigmoid function produces 0 when negative number
     //input for or gate{0,0} must produce 0 and {0,1} or {1,0} must produce 1
    p->set_weights({20,20,-15});
    std::cout<<"OR Gate: "<<std::endl;
    runPerceptron(p);

    //Challenge NAND and XOR gates
    p->set_weights({-11,-11,15});//Nand produces false only if all outputs are true
    std::cout<<"NAND Gate: "<<std::endl;
    runPerceptron(p);
    p->set_weights({-11,-11,-2});//Xor only produces true at or condition
    std::cout<<"XOR Gate: "<<std::endl;
    runPerceptron(p);
    delete p;
    MultiLayerPerceptron mlp({2,2,1});// 2 neurons for first layer 2 neurons for hiddeen layer, 1 for output
    mlp.set_weights({{{-11,-11,15},{20,20,-15},{10,10,-15}}}); //xor is combonation of  nand or and and gates
    std::cout<<"Hardcoded Weights:\n";
    mlp.print_weights();
    std::cout<<"XOR:\n";
    for(double i=0;i<2;i++){
        for(double j=0;j<2;j++){
            std::cout<<i<<" "<<j<<" = "<<mlp.run({i,j})[0]<<"\n";
        }
    }
    return 0;
}