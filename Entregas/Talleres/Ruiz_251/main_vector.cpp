#include<iostream>
using std::cout;
using std::endl;

/*Inicio Modificaciones*/
#include "vector.hpp"
using myvec::Vector;


/*Final Modificaciones*/

int main()
{
//LLamo el constructor, que crea vectores de 4 elementos
  Vector V1(4);
  Vector V2(4);
  Vector V3(4);
  Vector V4(4);
  Vector V6(5);
    
  cout<< endl<< "Prueba iniciación y V.PrintVec()"<< endl<< endl;

  cout<< "V1= ";
  V1.PrintVec();
  cout<< endl<< "V2= ";
  V1.PrintVec();
  cout<< endl<< "V3= ";
  V3.PrintVec();
  cout<< endl<< "V4= ";
  V4.PrintVec();

  cout<< endl<< "Prueba SetPos"<< endl<< endl;

  V1.SetPos(0,1);
  V1.SetPos(1,2);
  V1.SetPos(2,3);
  V1.SetPos(3,4);
  V1.SetPos(4,5);
  cout<< "V1= ";
  V1.PrintVec();
  cout<< endl;
  
  V2.SetPos(0,4);
  V2.SetPos(1,3);
  V2.SetPos(2,2);
  V2.SetPos(3,1);
  V2.SetPos(4,5);
  cout<< "V2= ";
  V2.PrintVec();

  cout<< endl<< "V3= ";
  V3.PrintVec();
  cout<< endl<< "V4= ";
  V4.PrintVec();

  cout<< endl<< endl<< "Prueba GetPos"<< endl<< endl;
  cout<< "V2.GetPos(4)= "<< V2.GetPos(4)<< endl;
  cout<< "V2.GetPos(3)= "<< V2.GetPos(3)<< endl;
  
  cout<< endl<< "Prueba GetSize"<< endl<< endl;
  cout<< "V2.GetSize()= "<< V2.GetSize()<< endl;
  cout<< "V6.GetSize()= "<< V6.GetSize()<< endl;
 
  cout<< endl<< "Prueba Sobrecarga de Operadores"<< endl<< endl;

  cout<< endl<< "Operador '='"<< endl<< endl;

  cout<< "V1= ";
  V1.PrintVec();
  cout<< endl<< "V4= ";
  V4.PrintVec();
  cout<< endl<< "V4=V1"<<endl;
  V4=V1;
  cout<< "V4= ";
  V4.PrintVec();
  cout<< endl<< "V6=V1"<<endl;
  V6=V1;
  cout<< "V6= ";
  V6.PrintVec();
  
   
  cout<< endl<< "Operador '+'"<< endl<< endl;

  cout<< "V1= ";
  V1.PrintVec();
  cout<< endl<< "V2= ";
  V2.PrintVec();
  cout<< endl<< "V4= ";
  V4.PrintVec();
  cout<< endl<< "V2=V1+V4"<<endl;
  V2=V1+V4;
  cout<< "V2= ";
  V2.PrintVec();
  cout<< endl;
  cout<< endl<< "V2=V1+V6"<<endl;
  V2=V1+V6;
  cout<< "V2= ";
  V2.PrintVec();
  
  cout<< endl<< endl<< "Operador '-'"<< endl<< endl;

  cout<< "V1= ";
  V1.PrintVec();
  cout<< endl<< "V2= ";
  V2.PrintVec();
  cout<< endl<< "V4= ";
  V4.PrintVec();
  cout<< endl<< "V2=V1-V4"<<endl;
  V2=V1-V4;
  cout<< "V2= ";
  V2.PrintVec();
  cout<< endl;
  cout<< endl<< "V2=V1-V6"<<endl;
  V2=V1-V6;
  cout<< "V2= ";
  V2.PrintVec();

  cout<< endl<< endl<< "Operador '[]'"<< endl<< endl;

  cout<< "V1= ";
  V1.PrintVec();
  cout<<endl<< "V1[0]= "<<V1[0]<<endl;
  cout<< "V1[1]= "<<V1[1]<<endl;
  cout<< "V1[2]= "<<V1[2]<<endl;
  cout<< "V1[3]= "<<V1[3]<<endl;
  cout<< "V1[4]= "<<V1[4]<<endl;
  V1[1]=10;
  cout<< "V1[1]= "<<V1[1]<<endl;
  
  return 0;
  
}//end main
