#include <iostream>
#include <vector>
using namespace std;
int main(){
    vector<int> x;
    x.reserve(10);
    cout<<x.size()<<endl;
    return 0;
}