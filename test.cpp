// #include <iostream>
// #include <vector>
// using namespace std;
// int main(){
//     vector<int> x;
//     x.reserve(10);
//     cout<<x.size()<<endl;
//     return 0;
// }
#include <iostream>
#include <vector>
#include <filesystem>
using namespace std;
namespace fs=std::filesystem;
class Dataset{
public:
    string data_root;
    Dataset(){
        data_root="./DATA/TRAIN";
    }
    void test(){
        try{
            for(auto &entry:fs::directory_iterator(data_root)){
                cout<<entry<<endl;
            }
        }catch(const fs::filesystem_error &e){
            cerr<<e.what()<<endl;
        }

    }

};
int main(){
    Dataset test;
    test.test();

    return 0;
}