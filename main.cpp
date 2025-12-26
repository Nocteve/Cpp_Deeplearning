#define STB_IMAGE_IMPLEMENTATION  
#include "./stb/stb_image.h" 
#include <iostream>
#include <cstdio>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <algorithm>
#define matrix vector<vector<float> >
#define vec vector<float>
#define ll long long 
#define vecP vector<pair<string,int> >
using namespace std;
namespace fs=std::filesystem;

matrix operator+(const matrix& x, const matrix& y) {
    matrix output;
    int rows = min(x.size(), y.size());
    if(rows == 0) return output;
    
    int cols = min(x[0].size(), y[0].size());
    output.reserve(rows);
    
    for(int i=0; i<rows; i++) {
        vec temp;
        temp.reserve(cols);
        for(int j=0; j<cols; j++) {
            temp.push_back(x[i][j] + y[i][j]);
        }
        output.push_back(temp);
    }
    return output;
}

// template<typename T>
// vector<T> operator+(vector<T> x,vector<T> y){
//     vector<T> output;
//     output.reserve(min(x.size(),y.size()));
//     for(int i=0;i<min(x.size(),y.size());i++){
//         output.push_back(x[i]+y[i]);
//     }
//     return output;
// }

vector<float> operator+(vector<float> x,vector<float> y){
    vector<float> output;
    output.reserve(min(x.size(),y.size()));
    for(int i=0;i<min(x.size(),y.size());i++){
        output.push_back(x[i]+y[i]);
    }
    return output;
}

template<typename T>
vector<T> operator-(vector<T> x,vector<T> y){
    vector<T> output;
    output.reserve(min(x.size(),y.size()));
    for(int i=0;i<min(x.size(),y.size());i++){
        output.push_back(x[i]-y[i]);
    }
    return output;
}
vec get_full_vec(float a,int size){
    vec output;
    while(size--){
        output.push_back(a);
    }
    return output;
}
matrix get_full_matrix(float a,int h,int w){
    matrix output;
    for(int i=0;i<w;i++){
        output.push_back(get_full_vec(a,h));
    }
    return output;
}
int get_row_num(matrix a){
    return a.size();
}
int get_col_num(matrix a){
    if(a.empty()){
        return 0;
    }
    else{
        return a[0].size();
    }
}
void show_matrix(matrix a){
    for(auto &row:a){
        for(auto &val:row){
            cout<<val<<" ";
        }
        cout<<endl;
    }
}  
void show_vec(vec a){
    for(auto &val:a){
        cout<<val<<endl;
    }
}
matrix read_img(char *address){
    int width, height, channels;
    unsigned char *img = stbi_load(address, &width, &height, &channels, 0);
    //cout<<channels<<endl;
    matrix pic;
    if(img != nullptr) {
        for(int i=0;i<width;i++){
            vector<float> temp;
            for(int j=0;j<height;j++){
                //cout<<dec<<static_cast<float>(img[i*(width)-1+j+1]);
                temp.push_back(static_cast<float>(img[i*(width)-1+j+1]));
            }
            pic.push_back(temp);
        }
        stbi_image_free(img); // 最后记得释放
    }
    for(auto &row:pic){
        for(auto &val:row){
            val/=255;
        }
    }
    return pic;
}
vec flatten(matrix x){
    vec output;
    for(auto &row:x){
        for(auto &val:row){
            output.push_back(val);
        }
    }
    return output;
}
matrix get_random_matrix(int height,int width){
    unsigned seed = chrono::system_clock::now().time_since_epoch().count();
    default_random_engine generator(seed);

    normal_distribution<float> normal_distribution(0.0, sqrt(1.0f/width));
    matrix output;
    for(int i=0;i<width;i++){
        vector<float> temp;
        for(int j=0;j<height;j++){
            float temp_num=normal_distribution(generator);
            temp.push_back(temp_num);
        }
        output.push_back(temp);
    }
    return output;
}
vec get_full_vec(int num,int size){
    vec output;
    while(size--){
        output.push_back(static_cast<float>(num));
    }
    return output;
}
int get_random_num(int min_num,int max_num){//0~9 0~5400
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);
    std::uniform_int_distribution<int> int_distribution(min_num, max_num);
    return int_distribution(generator);
}
vector<int> get_random_num_vec(int min_num,int max_num,int sum){//注意不能用vec,要int的
    vector<int> output;
    while(sum--){
        output.push_back(get_random_num(min_num,max_num));
    }
    return output;
}

class Dataset{
public:
    string data_root;
    Dataset(string _data_root){
        data_root=_data_root;//注意此处的初始化
    }
    vector<pair<string,int> > get_all(){
        vector<pair<string,int> > output;
        try{
            for(auto &entry:fs::directory_iterator(data_root)){
                if(entry.is_directory()){
                    for(auto &item:fs::directory_iterator(entry.path())){
                        pair<string,int> temp;
                        string path=entry.path();
                        int label=static_cast<int>(path[path.length()-1])-'0';
                        temp=make_pair(item.path(),label);
                        //cout<<temp.first<<temp.second<<endl;
                        output.push_back(temp);
                    }
                }
            }
        }catch(const fs::filesystem_error &e){
            cerr<<"file_system_error:"<<e.what()<<endl;
        }
        
        return output;
        
    }

};

class Model_FCN{
public:
    Dataset train_data_all,test_data_all;
    matrix w1,w2,w3;
    vec b1,b2,b3;
    //float epoch_loss;
    string train_data_root,test_data_root;

    vec orgin_x,y1,y2,y3,r_y1,r_y2,sig_y3,sof;//需要保存一些中间值用于梯度的计算
    vec t_vec;//实际值向量（是一个one hot向量）
    matrix grad_w1,grad_w2,grad_w3;
    vec grad_b1,grad_b2,grad_b3;
    Model_FCN():train_data_all("./DATA/TRAIN"),test_data_all("./DATA/TEST"){
        w1=get_random_matrix(784,128);
        w2=get_random_matrix(128,64);
        w3=get_random_matrix(64,10);
        b1=get_random_matrix(128,1)[0];
        b2=get_random_matrix(64,1)[0];
        b3=get_random_matrix(10,1)[0];
        //epoch_loss=0;
        train_data_root="./DATA/TRAIN/";
        test_data_root ="./DATA/TEST/";
    }
    vec relu(vector<float> &x){
        vec output;
        for(auto val:x){
            output.push_back(max(float(0),val));
        }
        return output;
    }
    float vector_multi(vec &x,vec &y){
        float output=0;
        for(int i=0;i<x.size();i++){
            output+=x[i]*y[i];
        }
        return output;
    }
    vec vector_multi_one_by_one(vec &x,vec &y){
        vec output;
        for(int i=0;i<min(x.size(),y.size());i++){
            output.push_back(x[i]*y[i]);
        }
        return output;
    }
    vec layer_forward(matrix &w,vec &b,vec &x){
        vec output;
        for(int i=0;i<w.size();i++){
            output.push_back(vector_multi(w[i],x)+b[i]);
        }
        return output;
    }
    vec softmax(vec &x){
        vec output;
        float sum=0;
        for(auto &val:x){
            sum+=exp(val);
        }
        for(auto &val:x){
            output.push_back(exp(val)/sum);
        }
        return output;
    }
    vec sigmoid(vec &x){
        vec output;
        for(auto &val:x){
            output.push_back(1/(1+exp((-1)*val)));
        }
        return output;
    }
    vec forward(vec &x){
        vec output;
        orgin_x=x;
        y1=layer_forward(w1,b1,x);
        r_y1=relu(y1);
        y2=layer_forward(w2,b2,r_y1);
        r_y2=relu(y2);
        y3=layer_forward(w3,b3,r_y2);
        sig_y3=y3;//sigmoid(y3);
        sof=softmax(sig_y3);
        output=sof;
        return output;
    }
    float get_one_loss(vec &v,vec &target){
        float loss=0;
        for(int i=0;i<target.size();i++){
            if(target[i]){
                loss+=-log(max(v[i],static_cast<float>(1e-8)));
            }
        }
        return loss;
    }


    // float get_loss(vec &v,vec &target){
    //     //交叉熵损失
    // }

    void get_gradient(vec &y){
        vec grad_sof=sof-t_vec;
        
        vec grad_sig;
        grad_sig.reserve(grad_sof.size());
        for(int i=0;i<grad_sof.size();i++){
            grad_sig.push_back(grad_sof[i]);//*(sig_y3[i]*(1-sig_y3[i])));
        }
        
        for(int i=0;i<w3.size();i++){
            vec temp;
            for(int j=0;j<r_y2.size();j++){
                temp.push_back(r_y2[j]*grad_sig[i]);
            }
            grad_w3.push_back(temp);
        }
        grad_b3=grad_sig;

        vec grad_ry2;
        grad_ry2.reserve(r_y2.size());
        for(int i=0;i<r_y2.size();i++){
            vec temp;
            for(vec &v:w3){
                temp.push_back(v[i]);
            }
            grad_ry2.push_back(vector_multi(temp,grad_sig));
        }
        for(int i=0;i<grad_ry2.size();i++){
            if(y2[i]<=0) grad_ry2[i]=0;
        }
        for(int i=0;i<w2.size();i++){
            vec temp;
            for(int j=0;j<r_y1.size();j++){
                temp.push_back(r_y1[j]*grad_ry2[i]);
            }
            grad_w2.push_back(temp);
        }
        grad_b2=grad_ry2;

        vec grad_ry1;
        grad_ry1.reserve(r_y1.size());
        for(int i=0;i<r_y1.size();i++){
            vec temp;
            for(vec &v:w2){
                temp.push_back(v[i]);
            }
            grad_ry1.push_back(vector_multi(temp,grad_ry2));
        }
        for(int i=0;i<grad_ry1.size();i++){
            if(y1[i]<=0) grad_ry1[i]=0;
        }
        for(int i=0;i<w1.size();i++){
            vec temp;
            for(int j=0;j<orgin_x.size();j++){
                temp.push_back(orgin_x[j]*grad_ry1[i]);
            }
            grad_w1.push_back(temp);
        }
        grad_b1=grad_ry1;

    }
    void clear_grad(){
        grad_w1.clear();
        grad_w2.clear();
        grad_w3.clear();
        grad_b1.clear();
        grad_b2.clear();
        grad_b3.clear();
    }
    void update(float learning_rate,matrix &w,matrix &g){
        for(int i=0;i<w.size();i++){
            for(int j=0;j<w[0].size();j++){
                w[i][j]-=learning_rate*g[i][j];
            }
        }
    }
    void update(float learning_rate,vec &b,vec &g){
        for(int i=0;i<b.size();i++){
            b[i]-=learning_rate*g[i];
        }
    }
    void backward(float learning_rate){
        update(learning_rate,w1,grad_w1);
        update(learning_rate,w2,grad_w2);
        update(learning_rate,w3,grad_w3);
        update(learning_rate,b1,grad_b1);
        update(learning_rate,b2,grad_b2);
        update(learning_rate,b3,grad_b3);
    }

    bool flag=false;
    vec get_target_vec(int x){
        vec target;
        for(int i=0;i<=9;i++){
            target.push_back((i==x));
        }
        return target;
    }
    matrix average(matrix &w,int n){
        matrix output;
        for(int i=0;i<w.size();i++){
            vec temp;
            for(int j=0;j<w[0].size();j++){
                temp.push_back(w[i][j]/n);
            }
            output.push_back(temp);
        }
        return output;
    }
    vec average(vec &v,int n){
        vec output;
        for(int i=0;i<v.size();i++){
            output.push_back(v[i]/n);
        }
        return output;
    }
    void check_gradients() {
        float grad_norm = 0;
        for(auto &row : grad_w1) for(auto &val : row) grad_norm += val*val;
        for(auto &row : grad_w2) for(auto &val : row) grad_norm += val*val;
        for(auto &row : grad_w3) for(auto &val : row) grad_norm += val*val;
        for(auto &val : grad_b1) grad_norm += val*val;
        for(auto &val : grad_b2) grad_norm += val*val;
        for(auto &val : grad_b3) grad_norm += val*val;
        
        grad_norm = sqrt(grad_norm);
        cout << "Gradient norm: " << grad_norm << endl;
    }
    float l_rate=0.001;
    void train(int batch_size,int epochs){
        auto data_all=train_data_all.get_all();
        for(int epoch=1;epoch<=epochs;epoch++){
            //if(epoch%10==0) l_rate*=0.9;
            //vector<int> train_data=get_random_num_vec(0,9,batch_size);
            // vector<int> train_data;
            // for(int i=0;i<batch_size;i++){
            //     train_data.push_back(i%10);
            // }
            unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
            shuffle(data_all.begin(),data_all.end(),std::default_random_engine(seed));
            for(int batch_idx=0;batch_idx<=data_all.size()/batch_size;batch_idx++){
                vecP train_data(data_all.begin()+batch_idx*batch_size,min(data_all.end(),data_all.begin()+(batch_idx+1)*batch_size));
                matrix sum_grad_w1,sum_grad_w2,sum_grad_w3;
                vec sum_grad_b1,sum_grad_b2,sum_grad_b3;
                sum_grad_w1=get_full_matrix(0,w1[0].size(),w1.size());
                sum_grad_w2=get_full_matrix(0,w2[0].size(),w2.size());
                sum_grad_w3=get_full_matrix(0,w3[0].size(),w3.size());
                sum_grad_b1=get_full_vec(0,b1.size());
                sum_grad_b2=get_full_vec(0,b2.size());
                sum_grad_b3=get_full_vec(0,b3.size());
                float batch_loss=0;
                for(auto &x:train_data){
                    int batch_size=train_data.size();
                    matrix pic=read_img(std::data(x.first));
                    if(!flag) show_matrix(pic),flag=!flag;
                    vec pic_vec=flatten(pic);
                    vec pic_out=forward(pic_vec);
                    //show_vec(pic_out);
                    
                    t_vec=get_target_vec(x.second);
                    batch_loss+=get_one_loss(pic_out,t_vec);
                    // cout<<"Loss:"<<get_one_loss(pic_out,t_vec)<<endl; 
                    // cout<<endl;

                    get_gradient(pic_out);
                    sum_grad_w1=sum_grad_w1+grad_w1;
                    sum_grad_w2=sum_grad_w2+grad_w2;
                    sum_grad_w3=sum_grad_w3+grad_w3;
                    sum_grad_b1=sum_grad_b1+grad_b1;
                    sum_grad_b2=sum_grad_b2+grad_b2;
                    sum_grad_b3=sum_grad_b3+grad_b3;
                    clear_grad();
                }
                grad_w1=average(sum_grad_w1,batch_size);
                grad_w2=average(sum_grad_w2,batch_size);
                grad_w3=average(sum_grad_w3,batch_size);
                grad_b1=average(sum_grad_b1,batch_size);
                grad_b2=average(sum_grad_b2,batch_size);
                grad_b3=average(sum_grad_b3,batch_size);
                //check_gradients();
                backward(l_rate);//学习率
                clear_grad();
                if((batch_idx+1)%100==0){
                    cout<<endl<<"Epoch"<<epoch<<" batch_idx:"<<batch_idx+1<<" "<<"Loss:"<<batch_loss/batch_size<<endl<<endl;
                }
                else{
                    if((batch_idx+1)%10==0) cout<<"."<<flush;
                }
            }
            // matrix sum_grad_w1,sum_grad_w2,sum_grad_w3;
            // vec sum_grad_b1,sum_grad_b2,sum_grad_b3;
            // sum_grad_w1=get_full_matrix(0,w1[0].size(),w1.size());
            // sum_grad_w2=get_full_matrix(0,w2[0].size(),w2.size());
            // sum_grad_w3=get_full_matrix(0,w3[0].size(),w3.size());
            // sum_grad_b1=get_full_vec(0,b1.size());
            // sum_grad_b2=get_full_vec(0,b2.size());
            // sum_grad_b3=get_full_vec(0,b3.size());
            // float epoch_loss=0;
            // for(auto &x:train_data){
            //     int pic_num=get_random_num(0,5400);
            //     string pic_root=train_data_root+std::to_string(x)+"/"+std::to_string(pic_num)+".jpg";
            //     matrix pic=read_img(std::data(pic_root));
            //     if(!flag) show_matrix(pic),flag=!flag;
            //     vec pic_vec=flatten(pic);
            //     vec pic_out=forward(pic_vec);
            //     //show_vec(pic_out);
                
            //     t_vec=get_target_vec(x);
            //     epoch_loss+=get_one_loss(pic_out,t_vec);
            //     // cout<<"Loss:"<<get_one_loss(pic_out,t_vec)<<endl; 
            //     // cout<<endl;

            //     get_gradient(pic_out);
            //     sum_grad_w1=sum_grad_w1+grad_w1;
            //     sum_grad_w2=sum_grad_w2+grad_w2;
            //     sum_grad_w3=sum_grad_w3+grad_w3;
            //     sum_grad_b1=sum_grad_b1+grad_b1;
            //     sum_grad_b2=sum_grad_b2+grad_b2;
            //     sum_grad_b3=sum_grad_b3+grad_b3;
            //     clear_grad();
            // }
            // grad_w1=average(sum_grad_w1,batch_size);
            // grad_w2=average(sum_grad_w2,batch_size);
            // grad_w3=average(sum_grad_w3,batch_size);
            // grad_b1=average(sum_grad_b1,batch_size);
            // grad_b2=average(sum_grad_b2,batch_size);
            // grad_b3=average(sum_grad_b3,batch_size);
            // //check_gradients();
            // backward(l_rate);//学习率
            // clear_grad();
            // cout<<"Epoch:"<<epoch<<" "<<"Loss:"<<epoch_loss/batch_size<<endl<<endl;
        }
    }
    void test(){
        // vector<int> test_data=get_random_num_vec(0,9,test_size);
        cout<<"testing"<<endl;
        auto data_all=test_data_all.get_all();
        int ac_sum=0;
        for(auto &x:data_all){
            matrix pic=read_img(std::data(x.first));
            vec pic_vec=flatten(pic);
            vec out=forward(pic_vec);
            int predict=0,k=0;
            float max_probability=0;
            // show_vec(out);
            // cout<<endl;
            for(auto &val:out){
                if(val>max_probability){
                    predict=k;
                    max_probability=val;
                }
                k++;
            }
            //cout<<predict<<endl;
            if(predict==x.second){
                ac_sum++;
            }
        }
        // for(auto &x:test_data){
        //     int pic_num=get_random_num(0,50);
        //     string pic_root=test_data_root+std::to_string(x)+"/"+std::to_string(pic_num)+".jpg";
        //     matrix pic=read_img(std::data(pic_root));
        //     vec pic_vec=flatten(pic);
        //     vec out=forward(pic_vec);
        //     int predict=0,k=0;
        //     float max_probability=0;
        //     // show_vec(out);
        //     // cout<<endl;
        //     for(auto &val:out){
        //         if(val>max_probability){
        //             predict=k;
        //             max_probability=val;
        //         }
        //         k++;
        //     }
        //     //cout<<predict<<endl;
        //     if(predict==x){
        //         ac_sum++;
        //     }
        // }
        float accuracy=ac_sum*1.0f/data_all.size()*1.0f;
        cout<<"Accuracy:"<<accuracy<<endl;
    }
};

void test(){
    matrix a={{1,1,1},{1,1,1}};
    show_matrix(a);
    cout<<get_row_num(a)<<endl;
    cout<<get_col_num(a)<<endl;

    matrix pic=read_img((char*)("./DATA/TRAIN/0/0.jpg"));
    show_matrix(pic);

    Model_FCN model;
    vec pic_vec=flatten(pic);
    //show_vec(pic_vec);
    vec out=model.forward(pic_vec);
    show_vec(out);
    cout<<endl;

    matrix
        test_a=get_full_matrix(1,2,3),
        test_b=get_full_matrix(6,2,3);
    matrix c=test_a+test_b;
    show_matrix(c);

    model.train(16,2);
    model.test();
}
int main(){
    test();
    return 0;
}