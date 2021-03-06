// main_2.cpp
#include<bits/stdc++.h>
#include <fstream>  
#include <sstream>
#include <string>
#define rep(i, a, b) for(int i=a;i<=b; ++i)  
#define per(i, a, b) for(int i=a;i>=b;--i) 
#define de(a) cout <<#a<<" => "<<a<<endl
#define dep(a) cout <<"(first, second) => "<<"("<<a.fi<<","<<a.se<<")"<<endl
#define fi first
#define se second
#define mp make_pair
#define pb push_back
#define pob pop_back
#define ms(a, b) memset(a, b, sizeof(a))
#define pq priority_queue
using namespace std;
typedef long long ll;
typedef pair<int, int> pii;
typedef vector<int>vi;
const int maxn = 1e6+7;
const int inf = 1e9+7;
set<int>::iterator it;

// ===========变量结构定义====================
int n, m;
map<string , ll> mark; //结点编号
ll id=1;
ll id_sen=0; 
struct node { //结点
    // 名字
    string name;
    // 线、门、触发器等等类型
    string type;
    // tag标签
    // 0表示正常线网，
    // 1表示木马线网
    int tag;
}v[maxn];
//vi e[maxn];     // 无向图
vi ed[maxn];    // 有向图 u->v, 输出方向
vi edf[maxn];   // 有向图 v->u，输入方向
vi line;  // 记录线编号，用于处理重复线
map<int, bool>pin; //主输入
map<int, bool>pout; //主输出
vi IN;
vi OUT;
vector<int> R[50];
int positive = 0;
int negtive = 0;
int rpath[maxn];
int feature[60]; // 记录各项特征值
bool vis[maxn];
int id_check;

// ============函数定义===================

// ============ init =====================
void init(); //读入文件
void build(string str, int flag); //拆解、建图

// ============ feature_count_function define ================
void count_func();
void count_sentence(int len,string base_path,bool isText,int file_index);
void outputcsv(string str);  
void output_Sentence_length(ofstream& outFile,int len,bool isText);
char* normal_in_txt_path;
char* trojan_in_txt_path;
string out_csv_path;
void output_PandN_sentence(ofstream& outFile,int len,bool isText);
// ============主函数=====================
int main(int argc, char* argv[]){
	string file_prefix[7] = {"RS232-T1000","RS232-T1100","RS232-T1200","RS232-T1300","RS232-T1400","RS232-T1500","RS232-T1600"};
	int i = 6;
    string str1,str2;
    str1 = file_prefix[i]+"/uart.txt";
    str2 = file_prefix[i]+"/trojan.txt";
    normal_in_txt_path = (char *)(str1.c_str());
   	trojan_in_txt_path = (char *)(str2.c_str());
    cout<<normal_in_txt_path<<trojan_in_txt_path<<endl;
   	init();
    //    count_func();
	count_sentence(5,file_prefix[i],false,i);

	return 0;
}

// ============ function detail ===================
void input(int flag) {
    string ans;
    char c;
    while(scanf("%c", &c)!=EOF) {
        if(c=='\n') continue;//输入跳过换行和空格
        if(c==' ') continue;
        ans = "";
        ans+=c;
        while((c=getchar())!=';') { //以分号;为界读取文件（一行器件）
            if(c == '\n') continue;
            if(c == '(' || c == ')' || c==',') ans += ' ';
            ans+=c;
            if(c == ',' || c == '(') ans+=' ';//不懂
        }
        if(ans == "") continue;//说明这行没东西 没读到 所以不用build
        build(ans, flag);//调用solve函数进行拆解
    }
    
    return ;
}
// ============init=======================
void init() { // 拆解txt，建图
    freopen(normal_in_txt_path, "r", stdin);
    input(0);
    freopen(trojan_in_txt_path, "r", stdin);
    input(1);
    int sz = line.size()-1;
    rep(i, 0, sz) {
        int line_id = line[i];
        int in_len = edf[line_id].size();
        int out_len = ed[line_id].size();
        if(in_len == 0){
            IN.pb(line_id);
            pin[line_id] = true;
        }
        else if(out_len == 0){
            OUT.pb(line_id);
            pout[line_id] = true;
        }
    }
    
    return ;
}

int CreateId(string s, string type) {//创建唯一编号
   if(mark[s] == 0 ) {
        if(type=="line") line.pb(id);
        return mark[s]=id++;
   }
   else  return mark[s];
}

void build(string str, int tag) {
    istringstream is(str);
    vector<string> sbuff;
    int output_count = 0;
    do {
        string substr;
        is >> substr;
        if(substr == ".QN") output_count++;
		if(substr == ".Q") output_count++;
        if(substr[0] == '(' || substr[0] == '.' || substr[0] == ')' || substr=="" || substr==",") continue;
        sbuff.pb(substr);
    }while(is);
    int len = sbuff.size()-1;
    string name = sbuff[1];
    string tp = sbuff[0];
    int idx = CreateId(name, "");//创建器件名字对应的唯一编号

    v[idx].name = name;
    v[idx].tag = tag;
    string str_2 = tp.substr(0,2);
    string str_3 = tp.substr(0,3);
    string str_4 = tp.substr(0,4);
    
    //清洗类型
//    if(str_2 == "OR" || str_3 == "NOT" || str_3 == "AND" || str_4 == "NAND" || str_3 == "NOR" || str_3 == "XOR"){//    NOT,AND, NAND,OR,NOR,XOR,XNOR
//        v[idx].type = "LOGIC"; 
//    }else if(str_4 == "SDFF") {
//        v[idx].type = "SDFF";
//    } else if(str_3 == "BUF") {
//        v[idx].type = "BUF";
//    } else {
//        v[idx].type = sbuff[0];
//    }
	v[idx].type = sbuff[0];
    //cout<<"器件类型："<<v[idx].type<<endl;
    rep(i, 2, len) {  //该元器件的线网
        //创建结点
        string to = sbuff[i];
        string type = "line";
//        if(to[0]<='9' && to[0]>='0') type = "const";
        int id_to = CreateId(to, type);//创建一条线网的编号
        v[id_to].name = to;
        v[id_to].type = type;
        v[id_to].tag = tag;
        //无向边添加
//        e[idx].pb(id_to);
//        e[id_to].pb(idx);
        //有向边   左输出，右输入，例如门A->门B，那么有向图为：A->line->B
        if(output_count==2){//两输出情况
            if(i<len-1) {
                ed[id_to].pb(idx);//这个to输出到idx这个节点
                edf[idx].pb(id_to);//to输入到idx这个节点
            } else {
                ed[idx].pb(id_to);
                edf[id_to].pb(idx);
            }

        }else{
            if(i<len) {
                ed[id_to].pb(idx);
                edf[idx].pb(id_to);
            } else {
                ed[idx].pb(id_to);
                edf[id_to].pb(idx);
            }
        }
        
    }
    return ;
}


/*-------------输出图G中从顶点u到v的长度为len的所有简单路径，d是到当前为止已走过的路径长度，调用时初值为-1----------------*/
/**
*   采用从顶点u出发的回溯深度优先搜索方法，每搜索一个新顶点，路径长度增1，
*   若搜索到顶点v且d等于len，则输出路径path[0..d]，然后继续回溯查找其他路径。
*
*/
void path_all2(vi e[], int cur, int des, int len, int path[], int d, string type)
{
   
   if(type == "line"){
	
   }else{
       d++;                         //路径长度增1
       if(d>len){   
            return;
       }
       path[d] = cur;               //将当前顶点添加到路径中
       
       if(cur == des && d == len)      //满足条件，输出一条路径
       {         
            cout<<"?????"<<len<<"???P???:"<<endl;
            for(int i = 0; i < d; i++)
                cout<<v[path[i]].name<<" "; 
            cout<<v[des].name<<endl;
            return;
        }
        vis[cur] = true;             //置已访问标记
   }
   
   int sz = e[cur].size()-1;
   rep(i, 0, sz) {
        int to = e[cur][i];
        if(!vis[to]){
            path_all2(e, to, des, len, path, d, v[to].type);
            // if(v[to].type == "line") {
            //     get_feature_loop(e, line_id, to_id, x, base);
            // } else {
            // get_feature_loop(e, line_id, to_id, x+1, base);
            // }
        }
        
    }
   vis[cur] = false;
   return;
}
void path_all(vi e[], int ori, int cur, int len, int path[], int d, string type, bool isReverse)
{
   
   if(type == "line"){
	
   }else{
       d++;                        //路径长度增1
       if(d>len){   
            return;
       }
       path[d] = cur;                //将当前顶点添加到路径中
       
       if(d == len)     //满足条件，输出一条路径
       {    
            //cout<<v[ori].type<<" "<<v[ori].name<<"长度为"<<len<<"的路径为:"<<endl;
            if(isReverse){
				for(int i = d; i>=0; i--){
            		R[len+1].pb(path[i]);
				}
			}else{
				for(int i = 0; i <= d; i++){
//            		cout<<v[path[i]].name<<" ";
            		R[len].pb(path[i]);
				}
			}
           	
            return;
        }
        vis[cur]=true;
   }
   
   int sz = e[cur].size()-1;
   rep(i, 0, sz) {
        int to = e[cur][i];
        if(v[to].type=="line"){
        	path_all(e,ori, to, len, path, d, v[to].type, isReverse);
		}else if(!vis[to]){
            path_all(e,ori, to, len, path, d, v[to].type, isReverse);
        }
    }
   vis[cur] = false;
   return;
}
void count_func(){
	//cout<<"!!!"<<endl;
    int sz_in = IN.size()-1;
    int sz_out = OUT.size()-1;
    //cout<<sz_in<<endl;
	//cout<<sz_out<<endl;
    for(int i = 0; i<=sz_in; i++){
        for(int j = 0; j<=sz_out; j++){
           //cout<<"!!!"<<endl;
            path_all2(ed, IN[i], OUT[j], 3, rpath, -1, "");	 
        }
    }
}
void outputtxt(ofstream& outFile){
	
    int size1 = R[1].size()/2;
    int size2 = R[2].size()/3;
    int size3 = R[3].size()/4;
    
    
    if(R[1].size()==0){
        size1 = 1;
        R[1].pb(0);
    }
    if(R[2].size()==0){
        size2 = 1;
        R[2].pb(0);
        
    }
    if(R[3].size()==0){
        size3 = 1;
        R[3].pb(0);
    }
    
    for(int i=0;i<size1;i++){
        for(int j=0;j<size2;j++){
            for(int k=0;k<size3;k++){
            	bool tro_flag = false;
                if(R[1][0]==0) outFile<<"EMP"<<" DouHao ";
                else{
                	
                    if(v[R[1][2*i]].tag || v[R[1][2*i+1]].tag) tro_flag = true;
                    outFile<<v[R[1][2*i]].type<<" "<<v[R[1][2*i+1]].type<<" DouHao ";
                }
                if(R[2][0]==0) outFile<<"EMP"<<" DouHao ";
                else{
                    if(v[R[2][3*j]].tag || v[R[2][3*j+1]].tag || v[R[2][3*j+2]].tag) tro_flag = true;
                    outFile<<v[R[2][3*j]].type<<" "<<v[R[2][3*j+1]].type<<" "<<v[R[2][3*j+2]].type<<" DouHao ";
                }
                if(R[3][0]==0) outFile<<"EMP";
                else{
                    if(v[R[3][4*k]].tag || v[R[3][4*k+1]].tag || v[R[3][4*k+2]].tag || v[R[3][4*k+3]].tag) tro_flag = true;
                    outFile<<v[R[3][4*k]].type<<" "<<v[R[3][4*k+1]].type<<" "<<v[R[3][4*k+2]].type<<" "<<v[R[3][4*k+3]].type;
                }
                if(tro_flag){
                    outFile<<","<<"1";
                    negtive++;
                }
                else 
                {
                    outFile<<","<<"0";
                    positive++;
                }
                outFile<<endl;			 
            }
        }
    }
}
void output_Sentence_length(ofstream& outFile,int len, bool isText){
	int size = R[len].size()/(len+1);
//	if(R[len+1].size()==0){
//        size3 = 1;
//        R[len+1].pb(0);
//    }
//	cout<<size<<endl;
    for(int i=0;i<size;i++){
    	bool tro_flag = false;
                for(int j=0;j<=len;j++){
                	if(v[R[len][len*i+j]].tag) tro_flag = true;
                	outFile<<v[R[len][(len+1)*i+j]].type<<" ";
				}
				if(!isText){
					if(tro_flag){
                    	outFile<<","<<"1";
                    	negtive++;
                	}
                	else{
                    	outFile<<","<<"0";
                    	positive++;
					}
				}
                outFile<<endl;
	}
}
void output_PandN_sentence(ofstream& outFile,int len,bool isText){
	int len_1 = len+1;
	int size1 = R[len].size()/(len+1);
	int size2 = R[len_1].size()/(len+1);
	if(R[len].size()==0){
        size1 = 1;
        R[len].pb(0);
    }
    if(R[len_1].size()==0){
        size2 = 1;
        R[len_1].pb(0);
    }
	for(int i=0;i<size1;i++){
		for(int j=0;j<size2;j++){
			bool tro_flag = false;
			for(int k=0;k<=len;k++){
                if(R[len_1][0]!=0){
                	if(v[R[len_1][len_1*j+k]].tag) tro_flag = true;
                	outFile<<v[R[len_1][len_1*j+k]].type<<" ";
				}
			}
			for(int k=0;k<=len;k++){
                if(R[len][0]!=0){
                	if(v[R[len][len_1*i+k]].tag) tro_flag = true;
                	outFile<<v[R[len][len_1*i+k]].type<<" ";
				}
			}
			if(!isText){
				if(tro_flag){
                    outFile<<","<<"1";
                    negtive++;
                }
                else{
                    outFile<<","<<"0";
                    positive++;
				}
			}
			outFile<<endl;
		}
	}
	 
}
void count_sentence(int len,string base_path,bool isText,int file_index){
	ofstream outFile;  
	if(isText){
		string path = "dataset/"+base_path+"_level"+to_string(len)+"_text.txt";
		cout<<path<<endl;
		outFile.open(path, ios::out|ios::binary); 
	}
	else{
		string path ="data/data"+to_string(file_index)+".csv";
		outFile.open(path,ios::out|ios::binary);
		outFile<<"text"<<","<<"label"<<endl; 
	}
	for(int i=1;i<id;i++){
		if(v[i].type != "line"){
        //    R[0].pb(v[i].name);
//            cout<<v[i].name<<endl;
			path_all(ed, i, i, len, rpath, -1, "", false);
			//path_all(edf, i, i, 1, rpath, -1, "");
			ms(vis, false);
			path_all(edf, i, i, len, rpath, -1, "", true);
			ms(vis, false);
//			path_all(ed, i, i, 2, rpath, -1, "");
			//path_all(edf, i, i, 2, rpath, -1, "");
//			ms(vis, false);
//			path_all(ed, i, i, 3, rpath, -1, "");
			//path_all(edf, i, i, 3, rpath, -1, "");
//			ms(vis, false);
			output_PandN_sentence(outFile,len,isText);
//			outputtxt(outFile);
//			output_Sentence_length(outFile,len);
            R[len].clear();
            R[len+1].clear();
		}
	}
	outFile.close();
}
void outputcsv(string str){
	ofstream outFile;  
//    outFile.open("data.csv", ios::out);
//    for(int i=1;i<id;i++){
//		if(v[i].type != "line"){
//			outFile <<v[i].type << endl;
//		}
//	}
//	outFile.close();
	outFile.open("output.csv", ios::out); 
    for(int i=1;i<id;i++){
		if(v[i].type == "line"){
			outFile <<v[i].name << endl;
		}
	}
    outFile.close();
}
