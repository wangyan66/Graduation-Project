// main_2.cpp
#include<bits/stdc++.h>
#include <fstream>  
#include <sstream>
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

// ===========�����ṹ����====================
int n, m;
map<string , ll> mark; //�����
ll id=1;
ll id_sen=0; 
struct node { //���
    // ����
    string name;
    // �ߡ��š��������ȵ�����
    string type;
    // tag��ǩ
    // 0��ʾ����������
    // 1��ʾľ������
    int tag;
}v[maxn];
//vi e[maxn];     // ����ͼ
vi ed[maxn];    // ����ͼ u->v, �������
vi edf[maxn];   // ����ͼ v->u�����뷽��
vi line;  // ��¼�߱�ţ����ڴ����ظ���
map<int, bool>pin; //������
map<int, bool>pout; //�����
vi IN;
vi OUT;
vector<int> R[50];
int positive = 0;
int negtive = 0;
int rpath[maxn];
int feature[60]; // ��¼��������ֵ
bool vis[maxn];
int id_check;

// ============��������===================

// ============ init =====================
void init(); //�����ļ�
void build(string str, int flag); //��⡢��ͼ

// ============ feature_count_function define ================
void count_func();
void count_sentence(int len);
void outputcsv(string str);
void output_Sentence_length(ofstream& outFile,int len); 
char* normal_in_txt_path;
char* trojan_in_txt_path;
string out_csv_path;
// ============������=====================
int main(int argc, char* argv[]){
	 
    normal_in_txt_path =  "data_test/uart.txt"; //�������������ļ�
    trojan_in_txt_path = "data_test/trojan.txt"; //����ľ���ļ�
    //out_csv_path = argv[2];
    init();
//    count_func();
	count_sentence(20);
//	outputcsv(" ");
    //cout<<positive<<endl<<negtive;
	return 0;
}

// ============ function detail ===================
void input(int flag) {
    string ans;
    char c;
    while(scanf("%c", &c)!=EOF) {
        if(c=='\n') continue;//�����������кͿո�
        if(c==' ') continue;
        ans = "";
        ans+=c;
        while((c=getchar())!=';') { //�Էֺ�;Ϊ���ȡ�ļ���һ��������
            if(c == '\n') continue;
            if(c == '(' || c == ')' || c==',') ans += ' ';
            ans+=c;
            if(c == ',' || c == '(') ans+=' ';//����
        }
        if(ans == "") continue;//˵������û���� û���� ���Բ���build
        build(ans, flag);//����solve�������в��
    }
    
    return ;
}
// ============init=======================
void init() { // ���txt����ͼ
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

int CreateId(string s, string type) {//����Ψһ���
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
    int idx = CreateId(name, "");//�����������ֶ�Ӧ��Ψһ���

    v[idx].name = name;
    v[idx].tag = tag;
    string str_2 = tp.substr(0,2);
    string str_3 = tp.substr(0,3);
    string str_4 = tp.substr(0,4);
    
    //��ϴ����
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
    //cout<<"�������ͣ�"<<v[idx].type<<endl;
    rep(i, 2, len) {  //��Ԫ����������
        //�������
        string to = sbuff[i];
        string type = "line";
//        if(to[0]<='9' && to[0]>='0') type = "const";
        int id_to = CreateId(to, type);//����һ�������ı��
        v[id_to].name = to;
        v[id_to].type = type;
        v[id_to].tag = tag;
        //��������
//        e[idx].pb(id_to);
//        e[id_to].pb(idx);
        //�����   ������������룬������A->��B����ô����ͼΪ��A->line->B
        if(output_count==2){//��������
            if(i<len-1) {
                ed[id_to].pb(idx);//���to�����idx����ڵ�
                edf[idx].pb(id_to);//to���뵽idx����ڵ�
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


/*-------------���ͼG�дӶ���u��v�ĳ���Ϊlen�����м�·����d�ǵ���ǰΪֹ���߹���·�����ȣ�����ʱ��ֵΪ-1----------------*/
/**
*   ���ôӶ���u�����Ļ��������������������ÿ����һ���¶��㣬·��������1��
*   ������������v��d����len�������·��path[0..d]��Ȼ��������ݲ�������·����
*
*/
void path_all2(vi e[], int cur, int des, int len, int path[], int d, string type)
{
   
   if(type == "line"){
	
   }else{
       d++;                         //·��������1
       if(d>len){   
            return;
       }
       path[d] = cur;               //����ǰ������ӵ�·����
       
       if(cur == des && d == len)      //�������������һ��·��
       {         
            cout<<"?????"<<len<<"???P???:"<<endl;
            for(int i = 0; i < d; i++)
                cout<<v[path[i]].name<<" "; 
            cout<<v[des].name<<endl;
            return;
        }
        vis[cur] = true;             //���ѷ��ʱ��
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
void path_all(vi e[], int ori, int cur, int len, int path[], int d, string type)
{
   
   if(type == "line"){
	
   }else{
       d++;                        //·��������1
       if(d>len){   
            return;
       }
       path[d] = cur;                //����ǰ������ӵ�·����
       
       if(d == len)     //�������������һ��·��
       {    
            //cout<<v[ori].type<<" "<<v[ori].name<<"����Ϊ"<<len<<"��·��Ϊ:"<<endl;
            for(int i = 0; i <= d; i++){
//            	cout<<v[path[i]].name<<" "; 
            	R[len].pb(path[i]);
            	
			}
           
            return;
        }
        vis[cur]=true;
   }
   
   int sz = e[cur].size()-1;
   rep(i, 0, sz) {
        int to = e[cur][i];
        if(v[to].type=="line"){
        	path_all(e,ori, to, len, path, d, v[to].type);
		}else if(!vis[to]){
            path_all(e,ori, to, len, path, d, v[to].type);
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
			//���������txt����word2Vec��
//            	bool tro_flag = false;
//                if(R[1][0]==0) outFile<<"EMP"<<" DouHao ";
//                else{
//                	
//                    if(v[R[1][2*i]].tag || v[R[1][2*i+1]].tag) tro_flag = true;
//                    outFile<<v[R[1][2*i]].type<<" "<<v[R[1][2*i+1]].type<<" DouHao ";
//                }
//                if(R[2][0]==0) outFile<<"EMP"<<" DouHao ";
//                else{
//                    if(v[R[2][3*j]].tag || v[R[2][3*j+1]].tag || v[R[2][3*j+2]].tag) tro_flag = true;
//                    outFile<<v[R[2][3*j]].type<<" "<<v[R[2][3*j+1]].type<<" "<<v[R[2][3*j+2]].type<<" DouHao ";
//                }
//                if(R[3][0]==0) outFile<<"EMP";
//                else{
//                    if(v[R[3][4*k]].tag || v[R[3][4*k+1]].tag || v[R[3][4*k+2]].tag || v[R[3][4*k+3]].tag) tro_flag = true;
//                    outFile<<v[R[3][4*k]].type<<" "<<v[R[3][4*k+1]].type<<" "<<v[R[3][4*k+2]].type<<" "<<v[R[3][4*k+3]].type;
//                }
////                if(tro_flag){
////                    outFile<<","<<"1";
////                    negtive++;
////                }
////                else 
////                {
////                    outFile<<","<<"0";
////                    positive++;
////                }
//                outFile<<endl;			 
            }
        }
    }
}
void output_Sentence_length(ofstream& outFile,int len){
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
//                if(tro_flag){
//                    outFile<<","<<"1";
//                    negtive++;
//                }
//                else 
//                {
//                    outFile<<","<<"0";
//                    positive++;
//                }
                outFile<<endl;
	}
}
void count_sentence(int len){
	ofstream outFile;  
    outFile.open("data_test/text.txt", ios::out); 
//	outFile.open("data_test/data.csv",ios::out);
//	outFile<<"text"<<","<<"label"<<endl;
	for(int i=1;i<id;i++){
		
		if(v[i].type != "line"){
        //    R[0].pb(v[i].name);
        //    cout<<R[0][0]<<endl; 
			path_all(ed, i, i, len, rpath, -1, "");
			//path_all(edf, i, i, 1, rpath, -1, "");
			ms(vis, false);
			
//			path_all(ed, i, i, 2, rpath, -1, "");
			//path_all(edf, i, i, 2, rpath, -1, "");
			
//			ms(vis, false);
//			path_all(ed, i, i, 3, rpath, -1, "");
			//path_all(edf, i, i, 3, rpath, -1, "");
//			ms(vis, false);
//			outputtxt(outFile);
			output_Sentence_length(outFile,len);
			R[1].clear();
			R[2].clear();
			R[3].clear();
            R[len].clear();
		}
	}
	outFile.close();
}
void outputcsv(string str){
	ofstream outFile;  
//    outFile.open("data.csv", ios::out); // ?????????
//    for(int i=1;i<id;i++){
//		if(v[i].type != "line"){//????????????????
//			outFile <<v[i].type << endl;
//		}
//	}
//	outFile.close();
	outFile.open("output.csv", ios::out); // ?????????
    for(int i=1;i<id;i++){
		if(v[i].type == "line"){//????????????????
			outFile <<v[i].name << endl;
		}
	}
    outFile.close();
}
