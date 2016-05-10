#define ARMA_DONT_USE_CXX11
#include <iostream>
#include <armadillo>
#include <fstream>
#include <time.h>
#include <string>
#include <unordered_map>
#include <vector>

using namespace std;
using namespace arma;

class Evaluator{
	char detail_path[100];
	unordered_map <string, int> entities2index;
	unordered_map <int, string> index2entities;
	unordered_map <string, int> relation2index;
	vector <int> schema;
	vector<pair<int, uvec> > testData;
	vector<string> testData_instance;
	vector<vec> A;
	mat BR, NR, ENT;
	int ENT_NUM, REL_NUM, DIM;
	void loadFB13K(char *entities_list_path, char *relation_list_path, char *test_data_path, char *detail_out_path);
	void loadMat(char *bias_out, char *entity_out, char *normal_out, char *a_out);
	
	mat project(const mat &_X, const vec &nr){
		return _X - nr * nr.t() * _X;
	}
	double lossFn(int rel, const uvec &indices){
		
		mat X = ENT.cols(indices);
		vec ar = A[rel];
		vec br = BR.col(rel);
		vec nr = NR.col(rel);
		mat Xr = project(X, nr);
		vec tmp = Xr * ar + br;
		return dot(tmp, tmp);
	}

public:

	Evaluator(char *entities_list_path, char *relation_list_path, char *test_data_path,
		char *entity_out, char *bias_out, char *normal_out, char *a_out, char *detail_path) {
		loadFB13K(entities_list_path, relation_list_path, test_data_path, detail_path);
		loadMat(bias_out, entity_out, normal_out, a_out);
	}

	void evaluate();
};

void Evaluator::loadFB13K(char *entities_list_path, char *relation_list_path, char *test_data_path, char *detail_out_path){
	strcpy(detail_path, detail_out_path);
	FILE *entFile, *relFile, *testFile;
	char str[500];
	ENT_NUM = 0, REL_NUM = 0;
	int n;
	entFile = fopen(entities_list_path, "r");
	while (fscanf(entFile, "%s", str) != EOF){
		entities2index[string(str)] = ENT_NUM;
		index2entities[ENT_NUM] = string(str);
		ENT_NUM++;
	}
	fclose(entFile);

	schema.clear();
	relFile = fopen(relation_list_path, "r");
	while (fscanf(relFile, "%s\t%d", str, &n) != EOF){
		schema.push_back(n == 0 ? 2 : n);
		relation2index[string(str)] = REL_NUM;
		REL_NUM++;
	}
	fclose(relFile);
	
	testFile = fopen(test_data_path, "r");
	char instance[100];
	while (fscanf(testFile, "%s%s", instance, str) != EOF){
		int index = relation2index[string(str)];
		int cnt = schema[index];
		uvec ent_indices = zeros<uvec>(cnt);
		for (int i = 0; i < cnt; i++){
			fscanf(testFile, "%s", str);
			ent_indices(i) = entities2index[string(str)];
		}
		testData.push_back(pair<int, uvec>(index, ent_indices));
		testData_instance.push_back(string(instance));
	}
	fclose(testFile);
	printf("Number of entities: %d, number of relations: %d, number of training data: %d\n", ENT_NUM, REL_NUM, testData.size());
}

void Evaluator::loadMat(char *bias_out, char *entity_out, char *normal_out, char *a_out){
	BR.load(bias_out);
	ENT.load(entity_out);
	NR.load(normal_out);
	DIM = NR.n_rows;
	cout << "DIM:\t" << DIM << endl;
	cout << "colum:\t" << NR.n_cols << endl;
	FILE *f_transf = fopen(a_out, "rb");
	for (int i = 0; i < REL_NUM; i++){
		vec a = zeros<vec>(schema[i]);
		for (int j = 0; j < schema[i]; j++){
			double tmp;
			fscanf(f_transf, "%lf", &a(j));
		}
		A.push_back(a);
	}
	fclose(f_transf);
}

void Evaluator::evaluate(){
	ofstream f_detail(detail_path);
	int rank_head = 1, rank_tail = 1;
	unsigned long long int posTail = 0;
	unsigned long long int posHead = 0;
	int vis_num = 0;
	int rank10numTail = 0;
	int rank10numHead = 0;
	double avgPosTail;
	double avgPosHead;
	double less10perTail;
	double less10perHead;

	for (int k = 0; k < testData.size(); k++){
		rank_head = 1;
		rank_tail = 1;
		int rel = testData[k].first;
		uvec indices = testData[k].second;
		double loss = lossFn(rel, indices);
		int tail = indices(1);
		int head = indices(0);
		uvec other_tail = indices;
		uvec other_head = indices;
		for (int i = 0; i < ENT_NUM; i++){
			if (i != tail){
				other_tail(1) = i;
				if (lossFn(rel, other_tail) <= loss) rank_tail++;
			}
			if (i != head){
				other_head(0) = i;
				if (lossFn(rel, other_head) <= loss) rank_head++;
			}
		}
		if (rank_tail <= 10) rank10numTail++;
		if (rank_head <= 10) rank10numHead++;
		posTail += rank_tail;
		posHead += rank_head;
		vis_num += 1;
		less10perTail = rank10numTail * 100.0 / vis_num;
		less10perHead = rank10numHead * 100.0 / vis_num;
		avgPosTail = posTail * 1.0 / vis_num;
		avgPosHead = posHead * 1.0 / vis_num;
		//printf("case number:%d Tail: hit@10:%f% rank:%f Head hit@10:%f% rank:%f ALL:hit@10:%f% rank:%f %c", vis_num, less10perTail, avgPosTail, less10perHead, avgPosHead, (less10perTail + less10perHead) / 2, (avgPosTail + avgPosHead) / 2, 13);
		f_detail << testData_instance[k] << "\t" << index2entities[tail] << "\t" << rank_tail << "\t" << loss << endl;
		f_detail << testData_instance[k] << "\t" << index2entities[head] << "\t" << rank_head << "\t" << loss << endl;

		//printf("case number:%d, hit@10 %f% rank:%f %c", vis_num*2, (less10perTail+less10perHead)/2, (avgPosTail+ avgPosHead)/2, 13);
		//fflush(stdout);
	}
	printf("\ncase number:%d, hit@10 %f% rank:%f %c", vis_num, (less10perTail + less10perHead) / 2, (avgPosTail + avgPosHead) / 2, 13);
	printf("\n");
	f_detail.close();
	//return pair<double, double>(rank10num * 100.0 / vis_num, pos / vis_num);
}


int main(int argc, char** argv){

	// load data
	char *entities_list_path, *relation_list_path, *test_data_path;
	char *vec_entity, *vec_bias, *vec_normal, *vec_tranf;
	FILE *entFile, *relFile, *testFile;
	Evaluator eva = Evaluator(argv[1], argv[2], argv[7], argv[3], argv[4], argv[5], argv[6], argv[8]);
	eva.evaluate();
}
