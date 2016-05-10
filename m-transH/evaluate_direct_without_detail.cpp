#define ARMA_DONT_USE_CXX11
#include <iostream>
#include <fstream>
#include <armadillo>
#include <time.h>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace std;
using namespace arma;


class Evaluator{
	unordered_map <string, int> entities2index;
	unordered_map <int, string> index2entities;
	unordered_map <string, int> relation2index;
	unordered_map <int ,string> index2relation;
	unordered_set <string> mediator_split;
	vector <int> schema;
	vector<string> real_entity;
	vector<pair<int, uvec> > testData;
	vector<string> dataCvtID;
	vector<vec> A;
	mat BR, NR, ENT;
	char detail_out_path[100];
	int ENT_NUM, REL_NUM, DIM;
	void loadFB13K(char *entities_list_path, char *relation_list_path, char *test_data_path, char *split_list, char *real_entity_path, char *detail_path){
		strcpy(detail_out_path, detail_path);
		FILE *entFile, *relFile, *testFile;
		ifstream f_split(split_list);
		ifstream f_real(real_entity_path);
		string line;
		while(getline(f_real, line)){
			real_entity.push_back(line);
		}
		cout << "real_entity size:\t" << real_entity.size() << endl; 
		f_real.close();
	
		while(getline(f_split, line)){
			mediator_split.insert(line);
		}
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
			index2relation[REL_NUM] = string(str);
			REL_NUM++;
		}
		fclose(relFile);

		testFile = fopen(test_data_path, "r");
		char cvt[100];
		while (fscanf(testFile, "%s%s", cvt, str) != EOF){
			int index = relation2index[string(str)];
			int cnt = schema[index];
			uvec ent_indices = zeros<uvec>(cnt);
			for (int i = 0; i < cnt; i++){
				fscanf(testFile, "%s", str);
				ent_indices(i) = entities2index[string(str)];
			}
			dataCvtID.push_back(string(cvt));
			testData.push_back(pair<int, uvec>(index, ent_indices));
		}
		fclose(testFile);
		printf("Number of entities: %d, number of relations: %d, number of testing data: %d\n", ENT_NUM, REL_NUM, testData.size());
	}
	void loadMat(char *bias_out, char *entity_out, char *normal_out, char *a_out){
		BR.load(bias_out);
		ENT.load(entity_out);
		NR.load(normal_out);
		DIM = NR.n_rows;
		cout << "DIM:\t" << DIM <<endl;
	        cout << "colum:\t" << NR.n_cols<<endl;	
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
		char *entity_out, char *bias_out, char *normal_out, char *a_out, char *split_list, char *real_entity_path, char *detail_path) {
		loadFB13K(entities_list_path, relation_list_path, test_data_path, split_list, real_entity_path, detail_path);
		loadMat(bias_out, entity_out, normal_out, a_out);
	}
	
	void evaluate();

};

void Evaluator::evaluate(){
	ofstream detailFile(detail_out_path);
	int rank = 1;
	unsigned long long int posTotal = 0;
	int vis_num = 0;
	int rank10num = 0;
	double avgPos;
	double less10Ratio;
	for (int k = 0; k < testData.size(); k++){
		rank = 1;
		string cvt = dataCvtID[k];
		int rel = testData[k].first;
		uvec indices = testData[k].second;
		int n = indices.n_rows;
		double loss = lossFn(rel, indices);
		//int tail = indices(1);

		for (int jj = 0; jj < n; jj++){
			rank = 1;
			// no need for these two line for direct model without dummy nodes
			//if (jj == 0)
			//	if (mediator_split.count(index2relation[rel]) > 0) jj++;
			int correctEntity = indices(jj);
			int pos = jj;
			uvec other_tail = indices;
			vec final_score = zeros<vec>(real_entity.size());
			//uvec other_head = indices;
			for (int i = 0; i < real_entity.size(); i++){
				int replace_index = entities2index[real_entity[i]];
				if (replace_index != correctEntity){
					other_tail(pos) = replace_index;
					//if filter mode, uncomment
					//if(fb_all.count(pair<int, uvec>(rel, other_tail)) > 0) continue;
					final_score(i) = lossFn(rel, other_tail);
					if (final_score(i) <= loss) rank++;
				}else
					final_score(i) = loss;

			}
				//cout << rank_tail << endl;
			if (rank <= 10) rank10num++;
			posTotal += rank;
			less10Ratio = rank10num * 100.0 / vis_num;
			avgPos = posTotal * 1.0 / vis_num;
			vis_num += 1;
			//printf("case number:%d hit@10:%f% rank:%f %c", vis_num, less10Ratio, avgPos, 13);
			//fflush(stdout);
			
			//--------------------------------------------------------------------------------
			//here to output the details
			detailFile << cvt << "\t" << index2relation[rel] << "\t" << index2entities[correctEntity] << "\t" << rank << "\t" << loss;
			for (int i = 0; i < real_entity.size(); i++){
				detailFile << "\t" << final_score(i);	
			}
			detailFile << endl;
		}
	}
	printf("\ncase number:%d hit@10:%f% rank:%f %c", vis_num, less10Ratio, avgPos, 13);
	printf("\n");
	detailFile.close();
}

int main(int argc, char** argv){

	// load data
	char *entities_list_path, *relation_list_path, *test_data_path;
	char *vec_entity, *vec_bias, *vec_normal, *vec_tranf;
	FILE *entFile, *relFile, *testFile;
	Evaluator eva = Evaluator(argv[1], argv[2], argv[7], argv[3], argv[4], argv[5], argv[6], argv[8], argv[9], argv[10]);
	eva.evaluate();

}
