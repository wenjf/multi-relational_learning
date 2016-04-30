#define ARMA_DONT_USE_CXX11
#include <iostream>
#include <fstream>
#include <armadillo>
#include <time.h>
#include <string>
#include <unordered_map>
#include <vector>
#include <unordered_set>

using namespace std;
using namespace arma;

class Evaluator{
	unordered_map <string, int> entities2index;
	unordered_map <int, string> index2entities;
	unordered_map <string, int> relation2index;
	vector <int> schema;
	unordered_map<string, vector<pair<int, uvec> > > testData;
	vector<vec> A;
	mat BR, NR, ENT;
	int ENT_NUM, REL_NUM, DIM;
	char detail_out_path[100];

	void loadFB13K(char *entities_list_path, char *relation_list_path, char *test_data_path, char *detail_path){
		strcpy(detail_out_path, detail_path);
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
		char cvt[50];
		while (fscanf(testFile, "%s%s",cvt, str) != EOF){
			int index = relation2index[string(str)];
			int cnt = schema[index];
			uvec ent_indices = zeros<uvec>(cnt);
			for (int i = 0; i < cnt; i++){
				fscanf(testFile, "%s", str);
				ent_indices(i) = entities2index[string(str)];
			}
			testData[string(cvt)].push_back(pair<int, uvec>(index, ent_indices));
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
		char *entity_out, char *bias_out, char *normal_out, char *a_out, char *detail_path) {
		loadFB13K(entities_list_path, relation_list_path, test_data_path, detail_path);
		loadMat(bias_out, entity_out, normal_out, a_out);
	}
	void evaluate();
};

void Evaluator::evaluate(){
	ofstream detailFile(detail_out_path);
	long long posTotal = 0;
	int vis_num = 0;
	int rank10num = 0;
	double avgPos;
	double less10;

	for (auto it = testData.begin(); it != testData.end(); it++){
		//all binary instance of this multi-relational instance
	    string cvtID = it->first;	
		vector<pair<int, uvec>> allInstance = it->second;
		vector<vec> allHeadScore;
		vector<vec> allTailScore;

		for (int j = 0; j < allInstance.size(); j++){
			vec headScore(ENT_NUM);
			vec tailScore(ENT_NUM);
			int rel = allInstance[j].first;
			uvec indices = allInstance[j].second;
			double loss = lossFn(rel, indices);
			int tail = indices(1);
			int head = indices(0);
			uvec other_tail = indices;
			uvec other_head = indices;
			for (int i = 0; i < ENT_NUM; i++){
				//predict tail
				if (i != tail){
					other_tail(1) = i;
					tailScore[i] = lossFn(rel, other_tail);
				}
				else
					tailScore[i] = loss;

				//predict head
				if (i != head){
					other_head(0) = i;
					headScore[i] = lossFn(rel, other_head);
				}
				else
					headScore[i] = loss;
			}
			allHeadScore.push_back(headScore);
			allTailScore.push_back(tailScore);
		}
		//count how many entities have appeared in this instance
		unordered_set<int> all_entity;
		for (int j = 0; j < allInstance.size(); j++){
			uvec indices = allInstance[j].second;
			if (all_entity.count(indices(1)) == 0)	all_entity.insert(indices(1));
			if (all_entity.count(indices(0)) == 0)	all_entity.insert(indices(0));
		}
		for (auto entity_p = all_entity.begin(); entity_p != all_entity.end(); entity_p++){
			int rank = 1;
			int entity = *entity_p;
			vec final_score = zeros<vec>(ENT_NUM);
			for (int i = 0; i < allInstance.size(); i++){
				if (entity == allInstance[i].second(0))
					final_score = final_score + allHeadScore[i];
				if (entity == allInstance[i].second(1))
					final_score = final_score + allTailScore[i];
			}
			for (int i = 0; i < ENT_NUM; i++){
				if (final_score(i) < final_score(entity)) rank++;
			}
			if (rank <= 10)  rank10num++;
			vis_num++;
			posTotal += rank;
			avgPos = posTotal*1.0 / vis_num;
			less10 = rank10num*100.0 / vis_num;
			//printf("testing number:%d,\t hit@10:%4f%,\t mean rank:%4f%c", vis_num, less10, avgPos, 13);
			//fflush(stdout);

		//here to output the details of testing
			detailFile << cvtID << "\t" << index2entities[entity] << "\t" << rank << "\t" << final_score(entity);
			for (int i = 0; i < ENT_NUM; i++){
				detailFile << "\t" << final_score(i);
			}
			detailFile << endl;
		}
	}
	printf("\ntesting number:%d,\t hit@10:%4f%,\t mean rank:%4f\n", vis_num, less10, avgPos);

	//end of testing, print out the statistical result.
	detailFile.close();

}

int main(int argc, char** argv){

	// load data
	char *entities_list_path, *relation_list_path, *test_data_path;
	char *vec_entity, *vec_bias, *vec_normal, *vec_tranf;
	FILE *entFile, *relFile, *testFile;
	Evaluator eva = Evaluator(argv[1], argv[2], argv[7], argv[3], argv[4], argv[5], argv[6], argv[8]);
	eva.evaluate();


}
