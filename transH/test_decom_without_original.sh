dirname $0
cd `dirname $0`

g++ -std=c++11 -o evaluate_decom_without_orginal evaluate_0125.cpp -larmadillo 

dim=$1
lambda=$2
batchsize=$3
eta=$4
detail=$5
#workPath=/home/wenjf/AAAI_final
#workPath=/home/wenjf/KB/SIGMOD2016/AAAI_final
workPath=/home/wenjf/SIGMOD2016/VLDB2016
dataPrefix=data_algo2
resultPrefix=result_algo2_H_new

echo $workPath
echo dim:$dim, lambda:$lambda, batchsize:$batchsize, eta:$eta
./evaluate_decom_without_orginal ${workPath}/$dataPrefix/entity2id.txt \
		${workPath}/$dataPrefix/relationlist.txt \
		${workPath}/$resultPrefix/entity2vec.dim${dim}.size${batchsize}.lambda${lambda}.eta${eta} \
		${workPath}/$resultPrefix/bias2vec.dim${dim}.size${batchsize}.lambda${lambda}.eta${eta} \
		${workPath}/$resultPrefix/normal2vec.dim${dim}.size${batchsize}.lambda${lambda}.eta${eta} \
		${workPath}/$resultPrefix/tranf2vec.dim${dim}.size${batchsize}.lambda${lambda}.eta${eta} \
		${workPath}/$dataPrefix/new_test_algo2.txt \
		$detail 
