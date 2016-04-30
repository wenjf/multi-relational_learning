dirname $0
cd `dirname $0`

g++ -std=c++11 -o evaluate_direct_with_detail  evaluate_direct_with_detail.cpp -larmadillo 

dim=$1
lambda=$2
batchsize=$3
eta=$4
#workPath=/home/wenjf/AAAI_final
#workPath=/home/wenjf/KB/SIGMOD2016/AAAI_final
workPath=/home/wenjf/KB/SIGMOD2016/VLDB2016
dataPrefix=data_algo3
resultPrefix=result_algo3_0130

echo $workPath
echo dim:$dim, lambda:$lambda, batchsize:$batchsize, eta:$eta
#./evaluate ${workPath}/$dataPrefix/entity2id.txt ${workPath}/$dataPrefix/relation2id.final ${workPath}/$resultPrefix/entity2vec.dim${dim}.size${batchsize}.lambda${lambda}.eta${eta} ${workPath}/$resultPrefix/bias2vec.dim${dim}.size${batchsize}.lambda${lambda}.eta${eta} ${workPath}/$resultPrefix/normal2vec.dim${dim}.size${batchsize}.lambda${lambda}.eta${eta} ${workPath}/$resultPrefix/tranf2vec.dim${dim}.size${batchsize}.lambda${lambda}.eta${eta} ${workPath}/$dataPrefix/testM_new.txt ${workPath}/$dataPrefix/mediator_split.txt ${workPath}/$dataPrefix/real_entity
./evaluate_direct_with_detail ${workPath}/$dataPrefix/entity2id.txt ${workPath}/$dataPrefix/relation2id.final ${workPath}/$resultPrefix/entity2vec.dim${dim}.size${batchsize}.lambda${lambda}.eta${eta} ${workPath}/$resultPrefix/bias2vec.dim${dim}.size${batchsize}.lambda${lambda}.eta${eta} ${workPath}/$resultPrefix/normal2vec.dim${dim}.size${batchsize}.lambda${lambda}.eta${eta} ${workPath}/$resultPrefix/tranf2vec.dim${dim}.size${batchsize}.lambda${lambda}.eta${eta} ${workPath}/$dataPrefix/test_algo3_detail.txt ${workPath}/$dataPrefix/relation_splited.txt ${workPath}/$dataPrefix/real_entity.txt $5
echo "" 
