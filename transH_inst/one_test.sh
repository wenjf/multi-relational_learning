dirname $0
cd `dirname $0`

make

dim=$1
lambda=$2
batchsize=$3
eta=$4
detail=$5
dataPrefix=JF17K/version3
resultPrefix=result

echo $workPath
echo dim:$dim, lambda:$lambda, batchsize:$batchsize, eta:$eta
./evaluate_decom_without_raw \
		../$dataPrefix/entity.txt \
		../$dataPrefix/relation.txt \
		$resultPrefix/entity2vec.dim${dim}.size${batchsize}.lambda${lambda}.eta${eta} \
		$resultPrefix/bias2vec.dim${dim}.size${batchsize}.lambda${lambda}.eta${eta} \
		$resultPrefix/normal2vec.dim${dim}.size${batchsize}.lambda${lambda}.eta${eta} \
		$resultPrefix/tranf2vec.dim${dim}.size${batchsize}.lambda${lambda}.eta${eta} \
		../$dataPrefix/test.txt \
		$detail 
