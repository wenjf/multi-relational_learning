make -f new_mFoldMakeFile

lambdas=(0.0015)
batchsizes=(1000)
dims=(25 50)
etas=(1)
dataPrefix=data_algo3
resultPrefix=result_algo3_0130

for dim in ${dims[@]}; do
	for batchsize in ${batchsizes[@]}; do
		for lambda in ${lambdas[@]}; do
			for eta in ${etas[@]}; do
				time ./mFold -dim $dim -epoch 1000 -batch $batchsize -lr $lambda -margin 0.5 -epsilon 0.01 -beta 0.001 \
					-entity $dataPrefix/entity2id.txt \
					-rel $dataPrefix/relation2id.final \
					-train $dataPrefix/train_algo3.txt \
					-bias_out $resultPrefix/bias2vec.dim${dim}.size${batchsize}.lambda${lambda}.eta${eta} \
					-entity_out $resultPrefix/entity2vec.dim${dim}.size${batchsize}.lambda${lambda}.eta${eta} \
					-normal_out $resultPrefix/normal2vec.dim${dim}.size${batchsize}.lambda${lambda}.eta${eta} \
					-a_out $resultPrefix/tranf2vec.dim${dim}.size${batchsize}.lambda${lambda}.eta${eta} \
					-split_list $dataPrefix/relation_splited.txt \
					-eta $eta
				#echo dim:$dim, lambda:$lambda, batchsize:$batchsize, eta:$eta
#bash arma_test/test_algo3.sh $dim $lambda $batchsize $eta >> result.algo3.txt
			done
		done
	done
done
