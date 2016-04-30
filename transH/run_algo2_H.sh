lambdas=(0.0015)
batchsizes=(1000)
dims=(50)
etas=(1)

dataPrefix=data_algo2
resultPrefix=result_algo2_H_new

for dim in ${dims[@]}; do
	for batchsize in ${batchsizes[@]}; do
		for lambda in ${lambdas[@]}; do
			for eta in ${etas[@]}; do
				time ./myTransH -dim $dim -epoch 1000 -batch $batchsize  -lr $lambda  -margin 0.5 -epsilon 0.01 -beta 0.001 \
					-entity $dataPrefix/entity2id.txt \
					-rel $dataPrefix/relationlist.txt \
					-train $dataPrefix/new_train_algo2.txt \
					-bias_out $resultPrefix/bias2vec.dim${dim}.size${batchsize}.lambda${lambda}.eta${eta} \
					-entity_out $resultPrefix/entity2vec.dim${dim}.size${batchsize}.lambda${lambda}.eta${eta} \
					-normal_out $resultPrefix/normal2vec.dim${dim}.size${batchsize}.lambda${lambda}.eta${eta} \
					-a_out $resultPrefix/tranf2vec.dim${dim}.size${batchsize}.lambda${lambda}.eta${eta}
#bash arma_test/test_algo2_H.sh $dim $lambda $batchsize $eta
			done
		done
	done
done
