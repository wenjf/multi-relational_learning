dims=(100)
#log_prefix=/storage1/kbData/mfold
log_prefix=/storage1/kbData/decomposition
 
for dim in ${dims[@]}; do
#nohup time bash arma_test/test_decom_without_raw_detail.sh $dim 0.0015 1000 1 $log_prefix/detail.decom.without.raw.$dim > $log_prefix/log.decom.without.raw.$dim &
	time bash arma_test/test_decom_without_raw_detail.sh $dim 0.0015 1000 1 $log_prefix/detail.decom.without.raw.$dim
#sleep 30
done
