dims=(25)
#log_prefix=/storage1/kbData/mfold
log_prefix=/storage1/kbData/decom_orginal
 
for dim in ${dims[@]}; do
#nohup bash arma_test/test_decom_without_original.sh $dim 0.0015 1000 1 $log_prefix/detail.decom.without.raw.$dim > $log_prefix/log.decom.without.raw.$dim &
	time bash arma_test/test_decom_without_original.sh $dim 0.0015 1000 1 $log_prefix/detail.decom.without.raw.$dim
	sleep 30
done
