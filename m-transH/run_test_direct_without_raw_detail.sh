dims=(250)

#log_prefix=/storage1/kbData/mfold
log_prefix=/storage1/kbData/without_enforce
time bash arma_test/test_direct_without_raw_detail.sh 50 0.0015 1000 1 $log_prefix/detail.direct.without.raw.$dim 
exit 1

for dim in ${dims[@]}; do
	nohup bash arma_test/test_direct_without_raw_detail.sh $dim 0.0015 1000 1 $log_prefix/detail.direct.without.raw.$dim > $log_prefix/log.direct.without.raw.$dim &
	sleep 30
done
