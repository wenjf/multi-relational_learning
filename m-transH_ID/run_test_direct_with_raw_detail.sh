dims=(25 50)

log_prefix=/storage1/kbData/enforce

for dim in ${dims[@]}; do
	nohup bash arma_test/test_direct_with_raw_detail.sh $dim 0.0015 1000 1 $log_prefix/detail.direct.with.raw.$dim > $log_prefix/log.direct.with.raw.$dim &
	sleep 30
done
