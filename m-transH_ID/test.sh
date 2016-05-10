dims=(25)

log_prefix=testing_log

for dim in ${dims[@]}; do
	nohup bash one_test.sh $dim 0.0015 1000 1 $log_prefix/detail.direct.with.raw.$dim > $log_prefix/log.direct.with.raw.$dim &
	sleep 10
done
