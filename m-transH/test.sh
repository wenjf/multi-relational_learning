dims=(25)

#log_prefix=/storage1/kbData/mfold
log_prefix=testing_log
# mkdir $log_prefix

for dim in ${dims[@]}; do
	nohup bash one_test.sh $dim 0.0015 1000 1 $log_prefix/detail.direct.without.raw.$dim > $log_prefix/log.direct.without.raw.$dim &
# sleep 10
done
