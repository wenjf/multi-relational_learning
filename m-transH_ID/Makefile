objects=pureMfold.cpp
 
mFold evaluate_direct_with_detail : $(objects) evaluate_direct_with_detail.cpp
	g++ -std=c++11  $(objects) -o  mFold -O2 -larmadillo
	g++ -std=c++11 -o evaluate_direct_with_detail  evaluate_direct_with_detail.cpp -larmadillo


clean :
	rm mFold
	rm evaluate_direct_with_detail
