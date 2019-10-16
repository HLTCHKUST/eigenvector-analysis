sizes=( 1 0.1 0.01 )
# sizes=( 100 10 1 0.1 0.01 )
sim_datasets=( ws353_similarity ws353_relatedness ws353 bruni_men radinsky_mturk luong_rare simlex999 )
ana_datasets=( google msr )

for size in "${sizes[@]}"
do
	for sim_dataset in "${sim_datasets[@]}"
	do
		# Evaluate on Word Similarity
		echo
		echo "WS353 Results on $size% on $sim_dataset"
		echo "-------------"

		python hyperwords/ws_eval.py --normalize --neg 5 PPMI w2.dyn.dirty.$size/pmi testsets/ws/$sim_dataset.txt
		python hyperwords/ws_eval.py --normalize --eig 0.5 SVD w2.dyn.dirty.$size/svd testsets/ws/$sim_dataset.txt
		python hyperwords/ws_eval.py --normalize --w+c SGNS w2.dyn.dirty.$size/sgns testsets/ws/$sim_dataset.txt
	done

	for ana_dataset in "${ana_datasets[@]}"
	do
		# Evaluate on Analogies
		echo
		echo "Google Analogy Results on $size% on $ana_dataset"
		echo "----------------------"

		python hyperwords/analogy_eval.py --neg 5 PPMI w2.dyn.dirty.$size/pmi testsets/analogy/$ana_dataset.txt
		python hyperwords/analogy_eval.py --eig 0 SVD w2.dyn.dirty.$size/svd testsets/analogy/$ana_dataset.txt
		python hyperwords/analogy_eval.py --w+c SGNS w2.dyn.dirty.$size/sgns testsets/analogy/$ana_dataset.txt
	done
done
