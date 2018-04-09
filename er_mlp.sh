#!bash
echo "ER !!!!!!!!!!"
./train_er_mlp.py data/kin
./train_er_mlp.py data/nations
./train_er_mlp.py data/uml
./train_er_mlp.py data/Freebase
./train_er_mlp.py data/Wordnet
echo "--"
echo "--"
echo "--"
echo "--"
echo "--"
echo "--"
echo "NTN !!!!!!!!!!"
./train_ntn.py data/kin 2
./train_ntn.py data/nations 2
./train_ntn.py data/uml 2
./train_ntn.py data/Freebase 2
./train_ntn.py data/Wordnet 2
echo "--"
echo "--"
echo "--"
echo "--"
echo "--"
echo "--"
echo "RESCAL NN !!!!!!!!!!"
./train_ntn.py data/kin 1
./train_ntn.py data/nations 1
./train_ntn.py data/uml 1
./train_ntn.py data/Freebase 1
./train_ntn.py data/Wordnet 1
