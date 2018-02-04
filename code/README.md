# SUSTechNRL
version control: 2018/02/03

## Discription
src: contains the pages of GraRep, TADW, DeepWalk, node2vec, LINE, SDNE </br>
data: input datasets which including bolgCatalog, cora, wiki </br>
output: output results </br>

## Environment requirement
python 3.6 </br>
tensorflow </br>
gensim </br>
networkx >= 2.1 </br>
numpy </br>
scipy </br>
scikit-learn </br>

## Usage
example: python /src/main.py --method deepwalk --input data/cora/cora_edgelist.txt --output /output/cora_embedding_test.txt --label_file data/cora/cora_labels.txt
