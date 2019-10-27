dataset=market1501
#dataset=duke
#dataset=mars
#dataset=DukeMTMC-VideoReID

batchSize=16
size_penalty=0.003
# size_penalty=0.1
# size_penalty=0.7
merge_percent=0.05
# ep=buc_${size_penalty}
ep=buc_s1

# arch=avg_pool
# arch=buc
gpu=$1
arch=$2

# python run_dbc.py --dataset $dataset --gpu 7 --ep $ep --logs_dir logs/dbc \
#               -b $batchSize --size_penalty $size_penalty -mp $merge_percent --seed 1 

python run.py --dataset $dataset --seed 1 -b $batchSize -mp $merge_percent -a $arch \
                --gpu $gpu --size_penalty $size_penalty --ep $ep --mode buc --no_log