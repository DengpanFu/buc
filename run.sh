dataset=market1501
#dataset=duke
#dataset=mars
#dataset=DukeMTMC-VideoReID

batchSize=16
size_penalty=0.005
merge_percent=0.05

logs=logs/debug
snap=snapshots/debug


python run.py --dataset $dataset --logs_dir $logs --snap_dir $snap \
              -b $batchSize --size_penalty $size_penalty -mp $merge_percent 
