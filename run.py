from __future__ import print_function, absolute_import
from reid.bottom_up import *
from reid import datasets
from reid import models
import numpy as np
import argparse
import os, sys, time
from reid.utils.logging import Logger
import os.path as osp
from torch.backends import cudnn

def main(args):
    cudnn.benchmark = True
    cudnn.enabled = True
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if not args.no_log:
        log_path = os.path.join(args.logs_dir, args.exp_name)
        if not osp.exists(log_path):
            os.makedirs(log_path)
        log_name = args.exp_name + "_log_" \
            + time.strftime("%Y-%m-%d_%H-%M-%S", time.gmtime()) + '.txt'
        sys.stdout = Logger(osp.join(log_path, log_name))
    print(args)

    if args.seed is not None:
        cudnn.benchmark = False
        cudnn.deterministic = True
        # fix random seed
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        random.seed(args.seed)
        print("set random seed={}".format(args.seed))

    snap_dir = osp.join(args.snap_dir, args.exp_name)

    # get all unlabeled data for training
    dataset_all = datasets.create(args.dataset, osp.join(args.data_dir, args.dataset))
    new_train_data, cluster_id_labels = change_to_unlabel(dataset_all)

    num_train_ids = len(np.unique(np.array(cluster_id_labels)))
    nums_to_merge = int(num_train_ids * args.merge_percent)

    BuMain = Bottom_up(model_name=args.arch, batch_size=args.batch_size, 
            num_classes=num_train_ids,
            dataset=dataset_all,
            u_data=new_train_data, save_path=args.logs_dir, max_frames=args.max_frames,
            embeding_fea_size=args.fea)

    start_step, train_data, labels = BuMain.load_checkpoint(args.resume_path)
    if train_data is not None:
        new_train_data = train_data
    if labels is not None:
        cluster_id_labels = labels
    
    for step in range(start_step, int(1/args.merge_percent)-1):
        print('step: ',step)

        BuMain.train(new_train_data, step, loss=args.loss) 
        BuMain.evaluate(dataset_all.query, dataset_all.gallery)

        # get new train data for the next iteration
        print('---------------------bottom-up clustering-----------------------')
        if args.mode == 'buc':
            cluster_id_labels, new_train_data = BuMain.get_new_train_data(cluster_id_labels, 
                nums_to_merge, size_penalty=args.size_penalty)
        elif mode == 'dbc':
            cluster_id_labels, new_train_data = BuMain.get_new_train_data_dbc(cluster_id_labels, 
                nums_to_merge, penalty=args.size_penalty)
        if args.save_snap:
            BuMain.save_checkpoint(snap_dir, step, new_train_data, cluster_id_labels)
        print('\n')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='bottom-up clustering')
    parser.add_argument('-gpu', '--gpu', type=str, default='0')
    parser.add_argument('-d', '--dataset', type=str, default='market1501',
                        choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=16)
    parser.add_argument('-f', '--fea', type=int, default=2048)
    parser.add_argument('-a', '--arch', type=str, default='avg_pool',choices=models.names())
    working_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument('--data_dir', type=str, metavar='PATH',
                        default=os.path.join(working_dir,'data'))
    parser.add_argument('--logs_dir', type=str, metavar='PATH',
                        default=os.path.join(working_dir,'logs'))
    parser.add_argument('--snap_dir', type=str, metavar='PATH',
                        default=os.path.join(working_dir,'snapshots'))
    parser.add_argument('--max_frames', type=int, default=900)
    parser.add_argument('--loss', type=str, default='ExLoss')
    parser.add_argument('-m', '--momentum', type=float, default=0.5)
    parser.add_argument('-s', '--step_size', type=int, default=55)
    parser.add_argument('--size_penalty',type=float, default=0.005)
    parser.add_argument('-mp', '--merge_percent',type=float, default=0.05)
    parser.add_argument('--mode', dest='mode', type=str, default='buc', choices=['buc', 'dbc'])
    parser.add_argument('--rep', dest='resume_path', type=str, default=None)
    parser.add_argument('--ep', dest='exp_name', type=str, default='debug')
    parser.add_argument('--no_log', dest='no_log', action='store_true')
    parser.add_argument('--seed', dest='seed', type=int, default=None)
    parser.add_argument('--save_snap', dest='save_snap', action='store_true')
    main(parser.parse_args())

