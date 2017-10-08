'''
Author: Satoshi Tsutsui <stsutsui@indiana.edu>
'''

import argparse
import sys
import os
import time

sys.path.append('./code')
from skipgram import build_model,traning_op,train
from dataset import Dataset


def parse_args():
	#Parses the arguments.
	parser = argparse.ArgumentParser(description="metapath2vec")
	parser.add_argument('--walks',type=str,required=True,help='text file that has a random walk in each line. A random walk is just a seaquence of node ids separated by a space.')
	parser.add_argument('--types',type=str,required=True,help='text file that has node types. each line is "node id <space> node type"')
	parser.add_argument('--epochs',type=int,default=2, help='number of epochs')
	# parser.add_argument('--batch',type=int,default=1, help='Batch size.Only batch one is supported now...')
	parser.add_argument('--lr',type=float,default=0.01, help='learning rate')
	parser.add_argument('--log',required=True,type=str,help='log directory')
	parser.add_argument('--log-interval',default=-1,type=int,help='log intervals. -1 means per epoch')
	parser.add_argument('--max-keep-model',default=10,type=int,help='number of models to keep saving')
	parser.add_argument('--embedding-dim',default=100,type=int,help='embedding dimensions')
	parser.add_argument('--negative-samples',default=5,type=int,help='number of negative samples')
	parser.add_argument('--care-type',default=1,type=int,help='care type or not. if 1, it cares (i.e. heterogeneous negative sampling). If 0, it does not care (i.e. normal negative sampling). ')
	parser.add_argument('--window',default=5,type=int,help='context window size')

	return parser.parse_args()

def main(args):
	if os.path.isdir(args.log):
		print("%s already exist. are you sure to override? Ok, I'll wait for 5 seconds. Ctrl-C to abort."%args.log)
		time.sleep(5)
		os.system('rm -rf %s/'%args.log)
	else:
	    os.makedirs(args.log)
	    print("made the log directory",args.log)

	dataset=Dataset(random_walk_txt=args.walks,node_type_mapping_txt=args.types,window_size=args.window)
	center_node_placeholder,context_node_placeholder,negative_samples_placeholder,loss = build_model(BATCH_SIZE=1,VOCAB_SIZE=len(dataset.nodeid2index),EMBED_SIZE=args.embedding_dim,NUM_SAMPLED=args.negative_samples)
	optimizer = traning_op(loss,LEARNING_RATE=args.lr)
	train(center_node_placeholder,context_node_placeholder,negative_samples_placeholder,loss,dataset,optimizer,NUM_EPOCHS=args.epochs,BATCH_SIZE=1,NUM_SAMPLED=args.negative_samples,care_type=args.care_type,LOG_DIRECTORY=args.log,LOG_INTERVAL=args.log_interval,MAX_KEEP_MODEL=args.max_keep_model)

if __name__ == "__main__":
	args = parse_args()
	main(args)
