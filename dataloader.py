import torch, os, pickle, random, linecache, functools

import pandas as pd

from torch.utils import data as data_utils
from tqdm import tqdm
import numpy as np
from config import load_args

args = load_args()

from tokenizer import get_tokenizer
tokenizer = get_tokenizer()

MAX_FRAMES = 1500

def subsequent_mask(size):
	"Mask out subsequent positions."
	attn_shape = (1, size, size)
	subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
	return torch.from_numpy(subsequent_mask) == 0

class Dataset(object):
	def __init__(self):

		self.fpaths = []
		self.texts = []
		self.lens = []
		self.langs = []

		discarded = 0

		
		print('Reading csv {}...'.format(args.csv))
		df = pd.read_csv(args.csv)
		if args.lang_id != "*":
			df = df[df['language'] == args.lang_id]

		self.fpaths = df['file_path'].tolist()
		self.texts = df['transcript'].tolist()
		self.lens = df['length'].tolist()
		self.langs = df['language'].tolist()
		self.starts = df['start'].tolist()
			
		self.num_samples = len(self.lens)
		print('{} samples'.format(self.num_samples))

		np.random.seed(42)
		self.idxes = np.random.permutation(len(self))

	def sequential_sampling_fn(self):
		for i in self.idxes: yield [i]
	
	def __len__(self):
		return self.num_samples

	def __getitem__(self, idx):
		fpath = self.fpaths[idx]
		text = torch.LongTensor(list(tokenizer.encode(f"<|startoftranscript|><|{self.langs[idx]}|><|transcribe|><|notimestamps|>{self.texts[idx]}<|endoftext|>")))
		start = self.starts[idx]
		end = start + self.lens[idx]

		start = max(start - 3, 0) 
		end += 3

		feats = np.load(f"{args.feats_root}/{fpath}.npy")
		if len(feats.shape) > 2: feats = feats[0]
		feats = feats[start : end]
		
		feats = torch.FloatTensor(feats)

		self.cur_fpath = fpath
		return feats, text
