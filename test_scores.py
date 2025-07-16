import numpy as np
import torch, os, cv2, pickle, sys

import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from glob import glob

from models import build_model

from dataloader import Dataset, tokenizer
from config import load_args

from torch.cuda.amp import autocast
from search import beam_search

args = load_args()

def forward_pass(model, encoder_output, src_mask, start_symbol):

	beam_outs, beam_scores = beam_search(
			model=model,
			bos_index=start_symbol,
			eos_index=tokenizer.eot,
			max_output_length=args.max_decode_len,
			pad_index=0,
			encoder_output=encoder_output,
			src_mask=src_mask,
			size=args.beam_size,
			n_best=args.beam_size,
		)

	return beam_outs, beam_scores

def levenshtein(a, b):
	"""Calculates the Levenshtein distance between a and b.
	The code was taken from: http://hetland.org/coding/python/levenshtein.py
	"""
	n, m = len(a), len(b)
	if n > m:
		a, b = b, a
		n, m = m, n
	current = list(range(n + 1))
	for i in range(1, m + 1):
		previous, current = current, [i] + [0] * n
		for j in range(1, n + 1):
			add, delete = previous[j] + 1, current[j - 1] + 1
			change = previous[j - 1]
			if a[j - 1] != b[i - 1]:
				change = change + 1
			current[j] = min(add, delete, change)
	return current[n]

def run(dataset, model):
	total_wer = 0
	num_words = 0
	prog_bar = tqdm(range(len(dataset)))
	for i in prog_bar:
		feats, gt = dataset[i]
		feats = torch.FloatTensor(feats).unsqueeze(0).to(args.device)
		src_mask = torch.ones(1, 1, feats.shape[1]).long().to(args.device)
		
		encoder_output, src_mask = model.encode(feats, src_mask)
		
		with torch.no_grad():
			with autocast(enabled=args.fp16):
				lang_code = tokenizer.encode(f"<|{args.lang_id}|>")[0]
				lang_code_idx = (gt == lang_code).nonzero(as_tuple=True)[0]
				start_symbol = gt[:lang_code_idx + 1]

				beam_outs, beam_scores = forward_pass(model, encoder_output, src_mask, start_symbol)
				out = beam_outs[0][0]
	
		pred = tokenizer.decode(out.cpu().numpy().tolist()[3:-1]).strip().lower()
		gt = tokenizer.decode(gt.cpu().numpy().tolist()[4:-1]).strip().lower()

		total_wer += levenshtein(pred.split(), gt.split())
		num_words += len(gt.split())

		prog_bar.set_description(f"WER: {total_wer / num_words}")

	print(f"WER: {total_wer / num_words}")


def load_model(args):
	model = build_model().to(args.device).eval()
	checkpoint = torch.load(args.ckpt_path, map_location=args.device)
	s = checkpoint["state_dict"]
	new_s = {}

	for k, v in s.items():
		if k.startswith("module."):
			new_s[k[7:]] = v
		else:
			new_s[k] = v

	model.load_state_dict(new_s)
	return model

def main(args):
	dataset = Dataset()
	assert args.ckpt_path is not None, 'Specify a trained checkpoint!'

	print('Loading model...')
	model = load_model(args)

	return model, dataset

if __name__ == '__main__':
	model, dataset = main(args)
	run(dataset, model)