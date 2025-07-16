import numpy as np
import torch, cv2, pickle, sys, os

import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from glob import glob

from models import build_model, build_visual_encoder

from dataloader import tokenizer
from config import load_args

from torch.amp import autocast
from search import beam_search

from decord import VideoReader

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

def run(faces, model, visual_encoder):
	chunk_frames = args.chunk_size * 25
	preds = []
	if args.lang_id is not None:
		lang_id = args.lang_id
	else:
		lang_id = None

	for i in range(0, faces.size(2), chunk_frames):
		chunk_faces = faces[:, :, i:i+chunk_frames]
		chunk_faces = chunk_faces.to(args.device)
		feats = visual_encoder(chunk_faces)

		src_mask = torch.ones(1, 1, feats.shape[1]).long().to(args.device)
		
		encoder_output, src_mask = model.encode(feats, src_mask)
		
		with torch.no_grad():
			with autocast('cuda', enabled=args.fp16):
				start_symbol = [50258]
				if args.lang_id is not None:
					start_symbol.append(tokenizer.encode(f"<|{args.lang_id}|>")[0])

				beam_outs, beam_scores = forward_pass(model, encoder_output, src_mask, start_symbol)
				out = beam_outs[0][0]
	
		pred = tokenizer.decode(out.cpu().numpy().tolist()[:-1]).strip().lower()

		pred = pred.replace("<|transcribe|><|notimestamps|>", " ")

		if i==0:
			lang_id = pred[2:4]
		pred = pred[7:].strip()
		preds.append(pred)

	print(f"Language: {lang_id}")
	print(f"Transcription: {' '.join(preds)}")


def load_models(args):
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

	visual_encoder = build_visual_encoder().to(args.device).eval()
	s = torch.load(args.visual_encoder_ckpt_path, map_location=args.device)["state_dict"]
	new_s = {}
	for k, v in s.items():
		if "face_encoder" not in k:
			continue
		k = k.replace("module.face_encoder.", "")
		new_s[k] = v
	
	visual_encoder.load_state_dict(new_s)
	print("Models loaded successfully")

	return model, visual_encoder

def read_video(fpath, start=0, end=None):
	start *= 25
	start = max(start - 4, 0)

	if end is not None:
		end *= 25
		end += 4 # to read till end + 3
	else:
		end = 100000000000000000000000000 # some large finite num

	with open(fpath, 'rb') as f:
		video_stream = VideoReader(f, width=160, height=160)

		end = min(end, len(video_stream))

		frames = video_stream.get_batch(list(range(int(start), int(end)))).asnumpy().astype(np.float32)
	frames = torch.FloatTensor(frames).to(args.device).unsqueeze(0)
	frames /= 255.
	frames = frames.permute(0, 4, 1, 2, 3)
	crop_x = (frames.size(3) - 96) // 2
	crop_y = (frames.size(4) - 96) // 2
	faces = frames[:, :, :, crop_x:crop_x + 96 , 
						crop_y:crop_y + 96]

	return faces

if __name__ == '__main__':
	assert args.ckpt_path is not None, 'Specify a trained checkpoint!'
	print('Loading models...')
	model, visual_encoder = load_models(args)

	faces = read_video(args.fpath, args.start, args.end)
	print("Input frames: ", faces.shape)
	
	print("Running inference...")
	run(faces, model, visual_encoder)