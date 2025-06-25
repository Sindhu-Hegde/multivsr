import configargparse

pad_token = '<pad>'
bos_token = '<bos>'
eos_token = '<eos>'
unk_token = '<unk>'

start_symbol, end_symbol = 100, 101

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def load_args():

  parser = configargparse.ArgumentParser(description = "main")

  parser.add_argument('--device', type=str, default='cuda')

  parser.add_argument('--ckpt_path', default=None)
  parser.add_argument('--visual_encoder_ckpt_path', default=None)
  parser.add_argument('--fp16', default=True, type=str2bool)

  # Data
  parser.add_argument('--csv', type=str)
  parser.add_argument('--feats_root', type=str)

  # Preprocessing
  parser.add_argument('--lang_id', type=str, default=None, help='Language ID, * means all languages')

  # feature extraction config
  parser.add_argument('--file_list', type=str, help='Regex of video file paths relative to videos_folder')

  # inference params
  parser.add_argument('--max_decode_len', type=int, default=100)
  parser.add_argument('--beam_size', type=int, default=20, help='The beam width')
  parser.add_argument('--fpath', type=str, default=None, help='file path for real-world inference')
  parser.add_argument('--chunk_size', type=int, default=14, help='Chunk size for visual encoder')
  parser.add_argument('--start', type=float, default=0., help='start second for real-world inference')
  parser.add_argument('--end', type=float, default=100000., help='end second for real-world inference')

  args = parser.parse_args()

  return args
