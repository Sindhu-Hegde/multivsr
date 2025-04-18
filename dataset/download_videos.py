import argparse
import numpy as np
import os, subprocess
from tqdm import tqdm
import math

import traceback
import executor
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

os.environ["LC_ALL"]="en_US.utf-8"
os.environ["LANG"]="en_US.utf-8"

parser = argparse.ArgumentParser(description="Code to download the videos from the input filelist of ids")
parser.add_argument('--file', type=str, required=True, help="Path to the input filelist of ids")
parser.add_argument('--video_root', type=str, required=True, help="Path to the directory to save the videos")

args = parser.parse_args()

def is_valid_video(file_path):
	
	'''
	This function validates the video file using the following checks:
		(i) Check if duration > 0
		(ii) Check if audio is present
	
	Args:
		- file_path (str): Path to the video file.
	Returns:
		- True if the video is valid, False otherwise.
	'''

	if not os.path.exists(file_path):
		return False  # File does not exist

	# Use ffmpeg to get duration
	try:
		result = subprocess.run(
			["ffmpeg", "-i", file_path, "-f", "null", "-"],
			stdout=subprocess.DEVNULL, 
			stderr=subprocess.DEVNULL  
		)
		return result.returncode == 0  # If return code is 0, it's a valid video
	except Exception:
		return False  # If ffmpeg fails, assume it's invalid

	# Check if audio exists
	cmd_audio = f'ffmpeg -i "{video_path}" -map 0:a:0 -f null - 2>&1 | grep "Output"'
	try:
		audio_output = subprocess.check_output(cmd_audio, shell=True, text=True)
		if not audio_output.strip():
			print(f"Invalid video (no audio): {video_path}")
			os.remove(video_path)
			return False
	except Exception:
		print(f"Error checking audio: {video_path}")
		os.remove(video_path)
		return False


def mp_handler(vid, result_dir):

	'''
	This function handles the multiprocessing of the video download

	Args:
		- vid (str): Video ID.
		- result_dir (str): Directory to save the video.
	'''

	try:
		vid = vid.strip()
		video_link = "https://www.youtube.com/watch?v={}".format(vid)
		output_fname = os.path.join(result_dir, "{}.mp4".format(vid))
	
		if os.path.exists(output_fname):
			# Validate file
			if is_valid_video(output_fname):
				return

		# Download the video
		cmd = "yt-dlp --geo-bypass -f b --format=mp4 -o {} {}".format(output_fname, video_link)
		subprocess.call(cmd, shell=True)

		# Validate file and delete if invalid
		if not is_valid_video(output_fname):
			print(f"Invalid file detected: {output_fname}. Deleting...")
			os.remove(output_fname)

	except KeyboardInterrupt:
		exit(0)
	except:
		traceback.print_exc()

def download_data(args):

	'''
	This function downloads the videos from the given csv file

	Args:
		- args (argparse.Namespace): Arguments.
	'''

	# Load the dataset csv file with annotations
	with open(args.file, 'r') as f:
		filelist = f.readlines()
	print("Total files: ", len(filelist))

	# Create the result directory
	if not os.path.exists(args.video_root):
		os.makedirs(args.video_root)

	p = ThreadPoolExecutor(7)
	futures = [p.submit(mp_handler, f, args.video_root) for f in filelist]
	res = [r.result() for r in tqdm(as_completed(futures), total=len(futures))]

if __name__ == '__main__':

	# Download the videos
	download_data(args)
