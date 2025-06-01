import cv2
import numpy as np
import argparse
import os
import pickle
import subprocess
from scipy import signal
from glob import glob
from shutil import copy

def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess video files for lipreading')
    parser.add_argument('--videos_folder', type=str, required=True, help='Folder containing unprocessed video files')
    parser.add_argument('--data_root', type=str, required=True, help='Root directory for the processed dataset')
    parser.add_argument('--temp_dir', type=str, required=True, help='Directory for temporary files')
    parser.add_argument('--frame_rate', type=int, default=25, help='Frame rate of the video')

    parser.add_argument('--rank', type=int, default=0, help='Rank of the current process for distributed processing')
    parser.add_argument('--nshard', type=int, default=1, help='Total number of shards for distributed processing')
    args = parser.parse_args()
    return args

args = parse_args()

# Append rank and shard to temp_dir to create unique directories for distributed processing
args.temp_dir = os.path.join(args.temp_dir, f"rank{args.rank}_of_{args.nshard}")

# Create the temporary directory if it doesn't exist
os.makedirs(args.temp_dir, exist_ok=True)


def crop_video(track):
    videofile = os.path.join(args.videos_folder, track.split('/')[-2] + '.mp4')
    temp_videofile = os.path.join(args.temp_dir, track.split('/')[-2] + '.mp4')

    command = [
        'ffmpeg',
        '-i', videofile,
        '-r', str(args.frame_rate),
        '-y',  # Overwrite if exists
        temp_videofile
    ]
    subprocess.call(command)
    
    videofile = temp_videofile

    if not os.path.exists(videofile):
        raise ValueError(f"temp video file {videofile} could not be created")
        return
    
    cropfile = track.replace('.pckl', '.mp4')
    if os.path.exists(cropfile):
        print(f"Cropped video file {cropfile} already exists")
        return

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vOut = cv2.VideoWriter(cropfile, fourcc, args.frame_rate, (224,224))

    dets = {'x':[], 'y':[], 's':[]}
    with open(track, 'rb') as f:
      track = pickle.load(f)

    for det in track['bbox']:

      dets['s'].append(max((det[3]-det[1]),(det[2]-det[0]))/2) 
      dets['y'].append((det[1]+det[3])/2) # crop center x 
      dets['x'].append((det[0]+det[2])/2) # crop center y

    # Smooth detections
    dets['s'] = signal.medfilt(dets['s'],kernel_size=13)   
    dets['x'] = signal.medfilt(dets['x'],kernel_size=13)
    dets['y'] = signal.medfilt(dets['y'],kernel_size=13)

    
    frame_no_to_start = track['frame'][0]

    video_stream = cv2.VideoCapture(videofile)
    video_stream.set(cv2.CAP_PROP_POS_FRAMES, frame_no_to_start)
    for fidx, frame in enumerate(range(track['frame'][0], track['frame'][-1] + 1)):

      cs  = 0.4

      bs  = dets['s'][fidx]   # Detection box size

      bsi = int(bs*(1+2*cs))  # Pad videos by this amount 

      image = video_stream.read()[1] #video_stream.next().asnumpy()[..., ::-1]
 
      frame = np.pad(image,((bsi,bsi),(bsi,bsi),(0,0)), 'constant', constant_values=(110,110))

      my  = dets['y'][fidx]+bsi  # BBox center Y
      mx  = dets['x'][fidx]+bsi  # BBox center X

      face = frame[int(my-bs):int(my+bs*(1+2*cs)),int(mx-bs*(1+cs)):int(mx+bs*(1+cs))]

      vOut.write(cv2.resize(face,(224,224)))
    video_stream.release()

    # audiotmp    = os.path.join(args.temp_dir, 'audio.wav')
    # audiostart  = (track['frame'][0]) / args.frame_rate
    # audioend    = (track['frame'][-1]+1) / args.frame_rate

    vOut.release()

    # ========== CROP AUDIO FILE ==========

    # command = ("ffmpeg -y -i %s -ss %.3f -to %.3f %s" % (os.path.join(args.temp_dir, 'audio.wav'),audiostart,audioend,audiotmp)) 
    # output = subprocess.call(command, shell=True, stdout=None)

    # ========== COMBINE AUDIO AND VIDEO FILES ==========
    # copy(audiotmp, cropfile.replace('.mp4', '.wav'))
    
    print('Written %s'%cropfile)

if __name__ == "__main__":
    clips = list(glob(os.path.join(args.data_root, '*/*.pckl')))
    
    # Calculate the number of clips per shard
    clips_per_shard = int(np.ceil(len(clips) / args.nshard))
    # Calculate the start and end indices for this rank
    start_idx = args.rank * clips_per_shard
    end_idx = start_idx + clips_per_shard if args.rank < args.nshard - 1 else len(clips)
    # Get the subset of clips for this rank
    clips = clips[start_idx:end_idx]
    print(f"Rank {args.rank}: Processing {len(clips)} clips from index {start_idx} to {end_idx-1}")

    for clip in clips:
        crop_video(clip)
