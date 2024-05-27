import json
import numpy as np
import os
from tqdm import tqdm
import subprocess
from glob import glob
import argparse
import time
from utils import crop_video, crop_face, write_video, crop_and_save_audio, remove_characters
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import sys

'''
Crop the untrimmed videos into multiple clips using corresponding start and end times, bounding boxes and face landmarks.

Usage: 
python crop_videos.py --video_dir /path/to/25-fps-videos --save_path /path/to/save/the/clips --json_path /path/to/json/folder

To save videos using ffmpeg, add "--use_ffmpeg True". This takes additional time but saves disk space.

To additionally save audio as separate wav files, add "--save_audio True"

To merge audio with video and save as a single mp4, add "--merge_audio True"
'''


def write_clip(metadata, vid_p, args):
    '''
    param metadata: dict containing start, end, bounding boxes, landmarks
    param vid_p: path to original untrimmed video at 25fps
    param args: main args 
    '''
    for k, clip in enumerate(metadata):
        # get the clip frames and corresponding landmarks
        video, landmarks = crop_video(vid_p, clip)
        # get the cropped sequence around the mouth using the landmarks
        crop_seq = crop_face(video, landmarks)
        save_video_path = os.path.join(args.save_path, 'videos', vid_p.split('/')[-1][:-4], f'{str(k).zfill(5)}.mp4')
        save_audio_path = save_video_path.replace('.mp4','.wav')
        # get the audio part of the clip
        if args.save_audio or args.merge_audio:
            crop_and_save_audio(vid_p, save_audio_path, clip['start'], clip['end'])
        # write clip to disk
        write_video(save_video_path, crop_seq, save_audio_path, merge_audio=args.merge_audio, use_ffmpeg=args.use_ffmpeg)
    return 


def main(args):
    savepath = args.save_path
    json_path = args.json_path
    vid_dir = args.video_dir

    ## Get the list of downloaded videos
    video_list = glob(os.path.join(vid_dir, '*.mp4'))
    ## Load the jsons
    json_files = glob(os.path.join(json_path, '*.json'))
    print(f'Loading {len(json_files)} jsons from {json_path}')
    data = {}
    for ji in json_files:
        temp = json.load(open(ji,'r'))
        data.update(temp)
    print(f'Total number of videos {len(video_list)}. Json length {len(data)}')
    video_ids = list(data.keys())
    count_clips = 0

    futures = []
    writer_str = 'Ffmpeg' if args.use_ffmpeg else 'cv2.VideoWriter'
    print(f'Using {writer_str} to save the cropped clips.')
    
    with tqdm(total=len(video_ids), file=sys.stdout) as progress:
        with ProcessPoolExecutor() as executor:
            for z in video_ids:
                idx = [k for k, i in enumerate(video_list) if z in i]        
                metadata = data[z]
                vid_p = video_list[idx[0]]
                os.makedirs(os.path.join(savepath, 'videos', vid_p.split('/')[-1][:-4]), exist_ok=True)
                future = executor.submit(write_clip, metadata, vid_p, args)
                futures.append(future)

            for _ in as_completed(futures):
                progress.update()
                    
    print(f'Cropping videos completed.')
    print(f'Getting the metadata.')
    labels = {}
    for z in tqdm(video_ids):
        metadata = data[z]
        for k, clip in enumerate(metadata):
            labk = clip['label']
            fi = os.path.join(savepath, 'videos', z, f'{str(k).zfill(5)}.mp4')
            if os.path.isfile(fi):
                labels[fi] = {'label': remove_characters(labk, '<ar>'), 'length': len(clip['landmarks'])}
    label_file = f'{args.save_path}/metadata.json'
    with open(label_file, 'w', encoding='utf-8') as f:
        json.dump(labels, f)
    print(f'Metadata saved to {label_file}')
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ViSpeR crop videos')
    parser.add_argument('--save_path', type=str, default='', help='Path for saving.')
    parser.add_argument('--json_path', type=str, default='', help='Json folder path')
    parser.add_argument('--video_dir', type=str, default='', help='Path to directory where original videos are stored.')
    parser.add_argument('--save_audio', type=bool, default=False, help='Whether to save audio info.')
    parser.add_argument('--merge_audio', type=bool, default=False, help='Whether to merge audio with the video when saving.')
    parser.add_argument('--use_ffmpeg', type=bool, default=False, help='Whether to use ffmpeg instead of cv2 for saving the video.')
    
    args = parser.parse_args()
    tic = time.time()
    main(args)
    print(f'Elpased total time for processing: {time.time()-tic} seconds')
