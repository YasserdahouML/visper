import cv2
import numpy as np
from scipy import signal
import os
import subprocess 
import tempfile

CROP_SCALE = 0.4
WINDOW_MARGIN = 12
START_IDX, STOP_IDX = 3, 5
STABLE_POINTS = (36, 45, 33, 48, 54)
CROP_HEIGHT, CROP_WIDTH = 96, 96

# PATH='/home/users/u100438/home200093/dataset_release/'
REFERENCE = np.load(os.path.join( os.path.dirname(__file__), '20words_mean_face.npy'))


def crop_and_save_audio(mp4_path: str, saving_path:str, start_audio: float, end_audio: float) -> None:
    """
    Crops original audio corresponding to the start and end time.
    Saves it as wav file with single channel and 16kHz sampling rate.
    
    :param mp4_path: str, path to original video.
    :param saving_path: str, path where audio will be saved. SHOULD END WITH .wav
    :param start_audio: float, start time of clip in seconds
    :param end_audio: float, end time of clip in seconds
    :return: None.
    """

    # write audio.
    command = f"ffmpeg -loglevel error -y -i {mp4_path} -ss {start_audio} -to {end_audio} -vn -acodec pcm_s16le -ar 16000 -ac 1 {saving_path}"
    subprocess.call(command, shell=True)



def crop_video(vid_path: str, clip_data: dict):
    '''
    Reads the video frames of video (in vid_path) between clip_data['start'] and clip_data['end'] times.
    Crops the faces in these frames using bounding boxes given by clip_data['bboxs']
    Returns sequence of faces and clip['landmarks'] aligned to 224x224 resolution.
    '''
    cap = cv2.VideoCapture(vid_path)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame, end_frame = round(clip_data['start']*25), round(clip_data['end']*25)
    clip_frames = end_frame - start_frame
    assert end_frame <= num_frames, f'End frame ({end_frame}) exceeds total number of frames ({num_frames})'
    
    landmarks_n, bboxs_n = np.array(clip_data['landmarks']), np.array(clip_data['bboxs'])
    bboxs = np.multiply(bboxs_n, [frame_width, frame_height, frame_width, frame_height])
    landmarks =  np.multiply(landmarks_n, [frame_width, frame_height])   
    assert len(landmarks) == clip_frames, f'Landmarks length ({len(landmarks)}) does not match the number of frames in the clip ({clip_frames})'
    
    dets = {'x':[], 'y':[], 's':[]}
    for det in bboxs:
        dets['s'].append(max((det[3]-det[1]),(det[2]-det[0]))/2)     
        dets['y'].append((det[1]+det[3])/2) # crop center x     
        dets['x'].append((det[0]+det[2])/2) # crop center y
    
    # Smooth detections  
    dets['s'] = signal.medfilt(dets['s'],kernel_size=13)     
    dets['x'] = signal.medfilt(dets['x'],kernel_size=13)  
    dets['y'] = signal.medfilt(dets['y'],kernel_size=13)
    
    image_seq = []
    current_frame = start_frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    while current_frame < end_frame:
        ret, frame = cap.read()
        count = current_frame - start_frame
        current_frame += 1

        if not ret:
            break
    
        bs  = dets['s'][count]   # Detection box size   
        bsi = int(bs*(1+2*CROP_SCALE))  # Pad videos by this amount 
        
        image = frame
        lands = landmarks[count]
        
        frame_ = np.pad(image,((bsi,bsi),(bsi,bsi),(0,0)), 'constant', constant_values=(110,110))    
        my  = dets['y'][count]+bsi  # BBox center Y    
        mx  = dets['x'][count]+bsi  # BBox center X
        
        face = frame_[int(my-bs):int(my+bs*(1+2*CROP_SCALE)),int(mx-bs*(1+CROP_SCALE)):int(mx+bs*(1+CROP_SCALE))]

        ## lands translation and scaling
        lands[:,0] -= int(mx-bs*(1+CROP_SCALE) - bsi) 
        lands[:,1] -= int(my - bs - bsi) 
        lands[:,0] *= (224/face.shape[1])
        lands[:,1] *= (224/face.shape[0])
        
        image_seq.append(cv2.resize(face,(224,224)))
            
    image_seq = np.array(image_seq)
    
    return image_seq, landmarks



def landmarks_interpolate(landmarks):
        """landmarks_interpolate.

        :param landmarks: List, the raw landmark (in-place)
        """
        valid_frames_idx = [idx for idx, _ in enumerate(landmarks) if _ is not None]
        if not valid_frames_idx:
            return None
        for idx in range(1, len(valid_frames_idx)):
            if valid_frames_idx[idx] - valid_frames_idx[idx-1] == 1:
                continue
            else:
                landmarks = linear_interpolate(landmarks, valid_frames_idx[idx-1], valid_frames_idx[idx])
        valid_frames_idx = [idx for idx, _ in enumerate(landmarks) if _ is not None]
        # -- Corner case: keep frames at the beginning or at the end failed to be detected.
        if valid_frames_idx:
            landmarks[:valid_frames_idx[0]] = [landmarks[valid_frames_idx[0]]] * valid_frames_idx[0]
            landmarks[valid_frames_idx[-1]:] = [landmarks[valid_frames_idx[-1]]] * (len(landmarks) - valid_frames_idx[-1])
        valid_frames_idx = [idx for idx, _ in enumerate(landmarks) if _ is not None]
        assert len(valid_frames_idx) == len(landmarks), "not every frame has landmark"
        return landmarks


def crop_patch(image_seq, landmarks):
        """crop_patch.

        :param video_pathname: str, the filename for the processed video.
        :param landmarks: List, the interpolated landmarks.
        """
        frame_idx = 0
        sequence = []
        for frame in image_seq:
            
            window_margin = min(WINDOW_MARGIN // 2, frame_idx, len(landmarks) - 1 - frame_idx)
            smoothed_landmarks = np.mean([landmarks[x] for x in range(frame_idx - window_margin, frame_idx + window_margin + 1)], axis=0)
            smoothed_landmarks += landmarks[frame_idx].mean(axis=0) - smoothed_landmarks.mean(axis=0)
            transformed_frame, transformed_landmarks = affine_transform(frame, smoothed_landmarks, REFERENCE)
            sequence.append( cut_patch( transformed_frame, transformed_landmarks[START_IDX : STOP_IDX], CROP_HEIGHT//2, CROP_WIDTH//2,))
            frame_idx += 1

        return np.array(sequence)

def affine_transform(frame, landmarks, reference,
        target_size=(256, 256),
        reference_size=(256, 256),
        stable_points=STABLE_POINTS,
        interpolation=cv2.INTER_LINEAR,
        border_mode=cv2.BORDER_CONSTANT,
        border_value=0
    ):
        """affine_transform.

        :param frame: numpy.array, the input sequence.
        :param landmarks: List, the tracked landmarks.
        :param reference: numpy.array, the neutral reference frame.
        :param target_size: tuple, size of the output image.
        :param reference_size: tuple, size of the neural reference frame.
        :param stable_points: tuple, landmark idx for the stable points.
        :param interpolation: interpolation method to be used.
        :param border_mode: Pixel extrapolation method .
        :param border_value: Value used in case of a constant border. By default, it is 0.
        """

        lands = [landmarks[x] for x in range(5)]
            
        stable_reference = np.vstack([reference[x] for x in stable_points])
        stable_reference[:, 0] -= (reference_size[0] - target_size[0]) / 2.0
        stable_reference[:, 1] -= (reference_size[1] - target_size[1]) / 2.0

        # Warp the face patch and the landmarks
        transform = cv2.estimateAffinePartial2D(np.vstack(lands), stable_reference, method=cv2.LMEDS)[0]
        transformed_frame = cv2.warpAffine(
            frame,
            transform,
            dsize=(target_size[0], target_size[1]),
            flags=interpolation,
            borderMode=border_mode,
            borderValue=border_value,
        )
        transformed_landmarks = np.matmul(landmarks, transform[:, :2].transpose()) + transform[:, 2].transpose()

        return transformed_frame, transformed_landmarks
    
    
def cut_patch(img, landmarks, height, width, threshold=5):
    """cut_patch.

    :param img: ndarray, an input image.
    :param landmarks: ndarray, the corresponding landmarks for the input image.
    :param height: int, the distance from the centre to the side of of a bounding box.
    :param width: int, the distance from the centre to the side of of a bounding box.
    :param threshold: int, the threshold from the centre of a bounding box to the side of image.
    """
    center_x, center_y = np.mean(landmarks, axis=0)

    if center_y - height < 0:                                                
        center_y = height                                                    
    if center_y - height < 0 - threshold:                                    
        raise Exception('too much bias in height')                           
    if center_x - width < 0:                                                 
        center_x = width                                                     
    if center_x - width < 0 - threshold:                                     
        raise Exception('too much bias in width')                            
                                                                             
    if center_y + height > img.shape[0]:                                     
        center_y = img.shape[0] - height                                     
    if center_y + height > img.shape[0] + threshold:                         
        raise Exception('too much bias in height')                           
    if center_x + width > img.shape[1]:                                      
        center_x = img.shape[1] - width                                      
    if center_x + width > img.shape[1] + threshold:                          
        raise Exception('too much bias in width')                            
                                                                             
    cutted_img = np.copy(img[ int(round(center_y) - round(height)): int(round(center_y) + round(height)),
                         int(round(center_x) - round(width)): int(round(center_x) + round(width))])
    return cutted_img


def crop_face(image_seq, landmarks):
    # Interpolate the landmarks
    preprocessed_landmarks = landmarks_interpolate(list(landmarks))
    # crop the face to obtain a sequence of 96x96 sized mouth rois
    crop_seq = crop_patch(image_seq, preprocessed_landmarks)
    
    return crop_seq

def merge_audio_video(tmp_path, audio_path, save_video_path):
    # Will merge the corresponding audio and video tracks of the clip. The associated .wav file will be removed.
    command = f"ffmpeg -loglevel error -y -i {tmp_path} -i {audio_path} -c:v libx264 -c:a aac -ar 16000 -ac 1 {save_video_path}"
    tval = subprocess.call(command, shell=True)
    tval = subprocess.call(f'rm {tmp_path}', shell=True)
    tval = subprocess.call(f'rm {audio_path}', shell=True)

def convert_ffmpeg(vid_path):
    # converts the mpeg4 video to h264 using ffmpeg. Saves disk space, but takes additional time
    tmp_path = vid_path[:-4] + 'temp2.mp4'
    cmd = f"cp {vid_path} {tmp_path}"
    tval = subprocess.call(cmd, shell=True)
    cmd = f"ffmpeg -loglevel error -i {tmp_path} -r 25 -vcodec libx264 -q:v 1 -y {vid_path}"
    tval = subprocess.call(cmd, shell=True)
    tval = subprocess.call(f"rm {tmp_path}", shell=True)
    
    
def write_video(save_video_path, crop_seq, audio_path=None, merge_audio=False, use_ffmpeg=False):
    # Writes the clip video to disk. Merges with audio if enabled
    tmp_path = save_video_path.replace('.mp4','_temp.mp4') if merge_audio else save_video_path
    vid_writer = cv2.VideoWriter(tmp_path, cv2.VideoWriter_fourcc(*'mp4v'), 25, (96, 96))
    for ci in crop_seq: 
        vid_writer.write(ci)
    vid_writer.release()
    if use_ffmpeg and not merge_audio:
        convert_ffmpeg(tmp_path)
        
    if merge_audio:
        merge_audio_video(tmp_path, audio_path, save_video_path)
        


