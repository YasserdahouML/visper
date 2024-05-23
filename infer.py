import os
import hydra
import torch
import torchaudio
import torchvision
from lightning_vsr import ModelModule, get_beam_search_decoder
from datamodule.av_dataset import cut_or_pad
from datamodule.transforms import AudioTransform, VideoTransform
from WER.measures import get_wer as wer
from WER.measures import get_cer as cer
import numpy as np
import torch.multiprocessing as mp
import time, re
from tqdm import tqdm
from subprocess import CalledProcessError, run


TEMP_DIR = os.path.dirname(os.path.realpath(__file__)) + '/temp_predictions'
SAVE_DIR = TEMP_DIR.replace('temp','final')
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)
LANG_TOKEN = {'english': 2, 'arabic': 3, 'spanish': 4, 'french': 5, 'chinese': 6}
LANG_DICT = {2: '<en>', 3: '<ar>', 4: '<es>', 5: '<fr>', 6: '<zh>'}


class InferencePipeline(torch.nn.Module):
    def __init__(self, cfg, detector="mediapipe"):
        super(InferencePipeline, self).__init__()
        self.modality = cfg.data.modality
        self.multi_lang = cfg.data.multi_lang
        if self.modality in ["audio", "audiovisual"]:
            self.audio_transform = AudioTransform(subset="test")
        if self.modality in ["video", "audiovisual"]:
            self.video_transform = VideoTransform(subset="test")

        self.modelmodule = ModelModule(cfg)
        self.modelmodule.cuda().eval()


    def forward(self, data_filename, lang):
        data_filename = os.path.abspath(data_filename)
        assert os.path.isfile(data_filename), f"data_filename: {data_filename} does not exist."

        if self.modality in ["audio", "audiovisual"]:
            try:
                audio, sample_rate = self.load_audio(data_filename)
            except:
                if os.path.isfile(data_filename.replace('.mp4', '.wav')):
                    audio_filename = data_filename.replace('.mp4', '.wav')
                    audio, sample_rate = self.load_audio(audio_filename)
                else:
                    raise ValueError(f"Unable to load audio for {data_filename}")
            audio = self.audio_process(audio, sample_rate)
            audio = audio.transpose(1, 0)
            audio = self.audio_transform(audio)

        if self.modality in ["video", "audiovisual"]:
            video = self.load_video(data_filename)
            video = torch.tensor(video)
            video = video.permute((0, 3, 1, 2))
            video = self.video_transform(video).cuda()
        
        if self.modality == "video":
            with torch.no_grad():
                transcript = self.modelmodule(video, lang=lang)
        elif self.modality == "audio":
            with torch.no_grad():
                transcript = self.modelmodule(audio)
        elif self.modality == "audiovisual":
            rate_ratio = len(audio) // len(video)
            if rate_ratio > 670 or rate_ratio < 530:
                print(f"WARNING: Inconsistent frame ratio for {data_filename}. Found audio length: {len(audio)}, video length: {len(video)}. It might affect the performance.")
            if rate_ratio != 640:
                audio = cut_or_pad(audio, len(video) * 640)
            with torch.no_grad():
                transcript = self.modelmodule(video, audio, lang=lang)
                
        return transcript


    def load_audio(self, data_filename: str, sr: int = 16000):
        """
        Sourced from https://github.com/openai/whisper/blob/main/whisper/audio.py
        Open an audio file and read as mono waveform, resampling as necessary

        Parameters
        ----------
        file: str
            The audio file to open

        sr: int
            The sample rate to resample the audio if necessary

        Returns
        -------
        A NumPy array containing the audio waveform, in float32 dtype.
        """

        # This launches a subprocess to decode audio while down-mixing
        # and resampling as necessary.  Requires the ffmpeg CLI in PATH.
        # fmt: off
        cmd = ["ffmpeg", "-nostdin", "-threads", "0", "-i", data_filename, "-f", "s16le", "-ac", "1", "-acodec", "pcm_s16le", "-ar", str(sr), "-"]
        # fmt: on
        try:
            out = run(cmd, capture_output=True, check=True).stdout
        except CalledProcessError as e:
            raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

        waveform = np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0
        waveform = torch.FloatTensor(waveform).unsqueeze(0)
        
        return waveform, sr

    def load_video(self, data_filename):
        return torchvision.io.read_video(data_filename, pts_unit="sec")[0].numpy()

    def audio_process(self, waveform, sample_rate, target_sample_rate=16000):
        if sample_rate != target_sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, sample_rate, target_sample_rate
            )
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        return waveform




def inference_on_single_gpu(gpu, cfg, files_chunk, save_name):
    torch.cuda.set_device(gpu)
    pipeline = InferencePipeline(cfg)
    pipeline.modelmodule.cuda(gpu)
    pipeline.modelmodule = torch.nn.DataParallel(pipeline.modelmodule, device_ids=[gpu])

    lens, wers_scores, res_dict = [], [], {}
    lang = LANG_TOKEN[cfg.infer_lang]
    
    with tqdm(total=len(files_chunk), position=0, leave=True) as pbar:
        for p in tqdm(files_chunk, position=0, leave=True):
            transcript = pipeline(p[1], lang)
            pred = str(transcript).replace("<unk>", '').lower()
            pbar.update()
            
            gt = str(p[4]).lower()
            
            weri = wer(pred, gt) if lang != 6 else cer(pred, gt)
            lens.append(len(gt.split()) if lang != 6 else len(gt))
            wers_scores.append(weri*lens[-1])
            res_dict[p[1]] = (transcript, gt, weri, lens[-1])
    
    np.save(f'{TEMP_DIR}/{save_name}_{gpu}.npy', {'res_dict': res_dict, 'lens': lens, 'wers_scores': wers_scores})


@hydra.main(config_path="conf", config_name="config")
def main(cfg):
    num_gpus = torch.cuda.device_count()
    assert num_gpus > 0, "No GPUs found."
    
    print(f'Running inference on multilingual model with language token set to {LANG_TOKEN[cfg.infer_lang]} for {cfg.infer_lang} language')
    print(f'Using infer path {cfg.infer_path}')

    files = np.load(cfg.infer_path, allow_pickle=True)
    print(f'Total number of video files to evaluate: {len(files)}')

    save_name = cfg.infer_path.split('/')[-1][:-4] +  '_'.join(cfg.ckpt_path.split('/')[-2:]).replace('ckpt', '').replace('.pth','') + '_' + cfg.infer_lang

    # Split the files list into chunks for each GPU
    files_chunks = np.array_split(files, num_gpus)
    # Use multiprocessing to run inference on each GPU
    processes = []
    manager = mp.Manager()
    all_results = manager.list()
    tic = time.time()

    for gpu in range(num_gpus):
        p = mp.Process(target=inference_on_single_gpu, args=(gpu, cfg, files_chunks[gpu], save_name))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
    print(f'All processes done in {time.time()-tic}s')
    
    all_wers, all_lens, all_res_dict = [], [], {}
    for gpu in range(num_gpus):
        transcripts = np.load(f'{TEMP_DIR}/{save_name}_{gpu}.npy', allow_pickle=True).item()
        all_res_dict.update(transcripts['res_dict'])
        all_lens += list(transcripts['lens'])
        all_wers += list(transcripts['wers_scores'])
        print(len(list(transcripts['wers_scores'])))
        
    final_wer = np.sum(all_wers) / np.sum(all_lens)
    print(f'Total WER {final_wer*100} for {len(all_wers)} videos')
    all_res_dict.update({"wer": final_wer*100, "all_wers": all_wers, "all_lens": all_lens})
    
    ## Save the merged results
    np.save(f'{SAVE_DIR}/results_{save_name}.npy', all_res_dict)
    print(f'Results saved to {SAVE_DIR}/results_{save_name}.npy')
    
    ## Remove the temporary prediction files
    for gpu in range(num_gpus):
        os.system(f'rm -f {TEMP_DIR}/{save_name}_{gpu}.npy')  
        

if __name__ == "__main__":
    main()