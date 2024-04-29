import os

import hydra
import torch
import torchaudio
import torchvision
from lightning import ModelModule
from datamodule.transforms import AudioTransform, VideoTransform
from WER.measures import get_wer as wer_ma
from WER.measures import get_cer as cer_ma
import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp
import time

SAVE_DIR = os.path.dirname(os.path.realpath(__file__)) + '/predictions'


class InferencePipeline(torch.nn.Module):
    def __init__(self, cfg, detector="mediapipe"):
        super(InferencePipeline, self).__init__()
        self.modality = cfg.data.modality
        self.multi_lang = cfg.data.multi_lang
        if self.modality in ["audio", "audiovisual"]:
            self.audio_transform = AudioTransform(subset="test")
        elif self.modality in ["video", "audiovisual"]:
            # if detector == "mediapipe":
            #     from preparation.detectors.mediapipe.detector import LandmarksDetector
            #     from preparation.detectors.mediapipe.video_process import VideoProcess
            #     self.landmarks_detector = LandmarksDetector()
            #     self.video_process = VideoProcess(convert_gray=False)
            # elif detector == "retinaface":
            #     from preparation.detectors.retinaface.detector import LandmarksDetector
            #     from preparation.detectors.retinaface.video_process import VideoProcess
            #     self.landmarks_detector = LandmarksDetector(device="cuda:0")
            #     self.video_process = VideoProcess(convert_gray=False)
            self.video_transform = VideoTransform(subset="test")

        self.modelmodule = ModelModule(cfg)
        self.modelmodule.load_state_dict(
            torch.load(cfg.ckpt_path, map_location=lambda storage, loc: storage)['state_dict']
        )
        self.modelmodule.cuda().eval()

    def forward(self, data_filename, lang):
        data_filename = os.path.abspath(data_filename)
        assert os.path.isfile(data_filename), f"data_filename: {data_filename} does not exist."

        if self.modality == "audio":
            audio, sample_rate = self.load_audio(data_filename)
            audio = self.audio_process(audio, sample_rate)
            audio = audio.transpose(1, 0)
            audio = self.audio_transform(audio)
            with torch.no_grad():
                transcript = self.modelmodule(audio)

        if self.modality == "video":
            #landmarks = self.landmarks_detector(data_filename)
            video = self.load_video(data_filename)
            #video = self.video_process(video, landmarks)
            video = torch.tensor(video)
            video = video.permute((0, 3, 1, 2))
            video = self.video_transform(video).cuda()
            with torch.no_grad():
                transcript = self.modelmodule(video, lang=lang)

        return transcript

    def load_audio(self, data_filename):
        waveform, sample_rate = torchaudio.load(data_filename, normalize=True)
        return waveform, sample_rate

    def load_video(self, data_filename):
        return torchvision.io.read_video(data_filename, pts_unit="sec")[0].numpy()

    def audio_process(self, waveform, sample_rate, target_sample_rate=16000):
        if sample_rate != target_sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, sample_rate, target_sample_rate
            )
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        return waveform




def inference_on_single_gpu(gpu, cfg, files_chunk):
    torch.cuda.set_device(gpu)
    pipeline = InferencePipeline(cfg)
    pipeline.modelmodule.cuda(gpu)
    pipeline.modelmodule = torch.nn.DataParallel(pipeline.modelmodule, device_ids=[gpu])

    results, lens, wers_scores = [], [], []
    for p in files_chunk:
        lang = p[3][0] if cfg.data.multi_lang else None
        transcript = pipeline(p[1], lang)        
        pred = str(transcript).replace("<unk>", '').lower()
        hypo = pred.split()
        words_t1, idx = np.unique(hypo, return_index=True)
        if len(hypo) - len(words_t1) > 3:
            hypo = [hypo[i] for i in sorted(idx)]
            pred = ' '.join(hypo)
        print(f"transcript: {pred}")
        
        wer = wer_ma(pred, str(p[4])) if lang != 6 else cer_ma(pred, str(p[4]))
        lens.append(len(p[4].split()) if lang != 6 else len(p[4]))
        wers_scores.append(wer*lens[-1])
        
        results.append(transcript)
    
    epoch = cfg.ckpt_path[-7:-5]
    np.save(f'{SAVE_DIR}/preds_ckpt{epoch}_{gpu}.npy', results)
    np.save(f'{SAVE_DIR}/results_ckpt{epoch}_{gpu}.npy', {'lens': lens, 'wers_scores': wers_scores})
    
    
    return results, lens, wers_scores


@hydra.main(config_path="conf", config_name="config")
def main(cfg):
    num_gpus = torch.cuda.device_count()
    print(num_gpus)
    assert num_gpus > 0, "No GPUs found."

    files = np.load(cfg.infer_path, allow_pickle=True)
    found = []
    for fi in files:
        found.append(os.path.isfile(fi[1]))
    missing = [files[k][1] for k in range(len(files)) if not found[k]]
    print(f'Number of missing video files: {len(missing)}')
    if len(missing) > 0:
        print(f'Saving missing files list to {cfg.infer_path[:-4]}_missing.npy')
        np.save(f'{cfg.infer_path[:-4]}_missing.npy', missing)

    # Split the files list into chunks for each GPU
    files_chunks = np.array_split(files, num_gpus)
    # Use multiprocessing to run inference on each GPU
    processes = []
    manager = mp.Manager()
    # all_results = manager.list()
    # tic = time.time()

    # for gpu in range(num_gpus):
    #     p = mp.Process(target=inference_on_single_gpu, args=(gpu, cfg, files_chunks[gpu]))
    #     p.start()
    #     processes.append(p)

    # for p in processes:
    #     p.join()

    # print(f'All processes done in {time.time()-tic}s')
    
    epoch = cfg.ckpt_path[-7:-5]
    print(epoch)
    all_transcripts, all_wers, all_lens = [], [], []
    for gpu in range(num_gpus):
        transcripts = np.load(f'{SAVE_DIR}/preds_ckpt{epoch}_{gpu}.npy')
        res = np.load(f'{SAVE_DIR}/results_ckpt{epoch}_{gpu}.npy',allow_pickle=True).item()
        all_transcripts += list(transcripts)
        all_lens += list(res['lens'])
        all_wers += list(res['wers_scores'])
        
    final_wer = np.sum(all_wers) / np.sum(all_lens)
    print(f'Total WER {final_wer*100} for {len(all_wers)} videos')
    
    GT = np.load('/home/yasser/vhisper_vsr/labels/test_multilang_all.npy',allow_pickle=True)
    LANG = {2: 'English', 3: 'Arabic', 4: 'French', 5: 'Spanish', 6: 'Chinese'}
    for lang in range(2,7):
        idx = [k for k in range(len(GT)) if GT[k][3][0] == lang]
        if lang != 6:
            wer = 100.0*np.sum([all_wers[k] for k in idx])/np.sum([all_lens[k] for k in idx])
        else:
            cers, lens = [], []
            for k in idx:
                ceri = cer_ma(all_transcripts[k], str(GT[k][4]))
                cers.append(ceri*len(str(GT[k][4])))
                lens.append(len(str(GT[k][4])))
            wer = 100.0*np.sum(cers)/np.sum(lens)
                
        print(f'{LANG[lang]} WER / CER: {wer}')

if __name__ == "__main__":
    main()