import torch
import torchaudio
from cosine import WarmupCosineScheduler
from datamodule.transforms import TextTransform
import os

# for testing
from espnet.asr.asr_utils import add_results_to_json
from espnet.nets.batch_beam_search import BatchBeamSearch
from espnet.nets.scorers.length_bonus import LengthBonus
from espnet.nets.scorers.ctc import CTCPrefixScorer
from pytorch_lightning import LightningModule
from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask


def compute_word_level_distance(seq1, seq2):
    return torchaudio.functional.edit_distance(
        seq1.lower().split(), seq2.lower().split()
    )


class ModelModule(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.cfg = cfg
        if self.cfg.data.modality == "audio":
            self.backbone_args = self.cfg.model.audio_backbone
        elif self.cfg.data.modality == "video":
            self.backbone_args = self.cfg.model.visual_backbone
        elif self.cfg.data.modality == "audiovisual":
            self.backbone_args = self.cfg.model.audiovisual_backbone

        spm_dict = os.path.join(os.path.dirname(os.path.abspath(__file__)), cfg.model.spm_dict)
        spm_model = os.path.join(os.path.dirname(os.path.abspath(__file__)), cfg.model.spm_model)
        
        self.text_transform = TextTransform(sp_model_path=spm_model, dict_path=spm_dict)
        self.token_list = self.text_transform.token_list
        if self.cfg.data.modality == "audiovisual":
            from espnet.nets.pytorch_backend.e2e_asr_conformer_av import E2E
            self.model = E2E(len(self.token_list), self.backbone_args)
        else:
            from espnet.nets.pytorch_backend.e2e_asr_conformer import E2E
            self.model = E2E(len(self.token_list), self.backbone_args)

        # -- initialise
        if self.cfg.ckpt_path:
            ckpt = torch.load(self.cfg.ckpt_path, map_location=lambda storage, loc: storage)
            if self.cfg.transfer_frontend:
                tmp_ckpt = {k: v for k, v in ckpt["model_state_dict"].items() if k.startswith("trunk.") or k.startswith("frontend3D.")}
                self.model.encoder.frontend.load_state_dict(tmp_ckpt)
            elif self.cfg.transfer_encoder:
                tmp_ckpt = {k.replace('encoder.',''): v for k, v in ckpt.items() if k.startswith("encoder.")}
                self.model.encoder.load_state_dict(tmp_ckpt)
                if self.cfg.data.modality == "audiovisual":
                    tmp_ckpt = {k.replace('aux_encoder.',''): v for k, v in ckpt.items() if k.startswith("aux_encoder.")}
                    self.model.aux_encoder.load_state_dict(tmp_ckpt)
            else:
                if "state_dict" in ckpt.keys():
                    tmp_ckpt = {k.replace('model.',''): v for k, v in ckpt["state_dict"].items() if k.find("beam_search") < 0}
                    ckpt = tmp_ckpt
                self.model.load_state_dict(ckpt)

        self.beam_search = get_beam_search_decoder(self.model, self.token_list)
        
                
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW([{"name": "model", "params": self.model.parameters(), "lr": self.cfg.optimizer.lr}], weight_decay=self.cfg.optimizer.weight_decay, betas=(self.cfg.optimizer.betas[0], self.cfg.optimizer.betas[1]))
        scheduler = WarmupCosineScheduler(optimizer, self.cfg.optimizer.warmup_epochs, self.cfg.trainer.max_epochs, len(self.trainer.datamodule.train_dataloader()))
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]

    def forward(self, video, audio=None, lang=None):
        enc_feat, _ = self.model.encoder(video.unsqueeze(0).to(self.device), None)
        if self.cfg.data.modality == "audiovisual":
            audio_feat, _ = self.model.aux_encoder(audio.unsqueeze(0).to(self.device), None)
            enc_feat = self.model.fusion(torch.cat((enc_feat, audio_feat), dim=-1))
        enc_feat = enc_feat.squeeze(0)
        nbest_hyps = self.beam_search(enc_feat, lang=lang)
        nbest_hyps = [h.asdict() for h in nbest_hyps[: min(len(nbest_hyps), 1)]]
        predicted = add_results_to_json(nbest_hyps, self.token_list, lang)
        predicted = predicted.replace("▁", " ").replace("\u2581", " ").strip().replace("<eos>", "")
        return predicted

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, step_type="train")          

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, step_type="val")

    def test_step(self, sample, sample_idx):
        enc_feat, _ = self.model.encoder(sample["input"].unsqueeze(0).to(self.device), None)
        if self.cfg.data.modality == "audiovisual":
            audio_feat, _ = self.model.aux_encoder(sample["audio"].unsqueeze(0).to(self.device), None)
            enc_feat = self.model.fusion(torch.cat((enc_feat, audio_feat), dim=-1))

        enc_feat = enc_feat.squeeze(0)
        token_id = sample["target"]
        nbest_hyps = self.beam_search(enc_feat, lang=token_id[0])
        nbest_hyps = [h.asdict() for h in nbest_hyps[: min(len(nbest_hyps), 1)]]
        predicted = add_results_to_json(nbest_hyps, self.token_list)
        predicted = predicted.replace("▁", " ").strip().replace("<eos>", "")

        if True:
            actual = self.text_transform.post_process(token_id[1:])
        else:
            actual = self.text_transform.post_process(token_id)

        self.total_edit_distance += compute_word_level_distance(actual, predicted)
        self.total_length += len(actual.split())
        return

    def _step(self, batch, batch_idx, step_type):
        if self.cfg.data.modality == "audiovisual":
            loss, loss_ctc, loss_att, acc = self.model(batch["videos"], batch["audios"], batch["video_lengths"], batch["audio_lengths"], batch["targets"], True)
            batch_size = len(batch["videos"])
        else:
            loss, loss_ctc, loss_att, acc = self.model(batch["inputs"], batch["input_lengths"], batch["targets"], True)
            batch_size = len(batch["inputs"])

        if step_type == "train":
            self.log("loss", loss, on_step=True, on_epoch=True, batch_size=batch_size)
            self.log("loss_ctc", loss_ctc, on_step=False, on_epoch=True, batch_size=batch_size)
            self.log("loss_att", loss_att, on_step=False, on_epoch=True, batch_size=batch_size)
            self.log("decoder_acc", acc, on_step=True, on_epoch=True, batch_size=batch_size)
        else:
            self.log("loss_val", loss, batch_size=batch_size)
            self.log("loss_ctc_val", loss_ctc, batch_size=batch_size)
            self.log("loss_att_val", loss_att, batch_size=batch_size)
            self.log("decoder_acc_val", acc, batch_size=batch_size)

        if step_type == "train":
            self.log("monitoring_step", torch.tensor(self.global_step, dtype=torch.float32))

        return loss
    
    def on_train_epoch_start(self):
        sampler = self.trainer.train_dataloader.batch_sampler
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(self.current_epoch)
        return super().on_train_epoch_start()

    def on_test_epoch_start(self):
        self.total_length = 0
        self.total_edit_distance = 0
        self.text_transform = TextTransform()
        self.beam_search = get_beam_search_decoder(self.model, self.token_list)

    def on_test_epoch_end(self):
        self.log("wer", self.total_edit_distance / self.total_length)


def get_beam_search_decoder(model, token_list, ctc_weight=0.1, beam_size=40):
    
    scorers = {
        "decoder": model.decoder,
        "ctc": CTCPrefixScorer(model.ctc, model.eos),
        "length_bonus": LengthBonus(len(token_list)),
        "lm": None
    }
    
    weights = {
        "decoder": 1.0 - ctc_weight,
        "ctc": ctc_weight,
        "lm": 0.0,
        "length_bonus": 0.0,
    }
    
    return BatchBeamSearch(
        beam_size=beam_size,
        vocab_size=len(token_list),
        weights=weights,
        scorers=scorers,
        sos=model.sos,
        eos=model.eos,
        token_list=token_list,
        pre_beam_score_key=None if ctc_weight == 1.0 else "decoder",
    )
