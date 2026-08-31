import torch
import torch.nn as nn

from .modules.attention import TransformerModel
from .modules.audio import build_audio_encoder, check_frame_rate, encoder_projection
from .modules.projection import LinearHead
from .modules.visual_encoder import VisualEncoder

#: The visual frontend samples faces onto this timeline, so the audio frontend
#: has to land on it too or the two streams cannot be fused frame for frame.
MODEL_FRAME_HZ = 25.0


class AudioVisualASD(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        cross_dim = cfg["cross"]["d_model"]
        temporal_dim = cfg["temporal"]["d_model"]
        if temporal_dim != cross_dim * 2:
            raise ValueError("ASD temporal d_model must be twice cross d_model")

        self.audio_encoder = build_audio_encoder(cfg)
        check_frame_rate(self.audio_encoder, MODEL_FRAME_HZ, "the ASD visual timeline")
        self.encoder_proj = encoder_projection(self.audio_encoder.out_dim, cross_dim)
        self.visual_encoder = VisualEncoder()
        if self.visual_encoder.out_dim != cross_dim:
            raise ValueError(
                f"ASD cross d_model must be {self.visual_encoder.out_dim} to match the "
                f"visual frontend, got {cross_dim}"
            )
        self.norm_audio = nn.LayerNorm(cross_dim)
        self.norm_visual = nn.LayerNorm(cross_dim)
        self.cross_v2a = TransformerModel(cfg["cross"], cross_attention=True)
        self.cross_a2v = TransformerModel(cfg["cross"], cross_attention=True)
        self.transformer = TransformerModel(cfg["temporal"])
        self.asd_head = LinearHead(temporal_dim, 6)
        self.audio_head = LinearHead(cross_dim, 1)
        self.visual_head = LinearHead(cross_dim, 1)

    def forward(self, audio, visual, return_embeddings=False):
        audio_embedding = self.norm_audio(
            self.encoder_proj(self.audio_encoder(audio))
        )
        visual_embedding = self.norm_visual(self.visual_encoder(visual))
        if audio_embedding.shape[1] != visual_embedding.shape[1]:
            raise ValueError(
                f"audio and visual lengths differ: {audio_embedding.shape[1]} and "
                f"{visual_embedding.shape[1]}"
            )

        audio_cross = self.cross_v2a(audio_embedding, src=visual_embedding)
        visual_cross = self.cross_a2v(visual_embedding, src=audio_embedding)
        fused = self.transformer(torch.cat([audio_cross, visual_cross], dim=-1))
        projection_logits = self.asd_head(fused)
        audio_logits = self.audio_head(audio_cross)
        visual_logits = self.visual_head(visual_cross)

        if return_embeddings:
            return projection_logits, audio_logits, visual_logits, fused, audio_embedding, visual_embedding
        return projection_logits, audio_logits, visual_logits
