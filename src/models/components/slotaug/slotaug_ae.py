from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch import nn

from src.models.components.slotaug.slotaug import SlotAug
from src.models.components.slota.slota_utils import Decoder, Encoder


class SlotAugAutoEncoder(nn.Module):
    """Builds Slot Attention-based auto-encoder for object discovery.

    Args:
        num_slots (int): Number of slots in Slot Attention.
    """

    def __init__(
        self,
        img_size: int = 128,
        num_slots: int = 7,
        num_iter: int = 3,
        num_iter_insts: int = 1,
        num_attn_heads: int = 1,
        hid_dim: int = 64,
        slot_dim: int = 64,
        mlp_hid_dim: int = 128,
        eps: float = 1e-8,
        enc_depth: int = 4,
        dec_hid_dim: int = 64,
        dec_init_size: int = 8,
        dec_depth: int = 6,
        aux_identity: bool = True,
        use_pe: bool = True,
    ):
        super().__init__()
        self.num_slots = num_slots
        self.aux_identity = aux_identity

        self.encoder_cnn = Encoder(
            img_size=img_size,
            hid_dim=hid_dim,
            enc_depth=enc_depth,
            use_pe=use_pe,
        )
        self.decoder_cnn = Decoder(
            img_size=img_size,
            slot_dim=slot_dim,
            dec_hid_dim=dec_hid_dim,
            dec_init_size=dec_init_size,
            dec_depth=dec_depth,
        )

        self.slotaug = SlotAug(
            num_slots=num_slots,
            num_iter=num_iter,
            num_iter_insts=num_iter_insts,
            num_attn_heads=num_attn_heads,
            slot_dim=slot_dim,
            hid_dim=hid_dim,
            mlp_hid_dim=mlp_hid_dim,
            aux_identity=self.aux_identity,
            eps=eps,
        )
                        
        self.num_iter = num_iter
        self.num_iter_insts = num_iter_insts

    def forward(self, img_ori, img_aug, insts_ori2aug, insts_aug2ori):
        # `img_ori`: (B, C, H, W)
        # `img_aug`: (B, C, H, W)
        # `insts`: (B, K, 11) obj_pos (2) + rotate (1) + translate (2) + scale (1) + color (3) + flip (2)
        
        B, C, H, W = img_ori.shape

        # Convolutional encoder with position embedding
        x = self.encoder_cnn(img_ori)  # CNN Backbone
        if img_aug != None:
            x_aug = self.encoder_cnn(img_aug)  # CNN Backbone
        else:
            x_aug = None
        # `x`: (B, height * width, hid_dim)

        # Slot Attention module.
        slotaug_outputs = self.slotaug(inputs_ori=x, inputs_aug=x_aug, insts_ori2aug=insts_ori2aug, insts_aug2ori=insts_aug2ori)
        slots_ori = slotaug_outputs["slots_ori"]
        slots_ori2aug = slotaug_outputs["slots_ori2aug"]
        # `slots`: (N, K, slot_dim)

        x_ori = self.decoder_cnn(slots_ori)
        x_ori2aug = self.decoder_cnn(slots_ori2aug)
        # `x`: (B*K, height, width, num_channels+1)

        # Undo combination of slot and batch dimension; split alpha masks
        recons_ori, masks_ori = x_ori.reshape(B, self.num_slots, H, W, C + 1).split([3, 1], dim=-1)
        recons_ori2aug, masks_ori2aug = x_ori2aug.reshape(B, self.num_slots, H, W, C + 1).split([3, 1], dim=-1)
        # `recons`: (B, K, height, width, num_channels)
        # `masks`: (B, K, height, width, 1)

        # Normalize alpha masks over slots.
        masks_ori = nn.Softmax(dim=1)(masks_ori)
        masks_ori2aug = nn.Softmax(dim=1)(masks_ori2aug)

        recon_combined_ori = torch.sum(recons_ori * masks_ori, dim=1)  # Recombine image
        recon_combined_ori2aug = torch.sum(recons_ori2aug * masks_ori2aug, dim=1)  # Recombine image
        recon_combined_ori = recon_combined_ori.permute(0, 3, 1, 2)
        recon_combined_ori2aug = recon_combined_ori2aug.permute(0, 3, 1, 2)
        # `recon_combined`: (batch_size, num_channels, height, width)

        outputs = dict()
        outputs["recon_combined_ori"] = recon_combined_ori
        outputs["recon_combined_ori2aug"] = recon_combined_ori2aug 
        outputs["recons_ori"] = recons_ori
        outputs["recons_ori2aug"] = recons_ori2aug
        outputs["masks_ori"] = masks_ori
        outputs["masks_ori2aug"] = masks_ori2aug
        # outputs["slots"] = slots
        outputs["slots_ori"] = slots_ori
        outputs["slots_ori2aug"] = slots_ori2aug
        outputs["slots_ori_revisited"] = slotaug_outputs["slots_ori_revisited"]
        outputs["attns"] = slotaug_outputs["attns"]
        outputs["normed_attns"] = slotaug_outputs["normed_attns"]
        outputs["attns_ori"] = slotaug_outputs["attns_ori"]
        outputs["normed_attns_ori"] = slotaug_outputs["normed_attns_ori"]
        
        outputs["slots_aug"] = slotaug_outputs["slots_aug"]
        outputs["attns_aug"] = slotaug_outputs["attns_aug"]
        outputs["normed_attns_aug"] = slotaug_outputs["normed_attns_aug"]
        
        # `attns`: (B, K, N_heads, N_in)
        

        return outputs


if __name__ == "__main__":
    _ = SlotAugAutoEncoder()
