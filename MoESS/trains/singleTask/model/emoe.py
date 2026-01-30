import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ...singleTask.model.router import router  # 保留你原 router


# ---------------------------
# Utils: GroupNorm groups chooser
# ---------------------------
def _pick_gn_groups(ch: int, preferred: int = 8) -> int:
    """
    选一个能整除 ch 的 group 数，优先 preferred，其次向下找
    """
    g = min(preferred, ch)
    while g > 1:
        if ch % g == 0:
            return g
        g -= 1
    return 1


# ---------------------------
# SpecAugment-like for 1D feature maps
# ---------------------------
class FeatureSpecAugment1D(nn.Module):
    """
    SpecAugment-like (1D) on features: x in [B, C, L]
    - Time mask: mask contiguous segments on time axis
    - Channel mask: mask contiguous segments on channel axis
    """
    def __init__(
        self,
        p: float = 0.5,
        time_mask_num: int = 2,
        time_mask_max: int = 80,
        channel_mask_num: int = 1,
        channel_mask_max: int = 8,
        replace_with_zero: bool = True,
    ):
        super().__init__()
        self.p = float(p)
        self.time_mask_num = int(time_mask_num)
        self.time_mask_max = int(time_mask_max)
        self.channel_mask_num = int(channel_mask_num)
        self.channel_mask_max = int(channel_mask_max)
        self.replace_with_zero = bool(replace_with_zero)

    @staticmethod
    def _rand_int(low: int, high: int, device):
        return int(torch.randint(low, high, (1,), device=device).item())

    def _sample_time_masks(self, L: int, device):
        if self.time_mask_num <= 0 or self.time_mask_max <= 0 or L <= 1:
            return []
        tmax = min(self.time_mask_max, L)
        masks = []
        for _ in range(self.time_mask_num):
            length = self._rand_int(1, tmax + 1, device)
            start = self._rand_int(0, max(1, L - length + 1), device)
            masks.append((start, length))
        return masks

    def _apply_time_masks_(self, y: torch.Tensor, masks):
        # y: [B,C,L]  (注意：这里是对 y 原地改，但 y 是 clone 的)
        if not masks:
            return y
        for (start, length) in masks:
            if self.replace_with_zero:
                y[:, :, start:start + length] = 0.0
            else:
                mean = y.mean(dim=-1, keepdim=True)
                y[:, :, start:start + length] = mean
        return y

    def _apply_channel_masks_(self, y: torch.Tensor):
        B, C, L = y.shape
        if self.channel_mask_num <= 0 or self.channel_mask_max <= 0 or C <= 1:
            return y
        cmax = min(self.channel_mask_max, C)
        for _ in range(self.channel_mask_num):
            width = self._rand_int(1, cmax + 1, y.device)
            start = self._rand_int(0, max(1, C - width + 1), y.device)
            if self.replace_with_zero:
                y[:, start:start + width, :] = 0.0
            else:
                mean = y.mean(dim=1, keepdim=True)
                y[:, start:start + width, :] = mean
        return y

    def forward_multi(self, *xs):
        """
        对多个 feature map 同步 time mask（共享 time mask），各自 channel mask
        xs: multiple tensors each in [B,C,L] with same L.
        """
        if (not self.training) or (self.p <= 0):
            return xs
        if torch.rand(1, device=xs[0].device).item() >= self.p:
            return xs

        Ls = [x.shape[-1] for x in xs]
        if len(set(Ls)) != 1:
            # L 不一致：只做各自 channel mask（在 clone 上）
            outs = []
            for x in xs:
                y = x.clone()
                y = self._apply_channel_masks_(y)
                outs.append(y)
            return tuple(outs)

        L = Ls[0]
        masks = self._sample_time_masks(L, xs[0].device)

        outs = []
        for x in xs:
            y = x.clone()             
            y = self._apply_time_masks_(y, masks)
            y = self._apply_channel_masks_(y)
            outs.append(y)
        return tuple(outs)

    def forward(self, x: torch.Tensor):
        """
        单独对一个 x 做增强（不共享 mask）
        """
        if (not self.training) or (self.p <= 0):
            return x
        if torch.rand(1, device=x.device).item() >= self.p:
            return x

        y = x.clone() 
        L = y.shape[-1]
        masks = self._sample_time_masks(L, y.device)
        y = self._apply_time_masks_(y, masks)
        y = self._apply_channel_masks_(y)
        return y


# ---------------------------
# 1) GaborConv1d (支持 groups，保证 data expert split 有语义)
# ---------------------------
class GaborConv1d(nn.Module):
    """可训练的Gabor卷积层（支持 groups，用于语义化双路）"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 sampling_rate=100, freq_range=(0.5, 35), learnable_phase=False, groups=1):
        super().__init__()
        assert kernel_size % 2 == 1, "建议使用奇数 kernel_size 以保持对称和 padding 对齐"
        assert in_channels % groups == 0, "in_channels 必须能被 groups 整除"
        assert out_channels % groups == 0, "out_channels 必须能被 groups 整除"

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.sampling_rate = sampling_rate
        self.learnable_phase = learnable_phase
        self.groups = groups

        self.in_per_group = in_channels // groups
        self.out_per_group = out_channels // groups

        half = kernel_size // 2
        t = torch.arange(-half, half + 1).float() / float(sampling_rate)
        self.register_buffer('t', t.view(1, 1, -1))  # (1,1,K)

        self.f_min, self.f_max = float(freq_range[0]), float(freq_range[1])
        self.sigma_min, self.sigma_max = 0.05, 0.8
        self.mu_min, self.mu_max = -0.25, 0.25

        # grouped conv weight: (O, I_per_group, K)
        self.freq_raw = nn.Parameter(torch.zeros(out_channels, self.in_per_group))
        self.sigma_raw = nn.Parameter(torch.zeros(out_channels, self.in_per_group))
        self.mu_raw = nn.Parameter(torch.zeros(out_channels, self.in_per_group))

        if learnable_phase:
            self.phi = nn.Parameter(torch.zeros(out_channels, self.in_per_group))
        else:
            self.register_buffer('phi', torch.zeros(out_channels, self.in_per_group))

        self._initialize_parameters()

    def _map(self, raw, lo, hi):
        return lo + (hi - lo) * torch.sigmoid(raw)

    def _initialize_parameters(self):
        with torch.no_grad():
            q = self.out_channels // 4

            def inv_sigmoid(p):
                p = torch.clamp(p, 1e-4, 1 - 1e-4)
                return torch.log(p / (1 - p))

            def set_raw(param_raw, target, lo, hi):
                p = (target - lo) / (hi - lo + 1e-8)
                param_raw.copy_(inv_sigmoid(p))

            f = torch.zeros(self.out_channels, self.in_per_group)
            f[:q] = torch.rand(q, self.in_per_group) * (4.0 - 0.5) + 0.5
            f[q:2*q] = torch.rand(q, self.in_per_group) * (8.0 - 4.0) + 4.0
            f[2*q:3*q] = torch.rand(q, self.in_per_group) * (13.0 - 8.0) + 8.0
            f[3*q:] = torch.rand(self.out_channels - 3*q, self.in_per_group) * (16.0 - 11.0) + 11.0
            f = f.clamp(self.f_min, self.f_max)
            set_raw(self.freq_raw, f, self.f_min, self.f_max)

            s = torch.zeros_like(f)
            s[:q] = torch.rand(q, self.in_per_group) * (0.8 - 0.4) + 0.4
            s[q:2*q] = torch.rand(q, self.in_per_group) * (0.4 - 0.2) + 0.2
            s[2*q:3*q] = torch.rand(q, self.in_per_group) * (0.25 - 0.15) + 0.15
            s[3*q:] = torch.rand(self.out_channels - 3*q, self.in_per_group) * (0.18 - 0.08) + 0.08
            s = s.clamp(self.sigma_min, self.sigma_max)
            set_raw(self.sigma_raw, s, self.sigma_min, self.sigma_max)

            set_raw(self.mu_raw, torch.zeros_like(f), self.mu_min, self.mu_max)

            if self.learnable_phase:
                self.phi.data.uniform_(-torch.pi, torch.pi)

    def _create_gabor_kernels(self):
        t = self.t.expand(self.out_channels, self.in_per_group, self.kernel_size)

        freq = self._map(self.freq_raw, self.f_min, self.f_max)
        sigma = self._map(self.sigma_raw, self.sigma_min, self.sigma_max)
        mu = self._map(self.mu_raw, self.mu_min, self.mu_max)

        t_centered = t - mu.unsqueeze(-1)

        gaussian = torch.exp(-torch.pi * (t_centered / (sigma.unsqueeze(-1) + 1e-8)) ** 2)
        cosine = torch.cos(2 * torch.pi * freq.unsqueeze(-1) * t_centered + self.phi.unsqueeze(-1))
        kernels = gaussian * cosine

        norm = torch.sqrt(torch.sum(kernels ** 2, dim=-1, keepdim=True)) + 1e-8
        kernels = kernels / norm
        return kernels

    def forward(self, x):
        kernels = self._create_gabor_kernels()
        return F.conv1d(x, kernels, stride=self.stride, padding=self.padding, groups=self.groups)

    def get_kernel_parameters(self):
        kernels = self._create_gabor_kernels()
        return {
            'kernels': kernels.detach(),
            'freq': self._map(self.freq_raw, self.f_min, self.f_max).detach(),
            'sigma': self._map(self.sigma_raw, self.sigma_min, self.sigma_max).detach(),
            'mu': self._map(self.mu_raw, self.mu_min, self.mu_max).detach(),
            'phi': self.phi.detach() if self.learnable_phase else self.phi
        }


# ---------------------------
# 2) ResidualBlock / Attention（BN->GN）
# ---------------------------
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, gn_groups=8):
        super().__init__()
        g1 = _pick_gn_groups(out_channels, gn_groups)

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.gn1 = nn.GroupNorm(g1, out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 1, padding, bias=False)
        self.gn2 = nn.GroupNorm(g1, out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            g_sc = _pick_gn_groups(out_channels, gn_groups)
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(g_sc, out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        out = self.relu(out + residual)
        return out


class MultiScaleAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        hidden = max(4, channels // 4)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(),
            nn.Linear(hidden, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x).squeeze(-1)).unsqueeze(-1)
        max_out = self.fc(self.max_pool(x).squeeze(-1)).unsqueeze(-1)
        attn = avg_out + max_out
        return x * attn


# ---------------------------
# 3) Dilated Temporal Block（BN->GN）
# ---------------------------
class DilatedTemporalBlock(nn.Module):
    def __init__(self, channels: int, dilations=(1, 2, 4), dropout=0.1, gn_groups=8):
        super().__init__()
        layers = []
        g = _pick_gn_groups(channels, gn_groups)
        for d in dilations:
            k = 3
            p = d * (k // 2)
            layers += [
                nn.Conv1d(channels, channels, kernel_size=k, dilation=d, padding=p, bias=False),
                nn.GroupNorm(g, channels),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.net(x)


class AvgMaxPool1D(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.0):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool1d(1)
        self.max = nn.AdaptiveMaxPool1d(1)
        self.proj = nn.Sequential(
            nn.Linear(channels * 2, channels),
            nn.LayerNorm(channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        a = self.avg(x).squeeze(-1)
        m = self.max(x).squeeze(-1)
        z = torch.cat([a, m], dim=1)
        return self.proj(z)


# ---------------------------
# 4) Cross-gated fusion（BN->GN）
# ---------------------------
class CrossGatedFusion(nn.Module):
    def __init__(self, channels: int, gn_groups=8):
        super().__init__()
        self.gate_from_eog = nn.Conv1d(channels, channels, kernel_size=1)
        self.gate_from_eeg = nn.Conv1d(channels, channels, kernel_size=1)

        g = _pick_gn_groups(256, gn_groups)
        self.fuse = nn.Sequential(
            nn.Conv1d(2 * channels, 256, kernel_size=1, bias=False),
            nn.GroupNorm(g, 256),
            nn.ReLU(inplace=True),
        )

    def forward(self, eeg_feat, eog_feat):
        gate_eeg = torch.sigmoid(self.gate_from_eog(eog_feat))
        gate_eog = torch.sigmoid(self.gate_from_eeg(eeg_feat))
        eeg_feat = eeg_feat * gate_eeg
        eog_feat = eog_feat * gate_eog
        fused = torch.cat([eeg_feat, eog_feat], dim=1)
        return self.fuse(fused)


class EMOE(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        fs = getattr(args, 'sampling_rate', 100)
        gn_groups = getattr(args, 'gn_groups', 8)

        # ========== Gabor 参数 ==========
        eeg_gabor_channels = 48
        eog_gabor_channels = 16

        self.eeg_gabor = GaborConv1d(
            in_channels=1,
            out_channels=eeg_gabor_channels,
            kernel_size=101,
            stride=2,
            padding=101 // 2,
            sampling_rate=fs,
            freq_range=(0.5, 35),
            learnable_phase=False,
            groups=1
        )

        self.eog_gabor = GaborConv1d(
            in_channels=1,
            out_channels=eog_gabor_channels,
            kernel_size=151,
            stride=2,
            padding=151 // 2,
            sampling_rate=fs,
            freq_range=(0.5, 20),
            learnable_phase=False,
            groups=1
        )

        
        self.data_gabor = GaborConv1d(
            in_channels=2,
            out_channels=64,
            kernel_size=101,
            stride=2,
            padding=101 // 2,
            sampling_rate=fs,
            freq_range=(0.5, 35),
            learnable_phase=False,
            groups=2
        )

        # ========== Gabor 后通道对齐（GN） ==========
        self.eeg_gabor_adjust = nn.Sequential(
            nn.Conv1d(eeg_gabor_channels, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(_pick_gn_groups(32, gn_groups), 32),
            nn.ReLU(inplace=True)
        )
        self.eog_gabor_adjust = nn.Sequential(
            nn.Conv1d(eog_gabor_channels, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(_pick_gn_groups(32, gn_groups), 32),
            nn.ReLU(inplace=True)
        )

        self.data_gabor_adjust = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1, bias=False, groups=2),
            nn.GroupNorm(_pick_gn_groups(64, gn_groups), 64),
            nn.ReLU(inplace=True)
        )

        self.use_specaug = bool(getattr(args, "use_specaug", True))
        self.specaug = FeatureSpecAugment1D(
            p=float(getattr(args, "specaug_p", 0.5)),
            time_mask_num=int(getattr(args, "specaug_time_num", 2)),
            time_mask_max=int(getattr(args, "specaug_time_max", 80)),   # L=1500 时约 1~2 秒
            channel_mask_num=int(getattr(args, "specaug_ch_num", 1)),
            channel_mask_max=int(getattr(args, "specaug_ch_max", 8)),
            replace_with_zero=True,
        )

        # ========== EEG Expert ==========
        self.eeg_encoder = nn.Sequential(
            ResidualBlock(32, 64, kernel_size=15, stride=2, padding=7, gn_groups=gn_groups),
            MultiScaleAttention(64),

            ResidualBlock(64, 128, kernel_size=11, stride=2, padding=5, gn_groups=gn_groups),
            MultiScaleAttention(128),

            ResidualBlock(128, 256, kernel_size=9, stride=2, padding=4, gn_groups=gn_groups),
            MultiScaleAttention(256),

            DilatedTemporalBlock(256, dilations=(1, 2, 4), dropout=0.1, gn_groups=gn_groups)
        )

        # ========== EOG Expert ==========
        self.eog_encoder = nn.Sequential(
            ResidualBlock(32, 64, kernel_size=9, stride=2, padding=4, gn_groups=gn_groups),
            MultiScaleAttention(64),

            ResidualBlock(64, 128, kernel_size=7, stride=2, padding=3, gn_groups=gn_groups),
            MultiScaleAttention(128),

            nn.Conv1d(128, 256, kernel_size=1, bias=False),
            nn.GroupNorm(_pick_gn_groups(256, gn_groups), 256),
            nn.ReLU(inplace=True),
        )

        # ========== Data Expert ==========
        self.data_stream_eeg = nn.Sequential(
            ResidualBlock(32, 64, kernel_size=11, stride=2, padding=5, gn_groups=gn_groups),
            MultiScaleAttention(64),
        )
        self.data_stream_eog = nn.Sequential(
            ResidualBlock(32, 64, kernel_size=11, stride=2, padding=5, gn_groups=gn_groups),
            MultiScaleAttention(64),
        )
        self.data_fusion = CrossGatedFusion(64, gn_groups=gn_groups)
        self.data_tail = nn.Sequential(
            ResidualBlock(256, 256, kernel_size=9, stride=2, padding=4, gn_groups=gn_groups),
            MultiScaleAttention(256),
            DilatedTemporalBlock(256, dilations=(1, 2, 4), dropout=0.1, gn_groups=gn_groups)
        )

        # 全局池化
        self.global_pool = AvgMaxPool1D(256, dropout=0.1)

        # ========== Classifier Heads ==========
        self.eeg_classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, args.num_classes)
        )
        self.eog_classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, args.num_classes)
        )
        self.data_classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, args.num_classes)
        )

        # 融合分类器
        if args.fusion_method == "sum":
            self.fusion_classifier = nn.Sequential(
                nn.Linear(256, 128),
                nn.LayerNorm(128),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(128, args.num_classes)
            )
        elif args.fusion_method == "concat":
            self.fusion_classifier = nn.Sequential(
                nn.Linear(256 * 3, 512),
                nn.LayerNorm(512),
                nn.ReLU(),
                nn.Dropout(0.5),

                nn.Linear(512, 256),
                nn.LayerNorm(256),
                nn.ReLU(),
                nn.Dropout(0.5),

                nn.Linear(256, args.num_classes)
            )
        else:
            raise ValueError(f"Unknown fusion_method: {args.fusion_method}")

        # ========== Router ==========
        self.router_in_dim = 256 * 3 + 8
        self.Router = router(self.router_in_dim, 3, getattr(self.args, 'temperature', 1.0))

    def _quality_feats(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1)
        std = x.std(dim=-1)
        rms = torch.sqrt((x ** 2).mean(dim=-1) + 1e-8)
        max_abs = x.abs().amax(dim=-1)
        return torch.cat([mean, std, rms, max_abs], dim=1)

    def forward(self, eeg, eog):
        # ========== 1) Gabor ==========
        eeg_gabor_features = self.eeg_gabor(eeg)  # (B,48,1500)
        eog_gabor_features = self.eog_gabor(eog)  # (B,16,1500)

        data_input = torch.cat([eeg, eog], dim=1)          # (B,2,3000)
        data_gabor_features = self.data_gabor(data_input)  # (B,64,1500) grouped

        # 通道对齐
        eeg_features = self.eeg_gabor_adjust(eeg_gabor_features)       # (B,32,1500)
        eog_features = self.eog_gabor_adjust(eog_gabor_features)       # (B,32,1500)
        data_features = self.data_gabor_adjust(data_gabor_features)    # (B,64,1500)

        # ========== ★ 1D SpecAugment-like（只训练启用） ==========
        # 共享 time mask：保证 EEG/EOG/Data 在时间轴上对齐
        if self.use_specaug and self.training:
            eeg_features, eog_features, data_features = self.specaug.forward_multi(
                eeg_features, eog_features, data_features
            )

        # ========== 2) Encode ==========
        eeg_features = self.eeg_encoder(eeg_features)  # -> (B,256,L)
        eog_features = self.eog_encoder(eog_features)  # -> (B,256,L)

        # data expert: split -> two streams -> fusion -> tail
        data_eeg, data_eog = torch.split(data_features, 32, dim=1)     # 语义：组0/组1
        data_eeg = self.data_stream_eeg(data_eeg)                      # (B,64,L1)
        data_eog = self.data_stream_eog(data_eog)                      # (B,64,L1)
        data_fused = self.data_fusion(data_eeg, data_eog)              # (B,256,L1)
        data_features = self.data_tail(data_fused)                     # (B,256,L2)

        # ========== 3) Pool ==========
        eeg_pooled = self.global_pool(eeg_features)    # [B,256]
        eog_pooled = self.global_pool(eog_features)    # [B,256]
        data_pooled = self.global_pool(data_features)  # [B,256]

        # ========== 4) Router ==========
        q_eeg = self._quality_feats(eeg)  # (B,4)
        q_eog = self._quality_feats(eog)  # (B,4)
        router_in = torch.cat([eeg_pooled, eog_pooled, data_pooled, q_eeg, q_eog], dim=1)
        weights = self.Router(router_in)  # (B,3)

        # ========== 5) Classify ==========
        logits_eeg = self.eeg_classifier(eeg_pooled)
        logits_eog = self.eog_classifier(eog_pooled)
        logits_data = self.data_classifier(data_pooled)

        # ========== 6) Fuse ==========
        if self.args.fusion_method == "sum":
            fused_features = (weights[:, 0:1] * eeg_pooled +
                              weights[:, 1:2] * eog_pooled +
                              weights[:, 2:3] * data_pooled)
            logits_fused = self.fusion_classifier(fused_features)
        else:
            fused_features = torch.cat([
                weights[:, 0:1] * eeg_pooled,
                weights[:, 1:2] * eog_pooled,
                weights[:, 2:3] * data_pooled
            ], dim=1)
            logits_fused = self.fusion_classifier(fused_features)

        return {
            'logits_c': logits_fused,
            'logits_eeg': logits_eeg,
            'logits_eog': logits_eog,
            'logits_data': logits_data,
            'channel_weight': weights,
            'c_proj': fused_features,
            'eeg_proj': eeg_pooled,
            'eog_proj': eog_pooled,
            'data_proj': data_pooled,
            'gabor_features': {
                'eeg_gabor': eeg_gabor_features,
                'eog_gabor': eog_gabor_features,
                'data_gabor': data_gabor_features
            },
            'gabor_params': {
                'eeg': self.eeg_gabor.get_kernel_parameters(),
                'eog': self.eog_gabor.get_kernel_parameters(),
                'data': self.data_gabor.get_kernel_parameters()
            }
        }
