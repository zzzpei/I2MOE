import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import logging

__all__ = ['Dataloader_Multimodal']
logger = logging.getLogger('EMOE')


class SleepAugment:
    """
    训练期在线增强（只作用于 train）
    - EEG/EOG 同步 time shift（保持对齐）
    - 幅值缩放、加噪声（可独立）
    - 模态 dropout（对 MoE/router 很有帮助）
    """
    def __init__(
        self,
        fs=100,
        p_shift=0.6,
        max_shift_sec=0.5,   # 0.5s @ 100Hz -> 50
        p_amp=0.5,
        amp_range=(0.9, 1.1),
        p_noise=0.3,
        noise_std=0.01,
        p_moddrop=0.10,
        moddrop_zero=True,
    ):
        self.fs = fs
        self.p_shift = p_shift
        self.max_shift = int(max_shift_sec * fs)

        self.p_amp = p_amp
        self.amp_lo, self.amp_hi = amp_range

        self.p_noise = p_noise
        self.noise_std = noise_std

        self.p_moddrop = p_moddrop
        self.moddrop_zero = moddrop_zero

    def _maybe_shift_pair(self, eeg, eog):
        # eeg/eog: [1, T]
        if self.max_shift > 0 and torch.rand(1).item() < self.p_shift:
            s = int(torch.randint(-self.max_shift, self.max_shift + 1, (1,)).item())
            eeg = torch.roll(eeg, shifts=s, dims=-1)
            eog = torch.roll(eog, shifts=s, dims=-1)
        return eeg, eog

    def _maybe_amp(self, x):
        if torch.rand(1).item() < self.p_amp:
            scale = self.amp_lo + (self.amp_hi - self.amp_lo) * torch.rand(1).item()
            x = x * scale
        return x

    def _maybe_noise(self, x):
        if torch.rand(1).item() < self.p_noise:
            x = x + self.noise_std * torch.randn_like(x)
        return x

    def _maybe_moddrop(self, x):
        if torch.rand(1).item() < self.p_moddrop:
            if self.moddrop_zero:
                return torch.zeros_like(x)
        return x

    def __call__(self, eeg, eog):
        # 1) 同步 shift
        eeg, eog = self._maybe_shift_pair(eeg, eog)

        # 2) amp/noise（可以独立）
        eeg = self._maybe_amp(eeg)
        eog = self._maybe_amp(eog)

        eeg = self._maybe_noise(eeg)
        eog = self._maybe_noise(eog)

        # 3) 模态 dropout（偶尔让某模态消失）
        eeg = self._maybe_moddrop(eeg)
        eog = self._maybe_moddrop(eog)

        return eeg, eog


class Dataloader_Multimodal(Dataset):
    """
    用于加载和处理 EOG/EEG 多模态数据的 PyTorch Dataset。
    """
    def __init__(self, args, file_lists_eog, file_lists_eeg, mode='train'):
        super(Dataloader_Multimodal, self).__init__()
        self.mode = mode
        self.args = args

        self.__load_eog_eeg_data(file_lists_eog, file_lists_eeg)

        # ====== 增强器：只在 train 使用 ======
        self.use_aug = (self.mode == 'train') and bool(getattr(args, "use_aug", True))

        if self.use_aug:
            fs = int(getattr(args, "sampling_rate", 100))
            # 这些参数你可以在 args 里覆写
            self.augmentor = SleepAugment(
                fs=fs,
                p_shift=float(getattr(args, "aug_p_shift", 0.6)),
                max_shift_sec=float(getattr(args, "aug_max_shift_sec", 0.5)),
                p_amp=float(getattr(args, "aug_p_amp", 0.5)),
                amp_range=getattr(args, "aug_amp_range", (0.9, 1.1)),
                p_noise=float(getattr(args, "aug_p_noise", 0.3)),
                noise_std=float(getattr(args, "aug_noise_std", 0.01)),
                p_moddrop=float(getattr(args, "aug_p_moddrop", 0.10)),
                moddrop_zero=True,
            )
        else:
            self.augmentor = None

    def __load_eog_eeg_data(self, np_dataset_eog, np_dataset_eeg):
        """
        加载 EOG 和 EEG 的 numpy 数据文件，并进行拼接和形状调整。
        """
        np_dataset_eog = sorted(np_dataset_eog)
        np_dataset_eeg = sorted(np_dataset_eeg)

        EOG_data = np.load(np_dataset_eog[0])["x"]
        EEG_data = np.load(np_dataset_eeg[0])["x"]
        labels = np.load(np_dataset_eog[0])["y"]

        for np_file in np_dataset_eog[1:]:
            EOG_data = np.vstack((EOG_data, np.load(np_file)["x"]))
            labels = np.append(labels, np.load(np_file)["y"])

        for np_file in np_dataset_eeg[1:]:
            EEG_data = np.vstack((EEG_data, np.load(np_file)["x"]))

        # (n_epochs, seq_len) -> (n_epochs, seq_len, 1)
        EOG_data = EOG_data[:, :, np.newaxis]
        EEG_data = EEG_data[:, :, np.newaxis]

        # (n_epochs, seq_len, 2)
        X_data = np.concatenate((EOG_data, EEG_data), axis=2)

        # torch: (n_epochs, seq_len, 2) -> (n_epochs, 2, seq_len)
        x = torch.from_numpy(X_data).float()
        if x.dim() == 3:
            x = x.permute(0, 2, 1)

        self.x_data = x                     # (n_epochs, 2, seq_len)
        self.eog = self.x_data[:, 0:1, :]   # (n_epochs, 1, seq_len)
        self.eeg = self.x_data[:, 1:2, :]   # (n_epochs, 1, seq_len)
        self.labels = torch.from_numpy(labels).long()

    def __len__(self):
        return self.labels.shape[0]

    def get_seq_len(self):
        return self.x_data.shape[2]

    def get_feature_dim(self):
        return self.x_data.shape[1]

    def __getitem__(self, index):
        # 注意：这里不要 torch.Tensor(...) 重新拷贝，直接 clone()（可选）
        eeg = self.eeg[index]  # [1, T]
        eog = self.eog[index]  # [1, T]

        # ====== 只对训练集做增强 ======
        if self.augmentor is not None:
            # 为了避免修改底层缓存 tensor（虽然一般不会共享写入），这里 clone 一下更保险
            eeg_aug, eog_aug = self.augmentor(eeg.clone(), eog.clone())
        else:
            eeg_aug, eog_aug = eeg, eog

        x_mm = torch.cat([eog_aug, eeg_aug], dim=0)  # [2, T]，保持你原来的通道顺序 (EOG, EEG)

        sample = {
            'eog': eog_aug,                 # (1, seq_len)
            'eeg': eeg_aug,                 # (1, seq_len)
            'multi_modal_data': x_mm,       # (2, seq_len)
            'index': index,
            'labels': {
                'M': self.labels[index].reshape(-1)
            }
        }
        return sample


def Dataloader_Multimodal_Generator(
    args,
    training_files_eog, validation_files_eog, subject_files_eog,
    training_files_eeg, validation_files_eeg, subject_files_eeg, num_workers=4
):
    datasets = {
        'train': Dataloader_Multimodal(args, training_files_eog, training_files_eeg, mode='train'),
        'valid': Dataloader_Multimodal(args, validation_files_eog, validation_files_eeg, mode='valid'),
        'test':  Dataloader_Multimodal(args, subject_files_eog, subject_files_eeg, mode='test')
    }

    all_ys = torch.cat((datasets['train'].labels, datasets['valid'].labels, datasets['test'].labels))
    all_ys = all_ys.tolist()
    num_classes = len(np.unique(all_ys))
    counts = [all_ys.count(i) for i in range(num_classes)]
    print("\ncounts (All data):", counts)
    print("TEST counts:", [datasets['test'].labels.tolist().count(i) for i in range(num_classes)])

    # batch_size：你现在 args 是对象（trainer 用 getattr），这里你是 dict 风格
    # 为了兼容，优先 getattr，否则 fallback dict.get
    def _get_bs(a, default=64):
        return int(getattr(a, "batch_size", a.get("batch_size", default) if isinstance(a, dict) else default))

    dataLoader = {
        'train': DataLoader(
            datasets['train'],
            batch_size=_get_bs(args, 64),
            num_workers=num_workers,
            shuffle=True,
            drop_last=False
        ),
        'valid': DataLoader(
            datasets['valid'],
            batch_size=_get_bs(args, 64),
            num_workers=num_workers,
            shuffle=False,
            drop_last=False
        ),
        'test': DataLoader(
            datasets['test'],
            batch_size=_get_bs(args, 64),
            num_workers=num_workers,
            shuffle=False,
            drop_last=False
        )
    }

    return dataLoader, counts
