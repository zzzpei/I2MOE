import gc
import logging
import os
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from config import get_config_regression
from data_loader import Dataloader_Multimodal_Generator
from trains import ATIO
from utils import assign_gpu, setup_seed
from trains.singleTask.model import emoe, emoe_i2moe
import sys
from glob import glob
import torch.nn as nn
import json
from datetime import datetime

os.environ["CUDA_VISIBLE_DEVICES"] = "2" 
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:2"
logger = logging.getLogger('EMOE')

def load_folds_data_with_val(np_data_path, n_folds, dset_name='SleepEDF-20'):
    """
    简化版本，按比例划分验证集
    """
    files = sorted(glob(os.path.join(np_data_path, "*.npz")))
    
    r_p_path = r"r_permute_20.npy"
    
    if os.path.exists(os.path.join("/home/zjp0212/EMOE/",r_p_path)):
        r_permute = np.load(os.path.join("/home/zjp0212/EMOE/",r_p_path))
        print("r_permute", r_permute)
    else:
        print ("============== ERROR =================")
    # 加载split_idx文件（与EEGDataLoader相同）
    split_idx_path = os.path.join('./split_idx', f'idx_{dset_name}.npy')
    if not os.path.exists(split_idx_path):
        # 尝试其他可能的位置
        split_idx_path = os.path.join('/home/zjp0212/EMOE/split_idx', f'idx_{dset_name}.npy')
    
    if os.path.exists(split_idx_path):
        split_idx_list = np.load(split_idx_path, allow_pickle=True)

    files_dict = dict()
    for i in files:
        file_name = os.path.split(i)[-1] 
        file_num = file_name[3:5]
        if file_num not in files_dict:
            files_dict[file_num] = [i]
        else:
            files_dict[file_num].append(i)
    
    files_pairs = []
    for key in files_dict:
        files_pairs.append(files_dict[key])
    
    files_pairs = np.array(files_pairs, dtype=object)
    files_pairs = files_pairs[r_permute]

    # 预先展开所有文件路径
    all_files_flat = []
    for subject_files in files_pairs:
        all_files_flat.extend(subject_files)
    
    train_files = np.array_split(files_pairs, n_folds)
    folds_data = {}
    
    for fold_id in range(n_folds):
        # 测试集 - 当前fold的所有文件
        test_subjects = train_files[fold_id].tolist()
        test_files = []
        for subject in test_subjects:
            test_files.extend(subject)
        
        # 验证集 - 下一个fold的所有文件
        val_files = []
        '''
        val_subjects_list = split_idx_list[fold_id].tolist()
        for idx in val_subjects_list:
            val_subjects = train_files[idx].tolist()
            for subject in val_subjects:
                val_files.extend(subject)
        '''
        val_fold_id = (fold_id + 1) % n_folds
        val_subjects = train_files[val_fold_id].tolist()
        for subject in val_subjects:
            val_files.extend(subject)
        # for subject in val_subjects:
        #     val_files.extend(subject)
        
        # 训练集 - 剩余文件
        train_files_list = list(set(all_files_flat) - set(test_files) - set(val_files))
        
        folds_data[fold_id] = [train_files_list, val_files, test_files]
    
    return folds_data

def _set_logger(log_dir, model_name, dataset_name, verbose_level):
    log_file_path = Path(log_dir) / f"{model_name}-{dataset_name}.log"
    logger = logging.getLogger('EMOE')
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler(log_file_path)
    fh_formatter = logging.Formatter('%(asctime)s - %(name)s [%(levelname)s] - %(message)s')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fh_formatter)
    logger.addHandler(fh)

    stream_level = {0: logging.ERROR, 1: logging.INFO, 2: logging.DEBUG}
    ch = logging.StreamHandler()
    ch.setLevel(stream_level[verbose_level])
    ch_formatter = logging.Formatter('%(name)s - %(message)s')
    ch.setFormatter(ch_formatter)
    logger.addHandler(ch)

    return logger

def save_fold_results(results_dict, save_dir, model_name, dataset_name):
    """保存每个fold的结果到文件"""
    save_path = Path(save_dir) / "fold_results"
    save_path.mkdir(parents=True, exist_ok=True)
    
    # 保存详细的JSON结果
    json_file = save_path / f"{model_name}-{dataset_name}-fold_results.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, indent=2, ensure_ascii=False)
    
    # 保存CSV格式的汇总结果
    csv_file = save_path / f"{model_name}-{dataset_name}-fold_results.csv"
    
    # 准备CSV数据
    csv_data = []
    for fold_id, fold_results in results_dict['folds'].items():
        row = {
            'fold_id': fold_id,
            'train_acc': fold_results['train']['Acc_5'],
            'train_f1_weighted': fold_results['train']['F1_weighted'],
            'train_f1_macro': fold_results['train']['F1_macro'],
            'val_acc': fold_results['val']['Acc_5'],
            'val_f1_weighted': fold_results['val']['F1_weighted'],
            'val_f1_macro': fold_results['val']['F1_macro'],
            'test_acc': fold_results['test']['Acc_5'],
            'test_f1_weighted': fold_results['test']['F1_weighted'],
            'test_f1_macro': fold_results['test']['F1_macro'],
            'test_loss': fold_results['test']['Loss']
        }
        # 添加每个类别的F1分数
        for key, value in fold_results['test'].items():
            if key.startswith('F1_') and key not in ['F1_weighted', 'F1_macro']:
                row[f'test_{key}'] = value
        csv_data.append(row)
    
    df = pd.DataFrame(csv_data)
    df.to_csv(csv_file, index=False, encoding='utf-8')
    
    # 计算并保存平均结果
    avg_results = {
        'train_acc_mean': df['train_acc'].mean(),
        'train_acc_std': df['train_acc'].std(),
        'val_acc_mean': df['val_acc'].mean(),
        'val_acc_std': df['val_acc'].std(),
        'test_acc_mean': df['test_acc'].mean(),
        'test_acc_std': df['test_acc'].std(),
        'test_f1_weighted_mean': df['test_f1_weighted'].mean(),
        'test_f1_weighted_std': df['test_f1_weighted'].std(),
        'test_f1_macro_mean': df['test_f1_macro'].mean(),
        'test_f1_macro_std': df['test_f1_macro'].std(),
    }
    
    # 添加每个类别的平均F1分数
    f1_columns = [col for col in df.columns if col.startswith('test_F1_') and col not in ['test_F1_weighted', 'test_F1_macro']]
    for col in f1_columns:
        class_name = col.replace('test_F1_', '')
        avg_results[f'test_F1_{class_name}_mean'] = df[col].mean()
        avg_results[f'test_F1_{class_name}_std'] = df[col].std()
    
    avg_file = save_path / f"{model_name}-{dataset_name}-average_results.json"
    with open(avg_file, 'w', encoding='utf-8') as f:
        json.dump(avg_results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Fold results saved to: {json_file}")
    logger.info(f"CSV results saved to: {csv_file}")
    logger.info(f"Average results saved to: {avg_file}")
    
    return avg_results

def EMOE_run(
    model_name, dataset_name, config=None, config_file="", seeds=[], is_tune=False,
    tune_times=500, feature_eeg="", feature_eog="",
    model_save_dir="", res_save_dir="", log_dir="",
    gpu_ids=[0], num_workers=4, verbose_level=1, mode=''
):
    model_name = model_name.lower()
    dataset_name = dataset_name.lower()

    if config_file != "":
        config_file = Path(config_file)
    else:
        config_file = Path(__file__).parent / "config" / "config.json"
    if not config_file.is_file():
        raise ValueError(f"Config file {str(config_file)} not found.")
    if model_save_dir == "":
        model_save_dir = Path.home() / "EMOE" / "saved_models"
    Path(model_save_dir).mkdir(parents=True, exist_ok=True)
    if res_save_dir == "":
        res_save_dir = Path.home() / "EMOE" / "results"
    Path(res_save_dir).mkdir(parents=True, exist_ok=True)
    if log_dir == "":
        log_dir = Path.home() / "EMOE" / "logs"
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    seeds = seeds if seeds != [] else [1111, 1112, 1113, 1114, 1115]
    logger = _set_logger(log_dir, model_name, dataset_name, verbose_level)

    args = get_config_regression(model_name, dataset_name, config_file)
    args.mode = mode
    args['model_save_path'] = Path(model_save_dir) / f"{args['model_name']}-{args['dataset_name']}.pth"
    args['device'] = assign_gpu(gpu_ids)
    args['train_mode'] = 'Classification'
    args['feature_eeg'] = feature_eeg
    args['feature_eog'] = feature_eog
    args['use_context'] = True
    args['ctx_len'] = 5  # 建议 7，收益大且计算还好

    # 添加保存路径到args
    args['res_save_dir'] = res_save_dir
    args['model_save_dir'] = model_save_dir
    args['log_dir'] = log_dir
    
    if config:
        args.update(config)

    res_save_dir = Path(res_save_dir) / "normal"
    res_save_dir.mkdir(parents=True, exist_ok=True)
    model_results = []
    
    result = _run(args, num_workers, is_tune, seed_idx=0)
    model_results.append(result)
    
    if 1:
        criterions = list(model_results[0].keys())
        csv_file = res_save_dir / f"{dataset_name}.csv"
        if csv_file.is_file():
            df = pd.read_csv(csv_file)
        else:
            df = pd.DataFrame(columns=["Model"] + criterions)
        res = [model_name]
        for c in criterions:
            values = [r[c] for r in model_results]
            mean = round(np.mean(values)*100, 2)
            std = round(np.std(values)*100, 2)
            res.append((mean, std))
        df.loc[len(df)] = res
        df.to_csv(csv_file, index=None)
        logger.info(f"Results saved to {csv_file}.")

def _run(args, num_workers=4, is_tune=False, from_sena=False, seed_idx=0):
    folds_eeg_data = load_folds_data_with_val("/data2/zjp0212/sleep-edfx/eeg_fpz_cz", 20)
    folds_eog_data = load_folds_data_with_val("/data2/zjp0212/sleep-edfx/eog_horizontal", 20)

    # ===== 本地 helper：统计每类样本数量（给 CB-Focal 用）=====
    def estimate_samples_per_cls(train_loader, num_classes):
        counts = torch.zeros(int(num_classes), dtype=torch.long)
        for batch in train_loader:
            y = batch["labels"]["M"].view(-1)
            # y 可能还在 CPU，这里统一转到 CPU 统计
            y = y.detach().cpu()
            for c in range(int(num_classes)):
                counts[c] += (y == c).sum()
        return counts.tolist()

    # 初始化结果记录
    all_fold_results = {
        'experiment_info': {
            'model_name': args.model_name,
            'dataset_name': args.dataset_name,
            'seed_index': seed_idx,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'total_folds': 20
        },
        'folds': {}
    }

    for i in range(20):
        fold_id = int(i)
        logger.info(f"\n{'='*60}")
        logger.info(f"Starting Fold {fold_id + 1}/20 (Seed {seed_idx + 1})")
        logger.info(f"{'='*60}")

        # 创建数据加载器
        dataloader, _ = Dataloader_Multimodal_Generator(
            args,
            folds_eog_data[fold_id][0], folds_eog_data[fold_id][1], folds_eog_data[fold_id][2],
            folds_eeg_data[fold_id][0], folds_eeg_data[fold_id][1], folds_eeg_data[fold_id][2],
            num_workers
        )

        # ===== 关键新增：统计训练集类别分布，写入 args（CB-Focal 需要）=====
        try:
            args.samples_per_cls = estimate_samples_per_cls(dataloader["train"], args.num_classes)
            logger.info(f"[Fold {fold_id}] samples_per_cls = {args.samples_per_cls}")
        except Exception as e:
            logger.warning(f"[Fold {fold_id}] samples_per_cls estimation failed: {e}")
            # fallback：至少保证 trainer 能跑
            args.samples_per_cls = [1 for _ in range(args.num_classes)]

        # 创建模型和训练器
        if args.model_name == "emoe_i2moe":
            seq_len = dataloader["train"].dataset.get_seq_len()
            model = emoe_i2moe.EMOEI2MOE(args, seq_len=seq_len).cuda()
        else:
            model = getattr(emoe, "EMOE")(args).cuda()
        model = nn.DataParallel(model)
        trainer = ATIO().getTrain(args)

        # 训练模型
        logger.info(f"Training Fold {fold_id + 1}...")
        _ = trainer.do_train(model, dataloader, return_epoch_results=from_sena, fold_id=fold_id)

        # 加载最佳模型进行测试
        if args.model_name == "emoe_i2moe":
            model_path = f"pt/emoe_i2moe_fold{fold_id}.pth"
        else:
            model_path = f"pt/emoe_fold{fold_id}.pth"
        if os.path.exists(model_path):
            state_dict = torch.load(model_path)
            if isinstance(model, nn.DataParallel):
                model.module.load_state_dict(state_dict)
            else:
                model.load_state_dict(state_dict)

            # 在训练集、验证集、测试集上进行评估
            train_results = trainer.do_test(model, dataloader['train'], mode="TRAIN")
            val_results = trainer.do_test(model, dataloader['valid'], mode="VAL")
            test_results = trainer.do_test(model, dataloader['test'], mode="TEST")

            # 记录结果
            fold_result = {
                'train': train_results,
                'val': val_results,
                'test': test_results
            }
            all_fold_results['folds'][f'fold_{fold_id}'] = fold_result

            # 打印当前fold结果
            logger.info(f"\nFold {fold_id + 1} Results:")
            logger.info(f"Train - Acc: {train_results['Acc_5']:.4f}, F1_weighted: {train_results['F1_weighted']:.4f}")
            logger.info(f"Val   - Acc: {val_results['Acc_5']:.4f}, F1_weighted: {val_results['F1_weighted']:.4f}")
            logger.info(f"Test  - Acc: {test_results['Acc_5']:.4f}, F1_weighted: {test_results['F1_weighted']:.4f}")
        else:
            logger.warning(f"Model file {model_path} not found, skipping evaluation for fold {fold_id}")

        # 清理资源
        del model, trainer, dataloader
        torch.cuda.empty_cache()
        gc.collect()
        time.sleep(1)

    # 保存所有fold的结果 - 使用正确的路径
    res_save_dir = Path(args.res_save_dir) / "normal"
    avg_results = save_fold_results(all_fold_results, res_save_dir, args.model_name, args.dataset_name)

    # 打印最终汇总结果
    logger.info(f"\n{'='*60}")
    logger.info(f"FINAL RESULTS SUMMARY (Seed {seed_idx + 1})")
    logger.info(f"{'='*60}")
    logger.info(f"Average Test Accuracy: {avg_results['test_acc_mean']:.4f} ± {avg_results['test_acc_std']:.4f}")
    logger.info(f"Average Test F1-weighted: {avg_results['test_f1_weighted_mean']:.4f} ± {avg_results['test_f1_weighted_std']:.4f}")
    logger.info(f"Average Test F1-macro: {avg_results['test_f1_macro_mean']:.4f} ± {avg_results['test_f1_macro_std']:.4f}")

    return avg_results
