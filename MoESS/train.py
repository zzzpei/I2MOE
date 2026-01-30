from run import EMOE_run

EMOE_run(model_name='emoe', dataset_name='SleepEDF-20', is_tune=False, seeds=[1111], model_save_dir="./pt",
         res_save_dir="./result", log_dir="./log", mode='train')     