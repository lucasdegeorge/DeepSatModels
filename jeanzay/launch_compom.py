import argparse
from launch import JeanZayExperiment


def parse_mode():
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()

    return args

computer_name = "a100"

cmd_modifiers = []
exps = []

for expand in [2, 4, 6]:
    exp_name = f"TSViComPoM_expand{expand}"
    job_name = f"TSViComPoM_expand{expand}"
    jz_exp = JeanZayExperiment(exp_name, job_name)
    jz_exp.qos = "t3"
    jz_exp.account = "syq"
    jz_exp.gpu_type = computer_name
    jz_exp.time = "01:59:59"
    jz_exp.cmd_path = "train_and_eval/segmentation_training_transf.py"

    exps.append(jz_exp)

    exp_modifier = {
        "--config-path": "../configs/PASTIS24",
        "--config-name": "TSViComPoM_fold1", 
        "MODEL.expand": expand,
        "DATASETS.train.dataset": "PASTIS24_JZ_fold1",
        "DATASETS.eval.dataset": "PASTIS24_JZ_fold1",
        "DATASETS.test.dataset": "PASTIS24_JZ_fold1",
        "WANDB.wandb_run_name": exp_name,
    }
    cmd_modifiers.append(dict(**exp_modifier))


for n_sel_heads in [64, 128, 256]:
    exp_name = f"TSViComPoM_nsel{n_sel_heads}"
    job_name = f"TSViComPoM_nsel{n_sel_heads}"
    jz_exp = JeanZayExperiment(exp_name, job_name)
    jz_exp.qos = "t3"
    jz_exp.account = "syq"
    jz_exp.gpu_type = computer_name
    jz_exp.time = "01:59:59"
    jz_exp.cmd_path = "train_and_eval/segmentation_training_transf.py"

    exps.append(jz_exp)

    exp_modifier = {
        "--config-path": "../configs/PASTIS24",
        "--config-name": "TSViComPoM_fold1",
        "MODEL.n_sel_heads": n_sel_heads,
        "DATASETS.train.dataset": "PASTIS24_JZ_fold1",
        "DATASETS.eval.dataset": "PASTIS24_JZ_fold1",
        "DATASETS.test.dataset": "PASTIS24_JZ_fold1", 
        "WANDB.wandb_run_name": exp_name,
        "CHECKPOINT.save_path": f"models/saved_models/PASTIS24/{exp_name}",
    }
    cmd_modifiers.append(dict(**exp_modifier))


if __name__ == "__main__":
    for exp, cmd_modifier in zip(exps, cmd_modifiers):
        args = parse_mode()
        exp.build_cmd(cmd_modifier)
        if args.launch == True:
            exp.launch()