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

exp_name = f"TSViT"
job_name = f"TSViT"
jz_exp = JeanZayExperiment(exp_name, job_name)
jz_exp.qos = "t3"
jz_exp.account = "syq"
jz_exp.gpu_type = computer_name
jz_exp.time = "14:59:59"
jz_exp.cmd_path = "train_and_eval/segmentation_training_transf.py"

exps.append(jz_exp)

exp_modifier = {
    "--config-path": "../configs/PASTIS24",
    "--config-name": "TSViT_fold1",
    "DATASETS.train.dataset": "PASTIS24_JZ_fold1",
    "DATASETS.eval.dataset": "PASTIS24_JZ_fold1",
    "DATASETS.test.dataset": "PASTIS24_JZ_fold1",
    "WANDB.wandb_run_name": f"{exp_name}",
    "CHECKPOINT.save_path": f"models/saved_models/PASTIS24/{exp_name}",
}

cmd_modifiers.append(dict(**exp_modifier))


if __name__ == "__main__":
    for exp, cmd_modifier in zip(exps, cmd_modifiers):
        args = parse_mode()
        exp.build_cmd(cmd_modifier)
        if args.launch == True:
            exp.launch()
