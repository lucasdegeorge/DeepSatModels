import os
import subprocess
from pathlib import Path
import math
import shlex


class JeanZayExperiment:
    def __init__(
        self,
        exp_name,
        job_name,
        slurm_array_nb_jobs=None,
        qos="t3",
        account="fbe",
        gpu_type="a100",
        cmd_path="train_and_eval/segmentation_training_transf.py",
        time=None,
    ):
        self.expname = exp_name
        self.job_name = job_name
        self.qos = qos
        self.account = account
        self.gpu_type = gpu_type
        self.slurm_array_nb_jobs = slurm_array_nb_jobs
        self.cmd_path = cmd_path
        self.time = time

    def build_cmd(self, hydra_args):
        hydra_modifiers = []

        for hydra_arg, value in hydra_args.items():
            if hydra_arg.startswith("--"):
                hydra_modifiers.append(f" {hydra_arg} {value}")
            else:
                hydra_modifiers.append(f" {hydra_arg}={value}")
        self.cmd = f"python {self.cmd_path} {''.join(hydra_modifiers)}"
        print(f"srun {self.cmd}")

    def launch(self, debug=False, return_job_id=False):
        if debug:
            self.qos = "dev"
            self.time = "02:00:00"
        # Check if either a single command or a list of commands has been built
        if not hasattr(self, "cmd") and not hasattr(self, "cmds"):
            raise ValueError("Run build_cmd first")
        if self.cmd is None and self.cmds is None:
            raise ValueError("Run build_cmd first - no command generated.")

        if self.qos == "prepost":
            slurm_partition_directive = f"#SBATCH --partition=prepost"
            self.time = "19:59:59" if self.time is None else self.time
            slurm_gpu_directive = ""  # No specific GPU constraint for prepost
            slurm_account_directive = f"#SBATCH --account={self.account}@a100"  # Account may not be GPU specific
            cpus_per_task = 1  # Assuming 1 CPU for prepost tasks
            module_load_directive = ""  # No specific module load for prepost
            print(f"Launching on partition prepost")

        else:  # Handle GPU partitions
            if self.qos == "t4":
                self.qos_name = "qos_gpu-t4"
                self.time = "99:59:59" if self.time is None else self.time
            elif self.qos == "t3":
                self.qos_name = "qos_gpu-t3"
                self.time = "19:59:59" if self.time is None else self.time
            elif self.qos == "dev":
                self.qos_name = "qos_gpu-dev"
                self.time = "01:59:59" if self.time is None else self.time
            else:
                raise ValueError(f"Not a valid QoS for GPU partitions: {self.qos}")

        if self.gpu_type == "a100":
            self.gpu_slurm_directive = "#SBATCH -C a100"
            self.cpus_per_task = 8
            self.qos_name = self.qos_name.replace("gpu", "gpu_a100")

        elif self.gpu_type == "h100":
            self.gpu_slurm_directive = "#SBATCH -C h100"
            self.cpus_per_task = 24
            self.qos_name = self.qos_name.replace("gpu", "gpu_h100")

        elif self.gpu_type == "v100":
            self.gpu_slurm_directive = "#SBATCH -C v100-32g"
            self.cpus_per_task = 10
        else:
            raise ValueError(f"Not a valid GPU type: {self.gpu_type}")

        local_slurmfolder = Path("logs") / Path(self.expname)
        local_slurmfolder.mkdir(parents=True, exist_ok=True)

        array_string = ""

        if isinstance(self.slurm_array_nb_jobs, int):
            total_jobs = self.slurm_array_nb_jobs
            if total_jobs <= 0:
                array_string = ""
            else:
                array_string = f"0-{total_jobs - 1}"

        elif self.slurm_array_nb_jobs is not None:
            raise ValueError("slurm_array_nb_jobs must be an int or None.")

        sbatch_array = f"#SBATCH --array={array_string}" if array_string else ""

        current_job_name = f"{self.job_name}"
        slurm_path = local_slurmfolder / f"job_file.slurm"

        # Store the script path
        self.slurm_script_path = slurm_path

        # Construct and store output/error paths with separate directories per task
        slurm_output_base_dir = f"/lustre/fswork/projects/rech/fbe/uaa31dq/GitHub/DeepSatModels/{local_slurmfolder}/job_%j_%a"

        # Prepare commands for the SLURM script
        srun_command_line = ""
        bash_definitions = ""

        self.cmds = getattr(self, "cmds", None)  # Ensure cmds is defined
        if self.cmds:
            # Create a bash array definition, quoting each command safely
            quoted_cmds = [shlex.quote(cmd) for cmd in self.cmds]
            bash_definitions = f"CMDS=({' '.join(quoted_cmds)})"
            # Use the SLURM_ARRAY_TASK_ID to index into the bash array
            # Adjust task ID based on the start index of the chunk
            start_index = 0
            if self.slurm_array_nb_jobs > self.max_array_size:
                # Ensure sub_job_index and max_array_size are defined and valid before using
                if self.sub_job_index is None or self.max_array_size is None:
                    raise ValueError(
                        "sub_job_index and max_array_size must be set for chunked arrays."
                    )
                start_index = self.sub_job_index * self.max_array_size

            srun_command_line = (
                f"COMMAND_INDEX=$((SLURM_ARRAY_TASK_ID - {start_index}))\n"
                f'echo "Running command index $COMMAND_INDEX: ${{CMDS[$COMMAND_INDEX]}}"\n'
                f"srun ${{CMDS[$COMMAND_INDEX]}}"
            )

        elif self.cmd:
            # Existing single command execution
            srun_command_line = f"srun {self.cmd}"
        else:
            # This case should ideally not be reached if build_cmd was called
            raise ValueError(
                "No command or command list was built. Call build_cmd first."
            )

        slurm = f"""#!/bin/bash
#SBATCH --job-name={current_job_name}
{sbatch_array}
#SBATCH --nodes=1	# number of nodes
#SBATCH --account={self.account}@{self.gpu_type}	
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --qos={self.qos_name}
{self.gpu_slurm_directive}
#SBATCH --cpus-per-task={self.cpus_per_task}
#SBATCH --hint=nomultithread
#SBATCH --time={self.time}
{"#SBATCH --time-min=" + self.min_time if self.min_time is not None else ""}
#SBATCH --output={slurm_output_base_dir}.out
#SBATCH --error={slurm_output_base_dir}.err
#SBATCH --signal=SIGUSR1@90
module purge
{"module load arch/" + self.gpu_type if self.gpu_type in ["a100", "h100"] else ""}
source /lustre/fswork/projects/rech/fbe/uaa31dq/.venv/geopom/bin/activate

export HYDRA_FULL_ERROR=1 # to have the full traceback
export WANDB_CACHE_DIR=$NEWSCRATCH/wandb_cache
export WANDB_MODE=offline

# Define commands if using command list
{bash_definitions}

set -x
# Execute the appropriate command
{srun_command_line}
        """
        with open(slurm_path, "w") as slurm_file:
            slurm_file.write(slurm)
            
        os.system(f"sbatch {slurm_path}")
        return None