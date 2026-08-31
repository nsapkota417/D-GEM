import os
import time
import datetime
import pathlib
import yaml
import copy

SCRIPT_TEMPLATE = """#!/bin/bash 
#$ -M nsapkota@nd.edu
#$ -m abe
#$ -o crclogs/$JOB_NAME-$JOB_ID.log
#$ -j y
#$ -N {job_name}
#$ -q gpu{gpu}
#$ -l gpu_card={num_gpus}

source /users/nsapkota/afs/.bashrc 
conda activate pyt

echo -e "Assigned GPU(s): ${{SGE_HGR_gpu_card}}\\n"
echo -e "Starting Experiment =)"
echo -e "=-=-=-=-=-=-=-=-=-=-=-=-=\\n"
cd {src_path}

python src/train_video.py -cfg {config_file}
"""

class Job:
    def __init__(
        self,
        job_name,
        src_path,
        config_file,
        model_config_file=None,
        changes_d={},
        gpu='',
        num_gpus=1,
    ):
        self.time_created = datetime.datetime.now()
        self.created_config_fps = []
        self.created_script_fps = []

        # Save parameters
        self.set_setting('job_name', str(job_name))
        self.set_setting('src_path', str(src_path))
        self.set_setting('config_file', str(config_file))
        self.set_setting('model_config_file', str(model_config_file) if model_config_file else None)
        self.set_setting('gpu', str(gpu))
        self.set_setting('num_gpus', int(num_gpus))

        # Read configs
        self.orig_config = self._read_config_d(self.config_file)
        self.orig_model_config = (
            self._read_config_d(self.model_config_file) if self.model_config_file else {}
        )

        # Merge dataset + model config
        merged = copy.deepcopy(self.orig_config)
        merged.update(self.orig_model_config)

        # Apply overrides
        self.config = self._modify_config(merged, changes_d)

    def submit(self, n=1, numbers=None, pause=10, same_seed=False):
        if n < 1 and numbers is None:
            return

        if numbers is not None:
            n = len(numbers)
        else:
            numbers = tuple(range(1, n + 1))
        print(f'* {n} Job Number(s): {numbers} *')

        for index, i in enumerate(numbers):
            # Write merged config
            run_config_fp = self._write_temp_config(self.src_path, self.config, tag="combined")
            self.created_config_fps.append(run_config_fp)

            # Create unique script name
            model = self.config['train']['model']
            dataset = self.config['data']['name']
            st = f"{model}_{dataset}"
            timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            run_script_name = f"{st}_{timestamp_str}"

            # Fill template
            script_string = SCRIPT_TEMPLATE.format(
                job_name=self.job_name,
                gpu=self.gpu,
                num_gpus=self.num_gpus,
                src_path=self.src_path,
                config_file=run_config_fp,
            )

            run_script_fp = self._write_temp_script(self.src_path, script_string, run_script_name)
            self.created_script_fps.append(run_script_fp)

            submit_command = f"qsub \"{run_script_fp}\""
            print(f'⭐ Submitting job number {i} (total {n} runs).')
            print(f'   {submit_command}')
            print(f'   TmpPath: {run_config_fp.parent}')
            print(f'   Config:  {run_config_fp.name}')
            print(f'   Script:  {run_script_fp.name}')

            os.system(submit_command)

            if pause > 0:
                time.sleep(pause)

            print(f'😊 Success!\n')

    def _get_name(self, config):
        model = config['train']['model']
        dataset = config['data']['name']
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{model}_{dataset}_{timestamp_str}"

    def _read_config_d(self, config_file):
        if not config_file:
            return {}
        return yaml.load(open(config_file, "r"), Loader=yaml.FullLoader)

    def _modify_config(self, orig_config, changes_d=None):
        if not orig_config or not changes_d:
            return orig_config

        def set_config_item(cfg, k, v):
            split_str = k.split('.')
            if len(split_str) == 1:
                cfg[k] = v
            else:
                set_config_item(cfg[split_str[0]], '.'.join(split_str[1:]), v)

        config = copy.deepcopy(orig_config)
        for k, v in changes_d.items():
            set_config_item(config, k, v)
        return config

    def _write_temp_config(self, src_path, config, tag="combined", changes_d=None):
        if not config:
            return None

        if changes_d:
            config = self._modify_config(config, changes_d=changes_d)

        src_path = pathlib.Path(src_path)
        temp_dp = src_path / 'exp' / 'yamls'
        temp_dp.mkdir(exist_ok=True, parents=True)

        config_fn = self._get_name(config) + f".yaml"
        config_fp = temp_dp / config_fn

        with open(config_fp, 'w') as f:
            yaml.dump(config, f)

        return pathlib.Path(config_fp)

    def _write_temp_script(self, src_path, script_string, unique_name):
        src_path = pathlib.Path(src_path)
        scripts_dp = src_path / 'exp' / 'shs'
        scripts_dp.mkdir(exist_ok=True, parents=True)

        script_fp = str(scripts_dp / f"{unique_name}.sh")
        with open(script_fp, "w") as f:
            f.write(script_string)

        return pathlib.Path(script_fp)

    def print_settings(self, disp=True):
        name = self.__class__.__name__
        d = self.time_created
        string = (f'💼 {name} (created: {d.month}/{d.day} {d.hour}:{d.minute}:{d.second})\n')
        if hasattr(self, '_setting_names'):
            for setting_name in self._setting_names:
                string += f'   {setting_name}: {getattr(self, setting_name)}\n'
        if disp:
            print(string)
        return string

    def set_setting(self, name, value):
        if not hasattr(self, '_setting_names'):
            self._setting_names = []
        self._setting_names.append(name)
        setattr(self, name, value)

    def __repr__(self):
        return self.print_settings(disp=False)
