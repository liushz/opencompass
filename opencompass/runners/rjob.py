import os
import os.path as osp
import random
import subprocess
import time
import uuid
from functools import partial
from typing import Any, Dict, List, Optional, Tuple

import mmengine
from mmengine.config import ConfigDict
from mmengine.utils import track_parallel_progress

from opencompass.registry import RUNNERS, TASKS
from opencompass.utils import get_logger

from .base import BaseRunner


@RUNNERS.register_module()
class RJOBRunner(BaseRunner):
    """Runner for submitting jobs via rjob bash script. Structure similar to
    DLC/VOLC runners.

    Args:
        task (ConfigDict): Task type config.
        rjob_cfg (ConfigDict): rjob related configuration.
        max_num_workers (int): Maximum number of concurrent workers.
        retry (int): Number of retries on failure.
        debug (bool): Whether in debug mode.
        lark_bot_url (str): Lark notification URL.
        keep_tmp_file (bool): Whether to keep temporary files.
        phase (str): Task phase.
    """

    def __init__(
        self,
        task: ConfigDict,
        rjob_cfg: ConfigDict,
        max_num_workers: int = 32,
        retry: int = 3,
        debug: bool = False,
        lark_bot_url: str = None,
        keep_tmp_file: bool = True,
        phase: str = 'unknown',
    ):
        super().__init__(task=task, debug=debug, lark_bot_url=lark_bot_url)
        self.rjob_cfg = rjob_cfg
        self.max_num_workers = max_num_workers
        self.retry = retry
        self.keep_tmp_file = keep_tmp_file
        self.phase = phase

    def launch(self, tasks: List[Dict[str, Any]]) -> List[Tuple[str, int]]:
        """Launch multiple tasks."""
        if not self.debug:
            status = track_parallel_progress(
                self._launch,
                tasks,
                nproc=self.max_num_workers,
                keep_order=False,
            )
        else:
            status = [self._launch(task, random_sleep=True) for task in tasks]
        return status

    def _run_task(self, task_name, log_path, poll_interval=60):
        """Poll rjob status until both active and pending are 0.

        Break if no dict line is found.
        """
        logger = get_logger()
        status = None
        time.sleep(10)
        while True:
            get_cmd = f'rjob get {task_name}'
            get_result = subprocess.run(get_cmd,
                                        shell=True,
                                        text=True,
                                        capture_output=True)
            output = get_result.stdout
            if log_path:
                with open(log_path, 'a', encoding='utf-8') as f:
                    f.write(f'\n[rjob get] {output}\n')

            # check if the command is executed successfully
            if get_result.returncode != 0:
                logger.error(f'rjob get command failed: {get_result.stderr}')
                logger.info('retrying...')
                status = 'ERROR'
                continue

            found_dict = False
            for line in output.splitlines():
                logger.info(f'line: {line}')
                if 'Starting' in line:
                    status = 'Starting'
                    found_dict = True
                    break
                if 'Pending' in line:
                    status = 'Pending'
                    found_dict = True
                    break
                if 'Running' in line:
                    status = 'Running'
                    found_dict = True
                    break
                if 'Timeout' in line:
                    status = 'Timeout'
                    found_dict = True
                    break
                if 'Restarting' in line:
                    status = 'Restarting'
                    found_dict = True
                    break
                if 'Queued' in line:
                    status = 'Queued'
                    found_dict = True
                    break
                if 'Suspended' in line:
                    status = 'Suspended'
                    found_dict = True
                    break
                if 'Submitted' in line:
                    status = 'Submitted'
                    found_dict = True
                    break
                if 'Succeeded' in line:
                    status = 'FINISHED'
                    break
                if 'Stopped' in line:
                    status = 'STOPPED'
                    break
                if 'Failed' in line or 'failed' in line:
                    status = 'FAILED'
                    break
                if 'Cancelled' in line:
                    status = 'CANCELLED'
                    break
                logger.warning(f'Unrecognized status in: {output}')
            if found_dict:
                time.sleep(poll_interval)
                continue
            break
        logger.info(f'[RJOB] Final status returned: {status}')
        return status

    def _launch(self, cfg: ConfigDict, random_sleep: Optional[bool] = None):
        """Launch a single task via rjob bash script."""
        if random_sleep is None:
            random_sleep = self.max_num_workers > 8
        if random_sleep:
            sleep_time = random.randint(0, 30)
            logger = get_logger()
            logger.info(f'Sleeping for {sleep_time} seconds to launch task')
            time.sleep(sleep_time)
        task = TASKS.build(dict(cfg=cfg, type=self.task_cfg['type']))
        num_gpus = task.num_gpus
        # Normalize task name
        logger = get_logger()
        logger.info(f'Task config: {cfg}')
        logger.info(f'Rjob config: {self.rjob_cfg}')
        # Obtain task_id in safe way, if not exist, use default value
        task_id = self.rjob_cfg.get('task_id', 'unknown')
        task_name = f'oc-{self.phase}-{task_id}-{str(uuid.uuid4())[:8]}'
        logger.info(f'Task name: {task_name}')
        # Generate temporary parameter file
        pwd = os.getcwd()
        mmengine.mkdir_or_exist('tmp/')
        uuid_str = str(uuid.uuid4())
        param_file = f'{pwd}/tmp/{uuid_str}_params.py'
        try:
            cfg.dump(param_file)
            # Construct rjob submit command arguments
            args = []
            # Basic parameters
            args.append(f'--name={task_name}')
            if num_gpus == 0:
                args.append('--gpu=0')
            elif num_gpus == 1:
                args.append('--gpu=1')
                args.append('--memory=200000')
                args.append('--cpu=32')
            elif num_gpus == 2:
                args.append('--gpu=2')
                args.append('--memory=400000')
                args.append('--cpu=64')
            elif num_gpus == 4:
                args.append('--gpu=4')
                args.append('--memory=800000')
                args.append('--cpu=128')
            elif num_gpus == 8:
                args.append('--gpu=8')
                args.append('--memory=1600000')
                args.append('--cpu=256')
            else:
                raise ValueError(f'Unsupported number of GPUs: {num_gpus}')
            if self.rjob_cfg.get('charged_group'):
                args.append(
                    f'--charged-group={self.rjob_cfg["charged_group"]}')
            if self.rjob_cfg.get('private_machine'):
                args.append(
                    f'--private-machine={self.rjob_cfg["private_machine"]}')
            if self.rjob_cfg.get('mount'):
                # Support multiple mounts
                mounts = self.rjob_cfg['mount']
                if isinstance(mounts, str):
                    mounts = [mounts]
                for m in mounts:
                    args.append(f'--mount={m}')
            if self.rjob_cfg.get('image'):
                args.append(f'--image={self.rjob_cfg["image"]}')
            if self.rjob_cfg.get('replicas'):
                args.append(f'-P {self.rjob_cfg["replicas"]}')
            if self.rjob_cfg.get('host_network'):
                host_network_val = self.rjob_cfg['host_network']
                if isinstance(host_network_val, bool):
                    host_network_val = 'true' if host_network_val else 'false'
                args.append(f'--host-network={host_network_val}')
            if self.rjob_cfg.get('gang_start'):
                gang_start_val = self.rjob_cfg['gang_start']
                if isinstance(gang_start_val, bool):
                    gang_start_val = 'true' if gang_start_val else 'false'
                args.append(f'--gang-start={gang_start_val}')
            if self.rjob_cfg.get('auto_restart'):
                auto_restart_val = self.rjob_cfg['auto_restart']
                if isinstance(auto_restart_val, bool):
                    auto_restart_val = 'true' if auto_restart_val else 'false'
                args.append(f'--auto-restart={auto_restart_val}')
            if self.rjob_cfg.get('preemptible'):
                preemptible_val = self.rjob_cfg['preemptible']
                # 处理布尔值或字符串值
                if isinstance(preemptible_val, bool):
                    preemptible_val = 'yes' if preemptible_val else 'no'
                args.append(f'--preemptible={preemptible_val}')
            # Environment variables
            envs = self.rjob_cfg.get('env', {})
            if isinstance(envs, dict):
                for k, v in envs.items():
                    args.append(f'-e {k}={v}')
            elif isinstance(envs, list):
                for e in envs:
                    args.append(f'-e {e}')

            # Additional environment variables from extra_envs
            extra_envs = self.rjob_cfg.get('extra_envs', [])
            if isinstance(extra_envs, list):
                for env_var in extra_envs:
                    args.append(f'-e {env_var}')
            # Additional arguments
            if self.rjob_cfg.get('extra_args'):
                args.extend(self.rjob_cfg['extra_args'])
            # Get launch command through task.get_command
            # compatible with template
            tmpl = '{task_cmd}'
            get_cmd = partial(task.get_command,
                              cfg_path=param_file,
                              template=tmpl)
            entry_cmd = get_cmd()
            
            # 如果配置了特定的 Python 环境，替换命令中的 python 路径
            if self.rjob_cfg.get('python_env_path'):
                python_path = f"{self.rjob_cfg['python_env_path']}/bin/python"
                import sys
                current_python = sys.executable
                
                # 更安全的替换逻辑：只替换一次，避免重复
                if current_python in entry_cmd:
                    entry_cmd = entry_cmd.replace(current_python, python_path, 1)
                    logger.info(f'Replaced {current_python} with {python_path}')
                else:
                    # 如果没有找到完整路径，尝试替换 python 命令开头
                    import re
                    # 匹配命令开头的 python（避免替换路径中的 python）
                    entry_cmd = re.sub(r'^python\s+', f'{python_path} ', entry_cmd)
                    logger.info(f'Using specified Python path: {python_path}')
                
                logger.info(f'Final entry command after Python replacement: {entry_cmd}')
            
            # 在命令中添加环境变量设置
            if self.rjob_cfg.get('python_env_path'):
                env_setup = f'export PATH={self.rjob_cfg["python_env_path"]}/bin:$PATH && export PYTHONPATH={pwd}:$PYTHONPATH && '
                entry_cmd = f'bash -c "cd {pwd} && {env_setup}{entry_cmd}"'
            else:
                entry_cmd = f'bash -c "cd {pwd} && {entry_cmd}"'
            # Construct complete command
            cmd = f"rjob submit {' '.join(args)} -- {entry_cmd}"
            logger = get_logger()
            logger.info(f'Running command: {cmd}')
            # Log output
            if self.debug:
                out_path = None
            else:
                out_path = task.get_log_path(file_extension='out')
                mmengine.mkdir_or_exist(osp.split(out_path)[0])

            retry = self.retry
            result = None
            # 至少执行一次，即使 retry=0
            attempts = max(1, retry + 1)
            for attempt in range(attempts):
                # Only submit, no polling
                result = subprocess.run(cmd,
                                        shell=True,
                                        text=True,
                                        capture_output=True)
                logger.info(f'CMD: {cmd}')
                logger.info(f'Command output: {result.stdout}')
                if result.stderr:
                    logger.error(f'Command error: {result.stderr}')
                logger.info(f'Return code: {result.returncode}')
                if result.returncode == 0:
                    break
                if attempt < attempts - 1:  # 不是最后一次尝试
                    retry_time = random.randint(5, 60)
                    logger.info(f"Attempt {attempt + 1} failed, retrying in {retry_time} seconds")
                    time.sleep(retry_time)
            if result.returncode != 0:
                # Submit failed, return directly
                return task_name, result.returncode

            # Submit successful, start polling
            status = self._run_task(task_name, out_path)
            output_paths = task.get_output_paths()
            returncode = 0 if status == 'FINISHED' else 1
            if self._job_failed(returncode, output_paths):
                returncode = 1
        finally:
            if not self.keep_tmp_file:
                os.remove(param_file)

        return task_name, returncode

    def _job_failed(self, return_code: int, output_paths: List[str]) -> bool:
        return return_code != 0 or not all(
            osp.exists(output_path) for output_path in output_paths)
