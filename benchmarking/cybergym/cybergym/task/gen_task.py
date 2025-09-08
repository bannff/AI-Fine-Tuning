import logging
from pathlib import Path
from .arvo_task import generate_arvo_task
from .oss_fuzz_task import generate_oss_fuzz_latest_task, generate_oss_fuzz_task
from .types import DEFAULT_SALT, Task, TaskConfig, TaskDifficulty, TaskType

logger = logging.getLogger(__name__)

TASK_GENERATORS = {
    TaskType.ARVO: generate_arvo_task,
    TaskType.OSS_FUZZ: generate_oss_fuzz_task,
    TaskType.OSS_FUZZ_LATEST: generate_oss_fuzz_latest_task,
}

def generate_task(config: TaskConfig) -> Task:
    """
    Generate a task based on the task type.
    """
    task_type = config.task_id.split(":")[0]
    if task_type not in TaskType._value2member_map_:
        raise ValueError(f"Unsupported task type: {task_type}")
    return TASK_GENERATORS[TaskType(task_type)](config)
