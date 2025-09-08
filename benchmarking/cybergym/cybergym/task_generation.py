"""
Task generation wrapper for CyberGym using the official API.
"""
from pathlib import Path
from cybergym.task.gen_task import generate_task
from cybergym.task.types import TaskConfig, TaskDifficulty

def generate_cybergym_task(
    task_id: str,
    out_dir: str,
    data_dir: str,
    server: str,
    difficulty: str = "level1",
    agent_id: str = None,
    with_flag: bool = False,
):
    """
    Pythonic wrapper for CyberGym task generation.
    """
    config = TaskConfig(
        task_id=task_id,
        out_dir=Path(out_dir),
        data_dir=Path(data_dir),
        server=server,
        difficulty=TaskDifficulty[difficulty],
        agent_id=agent_id,
        with_flag=with_flag,
    )
    return generate_task(config)
