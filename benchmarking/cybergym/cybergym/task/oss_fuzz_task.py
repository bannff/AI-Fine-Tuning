def generate_oss_fuzz_task(config):
    # Placeholder for OSS-Fuzz task generation logic
    return {
        "task_id": config.task_id,
        "agent_id": config.agent_id,
        "out_dir": str(config.out_dir),
        "type": "oss-fuzz"
    }

def generate_oss_fuzz_latest_task(config):
    # Placeholder for OSS-Fuzz latest task generation logic
    return {
        "task_id": config.task_id,
        "agent_id": config.agent_id,
        "out_dir": str(config.out_dir),
        "type": "oss-fuzz-latest"
    }
