def submit_poc(db, payload, mode, log_dir, salt, oss_fuzz_path):
    # Placeholder for PoC submission logic
    # Ensure payload is serializable
    if hasattr(payload, "to_dict"):
        payload = payload.to_dict()
    return {
        "status": "submitted",
        "mode": mode,
        "payload": payload,
        "log_dir": str(log_dir),
        "salt": salt,
        "oss_fuzz_path": str(oss_fuzz_path)
    }

def run_poc_id(db, log_dir, poc_id, oss_fuzz_path):
    # Placeholder for running PoC by ID
    return {
        "status": "run",
        "poc_id": poc_id,
        "log_dir": str(log_dir),
        "oss_fuzz_path": str(oss_fuzz_path)
    }
