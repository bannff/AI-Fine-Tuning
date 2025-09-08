"""
PoC submission and verification workflow for CyberGym.
"""
from pathlib import Path
from cybergym.server.server_utils import submit_poc, run_poc_id

# Simulate DB and payload for testing
class DummyDB:
    pass

class DummyPayload:
    def __init__(self, agent_id, task_id, data):
        self.agent_id = agent_id
        self.task_id = task_id
        self.data = data
        self.require_flag = False

    def to_dict(self):
        # Convert bytes to string if needed
        if isinstance(self.data, bytes):
            try:
                data_str = self.data.decode("utf-8")
            except Exception:
                data_str = repr(self.data)
        else:
            data_str = self.data
        return {
            "agent_id": self.agent_id,
            "task_id": self.task_id,
            "data": data_str,
            "require_flag": self.require_flag
        }

# Wrapper for PoC submission
def submit_cybergym_poc(agent_id, task_id, poc_data, mode="vul", log_dir="logs", salt="CyberGym", oss_fuzz_path="oss-fuzz-data"):
    db = DummyDB()
    payload = DummyPayload(agent_id, task_id, poc_data)
    result = submit_poc(db, payload, mode, log_dir, salt, oss_fuzz_path)
    return result

# Wrapper for PoC verification
def verify_cybergym_poc(poc_id, log_dir="logs", oss_fuzz_path="oss-fuzz-data"):
    db = DummyDB()
    result = run_poc_id(db, log_dir, poc_id, oss_fuzz_path)
    return result
