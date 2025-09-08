# CyberGym: Large-Scale Cybersecurity Evaluation Framework

This directory contains the setup for CyberGym - a large-scale cybersecurity evaluation framework focused on real-world vulnerability analysis and proof-of-concept (PoC) generation.

## Quick Reference Links

- **Hugging Face Subset Dataset**: [cybergym-server subset](https://huggingface.co/datasets/sunblaze-ucb/cybergym-server)
- **Official PoC Submission Workflow**: [sunblaze-ucb/cybergym GitHub](https://github.com/sunblaze-ucb/cybergym)
- **PoC Submission Server Docs**: [README](https://github.com/sunblaze-ucb/cybergym#evaluation)


## Overview

CyberGym is a comprehensive cybersecurity simulation platform that provides:
- Real-world vulnerability scenarios
- Docker-based sandbox environments
- PoC submission and verification system
- Agent evaluation across diverse cybersecurity tasks

### Key Features

- **Large-Scale Dataset**: ~10TB full dataset with comprehensive vulnerability scenarios
- **Subset Available**: Curated subset with 10 key tasks (5 successful, 5 challenging)
- **Docker Sandbox**: Isolated environments for safe vulnerability testing
- **PoC Verification**: Automatic verification of proof-of-concept submissions
- **Agent Framework**: Support for autonomous cybersecurity agents

### Evaluation Categories

1. **Vulnerability Analysis**: Identify and analyze security vulnerabilities
2. **Exploit Development**: Create working exploits for identified vulnerabilities
3. **PoC Generation**: Generate proof-of-concept demonstrations
4. **Security Assessment**: Evaluate overall security posture
5. **Mitigation Planning**: Develop remediation strategies

### Quick Start (Subset)

```bash
# Download and extract the subset dataset (~100GB)

### System Requirements

**Local Development:**
- **Docker**: Latest version with sufficient storage (~10TB for full dataset, ~100GB for subset)
- **Python**: 3.8+
- **Memory**: 16GB+ RAM recommended for model inference
- **CPU**: Multi-core processor (8+ cores recommended)

**Production/AWS Hosting:**
- See [AWS Hosting Requirements](#aws-hosting-requirements) section below
```

### PoC Submission Server Setup

```bash
# Install dependencies (see official repo for details)
- **Memory**: 16GB+ RAM recommended for full evaluations

# Start the PoC submission server
- **Network**: High-bandwidth internet for dataset downloads

### Software Dependencies

- Docker and Docker Compose
- Python cybersecurity libraries
- MLX-LM for model integration
```

### Task Generation Workflow

```bash
# Generate a working directory for a specific task
- Requests and web scraping tools

## Setup

### Quick Start (Subset)

```bash
# Run setup script (installs subset dataset ~100GB)
./setup.sh --subset

# Activate environment

# The OUT_DIR will contain:
source ../venv/bin/activate

# Run sample evaluation
python run_evaluation.py --sample
```

### PoC Submission and Scoring

```bash
# Generate your PoC file (e.g., using your agent/model)
```

# Submit the PoC for scoring


# Example server return (JSON)
### Full Dataset Setup
```

### Verify Submitted PoCs

```bash
# (Optional) Verify PoCs using the server API

```bash
# Setup with full dataset (~10TB - requires significant storage)
./setup.sh --full

```
# Configure dataset location
export CYBERGYM_DATA_PATH="/path/to/storage/cybergym-data"

# Run full evaluation
python run_evaluation.py --full
```

## Dataset Options

### Subset Dataset (~100GB)
- 10 curated vulnerability scenarios
- 5 successful test cases (solvable)
- 5 challenging test cases (advanced)
- Representative sample of full dataset
- Suitable for development and testing

### Full Dataset (~10TB)
- Complete vulnerability database
- Thousands of real-world scenarios
- Historical vulnerability data
- Enterprise-grade evaluation suite
- Production-ready benchmarking

## Usage

### PHI-3 Model Integration

```python
from utils.model_loader import load_phi3_model
from cybergym_evaluation import CyberGymBenchmark

# Load PHI-3 cybersecurity model
model = load_phi3_model()

# Initialize CyberGym benchmark
benchmark = CyberGymBenchmark(dataset="subset")

# Run evaluation
results = benchmark.evaluate_model(model)
```

### Command Line Interface

```bash
# Run sample evaluation
python run_evaluation.py --sample

# Evaluate specific vulnerability types
python run_evaluation.py --categories "SQL Injection,XSS"

# Run with custom model parameters
python run_evaluation.py --temperature 0.1 --max_tokens 1024

# Generate PoC for specific vulnerability
python run_evaluation.py --generate_poc --vuln_id CVE-2023-1234
```

### Docker Integration

```bash
# Start CyberGym containers
docker-compose up -d cybergym-sandbox

# Run evaluation in containerized environment
docker exec cybergym-sandbox python evaluate_model.py

# Clean up containers
docker-compose down
```

## Evaluation Process

1. **Environment Setup**: Docker containers simulate vulnerable systems
2. **Task Assignment**: Model receives vulnerability scenario description
3. **Analysis Phase**: Model analyzes vulnerability and plans approach
4. **PoC Generation**: Model creates proof-of-concept exploit
5. **Verification**: Automated system verifies PoC effectiveness
6. **Scoring**: Results scored based on success rate and approach quality

## Results Format

```json
{
  "benchmark": "cybergym",
  "model": "phi3-cybersec",
  "dataset": "subset",
  "total_scenarios": 10,
  "successful_pocs": 7,
  "success_rate": 0.70,
  "categories": {
    "SQL Injection": {"success": 2, "total": 3},
    "XSS": {"success": 3, "total": 4},
    "File Upload": {"success": 2, "total": 3}
  },
  "scenarios": [...]
}
```

## Configuration

### Environment Variables

```bash
# Dataset configuration
export CYBERGYM_DATASET_PATH="/path/to/cybergym-data"
export CYBERGYM_SUBSET_MODE="true"

# Docker configuration
export CYBERGYM_DOCKER_NETWORK="cybergym-net"
export CYBERGYM_CONTAINER_TIMEOUT="300"

# Model configuration
export CYBERGYM_MODEL_TEMPERATURE="0.1"
export CYBERGYM_MAX_GENERATION_TIME="120"
```

### Custom Configuration

Edit `cybergym_config.py` to customize:
- Dataset paths and filtering
- Docker container settings
- Model parameters
- Evaluation criteria
- Output formats

## Current Status

🚧 **Implementation Required**

This is a scaffold for CyberGym integration. To complete the setup:

1. **Install CyberGym**: Set up Docker containers and dataset
2. **Model Integration**: Implement PHI-3 model wrapper for CyberGym API
3. **PoC Generation**: Configure exploit generation pipeline
4. **Verification System**: Set up automated PoC verification
5. **Results Processing**: Implement scoring and analysis

## Security Considerations

⚠️ **Important Security Notes**:
- CyberGym involves real vulnerability scenarios
- Always run in isolated Docker environments
- Never test against production systems
- Follow responsible disclosure practices
- Ensure proper network isolation

## References

- [CyberGym Research Paper](https://example.com/cybergym-paper)
- [CyberGym GitHub Repository](https://github.com/cybergym/cybergym)
- [Docker Security Best Practices](https://docs.docker.com/engine/security/)
- [Responsible Vulnerability Disclosure](https://example.com/responsible-disclosure)

## Next Steps

1. Run `./setup.sh --subset` to install subset dataset
2. Configure Docker environment
3. Implement PHI-3 model integration
4. Test with sample vulnerability scenarios
5. Scale to full evaluation suite

## Support

For implementation questions:
- Review setup logs in `logs/setup.log`
- Check Docker container status with `docker ps`
- Verify dataset integrity with `python verify_dataset.py`
- Consult CyberGym documentation for advanced configuration

## Evaluation Workflow & Troubleshooting

### Running the Evaluation

1. **Download the CyberGym subset sample:**
   ```bash
   PYTHONPATH=benchmarking/cybergym/cybergym python3 cybergym/cybergym/download_cybergym_subset.py
   ```
   This generates `cybergym_subset_sample.json` for evaluation.

2. **Run the evaluation script:**
   ```bash
   PYTHONPATH=benchmarking/cybergym/cybergym python3 cybergym/cybergym/run_evaluation.py --sample_file cybergym_subset_sample.json --model_path /path/to/phi3_model --adapter_path /path/to/lora_adapter
   ```
   This will use your PHI model and LoRA adapter to generate responses, submit PoCs, and verify them for each vulnerability in the sample.

### Output Format
- Results are output as valid JSON, including:
  - Task ID, project name, prompt, model response
  - PoC submission result (status, payload, etc.)
  - PoC verification result

### Troubleshooting
- **Serialization errors:**
  - If you see errors about objects not being JSON serializable, ensure all custom objects (e.g., DummyPayload) implement a `to_dict()` method and convert any bytes fields to strings.
- **Missing dependencies:**
  - Install missing Python packages (e.g., `pip install datasets`).
- **Model loading issues:**
  - Ensure your model and adapter paths are correct and compatible with MLX-LM.
- **Docker/PoC server issues:**
  - Refer to the official [PoC Submission Server Docs](https://github.com/sunblaze-ucb/cybergym#evaluation) for setup and troubleshooting.

### Edge Cases
- All fields in the output are now robustly serialized, including bytes and custom objects.
- If you encounter new serialization issues, update the relevant class to handle conversion in `to_dict()`.

---

## AWS Hosting Requirements

### 🏗️ Architecture Overview

CyberGym requires significant infrastructure for production hosting due to its Docker-based vulnerability testing and large dataset requirements.

### 📋 AWS Services Required

**Core Compute & Storage:**
- **EC2 Instances**: 
  - `c5.4xlarge` or larger (16+ vCPUs, 32+ GB RAM) for model inference
  - `m5.xlarge` for orchestration and API services
- **EBS Storage**: 
  - 10TB+ for full dataset (`gp3` recommended for cost efficiency)
  - 100GB for subset mode
- **S3 Buckets**: 
  - Dataset storage and backup
  - Model artifacts and results storage

**Container & Security:**
- **ECS Fargate** or **EKS**: For Docker container orchestration
- **VPC**: Isolated network for security sandboxes
- **Security Groups**: Strict ingress/egress rules for vulnerability testing
- **NAT Gateway**: Outbound internet access for containers

**Database & Monitoring:**
- **RDS PostgreSQL**: Store evaluation results and metadata
- **ElastiCache Redis**: Caching for frequently accessed data
- **CloudWatch**: Logging and monitoring
- **X-Ray**: Distributed tracing for complex workflows

### 💰 Estimated AWS Costs (Monthly)

**Subset Mode (~100GB dataset):**
- EC2 instances: $400-600/month
- EBS storage: $30-50/month  
- S3 storage: $10-20/month
- RDS: $50-100/month
- **Total: ~$500-800/month**

**Full Mode (~10TB dataset):**
- EC2 instances: $800-1200/month
- EBS storage: $1000-1500/month
- S3 storage: $200-300/month
- RDS: $100-200/month
- **Total: ~$2100-3200/month**

### 🚀 Deployment Options

#### Option 1: ECS Fargate (Recommended)
```yaml
# docker-compose-aws.yml
version: '3.8'
services:
  cybergym-api:
    image: your-repo/cybergym:latest
    cpu: 2048
    memory: 4096
    environment:
      - AWS_REGION=us-east-1
      - S3_BUCKET=your-cybergym-data
    
  cybergym-worker:
    image: your-repo/cybergym-worker:latest
    cpu: 4096
    memory: 8192
    environment:
      - DATASET_MODE=subset
      - MODEL_PATH=/models/foundation-sec-8b
```

#### Option 2: EKS with Auto-scaling
```yaml
# kubernetes/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: cybergym-workers
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: worker
        image: cybergym:latest
        resources:
          requests:
            cpu: "2"
            memory: "4Gi"
          limits:
            cpu: "4"
            memory: "8Gi"
```

#### Option 3: Lambda + Step Functions (Lightweight)
- **Use for**: API endpoints and orchestration only
- **Limitations**: Cannot run full Docker vulnerability testing
- **Cost**: ~$50-200/month for API layer only

### 🔐 Security Considerations

**Network Isolation:**
- Deploy in private subnets with NAT Gateway
- Use VPC endpoints for AWS services
- Implement strict security groups

**Data Protection:**
- Encrypt EBS volumes and S3 buckets
- Use IAM roles with least privilege
- Enable CloudTrail for audit logging

**Container Security:**
- Scan images for vulnerabilities before deployment
- Use AWS Secrets Manager for credentials
- Implement runtime security monitoring

### 📝 Deployment Checklist

- [ ] **Infrastructure**: Provision VPC, subnets, security groups
- [ ] **Storage**: Create S3 buckets and EBS volumes
- [ ] **Compute**: Launch ECS cluster or EKS cluster  
- [ ] **Database**: Set up RDS PostgreSQL instance
- [ ] **Monitoring**: Configure CloudWatch dashboards
- [ ] **Security**: Implement IAM roles and policies
- [ ] **CI/CD**: Set up deployment pipeline
- [ ] **Testing**: Deploy subset mode for validation

### 🛠️ Quick AWS Setup Commands

```bash
# Install AWS CLI and configure
aws configure

# Create infrastructure stack
aws cloudformation create-stack \
  --stack-name cybergym-infra \
  --template-body file://aws/infrastructure.yaml \
  --parameters ParameterKey=DatasetMode,ParameterValue=subset

# Deploy application
aws ecs create-service \
  --cluster cybergym \
  --service-name cybergym-api \
  --task-definition cybergym:1
```

### 📞 Support & Scaling

**For Production Deployment:**
- Consider AWS Professional Services for architecture review
- Use AWS Config for compliance monitoring  
- Implement multi-region deployment for high availability
- Set up automated backups and disaster recovery

**Cost Optimization:**
- Use Spot Instances for non-critical workloads
- Implement auto-scaling based on evaluation queue depth
- Use S3 Intelligent Tiering for dataset storage
- Schedule EC2 instances for development environments

---
