#!/bin/bash
# Upload Consolidated PRIMUS-Enhanced Cybersecurity Dataset to AWS S3
# Total: 911,847 samples (924MB + 86MB = 1.01GB)

BUCKET_NAME="your-bucket-name"  # Replace with your S3 bucket name
PREFIX="cybersecurity-datasets/primus-enhanced"

echo "=== Uploading Consolidated PRIMUS-Enhanced Cybersecurity Dataset to AWS S3 ==="
echo "Total records: 911,847 samples"
echo "Training: 840,658 samples (924MB)"
echo "Validation: 71,189 samples (86MB)"
echo ""

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null; then
    echo "ERROR: AWS CLI not found. Install with: pip install awscli"
    exit 1
fi

# Upload training dataset
echo "Uploading training dataset (924MB)..."
aws s3 cp train_FINAL.jsonl s3://$BUCKET_NAME/$PREFIX/train_FINAL.jsonl \
    --storage-class STANDARD \
    --metadata "records=840658,size=924MB,format=conversational,tokens=480max"

# Upload validation dataset  
echo "Uploading validation dataset (86MB)..."
aws s3 cp valid_FINAL.jsonl s3://$BUCKET_NAME/$PREFIX/valid_FINAL.jsonl \
    --storage-class STANDARD \
    --metadata "records=71189,size=86MB,format=conversational,tokens=480max"

# Create a manifest file
echo "Creating dataset manifest..."
cat > dataset_manifest.json << EOF
{
  "dataset_name": "PRIMUS-Enhanced Cybersecurity Training Dataset",
  "creation_date": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "total_samples": 911847,
  "training_samples": 840658,
  "validation_samples": 71189,
  "total_size_gb": 1.01,
  "format": "conversational (User/Assistant)",
  "token_limit": 480,
  "truncation_rate": "0% (zero truncation achieved)",
  "source_datasets": [
    "PRIMUS-Seed (714K samples)",
    "Cybersecurity datasets",
    "Code analysis datasets"
  ],
  "optimization": "Token-aware preprocessing with TinyLlama tokenizer",
  "s3_paths": {
    "training": "s3://$BUCKET_NAME/$PREFIX/train_FINAL.jsonl",
    "validation": "s3://$BUCKET_NAME/$PREFIX/valid_FINAL.jsonl"
  }
}
EOF

# Upload manifest
aws s3 cp dataset_manifest.json s3://$BUCKET_NAME/$PREFIX/dataset_manifest.json

echo ""
echo "✅ Upload complete!"
echo "📍 Training data: s3://$BUCKET_NAME/$PREFIX/train_FINAL.jsonl"
echo "📍 Validation data: s3://$BUCKET_NAME/$PREFIX/valid_FINAL.jsonl" 
echo "📍 Manifest: s3://$BUCKET_NAME/$PREFIX/dataset_manifest.json"
echo ""
echo "🔧 For AWS processing job, reference these S3 paths in your job definition"
