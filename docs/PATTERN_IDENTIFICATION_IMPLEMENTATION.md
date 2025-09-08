#### Requirements:

1. As SecEng I want to see all patterns which Cataphract detected across all time with statuses statistic
2. As SecEng I want to review new patterns as soon as their appeared in Cataphract.
3. As SecEng I want to assign correct title, description, mitigation steps for pattern and Cata should use it for finding generation
4. As SecEng I want to change severity of patterns and Cata should use it as default severity for finding
5. As SecEng I want to disable/enable specific patterns so Cata will not display such findings for SecEng.
6. As SecEng I want do deduplication findings which has the same pattern and affected resources.
7. As SecEng I want to see pattern structure like this

{
"rule_id": "RULE-HARDCODED-PROD-CREDS",
"rule_name": "Hardcoded Production Credentials in Source Code"
"rule_description": "Detects hardcoded credentials, particularly those accessing production systems, embedded directly in source code. This represents a significant security risk as credentials stored in source code can be accessed by unauthorized parties, may be inadvertently committed to version control systems, and violate the principle of secure credential management. Production credentials require especially strict protection and should never be hardcoded.",
"finding_description": "This vulnerability about..,
"finding_mitigation": "Do this stag..,
"severity": "MEDIUM",
"category": "HARDCODED CREDENTIALS",
"status": "ACTIVE|DISABLED|UNDER_REVIEW",
"statistics": {
"usage_count": 157,
"true_positive_rate": 0.93
}
}


#### Diagram:

```
┌─────────────────────────────────┐
│     Cataphract LLM Analysis     │
│             Sourcer             │
└───────────────┬─────────────────┘
                │
                ▼
┌─────────────────────────────────┐
│    Vulnerability Detection      │
└───────────────┬─────────────────┘
                │
                ▼
┌─────────────────────────────────┐
│     Pattern Recognition         │
│                                 │
│  ┌───────────┐    ┌───────────┐ │
│  │LLM Pattern│◄──►│ Patterns  │ │
│  │  Matching │    │ Database  │ │
│  └─────┬─────┘    └───────────┘ │
│        │                        │
│        ▼                        │
│  ┌───────────────────┐          │
│  │ Pattern Found?    │──No──┐   │
│  └─────┬─────────────┘      │   │
│        │                    │   │
│       Yes                   │   │
│        │                    ▼   │
│        │             ┌───────────┐
│        │             │  Create   │
│        │             │   New     |
│        │             │  Pattern  │
│        │             └─────┬─────┘
└────────┼───────────────────┘
         │                    
         ▼                    
┌────────────────────────────────┐
│ Apply Pattern to Finding:      │
│ - Use standardized title       │
│ - Apply pattern description    │
│ - Set pattern-defined severity │
│ - Include mitigation steps     │
└─────────────┬──────────────────┘
              │
              ▼
┌────────────────────────────────┐
│ Finding Filtration:            │
│ - Use pattern status           │
│ Enabled/Disabled               │
└─────────────┬──────────────────┘
              │
              ▼
┌────────────────────────────────┐
│ Standardized Findingst         │
│ - Consistent titles            │
│ - Consistent descriptions      │
│ - Normalized severity ratings  │
│ - Standardized mitigations     │
└────────────────────────────────┘
```



Note:

Code for recognition and matching patterns
[scanner_rule_extractor_text.py](https://quip-amazon.com/-/blob/KPM9AAuyMEk/Ja1wAKiL8hteLXuhkkh0Vg?name=scanner_rule_extractor_text.py&s=q1OqAa2rJyC5)

#### Matching patterns prompt

```
prompt = f"""
You are a security expert specializing in matching security findings to scanner rules.

Your task is to determine if the following security finding matches any of the existing scanner rules.

Security Finding:
Title: {finding.title}
Description: {finding.description}
Category: {finding.component}

Existing Scanner Rules:
{rules_text}

CRITICAL MATCHING REQUIREMENTS:
1. PRIMARY RESOURCE TYPE MATCH: The main affected AWS resource type MUST be identical
   - Example: "Unencrypted EBS Volumes in EC2 AMIs" primarily affects EBS volumes, so it should match EBS rules, not EC2 or S3 rules
   - Example: "S3 Bucket Missing KMS Encryption" primarily affects S3 buckets, so it should match S3 rules, not KMS or EBS rules
  
2. SECURITY ISSUE MATCH: The specific security vulnerability must be the same
   - Similar concepts (like different types of encryption problems) are NOT automatic matches
   - The root cause and remediation approach should be similar
  
3. DETECTION LOGIC: Would the same scanner detect both issues?
   - The technical method of detection should be similar
   - Consider what specific configuration is being checked

MATCHING PROCESS:
1. First, carefully identify the PRIMARY resource type affected in the finding
   - Look for which resource has the vulnerability
   - Look for which resource needs remediation
  
2. For each rule:
   - Identify its primary resource type
   - If different from finding's primary resource, assign 0.0 similarity and mark as non-match
   - Only if primary resources match, evaluate issue similarity
  
3. Assign similarity scores:
   - 0.0: Different primary resource types
   - 0.1-0.3: Same resource type but clearly different security issues
   - 0.4-0.6: Same resource type with related but distinct security issues
   - 0.7-1.0: Same resource type with same or very similar security issue
  
4. For each rule provide:
   - The primary resource type in the finding
   - The primary resource type in the rule
   - Whether they match (yes/no)
   - Your similarity score with justification

Then identify which rule is the best match (if any) and provide its index (1-based) and similarity score.
If no rules match (similarity < 0.85), set the best match index to -1 and the best match score to 0.0.
"""
```


Recognition of new pattern prompt


```
prompt = f"""
        You are a security expert specializing in identifying scanner rules that generate security findings.
       
        Your task is to analyze a security finding and extract the core detection pattern that would be used in a scanner rule, with focus on the affected resource type.
       
        Security Finding:
        Title: {title}
        Description: {description}
        Category: {category}
       
        Guidelines:
        1. Extract the core detection pattern from the description, ignoring context-specific details
        2. Focus on the underlying detection logic that would be common across similar findings
        3. Create a unique rule ID using a prefix 'RULE-' followed by a descriptive identifier
        4. Provide a clear, concise rule name that describes what the rule detects
        5. Include a description of what the rule detects and why it's a security concern
       
        Provide the scanner rule information in a structured format.
        """
```


Sample of detected patterns:



```
{
  "rule_id": "RULE-MISSING-OBJ-VERSIONING",
  "rule_name": "Missing Object Versioning Configuration",
  "detection_pattern": "resource.VersioningConfiguration.Status != 'Enabled'",
  "description": "Detects resources that do not have versioning enabled. Versioning is a critical security control that protects against accidental or malicious deletions and modifications of data by maintaining multiple versions of objects. Without versioning, there is no way to recover from unintended changes or deletions, potentially leading to data loss and compliance violations.",
  "severity": "HIGH",
  "category": "LOGGING DISABLED"
},
{
  "rule_id": "RULE-VPC-CROSS-REGION-UNDOC",
  "rule_name": "Undocumented Cross-Region VPC Peering Connections",
  "detection_pattern": "resource.type == 'AWS::EC2::VPCPeering' && \nresource.crossRegion == true && \n(resource.tags == null || resource.tags.isEmpty()) && \nresource.status == 'active'",
  "description": "Detects active VPC peering connections that span across different AWS regions and lack proper documentation through tags. Cross-region VPC peering connections without proper documentation and tagging pose management and security risks by making it difficult to track ownership, purpose, and maintain proper network segmentation controls.",
  "severity": "MEDIUM",
  "category": "MISSING NETWORK SEGMENTATION"
},
{
  "rule_id": "RULE-RDS-INSUFFICIENT-BACKUP-RETENTION",
  "rule_name": "RDS Aurora Insufficient Backup Retention Period",
  "detection_pattern": "RDS Aurora PostgreSQL cluster.backup_retention_period < 7",
  "description": "Detects RDS Aurora PostgreSQL clusters configured with insufficient backup retention periods (less than 7 days). Short retention periods increase the risk of permanent data loss and compromise disaster recovery capabilities. This violates security best practices which recommend maintaining backups for at least 7 days to ensure adequate protection against data corruption, accidental deletion, and to meet potential compliance requirements.",
  "severity": "LOW",
  "category": "BACKUPS DISABLED"
},
{
  "rule_id": "RULE-S3-BUCKET-KEY-DISABLED",
  "rule_name": "S3 Bucket Key Disabled Detection",
  "detection_pattern": "resource.type == 'AWS::S3::Bucket' AND resource.BucketKeyEnabled == false",
  "description": "Detects S3 buckets that have the BucketKeyEnabled setting disabled. S3 Bucket Keys are a security best practice that improve encryption performance and reduce costs by decreasing AWS KMS API calls. Disabled bucket keys may lead to increased operational costs and potentially impact the performance of encrypted operations.",
  "severity": "LOW",
  "category": "INSECURE CREDENTIAL MANAGEMENT"
},
{
  "rule_id": "RULE-AWS-PUBLIC-SUBNET-AUTO-IP",
  "rule_name": "AWS Subnet Auto-Assign Public IP Enabled",
  "detection_pattern": "resource.type == \"AWS::EC2::Subnet\" && resource.MapPublicIpOnLaunch == true",
  "description": "Detects AWS subnets that have the MapPublicIpOnLaunch feature enabled, which automatically assigns public IP addresses to EC2 instances launched in these subnets. This configuration increases security risk by potentially exposing instances directly to the internet and violates network security best practices of maintaining private networks by default.",
  "severity": "HIGH",
  "category": "NETWORK_SECURITY"
},
{
  "rule_id": "RULE-HARDCODED-PROD-CREDS",
  "rule_name": "Hardcoded Production Credentials in Source Code",
  "detection_pattern": "(?i)(password|credential|secret|key).*=.*['\\\"][^'\\\"]+['\\\"].*(?:prod|production)",
  "description": "Detects hardcoded credentials, particularly those accessing production systems, embedded directly in source code. This represents a significant security risk as credentials stored in source code can be accessed by unauthorized parties, may be inadvertently committed to version control systems, and violate the principle of secure credential management. Production credentials require especially strict protection and should never be hardcoded.",
  "severity": "CRITICAL",
  "category": "HARDCODED CREDENTIALS"
},
{
  "rule_id": "RULE-CLOUDTRAIL-KMS-ENCRYPTION",
  "rule_name": "CloudTrail Logs Missing KMS Encryption",
  "detection_pattern": "resource.type == \"AWS::CloudTrail::Trail\" && resource.KMSKeyId == null",
  "description": "Detects CloudTrail trails that are not configured to use AWS KMS Customer Master Keys (CMKs) for encryption. CloudTrail logs contain sensitive audit data that should be protected with additional encryption controls. Lack of KMS encryption reduces the security of audit logs and may impact compliance requirements. KMS encryption provides enhanced security through access control, key rotation, and audit capabilities.",
  "severity": "HIGH",
  "category": "ENCRYPTION MISSING IN AWS RESOURCES"
},
{
  "rule_id": "RULE-S3-KMS-ENCRYPTION-MISSING",
  "rule_name": "S3 Bucket Missing KMS Encryption",
  "detection_pattern": "resource.type == 'AWS::S3::Bucket' && encryption.type != 'AWS-KMS'",
  "description": "Detects S3 buckets that are not configured to use AWS KMS encryption for data at rest. AWS KMS encryption provides an additional layer of security and is essential for meeting security and compliance requirements for data protection. Lack of KMS encryption may expose sensitive data to unauthorized access and non-compliance risks.",
  "severity": "HIGH",
  "category": "ENCRYPTION MISSING IN AWS RESOURCES"
},
{
  "rule_id": "RULE-UNRESTRICTED-CACHE-POLICY-UPDATES",
  "rule_name": "Unrestricted Cache Policy Updates",
  "detection_pattern": "CachePolicy.permissions NOT has_restrictions AND CachePolicy.modifications NOT has_iam_controls",
  "description": "Detects cache policies that lack proper IAM restrictions for modifications. This security issue allows unauthorized users to modify caching behaviors, potentially leading to service disruptions and security misconfigurations. The absence of proper access controls on cache policy modifications represents a significant security risk as it could allow unauthorized changes to critical caching configurations and bypass security controls.",
  "severity": "HIGH",
  "category": "OVERLY PERMISSIVE IAM POLICIES"
},
{
  "rule_id": "RULE-AWS-SNS-MISSING-SSE",
  "rule_name": "AWS SNS Topics Missing Server-Side Encryption",
  "detection_pattern": "resource.type == \"AWS::SNS::Topic\" && \nresource.encryption.serverSideEncryption == null || \nresource.encryption.serverSideEncryption.enabled != true",
  "description": "Detects AWS SNS topics that do not have server-side encryption (SSE) enabled. SNS topics without encryption enabled can expose sensitive message data in transit and at rest. Server-side encryption using AWS KMS provides an additional layer of data protection by helping secure message contents. This is particularly critical for topics handling sensitive or regulated data.",
  "severity": "HIGH",
  "category": "ENCRYPTION MISSING IN AWS RESOURCES"
},