# **AWS CLI: Walkthrough**

Thi

---
- ## **1. Introduction**
  
  AWS CLI is a powerful tool to interact with AWS services using commands in your terminal. It supports all AWS services and helps automate your workflows.
  
  ---
- ## **2. Installation and Configuration**
- ### **Install AWS CLI**
- ****Windows/macOS/Linux****: [AWS CLI Installation Guide](https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html)
  
  ```bash
  
  *# Windows (Using MSI Installer)*
  
  *# Download and run the installer from:*
  
  *# https://awscli.amazonaws.com/AWSCLIV2.msi*
  
  *# Or use Chocolatey*
  
  choco install awscli
  
  *# macOS (Homebrew)*
  
  brew install awscli
  
  *# Ubuntu*
  
  curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
  
  unzip awscliv2.zip
  
  sudo ./aws/install
  
  ```
- ### **Configure AWS CLI**
  
  ```bash
  
  aws configure
  
  ```
  
  Input:
- Access Key ID
- Secret Access Key
- Default region (e.g., us-east-1)
- Default output format (json, text, table)
  
  ---
- ## **3. Basic CLI Usage**
- ### **Syntax**
  
  ```bash
  
  aws <service> <operation> [parameters]
  
  ```
- ### **Examples**
  
  ```bash
  
  aws s3 ls                       *# List all buckets*
  
  aws ec2 describe-instances     *# Describe all EC2 instances*
  
  ```
- ### **Output Formats**
  
  ```bash
  
  --output json|text|table
  
  ```
  
  ---
- ## **4. Common AWS Services with CLI**
- ### **S3 (Simple Storage Service)**
  
  ```bash
  
  aws s3 mb s3://my-bucket-name            *# Create bucket*
  
  aws s3 cp file.txt s3://my-bucket-name   *# Upload file*
  
  aws s3 ls s3://my-bucket-name            *# List objects*
  
  aws s3 rm s3://my-bucket-name/file.txt   *# Delete file*
  
  ```
- ### **EC2 (Elastic Compute Cloud)**
  
  ```bash
  
  aws ec2 describe-instances
  
  aws ec2 start-instances --instance-ids i-1234567890abcdef0
  
  aws ec2 stop-instances --instance-ids i-1234567890abcdef0
  
  ```
- ### **IAM (Identity and Access Management)**
  
  ```bash
  
  aws iam list-users
  
  aws iam create-user --user-name new-user
  
  aws iam attach-user-policy --user-name new-user --policy-arn arn:aws:iam::aws:policy/ReadOnlyAccess
  
  ```
- ### **Lambda**
  
  ```bash
  
  aws lambda list-functions
  
  aws lambda invoke --function-name my-function output.json
  
  ```
- ### **CloudFormation**
  
  ```bash
  
  aws cloudformation deploy   --template-file template.yaml   --stack-name mystack   --capabilities CAPABILITY_IAM
  
  ```
  
  ---
- ## **5. Advanced CLI Usage**
- ### **Profiles**
  
  ```bash
  
  aws configure --profile dev
  
  aws s3 ls --profile dev
  
  ```
- ### **Assume Role**
  
  ```bash
  
  aws sts assume-role --role-arn arn:aws:iam::123456789012:role/demo --role-session-name demoSession
  
  ```
- ### **Waiters**
  
  ```bash
  
  aws ec2 wait instance-running --instance-ids i-1234567890abcdef0
  
  ```
- ### **Filters**
  
  ```bash
  
  aws ec2 describe-instances   --filters "Name=instance-state-name,Values=running"
  
  ```
  
  ---
- ## **6. Automation and Scripting**
- ### **Bash Script Example**
  
  ```bash
  
  *#!**/bin/bash*
  
  for bucket in $(aws s3api list-buckets --query "Buckets[].Name" --output text)
  
  do
  
  echo "Scanning bucket: $bucket"
  
  aws s3 ls s3://$bucket
  
  echo "----"
  
  done
  
  ```
- ### **Scheduled Automation**
- Use `cron` jobs (Linux/macOS)
- Use AWS Systems Manager Automation
  
  ---
- ## **7. Pagination and Querying**
- ### **Pagination**
  
  Most commands paginate by default.
  
  ```bash
  
  aws ec2 describe-instances --max-items 10 --starting-token <token>
  
  ```
- ### **JMESPath Querying**
  
  ```bash
  
  aws ec2 describe-instances --query "Reservations[*].Instances[*].InstanceId" --output text
  
  ```
  
  ---
- ## **8. Security and Best Practices**
- ****Never expose your credentials**** (use IAM roles or environment variables)
- Use ****least privilege principle**** for IAM
- Enable ****MFA**** for users
- Rotate keys regularly
- Log CLI activity with ****CloudTrail****
  
  ---
- ## **9. Useful Tips and Tools**
- `aws help` or `aws <service> help`
- Use `aws configure list`
- Install `jq` for JSON parsing
- Use AWS CLI v2 for latest features
- Explore AWS Shell or AWS CloudShell
  
  ---
- ## **10. Using jq for JSON Processing**
  
  `jq` is a lightweight command-line JSON processor that works perfectly with AWS CLI's JSON output.
- ### **Install jq**
  
  ```bash
  
  *# Windows*
  
  choco install jq
  
  *# macOS*
  
  brew install jq
  
  *# Ubuntu*
  
  ~ sudo apt-get install jq
  
  ```
- ### **Basic Usage**
  
  ```bash
  
  aws ec2 describe-instances --output json | jq '.'
  
  ```
- ### **Extract Specific Fields**
  
  ```bash
  
  aws ec2 describe-instances --output json | jq '.Reservations[].Instances[].InstanceId'
  
  ```
- ### **Pretty Print and Filter**
  
  ```bash
  
  aws s3api list-buckets --output json | jq '.Buckets[] | select(.Name | contains("log"))'
  
  ```
- ### **Combine with JMESPath for power usage**
  
  ```bash
  
  aws ec2 describe-instances --query "Reservations[*].Instances[*].{ID:InstanceId,Type:InstanceType}" --output json | jq '.'
  
  ```