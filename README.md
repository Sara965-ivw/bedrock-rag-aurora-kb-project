# Bedrock RAG + Aurora Serverless Knowledge Base  
### Using Amazon Bedrock, Aurora PostgreSQL Serverless v2, S3 & Terraform  
![Terraform](https://img.shields.io/badge/Terraform-1.9+-7B42BC?logo=terraform)
![AWS](https://img.shields.io/badge/AWS-Bedrock-orange?logo=amazonaws)
![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)

---

## Overview

This project builds a **complete Retrieval-Augmented Generation (RAG) system** on AWS using:

- **Terraform** to deploy infrastructure  
- **Amazon Aurora Serverless PostgreSQL** (with pgvector + text search)  
- **Amazon S3** for document storage  
- **AWS Bedrock Knowledge Base** to connect S3 + Aurora  
- **Python** to query the Knowledge Base programmatically  

The final system allows you to ingest documents into S3, index them in Aurora, and query them using a Bedrock LLM — fully automated.

---

# Project Structure
project-root/
│── stack1/
│   ├── main.tf
│   ├── outputs.tf
│   ├── variables.tf
│── stack2/
│   ├── main.tf
│   ├── outputs.tf
│   ├── variables.tf
│── modules/
│   ├── aurora_serverless/
│   │   ├── main.tf
│   │   ├── variables.tf
│   │   ├── outputs.tf
│   ├── bedrock_kb/
│   │   ├── main.tf
│   │   ├── variables.tf
│   │   ├── outputs.tf
│── scripts/
│   ├── aurora_sql.sql
│   ├── upload_to_s3.py
│── Screenshots/
│── README.md
│── temperature_top_p_explanation.pdf---

#  **Architecture Diagram**

### PNG Version  
+––––––––––+          +————————+
|        Client      |  —>    |   Python (Query API)   |
+––––––––––+          +————————+
|                               |
v                               v
+––––––––––+         +————————+
| Amazon Bedrock KB |  <––  |    S3 Document Store   |
+––––––––––+         +————————+
|                               |
v                               v
+——————————————————+
|      Aurora PostgreSQL Serverless (pgvector)         |
+——————————————————+
|
v
+––––––––––+
|       VPC          |
+––––––––––+

### Mermaid VersionA[Client / Python Script] --> B[Amazon Bedrock Knowledge Base]
B --> C[S3 - Document Store]
B --> D[Aurora Serverless PostgreSQL - Vector DB]
C --> B
D --> B
B --> A
