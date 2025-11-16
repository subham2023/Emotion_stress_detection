# Docker & CI/CD Deployment Guide

Complete guide for deploying the Emotion Stress Detection System using Docker containers and CI/CD pipelines.

## Table of Contents

1. [Pre-Deployment Checklist](#pre-deployment-checklist)
2. [Docker Containerization](#docker-containerization)
3. [Environment Configuration](#environment-configuration)
4. [Local Development Setup](#local-development-setup)
5. [Docker Swarm Deployment](#docker-swarm-deployment)
6. [CI/CD Pipeline](#cicd-pipeline)
7. [Health Monitoring](#health-monitoring)
8. [Model Optimization](#model-optimization)
9. [Cloud Deployment](#cloud-deployment)
10. [Performance Optimization](#performance-optimization)
11. [Monitoring & Logging](#monitoring--logging)
12. [Scaling Strategies](#scaling-strategies)
13. [Security Hardening](#security-hardening)
14. [Troubleshooting](#troubleshooting)

---

## Pre-Deployment Checklist

Before deploying to production, ensure:

- [ ] All tests pass (`pytest src/` and `pnpm test`)
- [ ] Code coverage > 80% (`pytest --cov`)
- [ ] No TypeScript errors (`pnpm build`)
- [ ] Environment variables configured
- [ ] Database migrations applied
- [ ] API endpoints tested
- [ ] Security audit completed
- [ ] Performance benchmarks acceptable
- [ ] Backup and recovery plan documented
- [ ] Monitoring dashboards set up

---

## Model Optimization

### 1. Model Quantization

Reduce model size and improve inference speed with quantization:

```python
import tensorflow as tf
from tensorflow.lite.python import lite_constants

def quantize_model(model_path, output_path):
    """Convert model to quantized TFLite format."""
    
    # Load model
    converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
    
    # Enable quantization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8
    ]
    
    # Convert
    tflite_model = converter.convert()
    
    # Save
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"Quantized model saved: {output_path}")
    
    # Compare sizes
    original_size = os.path.getsize(model_path) / (1024 * 1024)
    quantized_size = os.path.getsize(output_path) / (1024 * 1024)
    compression = (1 - quantized_size / original_size) * 100
    
    print(f"Original: {original_size:.2f}MB")
    print(f"Quantized: {quantized_size:.2f}MB")
    print(f"Compression: {compression:.1f}%")
```

**Benefits**:
- 4x smaller model size
- 2-3x faster inference
- Minimal accuracy loss (< 1%)

### 2. Model Pruning

Remove unnecessary weights:

```python
import tensorflow_model_optimization as tfmot

def prune_model(model):
    """Apply magnitude-based pruning to model."""
    
    pruning_schedule = tfmot.sparsity.keras.PolynomialDecay(
        initial_sparsity=0.0,
        final_sparsity=0.5,
        begin_step=0,
        end_step=1000
    )
    
    model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(
        model,
        pruning_schedule=pruning_schedule
    )
    
    model_for_pruning.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model_for_pruning
```

### 3. ONNX Export

Export model for cross-platform deployment:

```python
import onnx
import tf2onnx

def export_to_onnx(model_path, output_path):
    """Export TensorFlow model to ONNX format."""
    
    import tensorflow as tf
    
    # Load model
    model = tf.keras.models.load_model(model_path)
    
    # Convert to ONNX
    spec = (tf.TensorSpec((None, 48, 48, 1), tf.float32, name="input"),)
    output_path = tf.io.gfile.join(output_path, "model.onnx")
    
    model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec)
    onnx.save(model_proto, output_path)
    
    print(f"ONNX model saved: {output_path}")
```

---

## Docker Containerization

The application is containerized using a multi-stage Docker build for optimized production deployment.

### Dockerfile

The application uses a multi-stage build:

```dockerfile
# Stage 1: Frontend Build
FROM node:20-alpine AS frontend-builder
# Builds React frontend and TypeScript backend

# Stage 2: Backend Build
FROM node:20-alpine AS backend-builder
# Builds TypeScript backend

# Stage 3: Runtime Container
FROM python:3.11-slim
# Combines all components with ML dependencies
```

**Key Features:**
- Multi-stage build for minimal final image size
- Includes Python ML dependencies (TensorFlow, OpenCV)
- Health check endpoint at `/api/health`
- Production-optimized with pnpm for fast dependency installation

### Docker Compose Files

#### Production (`docker-compose.yml`)
- Full production stack with monitoring
- Includes Nginx, Prometheus, Grafana
- Persistent volumes for data
- Health checks and restart policies
- Resource limits and reservations

#### Development (`docker-compose.dev.yml`)
- Development configuration with hot reloading
- Includes Adminer and Redis Commander
- Debug ports enabled
- Development-friendly logging

---

## Environment Configuration

### Setup Instructions

1. **Copy Environment Template**
```bash
cp config/.env.example .env
```

2. **Configure Production Environment**
```bash
cp config/production.env .env
```

3. **Configure Development Environment**
```bash
cp config/development.env .env.dev
```

### Required Environment Variables

**Application:**
- `NODE_ENV` - Environment (production/development)
- `PORT` - Application port (default: 3000)
- `DOMAIN` - Application domain for SSL certificates

**Database:**
- `MYSQL_ROOT_PASSWORD` - MySQL root password
- `MYSQL_DATABASE` - Database name
- `MYSQL_USER` - Application database user
- `MYSQL_PASSWORD` - Application user password

**Cache:**
- `REDIS_PASSWORD` - Redis authentication password

**Storage:**
- `S3_BUCKET` - AWS S3 bucket name
- `AWS_ACCESS_KEY_ID` - AWS access key
- `AWS_SECRET_ACCESS_KEY` - AWS secret key
- `AWS_REGION` - AWS region

**Authentication:**
- `MANUS_CLIENT_ID` - Manus OAuth client ID
- `MANUS_CLIENT_SECRET` - Manus OAuth client secret

**Monitoring:**
- `GRAFANA_PASSWORD` - Grafana admin password

---

## Local Development Setup

### Prerequisites
- Docker and Docker Compose
- Node.js 20+ (for local development)
- Python 3.11 (for ML development)

### Quick Start

1. **Start Development Environment**
```bash
docker-compose -f docker-compose.dev.yml up --build
```

2. **Access Development Services**
- Application: http://localhost:3000
- Adminer (DB Admin): http://localhost:8080
- Redis Commander: http://localhost:8081
- Traefik Dashboard: http://localhost:8080

3. **Development Features**
- Hot reloading for frontend and backend
- Debug port 9229 for Node.js debugging
- Live database access via Adminer
- Redis management via Redis Commander

### Local Development without Docker

1. **Install Dependencies**
```bash
# Node.js dependencies
npm install -g pnpm
pnpm install

# Python dependencies
pip install -r requirements.txt
```

2. **Start Application**
```bash
pnpm dev
```

---

## Docker Swarm Deployment

### Initialize Docker Swarm

1. **Initialize Swarm Cluster**
```bash
./deploy/swarm-init.sh
```

2. **Deploy Production Stack**
```bash
./deploy/deploy-stack.sh
```

### Deployment Scripts

**`deploy/swarm-init.sh`**
- Initializes Docker Swarm cluster
- Creates overlay networks
- Sets up data directories
- Configures firewall rules

**`deploy/deploy-stack.sh`**
- Deploys application stack
- Supports production and development
- Blue-green deployment support
- Health checks and monitoring

**`deploy/update-stack.sh`**
- Performs rolling updates
- Service management
- Rollback capabilities

**`deploy/remove-stack.sh`**
- Graceful stack removal
- Data backup before removal
- Cleanup operations

### Deployment Commands

```bash
# Deploy production
./deploy/deploy-stack.sh -t production

# Deploy development
./deploy/deploy-stack.sh -t development

# Update service with new image
./deploy/update-stack.sh -i myrepo/app:v1.2.3

# Perform rollback
./deploy/update-stack.sh --rollback

# Remove stack
./deploy/remove-stack.sh

# Show stack status
./deploy/deploy-stack.sh -s
```

### Service Access Points

After deployment:

- **Application**: https://your-domain.com
- **Grafana**: https://grafana.your-domain.com
- **Prometheus**: https://prometheus.your-domain.com
- **Traefik Dashboard**: http://localhost:8080

---

## CI/CD Pipeline

### GitHub Actions Workflows

**`.github/workflows/ci-cd.yml`**
- Automated testing (Node.js and Python)
- Docker image building and security scanning
- Multi-architecture image support (AMD64, ARM64)
- Automated deployment to staging/production
- Environment-based deployments

**`.github/workflows/security.yml`**
- Container vulnerability scanning (Trivy)
- Dependency security scanning (npm audit, safety)
- Code security analysis (Semgrep)
- Scheduled security scans
- Automated issue creation for critical findings

**`.github/workflows/release.yml`**
- Automated releases based on tags
- Semantic versioning support
- Blue-green deployments
- Rollback on failure
- Release artifact generation

### Pipeline Triggers

**Automatic Triggers:**
- Push to `main` branch → Production deployment
- Push to `develop` branch → Staging deployment
- Pull requests → Testing and validation
- Schedule → Security scans

**Manual Triggers:**
- Workflow dispatch for testing
- Manual release creation
- On-demand security scans

### Container Registry

Images are pushed to GitHub Container Registry (ghcr.io):

```bash
# Pull latest image
docker pull ghcr.io/username/emotion-stress-detection:latest

# Pull specific version
docker pull ghcr.io/username/emotion-stress-detection:v1.2.3
```

---

## Health Monitoring

### Health Check Endpoints

**`/api/health`** - Comprehensive health check
```json
{
  "status": "healthy",
  "timestamp": "2024-11-16T12:00:00Z",
  "uptime": 3600,
  "version": "1.0.0",
  "checks": {
    "database": {"status": "pass"},
    "redis": {"status": "pass"},
    "storage": {"status": "pass"},
    "ml": {"status": "pass"},
    "websocket": {"status": "pass"},
    "memory": {"status": "pass"},
    "disk": {"status": "pass"}
  },
  "metrics": {
    "responseTime": 45,
    "memoryUsage": {...},
    "cpuUsage": {...}
  }
}
```

**Individual Component Checks:**
- `/api/health/database` - Database connectivity
- `/api/health/redis` - Redis connectivity
- `/api/health/storage` - File storage access
- `/api/health/ml` - ML model status
- `/api/health/websocket` - WebSocket status
- `/api/health/liveness` - Liveness probe
- `/api/health/readiness` - Readiness probe

### Monitoring Stack

**Prometheus**
- Metrics collection at `/metrics`
- Application performance metrics
- Infrastructure metrics
- Custom business metrics

**Grafana**
- Pre-configured dashboards
- Real-time monitoring
- Alert management
- Historical analysis

### Health Check Configuration

**Docker Health Check:**
```bash
curl -f http://localhost:3000/api/health || exit 1
```

**Kubernetes Probes:**
```yaml
livenessProbe:
  httpGet:
    path: /api/health/liveness
    port: 3000
  initialDelaySeconds: 30
  periodSeconds: 10

readinessProbe:
  httpGet:
    path: /api/health/readiness
    port: 3000
  initialDelaySeconds: 5
  periodSeconds: 5
```

---

## Cloud Deployment

### AWS Deployment

#### 1. Prepare for AWS

```bash
# Install AWS CLI
pip install awscli

# Configure credentials
aws configure

# Create ECR repository
aws ecr create-repository --repository-name emotion-detector
```

#### 2. Deploy to ECS

```bash
# Build and push Docker image
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com

docker build -t emotion-detector:latest .
docker tag emotion-detector:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/emotion-detector:latest
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/emotion-detector:latest
```

#### 3. CloudFormation Template

```yaml
AWSTemplateFormatVersion: '2010-09-09'
Description: 'Emotion Detector Deployment'

Resources:
  ECSCluster:
    Type: AWS::ECS::Cluster
    Properties:
      ClusterName: emotion-detector-cluster

  TaskDefinition:
    Type: AWS::ECS::TaskDefinition
    Properties:
      Family: emotion-detector-task
      NetworkMode: awsvpc
      RequiresCompatibilities:
        - FARGATE
      Cpu: '2048'
      Memory: '4096'
      ContainerDefinitions:
        - Name: app
          Image: !Sub '${AWS::AccountId}.dkr.ecr.${AWS::Region}.amazonaws.com/emotion-detector:latest'
          PortMappings:
            - ContainerPort: 3000
          Environment:
            - Name: NODE_ENV
              Value: production
          LogConfiguration:
            LogDriver: awslogs
            Options:
              awslogs-group: /ecs/emotion-detector
              awslogs-region: !Ref AWS::Region
              awslogs-stream-prefix: ecs

  Service:
    Type: AWS::ECS::Service
    Properties:
      ServiceName: emotion-detector-service
      Cluster: !Ref ECSCluster
      TaskDefinition: !Ref TaskDefinition
      DesiredCount: 2
      LaunchType: FARGATE
      NetworkConfiguration:
        AwsvpcConfiguration:
          Subnets:
            - subnet-xxxxx
          SecurityGroups:
            - sg-xxxxx
      LoadBalancers:
        - ContainerName: app
          ContainerPort: 3000
          TargetGroupArn: !Ref TargetGroup

  TargetGroup:
    Type: AWS::ElasticLoadBalancingV2::TargetGroup
    Properties:
      Port: 3000
      Protocol: HTTP
      VpcId: vpc-xxxxx
      TargetType: ip
```

### Google Cloud Deployment

```bash
# Build and push to Container Registry
gcloud builds submit --tag gcr.io/PROJECT_ID/emotion-detector

# Deploy to Cloud Run
gcloud run deploy emotion-detector \
  --image gcr.io/PROJECT_ID/emotion-detector \
  --platform managed \
  --region us-central1 \
  --memory 4Gi \
  --cpu 2 \
  --set-env-vars DATABASE_URL=<connection-string>
```

---

## Performance Optimization

### 1. Caching Strategy

```python
from functools import lru_cache
import redis

# In-memory cache
@lru_cache(maxsize=128)
def get_model(model_path):
    """Cache loaded model in memory."""
    return keras.models.load_model(model_path)

# Redis cache for API responses
redis_client = redis.Redis(host='localhost', port=6379)

def cache_prediction(key, value, ttl=3600):
    """Cache prediction results."""
    redis_client.setex(key, ttl, json.dumps(value))

def get_cached_prediction(key):
    """Retrieve cached prediction."""
    data = redis_client.get(key)
    return json.loads(data) if data else None
```

### 2. Database Optimization

```sql
-- Create indexes for frequently queried columns
CREATE INDEX idx_user_id ON detection_results(userId);
CREATE INDEX idx_session_id ON detection_results(sessionId);
CREATE INDEX idx_timestamp ON detection_results(timestamp);
CREATE INDEX idx_emotion ON detection_results(dominantEmotion);

-- Partition large tables by date
ALTER TABLE detection_results 
PARTITION BY RANGE (YEAR(timestamp)) (
  PARTITION p2023 VALUES LESS THAN (2024),
  PARTITION p2024 VALUES LESS THAN (2025),
  PARTITION p2025 VALUES LESS THAN (2026)
);
```

### 3. API Response Optimization

```typescript
// Implement pagination
router.result.list.useQuery({
  sessionId: 123,
  limit: 50,
  offset: 0
});

// Use field selection
router.result.recent.useQuery({
  limit: 10,
  fields: ['dominantEmotion', 'stressScore', 'timestamp']
});

// Implement compression
app.use(compression());
```

---

## Monitoring & Logging

### 1. Application Monitoring

```python
from prometheus_client import Counter, Histogram, start_http_server

# Metrics
prediction_counter = Counter(
    'predictions_total',
    'Total predictions',
    ['emotion', 'stress_level']
)

inference_time = Histogram(
    'inference_seconds',
    'Inference time in seconds'
)

@inference_time.time()
def predict(image):
    """Predict with timing."""
    return model.predict(image)
```

### 2. Logging Setup

```python
import logging
from pythonjsonlogger import jsonlogger

# JSON logging for structured logs
logHandler = logging.StreamHandler()
formatter = jsonlogger.JsonFormatter()
logHandler.setFormatter(formatter)

logger = logging.getLogger()
logger.addHandler(logHandler)
logger.setLevel(logging.INFO)

# Log important events
logger.info('prediction_made', extra={
    'emotion': 'happy',
    'confidence': 0.92,
    'inference_time': 0.045
})
```

### 3. Error Tracking

```python
import sentry_sdk

sentry_sdk.init(
    dsn="https://key@sentry.io/project",
    traces_sample_rate=0.1,
    environment="production"
)

try:
    result = predictor.predict_single_image(image_path)
except Exception as e:
    sentry_sdk.capture_exception(e)
    logger.error(f"Prediction failed: {str(e)}")
```

---

## Scaling Strategies

### 1. Horizontal Scaling

```yaml
# Kubernetes deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: emotion-detector
spec:
  replicas: 3
  selector:
    matchLabels:
      app: emotion-detector
  template:
    metadata:
      labels:
        app: emotion-detector
    spec:
      containers:
      - name: app
        image: emotion-detector:latest
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
        livenessProbe:
          httpGet:
            path: /health
            port: 3000
          initialDelaySeconds: 30
          periodSeconds: 10

---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: emotion-detector-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: emotion-detector
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

### 2. Load Balancing

```nginx
upstream emotion_detector {
    server app1:3000;
    server app2:3000;
    server app3:3000;
}

server {
    listen 80;
    server_name api.emotion-detector.ai;

    location / {
        proxy_pass http://emotion_detector;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

---

## Security Hardening

### 1. Environment Security

```bash
# Use environment variables for secrets
export DATABASE_URL="mysql://user:password@host/db"
export JWT_SECRET="$(openssl rand -base64 32)"
export API_KEY="$(openssl rand -base64 32)"

# Never commit secrets
echo ".env.local" >> .gitignore
echo ".env.production" >> .gitignore
```

### 2. API Security

```typescript
// Rate limiting
import rateLimit from 'express-rate-limit';

const limiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 100 // limit each IP to 100 requests per windowMs
});

app.use('/api/', limiter);

// CORS configuration
import cors from 'cors';

app.use(cors({
  origin: process.env.ALLOWED_ORIGINS?.split(','),
  credentials: true
}));

// Helmet for security headers
import helmet from 'helmet';
app.use(helmet());
```

### 3. Database Security

```sql
-- Create restricted user for application
CREATE USER 'app_user'@'localhost' IDENTIFIED BY 'strong_password';
GRANT SELECT, INSERT, UPDATE ON emotion_detector.* TO 'app_user'@'localhost';

-- Enable SSL
[mysqld]
ssl-ca=/etc/mysql/certs/ca.pem
ssl-cert=/etc/mysql/certs/server-cert.pem
ssl-key=/etc/mysql/certs/server-key.pem
```

---

## Rollback Strategy

### Blue-Green Deployment

```bash
#!/bin/bash

# Deploy to green environment
docker build -t emotion-detector:v2 .
docker run -d --name emotion-detector-green emotion-detector:v2

# Test green environment
curl http://localhost:3001/health

# Switch traffic to green
aws elbv2 modify-target-group-attribute \
  --target-group-arn arn:aws:elasticloadbalancing:... \
  --attributes Key=deregistration_delay.timeout_seconds,Value=30

# Remove blue environment
docker stop emotion-detector-blue
docker rm emotion-detector-blue
```

---

## Monitoring Dashboard

Key metrics to monitor:

| Metric | Target | Alert Threshold |
|--------|--------|-----------------|
| API Response Time | < 200ms | > 500ms |
| Error Rate | < 0.1% | > 1% |
| Model Accuracy | > 85% | < 80% |
| Inference Time | < 100ms | > 200ms |
| Database Latency | < 50ms | > 100ms |
| Memory Usage | < 70% | > 85% |
| CPU Usage | < 60% | > 80% |
| Uptime | > 99.9% | < 99% |

---

## Disaster Recovery

### Backup Strategy

```bash
# Daily database backup
0 2 * * * mysqldump -u root -p emotion_detector > /backups/db_$(date +\%Y\%m\%d).sql

# S3 backup
0 3 * * * aws s3 sync /backups s3://emotion-detector-backups/

# Model backup
0 4 * * * tar -czf models_$(date +\%Y\%m\%d).tar.gz ./models && aws s3 cp models_$(date +\%Y\%m\%d).tar.gz s3://emotion-detector-backups/
```

### Recovery Procedure

1. Restore database from latest backup
2. Restore models from S3
3. Verify data integrity
4. Run smoke tests
5. Gradual traffic migration

---

## Post-Deployment Checklist

- [ ] Monitor error rates and latency
- [ ] Verify all endpoints responding
- [ ] Check database replication
- [ ] Confirm backups running
- [ ] Test failover procedures
- [ ] Review logs for anomalies
- [ ] Update documentation
- [ ] Notify stakeholders

---

**Last Updated**: November 2024
**Version**: 1.0.0
