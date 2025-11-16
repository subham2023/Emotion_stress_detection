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

## Docker Deployment

### Dockerfile

```dockerfile
# Multi-stage build for optimized image

# Stage 1: Python ML backend
FROM python:3.11-slim as ml-builder

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy ML code
COPY src/ ./src/
COPY models/ ./models/

# Stage 2: Node.js frontend
FROM node:22-alpine as frontend-builder

WORKDIR /app

COPY package.json pnpm-lock.yaml ./
RUN npm install -g pnpm && pnpm install --frozen-lockfile

COPY client/ ./client/
COPY server/ ./server/
COPY drizzle/ ./drizzle/

RUN pnpm build

# Stage 3: Production image
FROM node:22-alpine

WORKDIR /app

# Install Python runtime
RUN apk add --no-cache python3 py3-pip

# Copy from builders
COPY --from=ml-builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=ml-builder /app/src ./src
COPY --from=ml-builder /app/models ./models
COPY --from=frontend-builder /app/dist ./dist
COPY --from=frontend-builder /app/node_modules ./node_modules

# Copy package files
COPY package.json pnpm-lock.yaml ./

# Expose ports
EXPOSE 3000 5000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=40s --retries=3 \
    CMD node -e "require('http').get('http://localhost:3000/health', (r) => {if (r.statusCode !== 200) throw new Error(r.statusCode)})"

# Start application
CMD ["node", "dist/index.js"]
```

### Docker Compose

```yaml
version: '3.8'

services:
  app:
    build: .
    ports:
      - "3000:3000"
    environment:
      - DATABASE_URL=mysql://user:password@db:3306/emotion_detector
      - NODE_ENV=production
      - JWT_SECRET=${JWT_SECRET}
    depends_on:
      - db
    restart: unless-stopped

  db:
    image: mysql:8.0
    environment:
      - MYSQL_DATABASE=emotion_detector
      - MYSQL_USER=user
      - MYSQL_PASSWORD=password
      - MYSQL_ROOT_PASSWORD=root_password
    volumes:
      - db_data:/var/lib/mysql
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    restart: unless-stopped

volumes:
  db_data:
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
