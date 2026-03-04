# Deployment Guide for Sepsis Prediction API

## Overview
This guide covers different deployment options for your FastAPI application.

---

## Deployment Options

### 1. Local Development
**Best for**: Testing and development

```bash
# Windows
run_api.bat

# Linux/Mac
bash run_api.sh

# Manual
pip install -r requirements_api.txt
python app.py
```

Access at: `http://localhost:8000`

---

### 2. Docker (Single Container)
**Best for**: Containerized deployment

```bash
# Build image
docker build -t sepsis-prediction-api .

# Run container
docker run -p 8000:8000 \
  -v $(pwd)/model:/app/model \
  -v $(pwd)/logs:/app/logs \
  sepsis-prediction-api

# Or use docker run with environment variables
docker run -p 8000:8000 \
  -e API_PORT=8000 \
  -e ACTIVE_MODEL=xgboost \
  sepsis-prediction-api
```

Access at: `http://localhost:8000`

---

### 3. Docker Compose (Full Stack)
**Best for**: Complete application with optional database/cache

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f api

# Stop services
docker-compose down

# Rebuild images
docker-compose up -d --build
```

Access at: `http://localhost:8000`

**Services**:
- FastAPI application (port 8000)
- PostgreSQL database (optional, port 5432)
- Redis cache (optional, port 6379)

---

### 4. AWS EC2
**Best for**: Production-grade deployment

#### Step 1: Launch EC2 Instance
```bash
# Use Amazon Linux 2 or Ubuntu 20.04+
# Security group: Open ports 80, 443, 8000, 22
```

#### Step 2: Install Dependencies
```bash
# Update system
sudo apt-get update && sudo apt-get upgrade -y

# Install Python
sudo apt-get install python3-pip python3-venv git -y

# Clone repository
git clone <your-repo-url>
cd AS_prediction
```

#### Step 3: Set Up Application
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements_api.txt
```

#### Step 4: Run with Gunicorn
```bash
pip install gunicorn

# Run with Gunicorn
gunicorn -w 4 -b 0.0.0.0:8000 app:app

# Or with systemd (persistent service)
sudo nano /etc/systemd/system/sepsis-api.service
```

#### Systemd Service File
```ini
[Unit]
Description=Sepsis Prediction API
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/AS_prediction
ExecStart=/home/ubuntu/AS_prediction/venv/bin/gunicorn \
    -w 4 \
    -b 0.0.0.0:8000 \
    app:app
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
# Enable service
sudo systemctl enable sepsis-api
sudo systemctl start sepsis-api
sudo systemctl status sepsis-api
```

---

### 5. Heroku
**Best for**: Quick cloud deployment

#### Step 1: Create Procfile
```
web: gunicorn app:app
```

#### Step 2: Create runtime.txt
```
python-3.10.13
```

#### Step 3: Deploy
```bash
# Install Heroku CLI
# Login to Heroku
heroku login

# Create app
heroku create sepsis-prediction-api

# Deploy
git push heroku main

# View logs
heroku logs --tail
```

**Access**: `https://sepsis-prediction-api.herokuapp.com`

---

### 6. Railway.app
**Best for**: Simple cloud deployment (no credit card for free tier)

```bash
# Install Railway CLI
npm i -g @railway/cli

# Login
railway login

# Initialize
railway init

# Deploy
railway up

# View logs
railway logs
```

---

### 7. Google Cloud Run
**Best for**: Serverless, pay-per-use deployment

```bash
# Install Google Cloud SDK
# Initialize
gcloud init

# Deploy
gcloud run deploy sepsis-prediction-api \
    --source . \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated
```

---

### 8. Nginx + Gunicorn (Production)
**Best for**: Professional production setup

```bash
# Install Nginx
sudo apt-get install nginx

# Configure Nginx
sudo nano /etc/nginx/sites-available/Default
```

#### Nginx Configuration
```nginx
upstream app {
    server 127.0.0.1:8000;
}

server {
    listen 80;
    server_name yourdomain.com;

    location / {
        proxy_pass http://app;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

```bash
# Enable site
sudo ln -s /etc/nginx/sites-available/default /etc/nginx/sites-enabled/

# Test config
sudo nginx -t

# Start Nginx
sudo systemctl start nginx
sudo systemctl enable nginx

# Run Gunicorn in background
gunicorn -w 4 -b 127.0.0.1:8000 app:app &
```

---

## SSL/TLS Setup

### Using Let's Encrypt
```bash
# Install Certbot
sudo apt-get install certbot python3-certbot-nginx

# Get certificate
sudo certbot certonly --nginx -d yourdomain.com

# Auto-renew
sudo systemctl enable certbot.timer
sudo systemctl start certbot.timer
```

### Update Nginx Config
```nginx
server {
    listen 443 ssl;
    ssl_certificate /etc/letsencrypt/live/yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/yourdomain.com/privkey.pem;
    
    # ... rest of config
}

# Redirect HTTP to HTTPS
server {
    listen 80;
    return 301 https://$server_name$request_uri;
}
```

---

## Environment Variables

Create `.env` file in deployment directory:

```bash
# Copy template
cp .env.example .env

# Edit with production values
nano .env
```

**Key variables**:
```
API_HOST=0.0.0.0
API_PORT=8000
ACTIVE_MODEL=xgboost
LOG_LEVEL=INFO
DEBUG=false
```

---

## Database Integration (Optional)

### PostgreSQL Setup
```bash
# Using Docker
docker run -d \
  -e POSTGRES_USER=sepsis_user \
  -e POSTGRES_PASSWORD=password123 \
  -e POSTGRES_DB=sepsis_predictions \
  -p 5432:5432 \
  -v postgres_data:/var/lib/postgresql/data \
  postgres:15

# Update DATABASE_URL in .env
DATABASE_URL=postgresql://sepsis_user:password123@localhost:5432/sepsis_predictions
```

---

## Monitoring & Logging

### View Application Logs
```bash
# Docker
docker logs -f sepsis-api

# Docker Compose
docker-compose logs -f api

# Systemd
journalctl -u sepsis-api -f

# Heroku
heroku logs --tail

# Railway
railway logs
```

### Health Check Endpoint
```bash
curl http://localhost:8000/health
```

---

## Performance Optimization

### Gunicorn Workers
```bash
# Calculate optimal workers: (2 x CPU count) + 1
gunicorn -w 9 -b 0.0.0.0:8000 app:app
```

### Load Balancing with Nginx
```nginx
upstream api_backend {
    server 127.0.0.1:8001;
    server 127.0.0.1:8002;
    server 127.0.0.1:8003;
}

server {
    listen 80;
    location / {
        proxy_pass http://api_backend;
    }
}
```

### Caching with Redis
Enable in `.env`:
```
REDIS_URL=redis://localhost:6379/0
REDIS_CACHE_ENABLED=true
```

---

## Security Checklist

- [ ] Change SECRET_KEY in production
- [ ] Enable HTTPS/SSL
- [ ] Set DEBUG=false
- [ ] Configure CORS properly (not wildcard in production)
- [ ] Use environment variables for secrets
- [ ] Enable rate limiting
- [ ] Add authentication/API keys
- [ ] Keep dependencies updated
- [ ] Use strong database passwords
- [ ] Regular backups

---

## Troubleshooting

### Port Already in Use
```bash
# Find process using port
lsof -i :8000

# Kill process
kill -9 <PID>

# Or use different port
gunicorn -b 0.0.0.0:8001 app:app
```

### Permission Denied
```bash
# Add execute permission
chmod +x run_api.sh

# Or run with bash
bash run_api.sh
```

### Module Not Found
```bash
# Reinstall dependencies
pip install -r requirements_api.txt --force-reinstall

# Check virtual environment is activated
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate.bat # Windows
```

### Out of Memory
```bash
# Reduce Gunicorn workers
gunicorn -w 2 -b 0.0.0.0:8000 app:app

# Or use Docker with memory limits
docker run -m 1g sepsis-prediction-api
```

---

## Recommended Deployment Stack

For **production**, use:
1. **Web Server**: Nginx (reverse proxy)
2. **App Server**: Gunicorn (4-8 workers)
3. **Database**: PostgreSQL (optional)
4. **Cache**: Redis (optional)
5. **Monitoring**: Prometheus + Grafana (optional)
6. **Container**: Docker + Docker Compose
7. **Infrastructure**: AWS EC2 or similar

---

## Deployment Checklist

- [ ] Install all dependencies
- [ ] Update `.env` with production values
- [ ] Set up database (if using)
- [ ] Configure SSL/HTTPS
- [ ] Set up logging
- [ ] Test health endpoint
- [ ] Load test the API
- [ ] Set up monitoring
- [ ] Create backup strategy
- [ ] Document deployment process

---

## Performance Benchmarks

On a typical deployment:
- **Response time**: < 200ms per prediction
- **Throughput**: 100-500 requests/second (depending on hardware)
- **Availability**: 99.9%+
- **Memory usage**: ~200-500 MB per worker

---

**For support**, refer to:
- API_GUIDE.md
- IMPLEMENTATION_SUMMARY.md
- app.py (inline documentation)

---

**Last Updated**: March 2026  
**Version**: 1.0.0
