#!/bin/bash

# Production deployment script for MVTec Anomaly Detection System

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
COMPOSE_FILE="docker-compose.yml"
ENV_FILE=".env"
BACKUP_DIR="backups"
LOG_FILE="logs/deployment.log"

# Functions
log() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] $1${NC}" | tee -a "$LOG_FILE"
}

warn() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}" | tee -a "$LOG_FILE"
    exit 1
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed"
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        error "Docker Compose is not installed"
    fi
    
    # Check NVIDIA Docker runtime
    if ! docker info | grep -q "nvidia"; then
        warn "NVIDIA Docker runtime not found. GPU support may not work."
    fi
    
    # Check environment file
    if [ ! -f "$ENV_FILE" ]; then
        warn "Environment file not found. Creating default..."
        create_env_file
    fi
    
    log "Prerequisites check completed"
}

create_env_file() {
    cat > "$ENV_FILE" << EOF
# Environment Configuration
ENVIRONMENT=production
DEBUG=false

# Database
POSTGRES_DB=anomaly_detection
POSTGRES_USER=postgres
POSTGRES_PASSWORD=change_me_in_production

# Redis
REDIS_PASSWORD=change_me_in_production

# API Configuration
API_HOST=0.0.0.0
API_PORT=5000
SECRET_KEY=change_me_in_production

# Model Configuration
MODEL_PATH=/app/checkpoints/best_model.pth
DATA_DIR=/app/dataset
THRESHOLD_PERCENTILE=90

# Monitoring
WANDB_API_KEY=your_wandb_key
PROMETHEUS_RETENTION_TIME=15d

# SSL Configuration
SSL_CERT_PATH=/etc/nginx/ssl/cert.pem
SSL_KEY_PATH=/etc/nginx/ssl/key.pem
EOF
    log "Default environment file created. Please update the passwords and keys."
}

# Backup function
backup_data() {
    log "Creating backup..."
    
    mkdir -p "$BACKUP_DIR"
    BACKUP_NAME="backup_$(date +%Y%m%d_%H%M%S)"
    
    # Backup database
    docker-compose exec -T database pg_dump -U postgres anomaly_detection > "$BACKUP_DIR/${BACKUP_NAME}_database.sql"
    
    # Backup checkpoints
    tar -czf "$BACKUP_DIR/${BACKUP_NAME}_checkpoints.tar.gz" checkpoints/
    
    # Backup logs
    tar -czf "$BACKUP_DIR/${BACKUP_NAME}_logs.tar.gz" logs/
    
    log "Backup created: $BACKUP_NAME"
}

# Build and deploy
deploy() {
    log "Starting deployment..."
    
    # Pull latest images
    log "Pulling latest images..."
    docker-compose -f "$COMPOSE_FILE" pull
    
    # Build images
    log "Building images..."
    docker-compose -f "$COMPOSE_FILE" build --no-cache
    
    # Start services
    log "Starting services..."
    docker-compose -f "$COMPOSE_FILE" up -d
    
    # Wait for services to be ready
    log "Waiting for services to be ready..."
    sleep 30
    
    # Health check
    health_check
    
    log "Deployment completed successfully!"
}

# Health check
health_check() {
    log "Performing health check..."
    
    # Check inference API
    if curl -f http://localhost:5000/health > /dev/null 2>&1; then
        log "Inference API is healthy"
    else
        error "Inference API health check failed"
    fi
    
    # Check database
    if docker-compose exec -T database pg_isready -U postgres > /dev/null 2>&1; then
        log "Database is healthy"
    else
        error "Database health check failed"
    fi
    
    # Check Redis
    if docker-compose exec -T redis redis-cli ping > /dev/null 2>&1; then
        log "Redis is healthy"
    else
        error "Redis health check failed"
    fi
    
    log "All health checks passed"
}

# Rollback function
rollback() {
    log "Rolling back deployment..."
    
    # Stop current services
    docker-compose -f "$COMPOSE_FILE" down
    
    # Restore from backup
    if [ -n "$1" ]; then
        BACKUP_NAME="$1"
        log "Restoring from backup: $BACKUP_NAME"
        
        # Restore database
        docker-compose -f "$COMPOSE_FILE" up -d database
        sleep 10
        cat "$BACKUP_DIR/${BACKUP_NAME}_database.sql" | docker-compose exec -T database psql -U postgres -d anomaly_detection
        
        # Restore checkpoints
        tar -xzf "$BACKUP_DIR/${BACKUP_NAME}_checkpoints.tar.gz"
        
        # Restore logs
        tar -xzf "$BACKUP_DIR/${BACKUP_NAME}_logs.tar.gz"
        
        log "Rollback completed"
    else
        error "No backup specified for rollback"
    fi
}

# Monitor deployment
monitor() {
    log "Starting monitoring..."
    
    # Show service status
    docker-compose -f "$COMPOSE_FILE" ps
    
    # Show logs
    docker-compose -f "$COMPOSE_FILE" logs -f
}

# Cleanup function
cleanup() {
    log "Cleaning up..."
    
    # Remove old images
    docker image prune -f
    
    # Remove old containers
    docker container prune -f
    
    # Remove old volumes
    docker volume prune -f
    
    log "Cleanup completed"
}

# Main script
main() {
    case "$1" in
        "deploy")
            check_prerequisites
            backup_data
            deploy
            ;;
        "rollback")
            rollback "$2"
            ;;
        "health")
            health_check
            ;;
        "monitor")
            monitor
            ;;
        "cleanup")
            cleanup
            ;;
        "backup")
            backup_data
            ;;
        *)
            echo "Usage: $0 {deploy|rollback|health|monitor|cleanup|backup}"
            echo ""
            echo "Commands:"
            echo "  deploy   - Deploy the application"
            echo "  rollback - Rollback to a previous backup"
            echo "  health   - Check service health"
            echo "  monitor  - Monitor services"
            echo "  cleanup  - Clean up unused resources"
            echo "  backup   - Create a backup"
            exit 1
            ;;
    esac
}

# Create log directory
mkdir -p "$(dirname "$LOG_FILE")"

# Run main function
main "$@"