#!/bin/bash

# Development script for MVTec Anomaly Detection System

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
COMPOSE_FILE="docker-compose.dev.yml"
ENV_FILE=".env.dev"

# Functions
log() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
    exit 1
}

# Create development environment file
create_dev_env() {
    if [ ! -f "$ENV_FILE" ]; then
        log "Creating development environment file..."
        cat > "$ENV_FILE" << EOF
# Development Environment Configuration
ENVIRONMENT=development
DEBUG=true

# Database
POSTGRES_DB=anomaly_detection_dev
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres

# Redis
REDIS_PASSWORD=redis123

# API Configuration
API_HOST=0.0.0.0
API_PORT=5000
SECRET_KEY=dev_secret_key

# Model Configuration
MODEL_PATH=/app/checkpoints/best_model.pth
DATA_DIR=/app/dataset
THRESHOLD_PERCENTILE=90

# Monitoring
WANDB_API_KEY=your_wandb_key
WANDB_MODE=online

# Development specific
FLASK_ENV=development
FLASK_DEBUG=1
JUPYTER_TOKEN=dev_token
EOF
        log "Development environment file created"
    fi
}

# Setup development environment
setup() {
    log "Setting up development environment..."
    
    create_dev_env
    
    # Create necessary directories
    mkdir -p dataset checkpoints logs uploads results notebooks
    
    # Build development images
    log "Building development images..."
    docker-compose -f "$COMPOSE_FILE" build
    
    log "Development environment setup completed"
}

# Start development environment
start() {
    log "Starting development environment..."
    
    # Start all services
    docker-compose -f "$COMPOSE_FILE" up -d
    
    # Wait for services
    sleep 10
    
    # Show service status
    docker-compose -f "$COMPOSE_FILE" ps
    
    log "Development environment started!"
    log "Access points:"
    log "- API: http://localhost:5000"
    log "- Jupyter: http://localhost:8888"
    log "- Database: localhost:5433"
    log "- Redis: localhost:6380"
    log "- Monitoring: http://localhost:9091"
}

# Stop development environment
stop() {
    log "Stopping development environment..."
    docker-compose -f "$COMPOSE_FILE" down
    log "Development environment stopped"
}

# Restart development environment
restart() {
    log "Restarting development environment..."
    stop
    start
}

# Show logs
logs() {
    if [ -n "$1" ]; then
        docker-compose -f "$COMPOSE_FILE" logs -f "$1"
    else
        docker-compose -f "$COMPOSE_FILE" logs -f
    fi
}

# Execute command in container
exec_cmd() {
    if [ -n "$1" ] && [ -n "$2" ]; then
        docker-compose -f "$COMPOSE_FILE" exec "$1" "${@:2}"
    else
        error "Usage: $0 exec <service> <command>"
    fi
}

# Run tests
test() {
    log "Running tests..."
    
    # Run unit tests
    docker-compose -f "$COMPOSE_FILE" exec inference-api-dev python -m pytest tests/ -v
    
    # Run integration tests
    docker-compose -f "$COMPOSE_FILE" exec inference-api-dev python -m pytest tests/integration/ -v
    
    log "Tests completed"
}

# Format code
format() {
    log "Formatting code..."
    
    # Run black
    docker-compose -f "$COMPOSE_FILE" exec inference-api-dev python -m black src/
    
    # Run isort
    docker-compose -f "$COMPOSE_FILE" exec inference-api-dev python -m isort src/
    
    log "Code formatting completed"
}

# Lint code
lint() {
    log "Linting code..."
    
    # Run flake8
    docker-compose -f "$COMPOSE_FILE" exec inference-api-dev python -m flake8 src/
    
    # Run mypy
    docker-compose -f "$COMPOSE_FILE" exec inference-api-dev python -m mypy src/
    
    log "Linting completed"
}

# Train model
train() {
    log "Starting model training..."
    
    docker-compose -f "$COMPOSE_FILE" exec training-dev python -m src.training.train "$@"
    
    log "Training completed"
}

# Backup development data
backup() {
    log "Creating development backup..."
    
    BACKUP_NAME="dev_backup_$(date +%Y%m%d_%H%M%S)"
    
    # Backup database
    docker-compose -f "$COMPOSE_FILE" exec -T database-dev pg_dump -U postgres anomaly_detection_dev > "backups/${BACKUP_NAME}_database.sql"
    
    # Backup checkpoints
    tar -czf "backups/${BACKUP_NAME}_checkpoints.tar.gz" checkpoints/
    
    log "Development backup created: $BACKUP_NAME"
}

# Clean up development environment
clean() {
    log "Cleaning up development environment..."
    
    # Stop and remove containers
    docker-compose -f "$COMPOSE_FILE" down -v
    
    # Remove development images
    docker images | grep mvtec | grep dev | awk '{print $3}' | xargs docker rmi -f
    
    # Clean up volumes
    docker volume prune -f
    
    log "Development environment cleaned up"
}

# Main script
main() {
    case "$1" in
        "setup")
            setup
            ;;
        "start")
            start
            ;;
        "stop")
            stop
            ;;
        "restart")
            restart
            ;;
        "logs")
            logs "$2"
            ;;
        "exec")
            exec_cmd "${@:2}"
            ;;
        "test")
            test
            ;;
        "format")
            format
            ;;
        "lint")
            lint
            ;;
        "train")
            train "${@:2}"
            ;;
        "backup")
            backup
            ;;
        "clean")
            clean
            ;;
        *)
            echo "Usage: $0 {setup|start|stop|restart|logs|exec|test|format|lint|train|backup|clean}"
            echo ""
            echo "Commands:"
            echo "  setup   - Setup development environment"
            echo "  start   - Start development environment"
            echo "  stop    - Stop development environment"
            echo "  restart - Restart development environment"
            echo "  logs    - Show logs (optionally for specific service)"
            echo "  exec    - Execute command in container"
            echo "  test    - Run tests"
            echo "  format  - Format code"
            echo "  lint    - Lint code"
            echo "  train   - Train model"
            echo "  backup  - Create backup"
            echo "  clean   - Clean up environment"
            exit 1
            ;;
    esac
}

# Create backup directory
mkdir -p backups

# Run main function
main "$@"