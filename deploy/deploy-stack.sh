#!/bin/bash

# Docker Stack Deployment Script for Emotion & Stress Detection Application
# This script deploys the application stack to Docker Swarm

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
STACK_NAME="emotion-stress"
DEV_STACK_NAME="emotion-stress-dev"
COMPOSE_FILE="$SCRIPT_DIR/docker-stack.yml"
DEV_COMPOSE_FILE="$SCRIPT_DIR/docker-stack.dev.yml"
ENV_FILE="$PROJECT_DIR/.env"
DEV_ENV_FILE="$PROJECT_DIR/.env.dev"

# Default to production deployment
DEPLOYMENT_TYPE="production"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Show usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Docker Stack Deployment Script for Emotion & Stress Detection Application

OPTIONS:
    -h, --help              Show this help message
    -t, --type TYPE         Deployment type: production (default) or development
    -e, --env-file FILE     Custom environment file (default: .env or .env.dev)
    -f, --force             Force deployment (skip confirmations)
    -r, --remove            Remove the existing stack
    -s, --status            Show stack status and exit
    -l, --logs              Show stack logs
    --rolling               Perform rolling update (default behavior)
    --blue-green            Perform blue-green deployment
    --dry-run               Show what would be deployed without actually deploying

EXAMPLES:
    $0                      Deploy production stack
    $0 -t development       Deploy development stack
    $0 -r                   Remove production stack
    $0 -t development -r    Remove development stack
    $0 -s                   Show stack status
    $0 -l                   Show stack logs
    $0 --blue-green         Perform blue-green deployment

EOF
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                usage
                exit 0
                ;;
            -t|--type)
                DEPLOYMENT_TYPE="$2"
                shift 2
                ;;
            -e|--env-file)
                CUSTOM_ENV_FILE="$2"
                shift 2
                ;;
            -f|--force)
                FORCE_DEPLOY=true
                shift
                ;;
            -r|--remove)
                REMOVE_STACK=true
                shift
                ;;
            -s|--status)
                SHOW_STATUS=true
                shift
                ;;
            -l|--logs)
                SHOW_LOGS=true
                shift
                ;;
            --rolling)
                DEPLOYMENT_STRATEGY="rolling"
                shift
                ;;
            --blue-green)
                DEPLOYMENT_STRATEGY="blue-green"
                shift
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            *)
                log_error "Unknown option: $1"
                usage
                exit 1
                ;;
        esac
    done
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."

    # Check if Docker is installed and running
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install Docker first."
        exit 1
    fi

    if ! docker info &> /dev/null; then
        log_error "Docker is not running. Please start Docker service."
        exit 1
    fi

    # Check if this is a Docker Swarm manager
    if ! docker node ls &> /dev/null; then
        log_error "This node is not a Docker Swarm manager."
        log_error "Please initialize Swarm first with: ./swarm-init.sh"
        exit 1
    fi

    # Check if compose file exists
    if [ "$DEPLOYMENT_TYPE" = "development" ]; then
        if [ ! -f "$DEV_COMPOSE_FILE" ]; then
            log_error "Development compose file not found: $DEV_COMPOSE_FILE"
            exit 1
        fi
    else
        if [ ! -f "$COMPOSE_FILE" ]; then
            log_error "Production compose file not found: $COMPOSE_FILE"
            exit 1
        fi
    fi

    log_success "Prerequisites check passed"
}

# Load environment variables
load_environment() {
    log_info "Loading environment variables..."

    local env_file=""
    if [ -n "${CUSTOM_ENV_FILE:-}" ]; then
        env_file="$CUSTOM_ENV_FILE"
    elif [ "$DEPLOYMENT_TYPE" = "development" ]; then
        env_file="$DEV_ENV_FILE"
    else
        env_file="$ENV_FILE"
    fi

    if [ -f "$env_file" ]; then
        log_info "Loading environment from: $env_file"
        set -a
        source "$env_file"
        set +a
    else
        log_warning "Environment file not found: $env_file"
        log_warning "Using default or existing environment variables"
    fi

    # Set required environment variables with defaults
    export REGISTRY_OWNER="${REGISTRY_OWNER:-$(whoami)}"
    export DOMAIN="${DOMAIN:-localhost}"
    export MYSQL_ROOT_PASSWORD="${MYSQL_ROOT_PASSWORD:-$(openssl rand -base64 32)}"
    export MYSQL_DATABASE="${MYSQL_DATABASE:-emotion_stress_db}"
    export MYSQL_USER="${MYSQL_USER:-app_user}"
    export MYSQL_PASSWORD="${MYSQL_PASSWORD:-$(openssl rand -base64 32)}"
    export REDIS_PASSWORD="${REDIS_PASSWORD:-$(openssl rand -base64 32)}"
    export GRAFANA_PASSWORD="${GRAFANA_PASSWORD:-admin123}"
    export ACME_EMAIL="${ACME_EMAIL:-admin@${DOMAIN}}"

    log_success "Environment variables loaded"
}

# Remove existing stack
remove_stack() {
    local stack_name="$([ "$DEPLOYMENT_TYPE" = "development" ] && echo "$DEV_STACK_NAME" || echo "$STACK_NAME")"

    log_info "Removing stack: $stack_name"

    if docker stack ls | grep -q "$stack_name"; then
        log_info "Found existing stack, removing..."
        docker stack rm "$stack_name"

        # Wait for stack to be removed
        log_info "Waiting for stack to be removed..."
        local timeout=60
        local elapsed=0
        while docker stack ls | grep -q "$stack_name" && [ $elapsed -lt $timeout ]; do
            sleep 2
            elapsed=$((elapsed + 2))
            echo -n "."
        done
        echo

        if docker stack ls | grep -q "$stack_name"; then
            log_warning "Stack removal timed out, but continuing..."
        else
            log_success "Stack removed successfully"
        fi
    else
        log_info "Stack not found, nothing to remove"
    fi
}

# Deploy stack
deploy_stack() {
    local stack_name="$([ "$DEPLOYMENT_TYPE" = "development" ] && echo "$DEV_STACK_NAME" || echo "$STACK_NAME")"
    local compose_file="$([ "$DEPLOYMENT_TYPE" = "development" ] && echo "$DEV_COMPOSE_FILE" || echo "$COMPOSE_FILE")"

    log_info "Deploying stack: $stack_name"
    log_info "Using compose file: $compose_file"

    if [ "${DRY_RUN:-false}" = "true" ]; then
        log_info "DRY RUN: Would deploy with the following command:"
        echo "docker stack deploy -c \"$compose_file\" \"$stack_name\""
        return
    fi

    # Perform deployment based on strategy
    case "${DEPLOYMENT_STRATEGY:-rolling}" in
        "blue-green")
            deploy_blue_green "$stack_name" "$compose_file"
            ;;
        "rolling"|*)
            deploy_rolling "$stack_name" "$compose_file"
            ;;
    esac
}

# Rolling deployment
deploy_rolling() {
    local stack_name="$1"
    local compose_file="$2"

    log_info "Performing rolling deployment..."

    # Deploy the stack
    if docker stack deploy -c "$compose_file" "$stack_name"; then
        log_success "Stack deployment initiated successfully"
    else
        log_error "Stack deployment failed"
        exit 1
    fi

    # Wait for services to be ready
    wait_for_services "$stack_name"
}

# Blue-green deployment
deploy_blue_green() {
    local stack_name="$1"
    local compose_file="$2"
    local green_stack="${stack_name}-green"

    log_info "Performing blue-green deployment..."

    # Deploy to green environment
    log_info "Deploying to green environment: $green_stack"
    if docker stack deploy -c "$compose_file" "$green_stack"; then
        log_success "Green environment deployed successfully"
    else
        log_error "Green environment deployment failed"
        exit 1
    fi

    # Wait for green services to be ready
    wait_for_services "$green_stack"

    # Perform health checks
    if perform_health_checks "$green_stack"; then
        log_success "Green environment health checks passed"

        # Switch traffic to green
        log_info "Switching traffic to green environment"
        # Implement traffic switching logic here

        # Remove blue environment
        if docker stack ls | grep -q "$stack_name"; then
            log_info "Removing blue environment: $stack_name"
            docker stack rm "$stack_name"
        fi

        # Rename green to production
        log_info "Renaming green environment to production"
        # This would require manual intervention or automated DNS update

    else
        log_error "Green environment health checks failed"
        log_info "Keeping blue environment running, removing green"
        docker stack rm "$green_stack"
        exit 1
    fi
}

# Wait for services to be ready
wait_for_services() {
    local stack_name="$1"
    local timeout=300
    local elapsed=0

    log_info "Waiting for services to be ready (timeout: ${timeout}s)..."

    while [ $elapsed -lt $timeout ]; do
        local services=$(docker stack services "$stack_name" --format "{{.Name}} {{.Replicas}}" | grep "0/" | wc -l)

        if [ "$services" -eq 0 ]; then
            log_success "All services are ready"
            return
        fi

        echo -n "."
        sleep 5
        elapsed=$((elapsed + 5))
    done

    echo
    log_warning "Timeout waiting for services, but deployment may still be proceeding"
}

# Perform health checks
perform_health_checks() {
    local stack_name="$1"
    local app_service="${stack_name}_app"

    log_info "Performing health checks..."

    # Get service tasks
    local tasks=$(docker service ps "$app_service" --format "{{.ID}}" | head -1)

    if [ -z "$tasks" ]; then
        log_error "No tasks found for service: $app_service"
        return 1
    fi

    # Wait a bit for service to start
    sleep 10

    # Check if service is healthy
    local status=$(docker service ps "$app_service" --filter "desired-state=running" --format "{{.CurrentState}}" | head -1)

    if echo "$status" | grep -q "Running"; then
        log_success "Application service is running"
        return 0
    else
        log_error "Application service is not running: $status"
        return 1
    fi
}

# Show stack status
show_status() {
    local stack_name="$([ "$DEPLOYMENT_TYPE" = "development" ] && echo "$DEV_STACK_NAME" || echo "$STACK_NAME")"

    log_info "Showing status for stack: $stack_name"

    if docker stack ls | grep -q "$stack_name"; then
        echo
        log_info "Stack Services:"
        docker stack services "$stack_name"
        echo

        log_info "Service Details:"
        docker service ls --filter "label=com.docker.stack.namespace=$stack_name"
        echo

        log_info "Node Status:"
        docker node ls
    else
        log_warning "Stack not found: $stack_name"
    fi
}

# Show stack logs
show_logs() {
    local stack_name="$([ "$DEPLOYMENT_TYPE" = "development" ] && echo "$DEV_STACK_NAME" || echo "$STACK_NAME")"
    local service="${2:-${stack_name}_app}"

    log_info "Showing logs for service: $service"
    docker service logs "$service" -f --tail 100
}

# Display deployment information
display_deployment_info() {
    local stack_name="$([ "$DEPLOYMENT_TYPE" = "development" ] && echo "$DEV_STACK_NAME" || echo "$STACK_NAME")"

    echo
    log_success "Deployment completed successfully!"
    echo
    log_info "Stack Information:"
    echo "================================"
    echo "Stack Name: $stack_name"
    echo "Deployment Type: $DEPLOYMENT_TYPE"
    echo "Domain: $DOMAIN"
    echo
    log_info "Access Points:"
    echo "================================"
    echo "Application: http${DOMAIN//localhost/://$DOMAIN}"
    echo "Grafana: http${DOMAIN//localhost/://grafana.$DOMAIN}"
    echo "Prometheus: http${DOMAIN//localhost/://prometheus.$DOMAIN}"
    echo "Traefik Dashboard: http${DOMAIN//localhost/://localhost:8080}"
    echo
    log_info "Useful Commands:"
    echo "================================"
    echo "View stack services: docker stack services $stack_name"
    echo "View service logs: docker service logs ${stack_name}_app"
    echo "Scale services: docker service scale ${stack_name}_app=3"
    echo "Update service: docker service update ${stack_name}_app --image new-image"
    echo "Remove stack: $0 -r"
}

# Main execution
main() {
    # Parse command line arguments
    parse_args "$@"

    # Show help if requested
    if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
        usage
        exit 0
    fi

    echo "================================"
    echo "Docker Stack Deployment"
    echo "Emotion & Stress Detection App"
    echo "================================"
    echo

    # Execute based on arguments
    if [ "${SHOW_STATUS:-false}" = "true" ]; then
        check_prerequisites
        show_status
        exit 0
    fi

    if [ "${SHOW_LOGS:-false}" = "true" ]; then
        check_prerequisites
        show_logs "$@"
        exit 0
    fi

    if [ "${REMOVE_STACK:-false}" = "true" ]; then
        check_prerequisites
        remove_stack
        exit 0
    fi

    # Normal deployment flow
    check_prerequisites
    load_environment

    if [ "${FORCE_DEPLOY:-false}" != "true" ]; then
        echo
        log_info "Deployment Configuration:"
        echo "Type: $DEPLOYMENT_TYPE"
        echo "Strategy: ${DEPLOYMENT_STRATEGY:-rolling}"
        echo "Domain: $DOMAIN"
        echo
        read -p "Continue with deployment? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_info "Deployment cancelled"
            exit 0
        fi
    fi

    remove_stack
    deploy_stack
    display_deployment_info
}

# Run the main function with all arguments
main "$@"