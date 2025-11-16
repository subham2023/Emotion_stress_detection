#!/bin/bash

# Docker Stack Removal Script for Emotion & Stress Detection Application
# This script safely removes the application stack from Docker Swarm

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
STACK_NAME="emotion-stress"
DEV_STACK_NAME="emotion-stress-dev"

# Default values
DEPLOYMENT_TYPE="production"
TIMEOUT=120
FORCE_REMOVE=false
REMOVE_VOLUMES=false

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

Docker Stack Removal Script for Emotion & Stress Detection Application

OPTIONS:
    -h, --help              Show this help message
    -t, --type TYPE         Deployment type: production (default) or development
    -f, --force             Force removal without confirmation
    -v, --volumes           Remove associated volumes
    -s, --timeout SECONDS   Timeout for stack removal (default: 120)
    --all                   Remove both production and development stacks
    --dry-run               Show what would be removed without actually removing

EXAMPLES:
    $0                      Remove production stack
    $0 -t development       Remove development stack
    $0 -f -v                Force remove production stack with volumes
    $0 --all                Remove both production and development stacks
    $0 --dry-run            Show what would be removed

EOF
}

# Parse command line arguments
parse_args() {
    REMOVE_ALL=false
    DRY_RUN=false

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
            -f|--force)
                FORCE_REMOVE=true
                shift
                ;;
            -v|--volumes)
                REMOVE_VOLUMES=true
                shift
                ;;
            -s|--timeout)
                TIMEOUT="$2"
                shift 2
                ;;
            --all)
                REMOVE_ALL=true
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
        exit 1
    fi

    log_success "Prerequisites check passed"
}

# Get stack name based on deployment type
get_stack_name() {
    if [ "$DEPLOYMENT_TYPE" = "development" ]; then
        echo "$DEV_STACK_NAME"
    else
        echo "$STACK_NAME"
    fi
}

# Get all stacks to remove
get_stacks_to_remove() {
    local stacks=()

    if [ "$REMOVE_ALL" = "true" ]; then
        stacks=("$STACK_NAME" "$DEV_STACK_NAME")
    else
        stacks=("$(get_stack_name)")
    fi

    echo "${stacks[@]}"
}

# Check if stack exists
check_stack_exists() {
    local stack_name="$1"

    if docker stack ls | grep -q "$stack_name"; then
        return 0
    else
        return 1
    fi
}

# Show stack information before removal
show_stack_info() {
    local stack_name="$1"

    log_info "Stack Information: $stack_name"
    echo "================================"

    if ! check_stack_exists "$stack_name"; then
        log_warning "Stack not found: $stack_name"
        return
    fi

    # Show services
    log_info "Services:"
    docker stack services "$stack_name" --format "table {{.Name}}\t{{.Mode}}\t{{.Replicas}}\t{{.Image}}"
    echo

    # Show networks
    log_info "Networks:"
    docker network ls --filter "label=com.docker.stack.namespace=$stack_name" --format "table {{.Name}}\t{{.Driver}}\t{{.Scope}}"
    echo

    # Show volumes
    log_info "Volumes:"
    local volumes=$(docker volume ls --filter "label=com.docker.stack.namespace=$stack_name" --format "{{.Name}}" | tr '\n' ' ')
    if [ -n "$volumes" ]; then
        echo "$volumes"
    else
        echo "No stack-specific volumes found"
    fi
    echo

    # Show resource usage
    log_info "Resource Usage:"
    docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}" \
        $(docker ps --filter "label=com.docker.swarm.service.name" --format "{{.Names}}") 2>/dev/null || echo "No running containers"
}

# Create backup before removal
create_backup() {
    local stack_name="$1"
    local backup_dir="$PROJECT_DIR/backups"
    local timestamp=$(date +"%Y%m%d_%H%M%S")
    local backup_file="$backup_dir/${stack_name}_backup_${timestamp}.tar.gz"

    log_info "Creating backup of stack data..."

    mkdir -p "$backup_dir"

    # Backup Docker configurations
    local temp_dir="/tmp/${stack_name}_backup_${timestamp}"
    mkdir -p "$temp_dir"

    # Export stack configuration
    if check_stack_exists "$stack_name"; then
        # Save service configurations
        docker stack services "$stack_name" --format "{{.Name}}" > "$temp_dir/services.txt"

        # Save network configurations
        docker network ls --filter "label=com.docker.stack.namespace=$stack_name" --format "{{.Name}}" > "$temp_dir/networks.txt"

        # Save volume information
        docker volume ls --filter "label=com.docker.stack.namespace=$stack_name" --format "{{.Name}}" > "$temp_dir/volumes.txt"

        # Save environment variables (if available)
        if [ -f "$PROJECT_DIR/.env" ]; then
            cp "$PROJECT_DIR/.env" "$temp_dir/env.backup"
        fi
        if [ -f "$PROJECT_DIR/.env.dev" ]; then
            cp "$PROJECT_DIR/.env.dev" "$temp_dir/env.dev.backup"
        fi

        # Create archive
        tar -czf "$backup_file" -C "$temp_dir" .
        rm -rf "$temp_dir"

        log_success "Backup created: $backup_file"
    else
        log_warning "Stack not found, skipping backup"
    fi
}

# Remove stack
remove_stack() {
    local stack_name="$1"

    if ! check_stack_exists "$stack_name"; then
        log_warning "Stack not found: $stack_name"
        return
    fi

    log_info "Removing stack: $stack_name"

    if [ "${DRY_RUN:-false}" = "true" ]; then
        log_info "DRY RUN: Would remove stack: $stack_name"
        return
    fi

    # Remove the stack
    if docker stack rm "$stack_name"; then
        log_success "Stack removal initiated: $stack_name"

        # Wait for stack to be removed
        log_info "Waiting for stack to be completely removed..."
        local elapsed=0

        while check_stack_exists "$stack_name" && [ $elapsed -lt $TIMEOUT ]; do
            echo -n "."
            sleep 2
            elapsed=$((elapsed + 2))
        done

        echo

        if check_stack_exists "$stack_name"; then
            log_warning "Stack removal timed out, but services may still be cleaning up"
        else
            log_success "Stack removed successfully: $stack_name"
        fi
    else
        log_error "Failed to remove stack: $stack_name"
        exit 1
    fi
}

# Remove associated networks
remove_networks() {
    local stack_name="$1"

    log_info "Removing associated networks..."

    if [ "${DRY_RUN:-false}" = "true" ]; then
        log_info "DRY RUN: Would remove networks for stack: $stack_name"
        return
    fi

    local networks=$(docker network ls --filter "label=com.docker.stack.namespace=$stack_name" --format "{{.Name}}")

    for network in $networks; do
        if [ "$network" != "ingress" ]; then  # Don't remove the ingress network
            log_info "Removing network: $network"
            if docker network rm "$network" 2>/dev/null; then
                log_success "Network removed: $network"
            else
                log_warning "Failed to remove network: $network (may still be in use)"
            fi
        fi
    done
}

# Remove associated volumes
remove_volumes() {
    local stack_name="$1"

    if [ "$REMOVE_VOLUMES" != "true" ]; then
        log_info "Volume removal skipped (use -v flag to enable)"
        return
    fi

    log_info "Removing associated volumes..."

    if [ "${DRY_RUN:-false}" = "true" ]; then
        log_info "DRY RUN: Would remove volumes for stack: $stack_name"
        return
    fi

    local volumes=$(docker volume ls --filter "label=com.docker.stack.namespace=$stack_name" --format "{{.Name}}")

    for volume in $volumes; do
        log_warning "Removing volume: $volume (this will delete all data)"
        if docker volume rm "$volume" 2>/dev/null; then
            log_success "Volume removed: $volume"
        else
            log_warning "Failed to remove volume: $volume (may still be in use)"
        fi
    done
}

# Clean up orphaned containers and images
cleanup_docker() {
    log_info "Cleaning up Docker resources..."

    if [ "${DRY_RUN:-false}" = "true" ]; then
        log_info "DRY RUN: Would clean up Docker resources"
        return
    fi

    # Remove stopped containers
    log_info "Removing stopped containers..."
    docker container prune -f &>/dev/null || true

    # Remove unused networks
    log_info "Removing unused networks..."
    docker network prune -f &>/dev/null || true

    # Remove unused images
    log_info "Removing unused images..."
    docker image prune -f &>/dev/null || true

    log_success "Docker cleanup completed"
}

# Confirm removal
confirm_removal() {
    local stacks=("$@")

    if [ "${FORCE_REMOVE:-false}" = "true" ]; then
        return
    fi

    echo
    log_warning "WARNING: This will remove the following stacks and all associated services:"
    for stack in "${stacks[@]}"; do
        echo "  - $stack"
    done
    echo
    if [ "$REMOVE_VOLUMES" = "true" ]; then
        log_warning "WARNING: Volumes will also be removed, deleting all data!"
    fi
    echo
    read -p "Are you sure you want to continue? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_info "Removal cancelled"
        exit 0
    fi
}

# Display removal summary
display_removal_summary() {
    local stacks=("$@")

    echo
    log_success "Stack removal completed!"
    echo
    log_info "Summary:"
    echo "================================"

    for stack in "${stacks[@]}"; do
        if check_stack_exists "$stack"; then
            log_warning "Stack may still be cleaning up: $stack"
        else
            log_success "Stack removed: $stack"
        fi
    done

    echo
    log_info "Remaining Docker Stacks:"
    docker stack ls
    echo
    log_info "Docker System Usage:"
    docker system df
    echo
    log_info "To redeploy: ./deploy-stack.sh"
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
    echo "Docker Stack Removal"
    echo "Emotion & Stress Detection App"
    echo "================================"
    echo

    # Execute removal
    check_prerequisites

    local stacks_to_remove=($(get_stacks_to_remove))

    if [ ${#stacks_to_remove[@]} -eq 0 ]; then
        log_error "No stacks specified for removal"
        exit 1
    fi

    # Show stack information
    for stack in "${stacks_to_remove[@]}"; do
        show_stack_info "$stack"
        echo
    done

    # Confirm removal
    confirm_removal "${stacks_to_remove[@]}"

    # Create backup
    if [ "${DRY_RUN:-false}" != "true" ]; then
        for stack in "${stacks_to_remove[@]}"; do
            create_backup "$stack"
        done
    fi

    # Remove stacks
    for stack in "${stacks_to_remove[@]}"; do
        remove_stack "$stack"
        remove_networks "$stack"
        remove_volumes "$stack"
        echo
    done

    # Cleanup
    cleanup_docker

    # Display summary
    display_removal_summary "${stacks_to_remove[@]}"
}

# Run the main function with all arguments
main "$@"