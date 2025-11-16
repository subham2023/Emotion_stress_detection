#!/bin/bash

# Docker Stack Update Script for Emotion & Stress Detection Application
# This script performs rolling updates of the application stack

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
STACK_NAME="emotion-stress"
DEV_STACK_NAME="emotion-stress-dev"

# Default values
DEPLOYMENT_TYPE="production"
SERVICE_NAME="app"
NEW_IMAGE=""
PARALLELISM=1
DELAY=10s
FAILURE_ACTION="rollback"
MONITOR=60s

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

Docker Stack Update Script for Emotion & Stress Detection Application

OPTIONS:
    -h, --help              Show this help message
    -t, --type TYPE         Deployment type: production (default) or development
    -s, --service SERVICE   Service to update (default: app)
    -i, --image IMAGE       New image to deploy (required)
    -p, --parallelism NUM   Number of tasks to update simultaneously (default: 1)
    -d, --delay DELAY       Delay between updates (default: 10s)
    -f, --failure-action    Action on failure: pause, continue, rollback (default: rollback)
    -m, --monitor TIME      Monitor duration for each task update (default: 60s)
    --force                 Force update without confirmation
    --rollback              Perform rollback instead of update
    --status                Show update status
    --history               Show update history

EXAMPLES:
    $0 -i myrepo/app:v1.2.3                    # Update app service with new image
    $0 -t dev -i myrepo/app:develop            # Update development environment
    $0 -s mysql -i mysql:8.0                   # Update MySQL service
    $0 -i myrepo/app:v1.2.3 -p 2 -d 20s        # Update with 2 parallel tasks, 20s delay
    $0 --rollback                              # Rollback to previous version
    $0 --status                                # Show update status

EOF
}

# Parse command line arguments
parse_args() {
    ROLLBACK=false
    SHOW_STATUS=false
    SHOW_HISTORY=false

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
            -s|--service)
                SERVICE_NAME="$2"
                shift 2
                ;;
            -i|--image)
                NEW_IMAGE="$2"
                shift 2
                ;;
            -p|--parallelism)
                PARALLELISM="$2"
                shift 2
                ;;
            -d|--delay)
                DELAY="$2"
                shift 2
                ;;
            -f|--failure-action)
                FAILURE_ACTION="$2"
                shift 2
                ;;
            -m|--monitor)
                MONITOR="$2"
                shift 2
                ;;
            --force)
                FORCE_UPDATE=true
                shift
                ;;
            --rollback)
                ROLLBACK=true
                shift
                ;;
            --status)
                SHOW_STATUS=true
                shift
                ;;
            --history)
                SHOW_HISTORY=true
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

# Get full service name
get_service_name() {
    local stack_name=$(get_stack_name)
    echo "${stack_name}_${SERVICE_NAME}"
}

# Check if stack exists
check_stack_exists() {
    local stack_name=$(get_stack_name)

    if ! docker stack ls | grep -q "$stack_name"; then
        log_error "Stack not found: $stack_name"
        log_error "Please deploy the stack first with: ./deploy-stack.sh"
        exit 1
    fi
}

# Check if service exists
check_service_exists() {
    local service_name=$(get_service_name)

    if ! docker service ls | grep -q "$service_name"; then
        log_error "Service not found: $service_name"
        exit 1
    fi
}

# Get current service information
get_current_service_info() {
    local service_name=$(get_service_name)

    log_info "Getting current service information..."

    docker service inspect "$service_name" --format '{
        "Image": "{{.Spec.TaskTemplate.ContainerSpec.Image}}",
        "Replicas": "{{.Spec.Mode.Replicated.Replicas}}",
        "UpdateStatus": "{{.UpdateStatus.State}}"
    }' | jq '.'
}

# Validate new image
validate_new_image() {
    if [ -z "$NEW_IMAGE" ] && [ "$ROLLBACK" != "true" ]; then
        log_error "New image is required. Use -i option to specify image."
        exit 1
    fi

    if [ -n "$NEW_IMAGE" ]; then
        log_info "Validating new image: $NEW_IMAGE"

        # Pull the new image to check if it exists
        if docker pull "$NEW_IMAGE" &>/dev/null; then
            log_success "New image is available: $NEW_IMAGE"
        else
            log_error "Failed to pull new image: $NEW_IMAGE"
            exit 1
        fi
    fi
}

# Perform service update
perform_update() {
    local service_name=$(get_service_name)

    log_info "Starting service update..."
    log_info "Service: $service_name"
    log_info "New Image: $NEW_IMAGE"
    log_info "Parallelism: $PARALLELISM"
    log_info "Delay: $DELAY"
    log_info "Failure Action: $FAILURE_ACTION"
    log_info "Monitor: $MONITOR"

    # Perform the update
    if docker service update \
        --image "$NEW_IMAGE" \
        --update-parallelism "$PARALLELISM" \
        --update-delay "$DELAY" \
        --update-failure-action "$FAILURE_ACTION" \
        --update-monitor "$MONITOR" \
        "$service_name"; then

        log_success "Service update initiated successfully"
    else
        log_error "Service update failed"
        exit 1
    fi

    # Monitor update progress
    monitor_update_progress
}

# Perform rollback
perform_rollback() {
    local service_name=$(get_service_name)

    log_info "Starting service rollback..."
    log_info "Service: $service_name"

    if docker service update --rollback "$service_name"; then
        log_success "Service rollback initiated successfully"
    else
        log_error "Service rollback failed"
        exit 1
    fi

    # Monitor rollback progress
    monitor_update_progress
}

# Monitor update progress
monitor_update_progress() {
    local service_name=$(get_service_name)
    local timeout=600
    local elapsed=0

    log_info "Monitoring update progress (timeout: ${timeout}s)..."

    while [ $elapsed -lt $timeout ]; do
        local update_state=$(docker service inspect "$service_name" --format '{{.UpdateStatus.State}}' 2>/dev/null || echo "unknown")
        local update_message=$(docker service inspect "$service_name" --format '{{.UpdateStatus.Message}}' 2>/dev/null || echo "unknown")

        case "$update_state" in
            "updating")
                echo -n "."
                ;;
            "completed")
                echo
                log_success "Update completed successfully"
                show_service_status
                return 0
                ;;
            "rollback_completed")
                echo
                log_success "Rollback completed successfully"
                show_service_status
                return 0
                ;;
            "rollback_paused")
                echo
                log_warning "Rollback paused: $update_message"
                return 1
                ;;
            "paused")
                echo
                log_warning "Update paused: $update_message"
                return 1
                ;;
            "rollback_started")
                echo
                log_info "Rollback started"
                ;;
            "error")
                echo
                log_error "Update failed: $update_message"
                return 1
                ;;
        esac

        sleep 5
        elapsed=$((elapsed + 5))
    done

    echo
    log_warning "Update monitoring timed out"
}

# Show service status
show_service_status() {
    local service_name=$(get_service_name)

    log_info "Service Status:"
    echo "================================"

    # Get service details
    docker service ps "$service_name" --format "table {{.Name}}\t{{.Image}}\t{{.CurrentState}}\t{{.Error}}"

    echo
    log_info "Service Summary:"
    docker service ls --filter "id=$(docker service inspect $service_name -f '{{.ID}}')" --format "table {{.Name}}\t{{.Mode}}\t{{.Replicas}}\t{{.Image}}"
}

# Show update status
show_update_status() {
    local service_name=$(get_service_name)

    log_info "Update Status:"
    echo "================================"

    local update_state=$(docker service inspect "$service_name" --format '{{.UpdateStatus.State}}' 2>/dev/null || echo "unknown")
    local update_message=$(docker service inspect "$service_name" --format '{{.UpdateStatus.Message}}' 2>/dev/null || echo "unknown")

    echo "State: $update_state"
    echo "Message: $update_message"

    if [ "$update_state" != "unknown" ]; then
        local update_started=$(docker service inspect "$service_name" --format '{{.UpdateStatus.StartedAt}}' 2>/dev/null || echo "unknown")
        local update_completed=$(docker service inspect "$service_name" --format '{{.UpdateStatus.CompletedAt}}' 2>/dev/null || echo "unknown")

        echo "Started: $update_started"
        echo "Completed: $update_completed"
    fi

    echo
    show_service_status
}

# Show update history
show_update_history() {
    local service_name=$(get_service_name)

    log_info "Update History:"
    echo "================================"

    # Get service tasks with their history
    docker service ps "$service_name" --format "table {{.Name}}\t{{.Image}}\t{{.DesiredState}}\t{{.CurrentState}}\t{{.Node}}"

    echo
    log_info "Service Configuration History:"
    docker service inspect "$service_name" --format '{{range .Spec.TaskHistory}}{{.ContainerSpec.Image}}{{end}}' | tr ' ' '\n' | sort -u
}

# Confirm update
confirm_update() {
    if [ "${FORCE_UPDATE:-false}" = "true" ]; then
        return
    fi

    echo
    log_info "Update Configuration:"
    echo "Stack Type: $DEPLOYMENT_TYPE"
    echo "Service: $SERVICE_NAME"
    echo "New Image: $NEW_IMAGE"
    echo "Parallelism: $PARALLELISM"
    echo "Delay: $DELAY"
    echo "Failure Action: $FAILURE_ACTION"
    echo "Monitor: $MONITOR"
    echo
    read -p "Continue with update? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_info "Update cancelled"
        exit 0
    fi
}

# Display update information
display_update_info() {
    echo
    log_success "Update operation completed!"
    echo
    log_info "Service Information:"
    show_service_status
    echo
    log_info "Monitoring Commands:"
    echo "docker service logs $(get_service_name) -f"
    echo "docker service ps $(get_service_name)"
    echo "$0 --status"
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
    echo "Docker Stack Update"
    echo "Emotion & Stress Detection App"
    echo "================================"
    echo

    # Execute based on arguments
    if [ "${SHOW_STATUS:-false}" = "true" ]; then
        check_prerequisites
        check_stack_exists
        show_update_status
        exit 0
    fi

    if [ "${SHOW_HISTORY:-false}" = "true" ]; then
        check_prerequisites
        check_stack_exists
        show_update_history
        exit 0
    fi

    # Normal update flow
    check_prerequisites
    check_stack_exists
    check_service_exists

    if [ "$ROLLBACK" != "true" ]; then
        validate_new_image
        confirm_update
        perform_update
    else
        perform_rollback
    fi

    display_update_info
}

# Run the main function with all arguments
main "$@"