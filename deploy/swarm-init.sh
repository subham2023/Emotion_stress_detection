#!/bin/bash

# Docker Swarm Initialization Script for Emotion & Stress Detection Application
# This script initializes a Docker Swarm cluster and prepares it for deployment

set -e

# Configuration
STACK_NAME="emotion-stress"
DEV_STACK_NAME="emotion-stress-dev"
NETWORK_NAME="emotion-stress-network"
DEV_NETWORK_NAME="emotion-stress-dev-network"

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

# Check if Docker is installed and running
check_docker() {
    log_info "Checking Docker installation..."

    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install Docker first."
        exit 1
    fi

    if ! docker info &> /dev/null; then
        log_error "Docker is not running. Please start Docker service."
        exit 1
    fi

    log_success "Docker is installed and running"
}

# Check if this node is already part of a swarm
check_swarm_status() {
    log_info "Checking Docker Swarm status..."

    if docker node ls &> /dev/null; then
        local current_node=$(docker node ls --filter "role=manager" --format "{{.Hostname}}" | head -n1)
        if [ -n "$current_node" ]; then
            log_info "This node is already part of a Swarm cluster as a manager"
            log_info "Current manager node: $current_node"
            return 0
        fi
    fi

    log_info "This node is not part of a Swarm cluster"
    return 1
}

# Get the current node's IP address
get_node_ip() {
    log_info "Detecting node IP address..."

    # Try multiple methods to get the IP
    local ip=$(ip route get 1.1.1.1 | awk '{print $7; exit}' 2>/dev/null || \
             hostname -I | awk '{print $1}' 2>/dev/null || \
             ipconfig getifaddr en0 2>/dev/null || \
             echo "127.0.0.1")

    echo "$ip"
}

# Initialize Docker Swarm
init_swarm() {
    log_info "Initializing Docker Swarm..."

    local advertise_addr=$(get_node_ip)
    log_info "Using advertise address: $advertise_addr"

    # Initialize swarm with default configuration
    if docker swarm init --advertise-addr "$advertise_addr" --default-addr-pool 10.10.0.0/16; then
        log_success "Docker Swarm initialized successfully"

        # Get join tokens for workers
        local worker_token=$(docker swarm join-token -q worker)
        local manager_token=$(docker swarm join-token -q manager)

        echo
        log_info "Swarm join commands:"
        echo "Worker join command:"
        echo "docker swarm join --token ${worker_token} ${advertise_addr}:2377"
        echo
        echo "Manager join command:"
        echo "docker swarm join --token ${manager_token} ${advertise_addr}:2377"
        echo
    else
        log_error "Failed to initialize Docker Swarm"
        exit 1
    fi
}

# Create required networks
create_networks() {
    log_info "Creating required overlay networks..."

    # Production network
    if ! docker network ls --filter "name=${NETWORK_NAME}" --quiet | grep -q .; then
        docker network create --driver overlay --attachable "${NETWORK_NAME}"
        log_success "Created production network: ${NETWORK_NAME}"
    else
        log_info "Production network ${NETWORK_NAME} already exists"
    fi

    # Development network
    if ! docker network ls --filter "name=${DEV_NETWORK_NAME}" --quiet | grep -q .; then
        docker network create --driver overlay --attachable "${DEV_NETWORK_NAME}"
        log_success "Created development network: ${DEV_NETWORK_NAME}"
    else
        log_info "Development network ${DEV_NETWORK_NAME} already exists"
    fi
}

# Create directories for persistent data
create_data_directories() {
    log_info "Creating directories for persistent data..."

    local directories=(
        "/var/lib/docker/volumes/${STACK_NAME}_mysql-data/_data"
        "/var/lib/docker/volumes/${STACK_NAME}_redis-data/_data"
        "/var/lib/docker/volumes/${STACK_NAME}_prometheus-data/_data"
        "/var/lib/docker/volumes/${STACK_NAME}_grafana-data/_data"
        "/var/lib/docker/volumes/${STACK_NAME}_traefik-letsencrypt/_data"
        "/var/lib/docker/volumes/${DEV_STACK_NAME}_mysql-dev-data/_data"
        "/var/lib/docker/volumes/${DEV_STACK_NAME}_redis-dev-data/_data"
    )

    for dir in "${directories[@]}"; do
        if [ ! -d "$dir" ]; then
            sudo mkdir -p "$dir"
            sudo chown 999:999 "$dir" 2>/dev/null || true
            log_success "Created directory: $dir"
        else
            log_info "Directory already exists: $dir"
        fi
    done
}

# Check system resources
check_system_resources() {
    log_info "Checking system resources..."

    # Check memory
    local total_mem=$(free -m | awk 'NR==2{printf "%.0f", $2/1024}')
    if [ "$total_mem" -lt 8 ]; then
        log_warning "System has less than 8GB RAM (${total_mem}GB detected)"
        log_warning "This may not be sufficient for production deployment"
    else
        log_success "System has sufficient RAM: ${total_mem}GB"
    fi

    # Check disk space
    local available_space=$(df -BG /var/lib/docker | awk 'NR==2{print $4}' | sed 's/G//')
    if [ "$available_space" -lt 20 ]; then
        log_warning "Less than 20GB disk space available in /var/lib/docker"
        log_warning "This may not be sufficient for Docker volumes and images"
    else
        log_success "Sufficient disk space available: ${available_space}GB"
    fi

    # Check Docker resources
    log_info "Docker system information:"
    docker system df
}

# Set up firewall rules (if applicable)
setup_firewall() {
    log_info "Checking firewall configuration..."

    # Check if UFW is available
    if command -v ufw &> /dev/null; then
        if ufw status | grep -q "Status: active"; then
            log_info "UFW is active, configuring rules..."
            ufw allow 2376/tcp comment "Docker Swarm API"
            ufw allow 2377/tcp comment "Docker Swarm management"
            ufw allow 7946/tcp comment "Docker Swarm discovery"
            ufw allow 7946/udp comment "Docker Swarm discovery"
            ufw allow 4789/udp comment "Docker overlay network"
            log_success "Firewall rules configured"
        fi
    fi

    # Check if firewalld is available
    if command -v firewall-cmd &> /dev/null; then
        if systemctl is-active --quiet firewalld; then
            log_info "Firewalld is active, configuring rules..."
            firewall-cmd --add-port=2376/tcp --permanent
            firewall-cmd --add-port=2377/tcp --permanent
            firewall-cmd --add-port=7946/tcp --permanent
            firewall-cmd --add-port=7946/udp --permanent
            firewall-cmd --add-port=4789/udp --permanent
            firewall-cmd --reload
            log_success "Firewalld rules configured"
        fi
    fi
}

# Display swarm information
display_swarm_info() {
    echo
    log_success "Docker Swarm initialization completed!"
    echo
    log_info "Swarm Information:"
    echo "================================"
    docker node ls
    echo
    log_info "Network Information:"
    echo "================================"
    docker network ls | grep -E "(emotion-stress|overlay)"
    echo
    log_info "Next Steps:"
    echo "================================"
    echo "1. Copy the join commands above to add worker nodes"
    echo "2. Set up your environment variables in .env file"
    echo "3. Run: ./deploy-stack.sh to deploy the application"
    echo "4. Access the application at http://localhost (after deployment)"
    echo
    log_info "Useful Commands:"
    echo "================================"
    echo "View stack services: docker stack services ${STACK_NAME}"
    echo "View service logs: docker service logs ${STACK_NAME}_app"
    echo "Scale services: docker service scale ${STACK_NAME}_app=3"
    echo "Remove stack: docker stack rm ${STACK_NAME}"
}

# Main execution
main() {
    echo "================================"
    echo "Docker Swarm Initialization"
    echo "Emotion & Stress Detection App"
    echo "================================"
    echo

    check_docker
    check_system_resources

    if ! check_swarm_status; then
        init_swarm
    fi

    create_networks
    create_data_directories
    setup_firewall
    display_swarm_info
}

# Run the main function
main "$@"