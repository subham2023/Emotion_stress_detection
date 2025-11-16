# Multi-stage Dockerfile for Emotion & Stress Detection Application
# Stage 1: Build Frontend
FROM node:20-alpine AS frontend-builder

WORKDIR /app/frontend

# Copy frontend package files
COPY package.json pnpm-lock.yaml ./
COPY client/package.json client/pnpm-lock.yaml ./client/

# Install pnpm and dependencies
RUN npm install -g pnpm@10.4.1
RUN pnpm install --frozen-lockfile

# Copy frontend source code
COPY client/ ./client/
COPY tsconfig.json vite.config.ts tailwind.config.js ./
COPY components.json ./

# Build frontend
RUN pnpm build

# Stage 2: Build Backend
FROM node:20-alpine AS backend-builder

WORKDIR /app/backend

# Copy backend package files
COPY package.json pnpm-lock.yaml ./

# Install pnpm and dependencies
RUN npm install -g pnpm@10.4.1
RUN pnpm install --frozen-lockfile

# Copy backend source code
COPY server/ ./server/
COPY shared/ ./shared/
COPY drizzle/ ./drizzle/
COPY drizzle.config.ts tsconfig.json ./

# Build backend
RUN pnpm build

# Stage 3: Runtime Container
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgthread-2.0-0 \
    libgtk-3-0 \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libatlas-base-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js and pnpm for runtime
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && npm install -g pnpm@10.4.1

WORKDIR /app

# Copy Python requirements and install ML dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy built artifacts from previous stages
COPY --from=frontend-builder /app/frontend/dist ./client/dist
COPY --from=backend-builder /app/backend/dist ./server/dist
COPY --from=backend-builder /app/backend/node_modules ./server/node_modules

# Copy ML pipeline source code
COPY src/ ./src/

# Copy configuration files
COPY package.json pnpm-lock.yaml ./
COPY tsconfig.json ./
COPY drizzle.config.ts ./
COPY drizzle/ ./drizzle/

# Create necessary directories
RUN mkdir -p ./shared ./logs

# Set environment variables
ENV NODE_ENV=production
ENV PORT=3000

# Expose application port
EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:3000/api/health || exit 1

# Start command
CMD ["npm", "start"]