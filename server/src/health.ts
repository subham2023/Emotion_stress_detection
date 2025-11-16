import { Request, Response } from "express";
import mysql from "mysql2/promise";
import Redis from "ioredis";
import fs from "fs/promises";
import path from "path";

// Health check interfaces
interface HealthStatus {
  status: "healthy" | "unhealthy" | "degraded";
  timestamp: string;
  uptime: number;
  version: string;
  checks: {
    database: HealthCheck;
    redis: HealthCheck;
    storage: HealthCheck;
    ml: HealthCheck;
    websocket: HealthCheck;
    memory: HealthCheck;
    disk: HealthCheck;
  };
  metrics: {
    responseTime: number;
    activeConnections: number;
    memoryUsage: NodeJS.MemoryUsage;
    cpuUsage: NodeJS.CpuUsage;
  };
}

interface HealthCheck {
  status: "pass" | "fail" | "warn";
  message: string;
  responseTime?: number;
  details?: any;
}

// Cache for health check results to avoid excessive calls
const healthCache = {
  lastCheck: 0,
  result: null as HealthStatus | null,
  ttl: 30000, // 30 seconds cache
};

// Database health check
async function checkDatabaseHealth(): Promise<HealthCheck> {
  const startTime = Date.now();

  try {
    const connection = await mysql.createConnection({
      host: process.env.MYSQL_HOST || "mysql",
      user: process.env.MYSQL_USER || "root",
      password: process.env.MYSQL_ROOT_PASSWORD,
      database: process.env.MYSQL_DATABASE,
      timeout: 5000,
    });

    await connection.execute("SELECT 1");
    await connection.end();

    const responseTime = Date.now() - startTime;

    return {
      status: "pass",
      message: "Database connection successful",
      responseTime,
    };
  } catch (error) {
    const responseTime = Date.now() - startTime;
    return {
      status: "fail",
      message: `Database connection failed: ${error instanceof Error ? error.message : "Unknown error"}`,
      responseTime,
      details: {
        error: error instanceof Error ? error.stack : error,
      },
    };
  }
}

// Redis health check
async function checkRedisHealth(): Promise<HealthCheck> {
  const startTime = Date.now();

  try {
    const redis = new Redis({
      host: process.env.REDIS_HOST || "redis",
      port: parseInt(process.env.REDIS_PORT || "6379"),
      password: process.env.REDIS_PASSWORD,
      retryDelayOnFailover: 100,
      maxRetriesPerRequest: 1,
      lazyConnect: true,
    });

    await redis.ping();
    await redis.quit();

    const responseTime = Date.now() - startTime;

    return {
      status: "pass",
      message: "Redis connection successful",
      responseTime,
    };
  } catch (error) {
    const responseTime = Date.now() - startTime;
    return {
      status: "fail",
      message: `Redis connection failed: ${error instanceof Error ? error.message : "Unknown error"}`,
      responseTime,
      details: {
        error: error instanceof Error ? error.stack : error,
      },
    };
  }
}

// Storage health check (S3 or local storage)
async function checkStorageHealth(): Promise<HealthCheck> {
  const startTime = Date.now();

  try {
    // Check if we can access the storage directory
    const storagePath = process.env.STORAGE_PATH || "./uploads";
    await fs.access(storagePath, fs.constants.R_OK | fs.constants.W_OK);

    // If S3 is configured, test S3 connectivity
    if (process.env.S3_BUCKET && process.env.AWS_ACCESS_KEY_ID) {
      // TODO: Implement S3 health check
      // For now, just check if S3 credentials are present
      if (process.env.AWS_SECRET_ACCESS_KEY && process.env.AWS_REGION) {
        const responseTime = Date.now() - startTime;
        return {
          status: "pass",
          message: "S3 configuration found",
          responseTime,
        };
      } else {
        return {
          status: "warn",
          message: "S3 configuration incomplete",
          responseTime: Date.now() - startTime,
        };
      }
    }

    const responseTime = Date.now() - startTime;
    return {
      status: "pass",
      message: "Local storage accessible",
      responseTime,
    };
  } catch (error) {
    const responseTime = Date.now() - startTime;
    return {
      status: "fail",
      message: `Storage check failed: ${error instanceof Error ? error.message : "Unknown error"}`,
      responseTime,
    };
  }
}

// ML model health check
async function checkMLHealth(): Promise<HealthCheck> {
  const startTime = Date.now();

  try {
    // Check if TensorFlow is available and models can be loaded
    const tf = await import("@tensorflow/tfjs-node");

    // Test basic TensorFlow functionality
    const model = tf.sequential();
    model.add(tf.layers.dense({units: 1, inputShape: [1]}));
    model.compile({optimizer: 'sgd', loss: 'meanSquaredError'});

    // Cleanup
    model.dispose();

    // Check if model files exist
    const modelPath = process.env.MODEL_PATH || "./src/models";
    try {
      await fs.access(modelPath, fs.constants.R_OK);
      const responseTime = Date.now() - startTime;
      return {
        status: "pass",
        message: "ML models accessible and TensorFlow functional",
        responseTime,
        details: {
          modelPath,
          tensorflowVersion: tf.version.tf,
        },
      };
    } catch (modelError) {
      const responseTime = Date.now() - startTime;
      return {
        status: "warn",
        message: "TensorFlow functional but model files not found",
        responseTime,
        details: {
          modelPath,
          error: modelError instanceof Error ? modelError.message : modelError,
        },
      };
    }
  } catch (error) {
    const responseTime = Date.now() - startTime;
    return {
      status: "fail",
      message: `ML health check failed: ${error instanceof Error ? error.message : "Unknown error"}`,
      responseTime,
      details: {
        error: error instanceof Error ? error.stack : error,
      },
    };
  }
}

// WebSocket health check
async function checkWebSocketHealth(): Promise<HealthCheck> {
  const startTime = Date.now();

  try {
    // Check if WebSocket server is configured
    const wsPort = process.env.WS_PORT || process.env.PORT || "3000";

    // Simple TCP check on the WebSocket port
    const net = await import("net");
    const socket = new net.Socket();

    return new Promise((resolve) => {
      const responseTime = Date.now() - startTime;

      socket.connect(parseInt(wsPort), "localhost", () => {
        socket.destroy();
        resolve({
          status: "pass",
          message: "WebSocket port is accessible",
          responseTime,
        });
      });

      socket.on("error", () => {
        resolve({
          status: "warn",
          message: "WebSocket port not accessible (may be normal in HTTP mode)",
          responseTime,
        });
      });

      // Timeout after 3 seconds
      setTimeout(() => {
        socket.destroy();
        resolve({
          status: "warn",
          message: "WebSocket check timed out",
          responseTime,
        });
      }, 3000);
    });
  } catch (error) {
    const responseTime = Date.now() - startTime;
    return {
      status: "warn",
      message: `WebSocket check failed: ${error instanceof Error ? error.message : "Unknown error"}`,
      responseTime,
    };
  }
}

// Memory health check
function checkMemoryHealth(): HealthCheck {
  const memUsage = process.memoryUsage();
  const totalMemory = require("os").totalmem();
  const freeMemory = require("os").freemem();

  // Calculate memory usage percentage
  const heapUsedMB = Math.round(memUsage.heapUsed / 1024 / 1024);
  const heapTotalMB = Math.round(memUsage.heapTotal / 1024 / 1024);
  const memoryUsagePercent = ((totalMemory - freeMemory) / totalMemory) * 100;

  let status: "pass" | "warn" | "fail" = "pass";
  let message = "Memory usage normal";

  if (memoryUsagePercent > 90) {
    status = "fail";
    message = "Critical memory usage";
  } else if (memoryUsagePercent > 80) {
    status = "warn";
    message = "High memory usage";
  }

  return {
    status,
    message,
    details: {
      heapUsed: `${heapUsedMB}MB`,
      heapTotal: `${heapTotalMB}MB`,
      systemMemoryUsage: `${memoryUsagePercent.toFixed(1)}%`,
    },
  };
}

// Disk health check
async function checkDiskHealth(): Promise<HealthCheck> {
  const startTime = Date.now();

  try {
    const stats = await fs.statfs(".");
    const totalSpace = stats.bavail * stats.bsize;
    const freeSpace = stats.bavail * stats.bsize;
    const usagePercent = ((stats.blocks - stats.bavail) / stats.blocks) * 100;

    let status: "pass" | "warn" | "fail" = "pass";
    let message = "Disk space adequate";

    if (usagePercent > 95) {
      status = "fail";
      message = "Critical disk usage";
    } else if (usagePercent > 85) {
      status = "warn";
      message = "Low disk space";
    }

    const responseTime = Date.now() - startTime;

    return {
      status,
      message,
      responseTime,
      details: {
        freeSpace: `${Math.round(freeSpace / 1024 / 1024 / 1024)}GB`,
        usagePercent: `${usagePercent.toFixed(1)}%`,
      },
    };
  } catch (error) {
    const responseTime = Date.now() - startTime;
    return {
      status: "fail",
      message: `Disk health check failed: ${error instanceof Error ? error.message : "Unknown error"}`,
      responseTime,
    };
  }
}

// Get system metrics
function getSystemMetrics() {
  return {
    responseTime: 0, // Will be calculated by the main health check
    activeConnections: 0, // TODO: Implement connection tracking
    memoryUsage: process.memoryUsage(),
    cpuUsage: process.cpuUsage(),
  };
}

// Main health check endpoint
export async function getHealthStatus(req: Request, res: Response) {
  const startTime = Date.now();

  // Check cache first
  const now = Date.now();
  if (healthCache.result && (now - healthCache.lastCheck) < healthCache.ttl) {
    const cachedResult = { ...healthCache.result };
    cachedResult.metrics.responseTime = Date.now() - startTime;
    return res.json(cachedResult);
  }

  try {
    // Run all health checks in parallel
    const [
      databaseCheck,
      redisCheck,
      storageCheck,
      mlCheck,
      websocketCheck,
    ] = await Promise.all([
      checkDatabaseHealth(),
      checkRedisHealth(),
      checkStorageHealth(),
      checkMLHealth(),
      checkWebSocketHealth(),
    ]);

    // Synchronous checks
    const memoryCheck = checkMemoryHealth();
    const diskCheck = await checkDiskHealth();

    // Determine overall status
    const allChecks = [databaseCheck, redisCheck, storageCheck, mlCheck, websocketCheck, memoryCheck, diskCheck];
    const failedChecks = allChecks.filter(check => check.status === "fail");
    const warningChecks = allChecks.filter(check => check.status === "warn");

    let overallStatus: "healthy" | "unhealthy" | "degraded";
    if (failedChecks.length > 0) {
      overallStatus = "unhealthy";
    } else if (warningChecks.length > 0) {
      overallStatus = "degraded";
    } else {
      overallStatus = "healthy";
    }

    const healthStatus: HealthStatus = {
      status: overallStatus,
      timestamp: new Date().toISOString(),
      uptime: process.uptime(),
      version: process.env.npm_package_version || "1.0.0",
      checks: {
        database: databaseCheck,
        redis: redisCheck,
        storage: storageCheck,
        ml: mlCheck,
        websocket: websocketCheck,
        memory: memoryCheck,
        disk: diskCheck,
      },
      metrics: {
        responseTime: Date.now() - startTime,
        ...getSystemMetrics(),
      },
    };

    // Update cache
    healthCache.lastCheck = now;
    healthCache.result = healthStatus;

    // Set appropriate HTTP status code
    const statusCode = overallStatus === "healthy" ? 200 : overallStatus === "degraded" ? 200 : 503;
    res.status(statusCode).json(healthStatus);

  } catch (error) {
    const errorStatus: HealthStatus = {
      status: "unhealthy",
      timestamp: new Date().toISOString(),
      uptime: process.uptime(),
      version: process.env.npm_package_version || "1.0.0",
      checks: {
        database: { status: "fail", message: "Not checked due to system error" },
        redis: { status: "fail", message: "Not checked due to system error" },
        storage: { status: "fail", message: "Not checked due to system error" },
        ml: { status: "fail", message: "Not checked due to system error" },
        websocket: { status: "fail", message: "Not checked due to system error" },
        memory: { status: "fail", message: "Not checked due to system error" },
        disk: { status: "fail", message: "Not checked due to system error" },
      },
      metrics: {
        responseTime: Date.now() - startTime,
        activeConnections: 0,
        memoryUsage: process.memoryUsage(),
        cpuUsage: process.cpuUsage(),
      },
    };

    res.status(503).json(errorStatus);
  }
}

// Individual health check endpoints
export async function getDatabaseHealth(req: Request, res: Response) {
  const result = await checkDatabaseHealth();
  res.status(result.status === "pass" ? 200 : 503).json(result);
}

export async function getRedisHealth(req: Request, res: Response) {
  const result = await checkRedisHealth();
  res.status(result.status === "pass" ? 200 : 503).json(result);
}

export async function getStorageHealth(req: Request, res: Response) {
  const result = await checkStorageHealth();
  res.status(result.status === "pass" ? 200 : 503).json(result);
}

export async function getMLHealth(req: Request, res: Response) {
  const result = await checkMLHealth();
  res.status(result.status === "pass" ? 200 : 503).json(result);
}

export async function getWebSocketHealth(req: Request, res: Response) {
  const result = await checkWebSocketHealth();
  res.status(result.status === "pass" ? 200 : 503).json(result);
}

// Liveness probe (simpler version for Kubernetes)
export async function getLivenessStatus(req: Request, res: Response) {
  try {
    // Simple check if the process is responsive
    res.status(200).json({
      status: "alive",
      timestamp: new Date().toISOString(),
      uptime: process.uptime(),
    });
  } catch (error) {
    res.status(503).json({
      status: "dead",
      timestamp: new Date().toISOString(),
      error: error instanceof Error ? error.message : "Unknown error",
    });
  }
}

// Readiness probe (checks if the service is ready to serve traffic)
export async function getReadinessStatus(req: Request, res: Response) {
  try {
    // Check essential components only
    const [databaseCheck] = await Promise.all([
      checkDatabaseHealth(),
    ]);

    const isReady = databaseCheck.status === "pass";

    if (isReady) {
      res.status(200).json({
        status: "ready",
        timestamp: new Date().toISOString(),
        checks: {
          database: databaseCheck,
        },
      });
    } else {
      res.status(503).json({
        status: "not_ready",
        timestamp: new Date().toISOString(),
        checks: {
          database: databaseCheck,
        },
      });
    }
  } catch (error) {
    res.status(503).json({
      status: "not_ready",
      timestamp: new Date().toISOString(),
      error: error instanceof Error ? error.message : "Unknown error",
    });
  }
}