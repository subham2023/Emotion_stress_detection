import { Server } from "http";
import mysql from "mysql2/promise";
import Redis from "ioredis";

interface ShutdownConfig {
  timeout: number; // milliseconds to wait before forceful shutdown
  signals: NodeJS.Signals[];
  cleanupTasks: (() => Promise<void>)[];
}

interface ActiveConnection {
  id: string;
  req: any;
  res: any;
  timestamp: number;
}

class GracefulShutdown {
  private server: Server | null = null;
  private connections: Map<string, ActiveConnection> = new Map();
  private isShuttingDown = false;
  private config: ShutdownConfig;
  private mysqlConnections: mysql.Connection[] = [];
  private redisConnections: Redis[] = [];

  constructor(config: Partial<ShutdownConfig> = {}) {
    this.config = {
      timeout: 30000, // 30 seconds default
      signals: ["SIGTERM", "SIGINT", "SIGUSR2"],
      cleanupTasks: [],
      ...config,
    };

    this.setupSignalHandlers();
  }

  // Set up the HTTP server for graceful shutdown
  setServer(server: Server): void {
    this.server = server;

    // Track incoming connections
    server.on("connection", (socket) => {
      const connectionId = this.generateConnectionId();

      // Add connection tracking
      socket.on("end", () => {
        this.connections.delete(connectionId);
      });

      socket.on("error", () => {
        this.connections.delete(connectionId);
      });
    });

    // Track active requests (simplified tracking)
    server.on("request", (req, res) => {
      const connectionId = this.generateConnectionId();
      const activeConnection: ActiveConnection = {
        id: connectionId,
        req,
        res,
        timestamp: Date.now(),
      };

      this.connections.set(connectionId, activeConnection);

      // Remove connection when response finishes
      res.on("finish", () => {
        this.connections.delete(connectionId);
      });

      res.on("error", () => {
        this.connections.delete(connectionId);
      });
    });
  }

  // Register a MySQL connection for cleanup
  addMySQLConnection(connection: mysql.Connection): void {
    this.mysqlConnections.push(connection);
  }

  // Register a Redis connection for cleanup
  addRedisConnection(connection: Redis): void {
    this.redisConnections.push(connection);
  }

  // Add a custom cleanup task
  addCleanupTask(task: () => Promise<void>): void {
    this.config.cleanupTasks.push(task);
  }

  // Setup signal handlers for graceful shutdown
  private setupSignalHandlers(): void {
    this.config.signals.forEach((signal) => {
      process.on(signal, (sig) => {
        console.log(`\nReceived ${sig}, starting graceful shutdown...`);
        this.gracefulShutdown(sig);
      });
    });

    // Handle uncaught exceptions
    process.on("uncaughtException", (error) => {
      console.error("Uncaught Exception:", error);
      this.gracefulShutdown("uncaughtException");
    });

    process.on("unhandledRejection", (reason, promise) => {
      console.error("Unhandled Rejection at:", promise, "reason:", reason);
      this.gracefulShutdown("unhandledRejection");
    });
  }

  // Main graceful shutdown method
  async gracefulShutdown(signal: string): Promise<void> {
    if (this.isShuttingDown) {
      console.log("Shutdown already in progress, ignoring signal:", signal);
      return;
    }

    this.isShuttingDown = true;

    try {
      console.log(`Starting graceful shutdown (signal: ${signal})`);

      // Step 1: Stop accepting new connections
      await this.stopAcceptingConnections();

      // Step 2: Wait for active connections to finish (with timeout)
      await this.waitForActiveConnections();

      // Step 3: Run custom cleanup tasks
      await this.runCleanupTasks();

      // Step 4: Close database connections
      await this.closeDatabaseConnections();

      // Step 5: Close Redis connections
      await this.closeRedisConnections();

      // Step 6: Force cleanup if needed
      await this.forceCleanup();

      console.log("Graceful shutdown completed successfully");
      process.exit(0);
    } catch (error) {
      console.error("Error during graceful shutdown:", error);
      process.exit(1);
    }
  }

  // Step 1: Stop accepting new connections
  private async stopAcceptingConnections(): Promise<void> {
    console.log("Stopping new connections...");

    if (this.server) {
      // Close the server to stop accepting new connections
      this.server.close((err) => {
        if (err) {
          console.error("Error closing server:", err);
        } else {
          console.log("Server stopped accepting new connections");
        }
      });

      // Give some time for the server to close
      await this.sleep(1000);
    }
  }

  // Step 2: Wait for active connections to finish
  private async waitForActiveConnections(): Promise<void> {
    console.log(`Waiting for ${this.connections.size} active connections to finish...`);

    const timeoutMs = this.config.timeout;
    const startTime = Date.now();

    while (this.connections.size > 0 && (Date.now() - startTime) < timeoutMs) {
      const activeConnections = Array.from(this.connections.entries())
        .filter(([_, conn]) => Date.now() - conn.timestamp < 30000) // Only count connections < 30s old
        .map(([id, _]) => id);

      if (activeConnections.length !== this.connections.size) {
        // Clean up old connections
        this.connections.forEach((conn, id) => {
          if (Date.now() - conn.timestamp >= 30000) {
            console.log(`Cleaning up stale connection: ${id}`);
            this.connections.delete(id);
          }
        });
      }

      console.log(`Still waiting for ${this.connections.size} active connections...`);
      await this.sleep(1000);
    }

    if (this.connections.size > 0) {
      console.log(`Timeout reached, ${this.connections.size} connections still active`);
      // Force close remaining connections
      this.connections.forEach((conn, id) => {
        try {
          conn.res.destroy();
          console.log(`Forcefully closed connection: ${id}`);
        } catch (error) {
          console.error(`Error closing connection ${id}:`, error);
        }
      });
      this.connections.clear();
    }
  }

  // Step 3: Run custom cleanup tasks
  private async runCleanupTasks(): Promise<void> {
    console.log("Running custom cleanup tasks...");

    for (const task of this.config.cleanupTasks) {
      try {
        await task();
      } catch (error) {
        console.error("Error running cleanup task:", error);
      }
    }
  }

  // Step 4: Close database connections
  private async closeDatabaseConnections(): Promise<void> {
    console.log("Closing database connections...");

    const closePromises = this.mysqlConnections.map(async (connection, index) => {
      try {
        await connection.end();
        console.log(`MySQL connection ${index + 1} closed`);
      } catch (error) {
        console.error(`Error closing MySQL connection ${index + 1}:`, error);
      }
    });

    await Promise.all(closePromises);
    this.mysqlConnections = [];
  }

  // Step 5: Close Redis connections
  private async closeRedisConnections(): Promise<void> {
    console.log("Closing Redis connections...");

    const closePromises = this.redisConnections.map(async (connection, index) => {
      try {
        await connection.quit();
        console.log(`Redis connection ${index + 1} closed`);
      } catch (error) {
        console.error(`Error closing Redis connection ${index + 1}:`, error);
        try {
          await connection.disconnect();
          console.log(`Redis connection ${index + 1} disconnected`);
        } catch (disconnectError) {
          console.error(`Error disconnecting Redis connection ${index + 1}:`, disconnectError);
        }
      }
    });

    await Promise.all(closePromises);
    this.redisConnections = [];
  }

  // Step 6: Force cleanup if needed
  private async forceCleanup(): Promise<void> {
    console.log("Performing final cleanup...");

    // Flush stdout and stderr
    try {
      process.stdout.write(() => {});
      process.stderr.write(() => {});
    } catch (error) {
      console.error("Error flushing streams:", error);
    }
  }

  // Utility method to generate connection IDs
  private generateConnectionId(): string {
    return `conn_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  // Utility method for sleeping
  private sleep(ms: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

  // Get current status
  getStatus(): {
    isShuttingDown: boolean;
    activeConnections: number;
    mysqlConnections: number;
    redisConnections: number;
  } {
    return {
      isShuttingDown: this.isShuttingDown,
      activeConnections: this.connections.size,
      mysqlConnections: this.mysqlConnections.length,
      redisConnections: this.redisConnections.length,
    };
  }

  // Emergency shutdown (immediate)
  async emergencyShutdown(): Promise<void> {
    console.log("EMERGENCY SHUTDOWN INITIATED");

    this.isShuttingDown = true;

    // Force close all connections immediately
    this.connections.forEach((conn, id) => {
      try {
        conn.res.destroy();
      } catch (error) {
        // Ignore errors during emergency shutdown
      }
    });

    // Force close database connections
    this.mysqlConnections.forEach((connection) => {
      try {
        connection.destroy();
      } catch (error) {
        // Ignore errors during emergency shutdown
      }
    });

    // Force close Redis connections
    this.redisConnections.forEach((connection) => {
      try {
        connection.disconnect();
      } catch (error) {
        // Ignore errors during emergency shutdown
      }
    });

    // Exit immediately
    process.exit(1);
  }
}

// Export singleton instance
export const gracefulShutdown = new GracefulShutdown({
  timeout: 30000, // 30 seconds
  signals: ["SIGTERM", "SIGINT", "SIGUSR2"],
});

// Export class for creating additional instances
export { GracefulShutdown };

// Export utility functions for easy integration
export const setupGracefulShutdown = (
  server: Server,
  config?: Partial<ShutdownConfig>,
) => {
  const shutdownManager = new GracefulShutdown(config);
  shutdownManager.setServer(server);
  return shutdownManager;
};

// Export health check for shutdown status
export const getShutdownStatus = () => {
  return gracefulShutdown.getStatus();
};