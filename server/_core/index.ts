import "dotenv/config";
import express from "express";
import { createServer } from "http";
import net from "net";
import { createExpressMiddleware } from "@trpc/server/adapters/express";
import { registerOAuthRoutes } from "./oauth";
import { appRouter } from "../routers";
import { createContext } from "./context";
import { serveStatic, setupVite } from "./vite";
import {
  getHealthStatus,
  getDatabaseHealth,
  getRedisHealth,
  getStorageHealth,
  getMLHealth,
  getWebSocketHealth,
  getLivenessStatus,
  getReadinessStatus,
} from "../src/health";
import { gracefulShutdown, setupGracefulShutdown } from "../src/shutdown";

function isPortAvailable(port: number): Promise<boolean> {
  return new Promise(resolve => {
    const server = net.createServer();
    server.listen(port, () => {
      server.close(() => resolve(true));
    });
    server.on("error", () => resolve(false));
  });
}

async function findAvailablePort(startPort: number = 3000): Promise<number> {
  for (let port = startPort; port < startPort + 20; port++) {
    if (await isPortAvailable(port)) {
      return port;
    }
  }
  throw new Error(`No available port found starting from ${startPort}`);
}

async function startServer() {
  const app = express();
  const server = createServer(app);
  // Configure body parser with larger size limit for file uploads
  app.use(express.json({ limit: "50mb" }));
  app.use(express.urlencoded({ limit: "50mb", extended: true }));

  // Health check endpoints
  app.get("/api/health", getHealthStatus);
  app.get("/api/health/database", getDatabaseHealth);
  app.get("/api/health/redis", getRedisHealth);
  app.get("/api/health/storage", getStorageHealth);
  app.get("/api/health/ml", getMLHealth);
  app.get("/api/health/websocket", getWebSocketHealth);
  app.get("/api/health/liveness", getLivenessStatus);
  app.get("/api/health/readiness", getReadinessStatus);

  // OAuth callback under /api/oauth/callback
  registerOAuthRoutes(app);

  // tRPC API
  app.use(
    "/api/trpc",
    createExpressMiddleware({
      router: appRouter,
      createContext,
    })
  );
  // development mode uses Vite, production mode uses static files
  if (process.env.NODE_ENV === "development") {
    await setupVite(app, server);
  } else {
    serveStatic(app);
  }

  const preferredPort = parseInt(process.env.PORT || "3000");
  const port = await findAvailablePort(preferredPort);

  if (port !== preferredPort) {
    console.log(`Port ${preferredPort} is busy, using port ${port} instead`);
  }

  // Set up graceful shutdown
  setupGracefulShutdown(server, {
    timeout: parseInt(process.env.SHUTDOWN_TIMEOUT || "30000"),
    cleanupTasks: [
      async () => {
        console.log("Cleaning up application resources...");
        // Add application-specific cleanup here
      },
    ],
  });

  server.listen(port, () => {
    console.log(`Server running on http://localhost:${port}/`);
    console.log("Graceful shutdown handlers registered");
  });
}

startServer().catch(console.error);
