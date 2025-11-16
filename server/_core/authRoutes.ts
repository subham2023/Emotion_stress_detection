import { COOKIE_NAME, ONE_YEAR_MS } from "@shared/const";
import type { Express, Request, Response } from "express";
import { getSessionCookieOptions } from "./cookies";
import { authService } from "./auth";

/**
 * Register simple authentication routes
 */
export function registerAuthRoutes(app: Express) {
  // Simple login endpoint
  app.post("/api/auth/login", async (req: Request, res: Response) => {
    try {
      const { email, password } = req.body;

      if (!email || !password) {
        res.status(400).json({ error: "Email and password are required" });
        return;
      }

      // For demo purposes - accept any non-empty password
      // In production, implement proper password verification
      if (typeof email !== "string" || typeof password !== "string") {
        res.status(400).json({ error: "Invalid input types" });
        return;
      }

      const user = await authService.authenticateUser(email, password);

      if (!user) {
        res.status(401).json({ error: "Invalid credentials" });
        return;
      }

      const sessionToken = await authService.createUserSession(user);
      const cookieOptions = getSessionCookieOptions(req);

      res.cookie(COOKIE_NAME, sessionToken, { ...cookieOptions, maxAge: ONE_YEAR_MS });

      res.json({
        success: true,
        user: {
          id: user.id,
          email: user.email,
          name: user.name,
          loginMethod: user.loginMethod,
        },
      });
    } catch (error) {
      console.error("[Auth] Login failed", error);
      res.status(500).json({ error: "Login failed" });
    }
  });

  // Logout endpoint
  app.post("/api/auth/logout", async (req: Request, res: Response) => {
    try {
      const cookieOptions = getSessionCookieOptions(req);
      res.clearCookie(COOKIE_NAME, cookieOptions);

      res.json({ success: true, message: "Logged out successfully" });
    } catch (error) {
      console.error("[Auth] Logout failed", error);
      res.status(500).json({ error: "Logout failed" });
    }
  });

  // Get current user info
  app.get("/api/auth/me", async (req: Request, res: Response) => {
    try {
      const user = await authService.authenticateRequest(req);

      res.json({
        success: true,
        user: {
          id: user.id,
          email: user.email,
          name: user.name,
          loginMethod: user.loginMethod,
          lastSignedIn: user.lastSignedIn,
        },
      });
    } catch (error) {
      res.status(401).json({ error: "Not authenticated" });
    }
  });
}