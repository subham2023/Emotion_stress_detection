import { COOKIE_NAME, ONE_YEAR_MS } from "@shared/const";
import { ForbiddenError } from "@shared/_core/errors";
import { parse as parseCookieHeader } from "cookie";
import type { Request } from "express";
import { SignJWT, jwtVerify } from "jose";
import type { User } from "../../drizzle/schema";
import * as db from "../db";

// Utility function
const isNonEmptyString = (value: unknown): value is string =>
  typeof value === "string" && value.length > 0;

export type SessionPayload = {
  userId: string;
  email: string;
  name: string;
};

/**
 * Simple local authentication service
 */
class AuthService {
  private getSessionSecret() {
    const secret = process.env.JWT_SECRET || process.env.SESSION_SECRET || "fallback-secret-key";
    return new TextEncoder().encode(secret);
  }

  /**
   * Create a session token for a user
   */
  async createSessionToken(
    userId: string,
    options: { expiresInMs?: number; email?: string; name?: string } = {}
  ): Promise<string> {
    const issuedAt = Date.now();
    const expiresInMs = options.expiresInMs ?? ONE_YEAR_MS;
    const expirationSeconds = Math.floor((issuedAt + expiresInMs) / 1000);
    const secretKey = this.getSessionSecret();

    return new SignJWT({
      userId,
      email: options.email || "",
      name: options.name || "",
    })
      .setProtectedHeader({ alg: "HS256", typ: "JWT" })
      .setExpirationTime(expirationSeconds)
      .sign(secretKey);
  }

  private parseCookies(cookieHeader: string | undefined) {
    if (!cookieHeader) {
      return new Map<string, string>();
    }

    const parsed = parseCookieHeader(cookieHeader);
    return new Map(Object.entries(parsed));
  }

  /**
   * Verify a session token
   */
  async verifySession(
    cookieValue: string | undefined | null
  ): Promise<{ userId: string; email: string; name: string } | null> {
    if (!cookieValue) {
      console.warn("[Auth] Missing session cookie");
      return null;
    }

    try {
      const secretKey = this.getSessionSecret();
      const { payload } = await jwtVerify(cookieValue, secretKey, {
        algorithms: ["HS256"],
      });
      const { userId, email, name } = payload as Record<string, unknown>;

      if (
        !isNonEmptyString(userId) ||
        !isNonEmptyString(email) ||
        !isNonEmptyString(name)
      ) {
        console.warn("[Auth] Session payload missing required fields");
        return null;
      }

      return {
        userId,
        email,
        name,
      };
    } catch (error) {
      console.warn("[Auth] Session verification failed", String(error));
      return null;
    }
  }

  /**
   * Authenticate a request and return the user
   */
  async authenticateRequest(req: Request): Promise<User> {
    const cookies = this.parseCookies(req.headers.cookie);
    const sessionCookie = cookies.get(COOKIE_NAME);
    const session = await this.verifySession(sessionCookie);

    if (!session) {
      throw ForbiddenError("Invalid session cookie");
    }

    const signedInAt = new Date();
    let user = await db.getUserById(session.userId);

    if (!user) {
      throw ForbiddenError("User not found");
    }

    // Update last signed in time
    await db.upsertUser({
      openId: user.openId,
      lastSignedIn: signedInAt,
    });

    return user;
  }

  /**
   * Simple user authentication for demo purposes
   * In production, this would integrate with your user database
   */
  async authenticateUser(email: string, password: string): Promise<User | null> {
    // For demo purposes - in production, use proper password hashing
    // and database verification
    console.log("[Auth] Authenticating user:", email);

    // Check if user exists in database
    let user = await db.getUserByEmail(email);

    if (!user) {
      // Create a demo user if not exists (for testing only)
      console.log("[Auth] Creating demo user for:", email);
      const openId = `local_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

      await db.upsertUser({
        openId,
        name: email.split("@")[0],
        email,
        loginMethod: "local",
        lastSignedIn: new Date(),
      });

      user = await db.getUserByEmail(email);
    }

    return user;
  }

  /**
   * Create a user session after authentication
   */
  async createUserSession(user: User): Promise<string> {
    return this.createSessionToken(user.id, {
      email: user.email || undefined,
      name: user.name || undefined,
    });
  }
}

export const authService = new AuthService();