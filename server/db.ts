import { eq, desc, and } from "drizzle-orm";
import { drizzle } from "drizzle-orm/mysql2";
import { 
  InsertUser, 
  users,
  detectionSessions,
  detectionResults,
  uploadedFiles,
  userStatistics,
  modelMetadata,
  InsertDetectionSession,
  InsertDetectionResult,
  InsertUploadedFile,
  InsertUserStatistics,
  InsertModelMetadata,
} from "../drizzle/schema";
import { ENV } from './_core/env';

let _db: ReturnType<typeof drizzle> | null = null;

// Lazily create the drizzle instance so local tooling can run without a DB.
export async function getDb() {
  if (!_db && process.env.DATABASE_URL) {
    try {
      _db = drizzle(process.env.DATABASE_URL);
    } catch (error) {
      console.warn("[Database] Failed to connect:", error);
      _db = null;
    }
  }
  return _db;
}

// ============================================================================
// USER FUNCTIONS
// ============================================================================

export async function upsertUser(user: InsertUser): Promise<void> {
  if (!user.openId) {
    throw new Error("User openId is required for upsert");
  }

  const db = await getDb();
  if (!db) {
    console.warn("[Database] Cannot upsert user: database not available");
    return;
  }

  try {
    const values: InsertUser = {
      openId: user.openId,
    };
    const updateSet: Record<string, unknown> = {};

    const textFields = ["name", "email", "loginMethod"] as const;
    type TextField = (typeof textFields)[number];

    const assignNullable = (field: TextField) => {
      const value = user[field];
      if (value === undefined) return;
      const normalized = value ?? null;
      values[field] = normalized;
      updateSet[field] = normalized;
    };

    textFields.forEach(assignNullable);

    if (user.lastSignedIn !== undefined) {
      values.lastSignedIn = user.lastSignedIn;
      updateSet.lastSignedIn = user.lastSignedIn;
    }
    if (user.role !== undefined) {
      values.role = user.role;
      updateSet.role = user.role;
    } else if (user.openId === ENV.ownerOpenId) {
      values.role = 'admin';
      updateSet.role = 'admin';
    }

    if (!values.lastSignedIn) {
      values.lastSignedIn = new Date();
    }

    if (Object.keys(updateSet).length === 0) {
      updateSet.lastSignedIn = new Date();
    }

    await db.insert(users).values(values).onDuplicateKeyUpdate({
      set: updateSet,
    });
  } catch (error) {
    console.error("[Database] Failed to upsert user:", error);
    throw error;
  }
}

export async function getUserByOpenId(openId: string) {
  const db = await getDb();
  if (!db) {
    console.warn("[Database] Cannot get user: database not available");
    return undefined;
  }

  const result = await db.select().from(users).where(eq(users.openId, openId)).limit(1);

  return result.length > 0 ? result[0] : undefined;
}

export async function getUserById(userId: number) {
  const db = await getDb();
  if (!db) {
    console.warn("[Database] Cannot get user: database not available");
    return undefined;
  }

  const result = await db.select().from(users).where(eq(users.id, userId)).limit(1);

  return result.length > 0 ? result[0] : undefined;
}

// ============================================================================
// DETECTION SESSION FUNCTIONS
// ============================================================================

export async function createDetectionSession(session: InsertDetectionSession) {
  const db = await getDb();
  if (!db) {
    throw new Error("Database not available");
  }

  const result = await db.insert(detectionSessions).values(session);
  return result;
}

export async function updateDetectionSession(sessionId: number, updates: Partial<InsertDetectionSession>) {
  const db = await getDb();
  if (!db) {
    throw new Error("Database not available");
  }

  await db.update(detectionSessions)
    .set(updates)
    .where(eq(detectionSessions.id, sessionId));
}

export async function getDetectionSessions(userId: number, limit: number = 10) {
  const db = await getDb();
  if (!db) {
    return [];
  }

  return await db.select()
    .from(detectionSessions)
    .where(eq(detectionSessions.userId, userId))
    .orderBy(desc(detectionSessions.startTime))
    .limit(limit);
}

export async function getDetectionSessionById(sessionId: number) {
  const db = await getDb();
  if (!db) {
    return undefined;
  }

  const result = await db.select()
    .from(detectionSessions)
    .where(eq(detectionSessions.id, sessionId))
    .limit(1);

  return result.length > 0 ? result[0] : undefined;
}

// ============================================================================
// DETECTION RESULT FUNCTIONS
// ============================================================================

export async function createDetectionResult(result: InsertDetectionResult) {
  const db = await getDb();
  if (!db) {
    throw new Error("Database not available");
  }

  await db.insert(detectionResults).values(result);
}

export async function getDetectionResults(sessionId: number) {
  const db = await getDb();
  if (!db) {
    return [];
  }

  return await db.select()
    .from(detectionResults)
    .where(eq(detectionResults.sessionId, sessionId))
    .orderBy(detectionResults.frameNumber);
}

export async function getUserDetectionResults(userId: number, limit: number = 100) {
  const db = await getDb();
  if (!db) {
    return [];
  }

  return await db.select()
    .from(detectionResults)
    .where(eq(detectionResults.userId, userId))
    .orderBy(desc(detectionResults.timestamp))
    .limit(limit);
}

// ============================================================================
// UPLOADED FILE FUNCTIONS
// ============================================================================

export async function createUploadedFile(file: InsertUploadedFile) {
  const db = await getDb();
  if (!db) {
    throw new Error("Database not available");
  }

  const result = await db.insert(uploadedFiles).values(file);
  return result;
}

export async function getUserUploadedFiles(userId: number, limit: number = 20) {
  const db = await getDb();
  if (!db) {
    return [];
  }

  return await db.select()
    .from(uploadedFiles)
    .where(eq(uploadedFiles.userId, userId))
    .orderBy(desc(uploadedFiles.uploadedAt))
    .limit(limit);
}

// ============================================================================
// USER STATISTICS FUNCTIONS
// ============================================================================

export async function getOrCreateUserStatistics(userId: number) {
  const db = await getDb();
  if (!db) {
    throw new Error("Database not available");
  }

  let stats = await db.select()
    .from(userStatistics)
    .where(eq(userStatistics.userId, userId))
    .limit(1);

  if (stats.length === 0) {
    await db.insert(userStatistics).values({
      userId,
      totalSessions: 0,
      totalDetections: 0,
    });

    stats = await db.select()
      .from(userStatistics)
      .where(eq(userStatistics.userId, userId))
      .limit(1);
  }

  return stats[0];
}

export async function updateUserStatistics(userId: number, updates: Partial<InsertUserStatistics>) {
  const db = await getDb();
  if (!db) {
    throw new Error("Database not available");
  }

  await db.update(userStatistics)
    .set(updates)
    .where(eq(userStatistics.userId, userId));
}

// ============================================================================
// MODEL METADATA FUNCTIONS
// ============================================================================

export async function createModelMetadata(metadata: InsertModelMetadata) {
  const db = await getDb();
  if (!db) {
    throw new Error("Database not available");
  }

  await db.insert(modelMetadata).values(metadata);
}

export async function getActiveModel() {
  const db = await getDb();
  if (!db) {
    return undefined;
  }

  const result = await db.select()
    .from(modelMetadata)
    .where(eq(modelMetadata.isActive, 1))
    .limit(1);

  return result.length > 0 ? result[0] : undefined;
}

export async function getAllModels() {
  const db = await getDb();
  if (!db) {
    return [];
  }

  return await db.select()
    .from(modelMetadata)
    .orderBy(desc(modelMetadata.trainingDate));
}

// TODO: add feature queries here as your schema grows.
