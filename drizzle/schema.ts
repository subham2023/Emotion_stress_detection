import { int, mysqlEnum, mysqlTable, text, timestamp, varchar, decimal, json } from "drizzle-orm/mysql-core";

/**
 * Core user table backing auth flow.
 * Extended with emotion detection specific tables.
 */
export const users = mysqlTable("users", {
  id: int("id").autoincrement().primaryKey(),
  openId: varchar("openId", { length: 64 }).notNull().unique(),
  name: text("name"),
  email: varchar("email", { length: 320 }),
  loginMethod: varchar("loginMethod", { length: 64 }),
  role: mysqlEnum("role", ["user", "admin"]).default("user").notNull(),
  createdAt: timestamp("createdAt").defaultNow().notNull(),
  updatedAt: timestamp("updatedAt").defaultNow().onUpdateNow().notNull(),
  lastSignedIn: timestamp("lastSignedIn").defaultNow().notNull(),
});

export type User = typeof users.$inferSelect;
export type InsertUser = typeof users.$inferInsert;

/**
 * Detection sessions - tracks user sessions for emotion detection
 */
export const detectionSessions = mysqlTable("detection_sessions", {
  id: int("id").autoincrement().primaryKey(),
  userId: int("userId").notNull(),
  sessionType: mysqlEnum("sessionType", ["image", "webcam", "video"]).notNull(),
  startTime: timestamp("startTime").defaultNow().notNull(),
  endTime: timestamp("endTime"),
  duration: int("duration"), // in seconds
  totalFrames: int("totalFrames"),
  averageStress: decimal("averageStress", { precision: 5, scale: 2 }),
  maxStress: decimal("maxStress", { precision: 5, scale: 2 }),
  minStress: decimal("minStress", { precision: 5, scale: 2 }),
  dominantEmotion: varchar("dominantEmotion", { length: 20 }),
  notes: text("notes"),
  createdAt: timestamp("createdAt").defaultNow().notNull(),
  updatedAt: timestamp("updatedAt").defaultNow().onUpdateNow().notNull(),
});

export type DetectionSession = typeof detectionSessions.$inferSelect;
export type InsertDetectionSession = typeof detectionSessions.$inferInsert;

/**
 * Detection results - individual emotion detection results
 */
export const detectionResults = mysqlTable("detection_results", {
  id: int("id").autoincrement().primaryKey(),
  sessionId: int("sessionId").notNull(),
  userId: int("userId").notNull(),
  frameNumber: int("frameNumber"),
  dominantEmotion: varchar("dominantEmotion", { length: 20 }).notNull(),
  emotionConfidence: decimal("emotionConfidence", { precision: 5, scale: 4 }).notNull(),
  stressScore: decimal("stressScore", { precision: 5, scale: 2 }).notNull(),
  stressLevel: mysqlEnum("stressLevel", ["low", "moderate", "high", "critical"]).notNull(),
  emotionProbabilities: json("emotionProbabilities"), // JSON with all 7 emotions
  facesDetected: int("facesDetected"),
  timestamp: timestamp("timestamp").defaultNow().notNull(),
  createdAt: timestamp("createdAt").defaultNow().notNull(),
});

export type DetectionResult = typeof detectionResults.$inferSelect;
export type InsertDetectionResult = typeof detectionResults.$inferInsert;

/**
 * Uploaded files - tracks user-uploaded images and videos
 */
export const uploadedFiles = mysqlTable("uploaded_files", {
  id: int("id").autoincrement().primaryKey(),
  userId: int("userId").notNull(),
  fileName: varchar("fileName", { length: 255 }).notNull(),
  fileType: mysqlEnum("fileType", ["image", "video"]).notNull(),
  fileSize: int("fileSize"), // in bytes
  s3Key: varchar("s3Key", { length: 255 }).notNull(),
  s3Url: text("s3Url"),
  sessionId: int("sessionId"),
  uploadedAt: timestamp("uploadedAt").defaultNow().notNull(),
  createdAt: timestamp("createdAt").defaultNow().notNull(),
});

export type UploadedFile = typeof uploadedFiles.$inferSelect;
export type InsertUploadedFile = typeof uploadedFiles.$inferInsert;

/**
 * User statistics - aggregate statistics per user
 */
export const userStatistics = mysqlTable("user_statistics", {
  id: int("id").autoincrement().primaryKey(),
  userId: int("userId").notNull().unique(),
  totalSessions: int("totalSessions").default(0),
  totalDetections: int("totalDetections").default(0),
  averageStress: decimal("averageStress", { precision: 5, scale: 2 }),
  mostCommonEmotion: varchar("mostCommonEmotion", { length: 20 }),
  lastDetectionTime: timestamp("lastDetectionTime"),
  updatedAt: timestamp("updatedAt").defaultNow().onUpdateNow().notNull(),
  createdAt: timestamp("createdAt").defaultNow().notNull(),
});

export type UserStatistics = typeof userStatistics.$inferSelect;
export type InsertUserStatistics = typeof userStatistics.$inferInsert;

/**
 * Model metadata - tracks model versions and performance
 */
export const modelMetadata = mysqlTable("model_metadata", {
  id: int("id").autoincrement().primaryKey(),
  modelName: varchar("modelName", { length: 100 }).notNull(),
  modelType: mysqlEnum("modelType", ["custom_cnn", "resnet50", "mobilenetv2", "vgg16"]).notNull(),
  version: varchar("version", { length: 50 }).notNull(),
  accuracy: decimal("accuracy", { precision: 5, scale: 4 }),
  trainingDate: timestamp("trainingDate"),
  isActive: int("isActive").default(0), // 0 or 1 for boolean
  description: text("description"),
  createdAt: timestamp("createdAt").defaultNow().notNull(),
  updatedAt: timestamp("updatedAt").defaultNow().onUpdateNow().notNull(),
});

export type ModelMetadata = typeof modelMetadata.$inferSelect;
export type InsertModelMetadata = typeof modelMetadata.$inferInsert;
