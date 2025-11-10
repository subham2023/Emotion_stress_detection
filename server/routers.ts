import { COOKIE_NAME } from "@shared/const";
import { getSessionCookieOptions } from "./_core/cookies";
import { systemRouter } from "./_core/systemRouter";
import { publicProcedure, protectedProcedure, router } from "./_core/trpc";
import { predictionRouter } from "./routers/prediction";
import { z } from "zod";
import {
  createDetectionSession,
  updateDetectionSession,
  getDetectionSessions,
  getDetectionSessionById,
  createDetectionResult,
  getDetectionResults,
  getUserDetectionResults,
  createUploadedFile,
  getUserUploadedFiles,
  getOrCreateUserStatistics,
  updateUserStatistics,
  getActiveModel,
  getAllModels,
} from "./db";

export const appRouter = router({
  system: systemRouter,
  prediction: predictionRouter,
  auth: router({
    me: publicProcedure.query(opts => opts.ctx.user),
    logout: publicProcedure.mutation(({ ctx }) => {
      const cookieOptions = getSessionCookieOptions(ctx.req);
      ctx.res.clearCookie(COOKIE_NAME, { ...cookieOptions, maxAge: -1 });
      return {
        success: true,
      } as const;
    }),
  }),

  // ========================================================================
  // DETECTION SESSION ENDPOINTS
  // ========================================================================

  session: router({
    create: protectedProcedure
      .input(
        z.object({
          sessionType: z.enum(["image", "webcam", "video"]),
          notes: z.string().optional(),
        })
      )
      .mutation(async ({ input, ctx }) => {
        try {
          await createDetectionSession({
            userId: ctx.user.id,
            sessionType: input.sessionType,
            notes: input.notes,
          });

          return {
            success: true,
          };
        } catch (error) {
          throw new Error(`Failed to create session: ${String(error)}`);
        }
      }),

    list: protectedProcedure
      .input(
        z.object({
          limit: z.number().default(10),
        })
      )
      .query(async ({ input, ctx }) => {
        try {
          const sessions = await getDetectionSessions(ctx.user.id, input.limit);
          return sessions;
        } catch (error) {
          throw new Error(`Failed to fetch sessions: ${String(error)}`);
        }
      }),

    get: protectedProcedure
      .input(z.object({ sessionId: z.number() }))
      .query(async ({ input, ctx }) => {
        try {
          const session = await getDetectionSessionById(input.sessionId);

          if (!session || session.userId !== ctx.user.id) {
            throw new Error("Session not found or unauthorized");
          }

          return session || null;
        } catch (error) {
          throw new Error(`Failed to fetch session: ${String(error)}`);
        }
      }),

    update: protectedProcedure
      .input(
        z.object({
          sessionId: z.number(),
          duration: z.number().optional(),
          totalFrames: z.number().optional(),
          averageStress: z.number().optional(),
          maxStress: z.number().optional(),
          minStress: z.number().optional(),
          dominantEmotion: z.string().optional(),
        })
      )
      .mutation(async ({ input, ctx }) => {
        try {
          const session = await getDetectionSessionById(input.sessionId);

          if (!session || session.userId !== ctx.user.id) {
            throw new Error("Session not found or unauthorized");
          }

          await updateDetectionSession(input.sessionId, {
            duration: input.duration,
            totalFrames: input.totalFrames,
            averageStress: input.averageStress ? String(input.averageStress) : undefined,
            maxStress: input.maxStress ? String(input.maxStress) : undefined,
            minStress: input.minStress ? String(input.minStress) : undefined,
            dominantEmotion: input.dominantEmotion,
            endTime: new Date(),
          });

          return { success: true };
        } catch (error) {
          throw new Error(`Failed to update session: ${String(error)}`);
        }
      }),
  }),

  // ========================================================================
  // DETECTION RESULT ENDPOINTS
  // ========================================================================

  result: router({
    create: protectedProcedure
      .input(
        z.object({
          sessionId: z.number(),
          frameNumber: z.number().optional(),
          dominantEmotion: z.string(),
          emotionConfidence: z.number(),
          stressScore: z.number(),
          stressLevel: z.enum(["low", "moderate", "high", "critical"]),
          emotionProbabilities: z.record(z.string(), z.number()).optional(),
          facesDetected: z.number().optional(),
        })
      )
      .mutation(async ({ input, ctx }) => {
        try {
          const session = await getDetectionSessionById(input.sessionId);

          if (!session || session.userId !== ctx.user.id) {
            throw new Error("Session not found or unauthorized");
          }

          await createDetectionResult({
            sessionId: input.sessionId,
            userId: ctx.user.id,
            frameNumber: input.frameNumber,
            dominantEmotion: input.dominantEmotion,
            emotionConfidence: String(input.emotionConfidence),
            stressScore: String(input.stressScore),
            stressLevel: input.stressLevel,
            emotionProbabilities: input.emotionProbabilities,
            facesDetected: input.facesDetected,
          });

          return { success: true };
        } catch (error) {
          throw new Error(`Failed to create result: ${String(error)}`);
        }
      }),

    list: protectedProcedure
      .input(z.object({ sessionId: z.number() }))
      .query(async ({ input, ctx }) => {
        try {
          const session = await getDetectionSessionById(input.sessionId);

          if (!session || session.userId !== ctx.user.id) {
            throw new Error("Session not found or unauthorized");
          }

          const results = await getDetectionResults(input.sessionId);
          return results;
        } catch (error) {
          throw new Error(`Failed to fetch results: ${String(error)}`);
        }
      }),

    recent: protectedProcedure
      .input(z.object({ limit: z.number().default(50) }))
      .query(async ({ input, ctx }) => {
        try {
          const results = await getUserDetectionResults(ctx.user.id, input.limit);
          return results;
        } catch (error) {
          throw new Error(`Failed to fetch recent results: ${String(error)}`);
        }
      }),
  }),

  // ========================================================================
  // FILE UPLOAD ENDPOINTS
  // ========================================================================

  file: router({
    create: protectedProcedure
      .input(
        z.object({
          fileName: z.string(),
          fileType: z.enum(["image", "video"]),
          fileSize: z.number(),
          s3Key: z.string(),
          s3Url: z.string(),
          sessionId: z.number().optional(),
        })
      )
      .mutation(async ({ input, ctx }) => {
        try {
          await createUploadedFile({
            userId: ctx.user.id,
            fileName: input.fileName,
            fileType: input.fileType,
            fileSize: input.fileSize,
            s3Key: input.s3Key,
            s3Url: input.s3Url,
            sessionId: input.sessionId,
          });

          return { success: true };
        } catch (error) {
          throw new Error(`Failed to create file record: ${String(error)}`);
        }
      }),

    list: protectedProcedure
      .input(z.object({ limit: z.number().default(20) }))
      .query(async ({ input, ctx }) => {
        try {
          const files = await getUserUploadedFiles(ctx.user.id, input.limit);
          return files;
        } catch (error) {
          throw new Error(`Failed to fetch files: ${String(error)}`);
        }
      }),
  }),

  // ========================================================================
  // STATISTICS ENDPOINTS
  // ========================================================================

  stats: router({
    get: protectedProcedure.query(async ({ ctx }) => {
      try {
        const stats = await getOrCreateUserStatistics(ctx.user.id);
        return stats;
      } catch (error) {
        throw new Error(`Failed to fetch statistics: ${String(error)}`);
      }
    }),

    update: protectedProcedure
      .input(
        z.object({
          totalSessions: z.number().optional(),
          totalDetections: z.number().optional(),
          averageStress: z.number().optional(),
          mostCommonEmotion: z.string().optional(),
        })
      )
      .mutation(async ({ input, ctx }) => {
        try {
          await updateUserStatistics(ctx.user.id, {
            totalSessions: input.totalSessions,
            totalDetections: input.totalDetections,
            averageStress: input.averageStress ? String(input.averageStress) : undefined,
            mostCommonEmotion: input.mostCommonEmotion,
            lastDetectionTime: new Date(),
          });

          return { success: true };
        } catch (error) {
          throw new Error(`Failed to update statistics: ${String(error)}`);
        }
      }),
  }),

  // ========================================================================
  // MODEL ENDPOINTS
  // ========================================================================

  model: router({
    active: publicProcedure.query(async () => {
      try {
        const model = await getActiveModel();
        return model;
      } catch (error) {
        throw new Error(`Failed to fetch active model: ${String(error)}`);
      }
    }),

    list: publicProcedure.query(async () => {
      try {
        const models = await getAllModels();
        return models;
      } catch (error) {
        throw new Error(`Failed to fetch models: ${String(error)}`);
      }
    }),
  }),
});

export type AppRouter = typeof appRouter;
