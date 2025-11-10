import { protectedProcedure, router } from "../_core/trpc";
import { z } from "zod";
import {
  createDetectionSession,
  updateDetectionSession,
  createDetectionResult,
  getDetectionSessionById,
  updateUserStatistics,
  getOrCreateUserStatistics,
} from "../db";
import { TRPCError } from "@trpc/server";

export const predictionRouter = router({
  /**
   * Create a new detection session
   */
  createSession: protectedProcedure
    .input(
      z.object({
        sessionType: z.enum(["image", "webcam", "video"]),
        notes: z.string().optional(),
      })
    )
    .mutation(async ({ input, ctx }) => {
      try {
        const result = await createDetectionSession({
          userId: ctx.user.id,
          sessionType: input.sessionType,
          notes: input.notes,
        });

        return {
          success: true,
          sessionId: (result as any).insertId || 0,
        };
      } catch (error) {
        throw new TRPCError({
          code: "INTERNAL_SERVER_ERROR",
          message: `Failed to create session: ${String(error)}`,
        });
      }
    }),

  /**
   * Process a single image and return predictions
   */
  predictImage: protectedProcedure
    .input(
      z.object({
        imageBase64: z.string(),
        sessionId: z.number().optional(),
      })
    )
    .mutation(async ({ input, ctx }) => {
      try {
        // Call Python inference server
        const response = await fetch("http://localhost:5000/api/predict/base64", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            image: input.imageBase64,
          }),
        });

        if (!response.ok) {
          throw new Error("Inference server error");
        }

        const result = await response.json();

        // Store result if session provided
        if (input.sessionId && !result.error) {
          await createDetectionResult({
            sessionId: input.sessionId,
            userId: ctx.user.id,
            frameNumber: 1,
            dominantEmotion: result.dominant_emotion,
            emotionConfidence: String(result.confidence),
            stressScore: String(result.stress_score),
            stressLevel: result.stress_level,
            emotionProbabilities: result.emotion_probabilities,
            facesDetected: result.faces_detected,
          });
        }

        return {
          success: !result.error,
          result,
        };
      } catch (error) {
        throw new TRPCError({
          code: "INTERNAL_SERVER_ERROR",
          message: `Prediction failed: ${String(error)}`,
        });
      }
    }),

  /**
   * Store a single prediction result
   */
  storeResult: protectedProcedure
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
        // Verify session ownership
        const session = await getDetectionSessionById(input.sessionId);
        if (!session || session.userId !== ctx.user.id) {
          throw new TRPCError({
            code: "FORBIDDEN",
            message: "Session not found or unauthorized",
          });
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
        if (error instanceof TRPCError) throw error;
        throw new TRPCError({
          code: "INTERNAL_SERVER_ERROR",
          message: `Failed to store result: ${String(error)}`,
        });
      }
    }),

  /**
   * Finalize session and update statistics
   */
  finalizeSession: protectedProcedure
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
        // Verify session ownership
        const session = await getDetectionSessionById(input.sessionId);
        if (!session || session.userId !== ctx.user.id) {
          throw new TRPCError({
            code: "FORBIDDEN",
            message: "Session not found or unauthorized",
          });
        }

        // Update session
        await updateDetectionSession(input.sessionId, {
          duration: input.duration,
          totalFrames: input.totalFrames,
          averageStress: input.averageStress
            ? String(input.averageStress)
            : undefined,
          maxStress: input.maxStress ? String(input.maxStress) : undefined,
          minStress: input.minStress ? String(input.minStress) : undefined,
          dominantEmotion: input.dominantEmotion,
          endTime: new Date(),
        });

        // Update user statistics
        const stats = await getOrCreateUserStatistics(ctx.user.id);
        await updateUserStatistics(ctx.user.id, {
          totalSessions: (stats.totalSessions || 0) + 1,
          totalDetections:
            (stats.totalDetections || 0) + (input.totalFrames || 0),
          averageStress: input.averageStress
            ? String(input.averageStress)
            : undefined,
          mostCommonEmotion: input.dominantEmotion,
          lastDetectionTime: new Date(),
        });

        return { success: true };
      } catch (error) {
        if (error instanceof TRPCError) throw error;
        throw new TRPCError({
          code: "INTERNAL_SERVER_ERROR",
          message: `Failed to finalize session: ${String(error)}`,
        });
      }
    }),

  /**
   * Get model information
   */
  getModelInfo: protectedProcedure.query(async () => {
    try {
      const response = await fetch("http://localhost:5000/api/model/info");
      if (!response.ok) {
        return {
          loaded: false,
          message: "Inference server unavailable",
        };
      }

      const info = await response.json();
      return info;
    } catch (error) {
      return {
        loaded: false,
        message: "Failed to fetch model info",
      };
    }
  }),

  /**
   * Check inference server health
   */
  checkHealth: protectedProcedure.query(async () => {
    try {
      const response = await fetch("http://localhost:5000/health");
      if (!response.ok) {
        return {
          healthy: false,
          message: "Inference server returned error",
        };
      }

      const health = await response.json();
      return {
        healthy: true,
        ...health,
      };
    } catch (error) {
      return {
        healthy: false,
        message: "Inference server unreachable",
      };
    }
  }),
});
