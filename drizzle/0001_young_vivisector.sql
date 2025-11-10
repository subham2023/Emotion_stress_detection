CREATE TABLE `detection_results` (
	`id` int AUTO_INCREMENT NOT NULL,
	`sessionId` int NOT NULL,
	`userId` int NOT NULL,
	`frameNumber` int,
	`dominantEmotion` varchar(20) NOT NULL,
	`emotionConfidence` decimal(5,4) NOT NULL,
	`stressScore` decimal(5,2) NOT NULL,
	`stressLevel` enum('low','moderate','high','critical') NOT NULL,
	`emotionProbabilities` json,
	`facesDetected` int,
	`timestamp` timestamp NOT NULL DEFAULT (now()),
	`createdAt` timestamp NOT NULL DEFAULT (now()),
	CONSTRAINT `detection_results_id` PRIMARY KEY(`id`)
);
--> statement-breakpoint
CREATE TABLE `detection_sessions` (
	`id` int AUTO_INCREMENT NOT NULL,
	`userId` int NOT NULL,
	`sessionType` enum('image','webcam','video') NOT NULL,
	`startTime` timestamp NOT NULL DEFAULT (now()),
	`endTime` timestamp,
	`duration` int,
	`totalFrames` int,
	`averageStress` decimal(5,2),
	`maxStress` decimal(5,2),
	`minStress` decimal(5,2),
	`dominantEmotion` varchar(20),
	`notes` text,
	`createdAt` timestamp NOT NULL DEFAULT (now()),
	`updatedAt` timestamp NOT NULL DEFAULT (now()) ON UPDATE CURRENT_TIMESTAMP,
	CONSTRAINT `detection_sessions_id` PRIMARY KEY(`id`)
);
--> statement-breakpoint
CREATE TABLE `model_metadata` (
	`id` int AUTO_INCREMENT NOT NULL,
	`modelName` varchar(100) NOT NULL,
	`modelType` enum('custom_cnn','resnet50','mobilenetv2','vgg16') NOT NULL,
	`version` varchar(50) NOT NULL,
	`accuracy` decimal(5,4),
	`trainingDate` timestamp,
	`isActive` int DEFAULT 0,
	`description` text,
	`createdAt` timestamp NOT NULL DEFAULT (now()),
	`updatedAt` timestamp NOT NULL DEFAULT (now()) ON UPDATE CURRENT_TIMESTAMP,
	CONSTRAINT `model_metadata_id` PRIMARY KEY(`id`)
);
--> statement-breakpoint
CREATE TABLE `uploaded_files` (
	`id` int AUTO_INCREMENT NOT NULL,
	`userId` int NOT NULL,
	`fileName` varchar(255) NOT NULL,
	`fileType` enum('image','video') NOT NULL,
	`fileSize` int,
	`s3Key` varchar(255) NOT NULL,
	`s3Url` text,
	`sessionId` int,
	`uploadedAt` timestamp NOT NULL DEFAULT (now()),
	`createdAt` timestamp NOT NULL DEFAULT (now()),
	CONSTRAINT `uploaded_files_id` PRIMARY KEY(`id`)
);
--> statement-breakpoint
CREATE TABLE `user_statistics` (
	`id` int AUTO_INCREMENT NOT NULL,
	`userId` int NOT NULL,
	`totalSessions` int DEFAULT 0,
	`totalDetections` int DEFAULT 0,
	`averageStress` decimal(5,2),
	`mostCommonEmotion` varchar(20),
	`lastDetectionTime` timestamp,
	`updatedAt` timestamp NOT NULL DEFAULT (now()) ON UPDATE CURRENT_TIMESTAMP,
	`createdAt` timestamp NOT NULL DEFAULT (now()),
	CONSTRAINT `user_statistics_id` PRIMARY KEY(`id`),
	CONSTRAINT `user_statistics_userId_unique` UNIQUE(`userId`)
);
