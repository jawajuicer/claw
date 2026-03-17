# Claw Assistant ProGuard Rules

# ONNX Runtime
-keep class ai.onnxruntime.** { *; }
-dontwarn ai.onnxruntime.**

# OkHttp
-dontwarn okhttp3.internal.platform.**
-dontwarn org.conscrypt.**
-dontwarn org.bouncycastle.**
-dontwarn org.openjsse.**

# Keep JSON classes used via reflection
-keep class org.json.** { *; }

# Media3 / ExoPlayer
-keep class androidx.media3.** { *; }
-dontwarn androidx.media3.**

# Room
-keep class * extends androidx.room.RoomDatabase { *; }
-keep @androidx.room.Entity class * { *; }
-keep @androidx.room.Dao interface * { *; }

# WorkManager
-keep class * extends androidx.work.Worker { *; }
-keep class * extends androidx.work.CoroutineWorker { *; }
-keep class * extends androidx.work.ListenableWorker { *; }

# OkHttp SSE
-keep class okhttp3.sse.** { *; }
-dontwarn okhttp3.sse.**

# Keep service declarations
-keep class com.claw.assistant.service.WakeWordService { *; }
-keep class com.claw.assistant.service.BootReceiver { *; }
-keep class com.claw.assistant.media.ClawMediaService { *; }
-keep class com.claw.assistant.service.NotificationService { *; }
-keep class com.claw.assistant.data.local.MessageQueueWorker { *; }
