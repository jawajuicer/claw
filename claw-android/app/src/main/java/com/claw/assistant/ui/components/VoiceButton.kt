package com.claw.assistant.ui.components

import androidx.compose.animation.animateColorAsState
import androidx.compose.animation.core.FastOutSlowInEasing
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Mic
import androidx.compose.material.icons.filled.MicOff
import androidx.compose.material.icons.filled.Stop
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.scale
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.unit.dp
import com.claw.assistant.network.AudioStreamState
import com.claw.assistant.ui.theme.ClawAccent
import com.claw.assistant.ui.theme.ClawAccentDim
import com.claw.assistant.ui.theme.ClawBackground
import com.claw.assistant.ui.theme.ClawError
import com.claw.assistant.ui.theme.ClawSurfaceVariant
import com.claw.assistant.ui.theme.ClawSuccess
import com.claw.assistant.ui.theme.ClawTextSecondary

@Composable
fun VoiceButton(
    state: AudioStreamState,
    wakeWordAvailable: Boolean,
    onPushToTalkStart: () -> Unit,
    onPushToTalkStop: () -> Unit,
    onTap: () -> Unit,
    modifier: Modifier = Modifier
) {
    val isRecording = state == AudioStreamState.STREAMING
    val isProcessing = state == AudioStreamState.PLAYING_TTS
    val isListening = state == AudioStreamState.LISTENING

    // Pulsing animation when recording
    val infiniteTransition = rememberInfiniteTransition(label = "pulse")
    val pulseScale by infiniteTransition.animateFloat(
        initialValue = 1f,
        targetValue = 1.15f,
        animationSpec = infiniteRepeatable(
            animation = tween(800, easing = FastOutSlowInEasing),
            repeatMode = RepeatMode.Reverse
        ),
        label = "pulseScale"
    )

    val scale by animateFloatAsState(
        targetValue = when {
            isRecording -> pulseScale
            isProcessing -> 0.95f
            else -> 1f
        },
        animationSpec = tween(200),
        label = "scale"
    )

    val backgroundColor by animateColorAsState(
        targetValue = when {
            isRecording -> ClawError
            isProcessing -> ClawAccentDim
            isListening -> ClawAccent
            else -> ClawSurfaceVariant
        },
        animationSpec = tween(300),
        label = "bgColor"
    )

    val borderColor by animateColorAsState(
        targetValue = when {
            isRecording -> ClawError.copy(alpha = 0.5f)
            isListening && wakeWordAvailable -> ClawSuccess.copy(alpha = 0.3f)
            else -> Color.Transparent
        },
        animationSpec = tween(300),
        label = "borderColor"
    )

    val icon = when {
        isRecording -> Icons.Default.Stop
        isProcessing -> Icons.Default.Mic
        !wakeWordAvailable && state == AudioStreamState.IDLE -> Icons.Default.MicOff
        else -> Icons.Default.Mic
    }

    val iconTint = when {
        isRecording -> Color.White
        isProcessing -> ClawTextSecondary
        else -> ClawBackground
    }

    Box(
        contentAlignment = Alignment.Center,
        modifier = modifier
            .size(72.dp)
            .scale(scale)
            .clip(CircleShape)
            .background(
                brush = Brush.radialGradient(
                    colors = listOf(
                        backgroundColor,
                        backgroundColor.copy(alpha = 0.8f)
                    )
                )
            )
            .border(
                width = 3.dp,
                color = borderColor,
                shape = CircleShape
            )
            .pointerInput(state) {
                detectTapGestures(
                    onPress = {
                        if (!isProcessing) {
                            onPushToTalkStart()
                            val released = tryAwaitRelease()
                            if (released) {
                                onPushToTalkStop()
                            }
                        }
                    },
                    onTap = {
                        onTap()
                    }
                )
            }
    ) {
        Icon(
            imageVector = icon,
            contentDescription = when {
                isRecording -> "Stop recording"
                isProcessing -> "Processing"
                else -> "Push to talk"
            },
            modifier = Modifier.size(32.dp),
            tint = iconTint
        )
    }
}
