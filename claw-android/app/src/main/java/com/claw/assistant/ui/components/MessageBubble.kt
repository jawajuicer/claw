package com.claw.assistant.ui.components

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.fadeIn
import androidx.compose.animation.slideInVertically
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Build
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.platform.LocalConfiguration
import androidx.compose.ui.text.font.FontStyle
import androidx.compose.ui.unit.dp
import com.claw.assistant.ui.theme.ClawAccent
import com.claw.assistant.ui.theme.ClawAssistantBubble
import com.claw.assistant.ui.theme.ClawError
import com.claw.assistant.ui.theme.ClawTextSecondary
import com.claw.assistant.ui.theme.ClawUserBubble

data class ChatMessage(
    val id: String,
    val content: String,
    val isUser: Boolean,
    val toolsUsed: List<String> = emptyList(),
    val timestamp: Long = System.currentTimeMillis(),
    val status: String = "sent"  // "sent", "failed", "pending"
)

@Composable
fun MessageBubble(
    message: ChatMessage,
    onRetry: (() -> Unit)? = null,
    modifier: Modifier = Modifier
) {
    val maxWidth = (LocalConfiguration.current.screenWidthDp * 0.8).dp

    AnimatedVisibility(
        visible = true,
        enter = fadeIn() + slideInVertically(initialOffsetY = { it / 2 })
    ) {
        Column(
            modifier = modifier
                .fillMaxWidth()
                .padding(horizontal = 12.dp, vertical = 4.dp),
            horizontalAlignment = if (message.isUser) Alignment.End else Alignment.Start
        ) {
            // Sender label
            Text(
                text = if (message.isUser) "You" else "The Claw",
                style = MaterialTheme.typography.labelSmall,
                color = if (message.isUser) ClawAccent else ClawTextSecondary,
                modifier = Modifier.padding(bottom = 2.dp, start = 4.dp, end = 4.dp)
            )

            // Message bubble
            Surface(
                shape = RoundedCornerShape(
                    topStart = 16.dp,
                    topEnd = 16.dp,
                    bottomStart = if (message.isUser) 16.dp else 4.dp,
                    bottomEnd = if (message.isUser) 4.dp else 16.dp
                ),
                color = if (message.isUser) ClawUserBubble else ClawAssistantBubble,
                modifier = Modifier.widthIn(max = maxWidth)
            ) {
                Column(modifier = Modifier.padding(12.dp)) {
                    Text(
                        text = message.content,
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurface
                    )

                    // Tools used indicator
                    if (message.toolsUsed.isNotEmpty()) {
                        Spacer(modifier = Modifier.height(8.dp))
                        Row(
                            verticalAlignment = Alignment.CenterVertically,
                            horizontalArrangement = Arrangement.spacedBy(4.dp)
                        ) {
                            Icon(
                                imageVector = Icons.Default.Build,
                                contentDescription = "Tools used",
                                modifier = Modifier.size(12.dp),
                                tint = ClawTextSecondary
                            )
                            Text(
                                text = message.toolsUsed.joinToString(", "),
                                style = MaterialTheme.typography.bodySmall,
                                fontStyle = FontStyle.Italic,
                                color = ClawTextSecondary
                            )
                        }
                    }
                }
            }

            // Retry button for failed messages
            if (message.status == "failed" && onRetry != null) {
                TextButton(
                    onClick = onRetry,
                    modifier = Modifier.padding(start = 4.dp, top = 2.dp)
                ) {
                    Icon(
                        imageVector = Icons.Default.Refresh,
                        contentDescription = "Retry",
                        modifier = Modifier.size(14.dp),
                        tint = ClawError
                    )
                    Spacer(modifier = Modifier.size(4.dp))
                    Text(
                        text = "Retry",
                        style = MaterialTheme.typography.labelSmall,
                        color = ClawError
                    )
                }
            }

            // Pending label
            if (message.status == "pending") {
                Text(
                    text = "Pending...",
                    style = MaterialTheme.typography.labelSmall,
                    color = ClawTextSecondary,
                    modifier = Modifier.padding(start = 4.dp, top = 2.dp)
                )
            }
        }
    }
}

@Composable
fun TypingIndicator(modifier: Modifier = Modifier) {
    Column(
        modifier = modifier
            .fillMaxWidth()
            .padding(horizontal = 12.dp, vertical = 4.dp),
        horizontalAlignment = Alignment.Start
    ) {
        Text(
            text = "The Claw",
            style = MaterialTheme.typography.labelSmall,
            color = ClawTextSecondary,
            modifier = Modifier.padding(bottom = 2.dp, start = 4.dp)
        )

        Surface(
            shape = RoundedCornerShape(16.dp, 16.dp, 16.dp, 4.dp),
            color = ClawAssistantBubble
        ) {
            Row(
                modifier = Modifier.padding(horizontal = 16.dp, vertical = 12.dp),
                horizontalArrangement = Arrangement.spacedBy(4.dp)
            ) {
                repeat(3) {
                    Box(
                        modifier = Modifier
                            .size(8.dp)
                            .clip(CircleShape)
                            .background(ClawTextSecondary)
                    )
                }
            }
        }
    }
}
