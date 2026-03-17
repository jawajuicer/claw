package com.claw.assistant.ui.components

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.slideInVertically
import androidx.compose.animation.slideOutVertically
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.MusicNote
import androidx.compose.material.icons.filled.Pause
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import com.claw.assistant.media.NowPlaying
import com.claw.assistant.ui.theme.ClawAccent
import com.claw.assistant.ui.theme.ClawBorder
import com.claw.assistant.ui.theme.ClawSurface
import com.claw.assistant.ui.theme.ClawSurfaceVariant
import com.claw.assistant.ui.theme.ClawTextPrimary
import com.claw.assistant.ui.theme.ClawTextSecondary
import kotlinx.coroutines.delay

@Composable
fun MiniPlayer(
    nowPlaying: NowPlaying?,
    onPlayPause: () -> Unit,
    onStop: () -> Unit,
    onGetPosition: () -> Pair<Long, Long>,  // Returns (position, duration)
    modifier: Modifier = Modifier
) {
    AnimatedVisibility(
        visible = nowPlaying != null,
        enter = slideInVertically(initialOffsetY = { it }),
        exit = slideOutVertically(targetOffsetY = { it }),
        modifier = modifier
    ) {
        nowPlaying?.let { playing ->
            var progress by remember { mutableFloatStateOf(0f) }

            // Update progress periodically
            LaunchedEffect(playing.isPlaying) {
                while (playing.isPlaying) {
                    val (position, duration) = onGetPosition()
                    progress = if (duration > 0) {
                        (position.toFloat() / duration.toFloat()).coerceIn(0f, 1f)
                    } else 0f
                    delay(1000)
                }
            }

            Surface(
                modifier = Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(topStart = 16.dp, topEnd = 16.dp),
                color = ClawSurface,
                shadowElevation = 8.dp
            ) {
                Column {
                    // Progress bar
                    LinearProgressIndicator(
                        progress = { progress },
                        modifier = Modifier
                            .fillMaxWidth()
                            .height(2.dp),
                        color = ClawAccent,
                        trackColor = ClawBorder,
                    )

                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(horizontal = 12.dp, vertical = 8.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        // Music icon
                        Surface(
                            modifier = Modifier.size(40.dp),
                            shape = RoundedCornerShape(8.dp),
                            color = ClawSurfaceVariant
                        ) {
                            Icon(
                                imageVector = Icons.Default.MusicNote,
                                contentDescription = null,
                                modifier = Modifier
                                    .padding(8.dp)
                                    .size(24.dp),
                                tint = ClawAccent
                            )
                        }

                        Spacer(modifier = Modifier.width(12.dp))

                        // Title and artist
                        Column(
                            modifier = Modifier.weight(1f),
                            verticalArrangement = Arrangement.Center
                        ) {
                            Text(
                                text = playing.title,
                                style = MaterialTheme.typography.bodyMedium,
                                color = ClawTextPrimary,
                                maxLines = 1,
                                overflow = TextOverflow.Ellipsis
                            )
                            Text(
                                text = playing.artist,
                                style = MaterialTheme.typography.bodySmall,
                                color = ClawTextSecondary,
                                maxLines = 1,
                                overflow = TextOverflow.Ellipsis
                            )
                        }

                        Spacer(modifier = Modifier.width(8.dp))

                        // Play/Pause button
                        IconButton(
                            onClick = onPlayPause,
                            modifier = Modifier
                                .size(40.dp)
                                .clip(CircleShape)
                                .background(ClawAccent)
                        ) {
                            Icon(
                                imageVector = if (playing.isPlaying) {
                                    Icons.Default.Pause
                                } else {
                                    Icons.Default.PlayArrow
                                },
                                contentDescription = if (playing.isPlaying) "Pause" else "Play",
                                tint = ClawSurface,
                                modifier = Modifier.size(24.dp)
                            )
                        }

                        Spacer(modifier = Modifier.width(4.dp))

                        // Stop button
                        IconButton(
                            onClick = onStop,
                            modifier = Modifier.size(36.dp)
                        ) {
                            Icon(
                                imageVector = Icons.Default.Close,
                                contentDescription = "Stop",
                                tint = ClawTextSecondary,
                                modifier = Modifier.size(20.dp)
                            )
                        }
                    }
                }
            }
        }
    }
}
