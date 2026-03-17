package com.claw.assistant.media

import android.content.Context
import android.net.Uri
import android.util.Log
import androidx.media3.common.MediaItem
import androidx.media3.common.MediaMetadata
import androidx.media3.common.PlaybackException
import androidx.media3.common.Player
import androidx.media3.exoplayer.ExoPlayer
import com.claw.assistant.ClawApplication
import com.claw.assistant.network.MusicInfo
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

data class NowPlaying(
    val title: String,
    val artist: String,
    val videoId: String,
    val isPlaying: Boolean,
    val durationMs: Long = 0,
    val positionMs: Long = 0
)

class MusicPlayerManager(private val context: Context) {

    private var exoPlayer: ExoPlayer? = null

    private val _nowPlaying = MutableStateFlow<NowPlaying?>(null)
    val nowPlaying: StateFlow<NowPlaying?> = _nowPlaying.asStateFlow()

    private val _isPlaying = MutableStateFlow(false)
    val isPlaying: StateFlow<Boolean> = _isPlaying.asStateFlow()

    private fun getOrCreatePlayer(): ExoPlayer {
        return exoPlayer ?: ExoPlayer.Builder(context).build().also { player ->
            player.addListener(object : Player.Listener {
                override fun onPlaybackStateChanged(playbackState: Int) {
                    when (playbackState) {
                        Player.STATE_READY -> {
                            updateNowPlaying(isPlaying = player.isPlaying)
                        }
                        Player.STATE_ENDED -> {
                            _isPlaying.value = false
                            updateNowPlaying(isPlaying = false)
                        }
                        Player.STATE_IDLE, Player.STATE_BUFFERING -> {
                            // No action needed
                        }
                    }
                }

                override fun onIsPlayingChanged(playing: Boolean) {
                    _isPlaying.value = playing
                    updateNowPlaying(isPlaying = playing)
                }

                override fun onPlayerError(error: PlaybackException) {
                    Log.e(TAG, "ExoPlayer error: ${error.message}", error)
                    _isPlaying.value = false
                    updateNowPlaying(isPlaying = false)
                }
            })
            exoPlayer = player
        }
    }

    fun playFromServer(music: MusicInfo) {
        val app = ClawApplication.getInstance()
        val streamUrl = app.apiClient.getStreamUrl(music.videoId)

        Log.d(TAG, "Playing: ${music.title} by ${music.artist} from $streamUrl")

        val player = getOrCreatePlayer()

        val mediaItem = MediaItem.Builder()
            .setUri(Uri.parse(streamUrl))
            .setMediaId(music.videoId)
            .setMediaMetadata(
                MediaMetadata.Builder()
                    .setTitle(music.title)
                    .setArtist(music.artist)
                    .build()
            )
            .build()

        player.setMediaItem(mediaItem)
        player.prepare()
        player.play()

        _nowPlaying.value = NowPlaying(
            title = music.title,
            artist = music.artist,
            videoId = music.videoId,
            isPlaying = true
        )
        _isPlaying.value = true
    }

    fun play() {
        exoPlayer?.play()
    }

    fun pause() {
        exoPlayer?.pause()
    }

    fun togglePlayPause() {
        val player = exoPlayer ?: return
        if (player.isPlaying) {
            player.pause()
        } else {
            player.play()
        }
    }

    fun stop() {
        exoPlayer?.stop()
        _isPlaying.value = false
        _nowPlaying.value = null
    }

    fun seekTo(positionMs: Long) {
        exoPlayer?.seekTo(positionMs)
    }

    fun getDurationMs(): Long = exoPlayer?.duration ?: 0

    fun getPositionMs(): Long = exoPlayer?.currentPosition ?: 0

    fun getPlayer(): ExoPlayer? = exoPlayer

    private fun updateNowPlaying(isPlaying: Boolean) {
        val current = _nowPlaying.value ?: return
        _nowPlaying.value = current.copy(
            isPlaying = isPlaying,
            durationMs = exoPlayer?.duration ?: 0,
            positionMs = exoPlayer?.currentPosition ?: 0
        )
    }

    fun release() {
        exoPlayer?.release()
        exoPlayer = null
        _isPlaying.value = false
        _nowPlaying.value = null
    }

    companion object {
        private const val TAG = "MusicPlayerManager"
    }
}
