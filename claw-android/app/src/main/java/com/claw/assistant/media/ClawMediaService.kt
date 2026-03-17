package com.claw.assistant.media

import android.content.Intent
import android.os.Bundle
import android.util.Log
import androidx.media3.common.MediaItem
import androidx.media3.common.MediaMetadata
import androidx.media3.common.Player
import androidx.media3.session.LibraryResult
import androidx.media3.session.MediaLibraryService
import androidx.media3.session.MediaSession
import com.claw.assistant.ClawApplication
import com.google.common.collect.ImmutableList
import com.google.common.util.concurrent.Futures
import com.google.common.util.concurrent.ListenableFuture
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.launch

class ClawMediaService : MediaLibraryService() {

    private var mediaSession: MediaLibrarySession? = null
    private val serviceScope = CoroutineScope(Dispatchers.Main + SupervisorJob())

    override fun onCreate() {
        super.onCreate()
        Log.d(TAG, "ClawMediaService created")

        val app = application as ClawApplication
        val player = app.musicPlayerManager.getPlayer()
            ?: return  // Player not initialized yet

        mediaSession = MediaLibrarySession.Builder(this, player, LibrarySessionCallback())
            .build()
    }

    override fun onGetSession(controllerInfo: MediaSession.ControllerInfo): MediaLibrarySession? {
        return mediaSession
    }

    override fun onTaskRemoved(rootIntent: Intent?) {
        val player = mediaSession?.player
        if (player == null || !player.playWhenReady || player.mediaItemCount == 0) {
            stopSelf()
        }
    }

    override fun onDestroy() {
        mediaSession?.run {
            player.release()
            release()
            mediaSession = null
        }
        serviceScope.cancel()
        super.onDestroy()
        Log.d(TAG, "ClawMediaService destroyed")
    }

    private inner class LibrarySessionCallback : MediaLibrarySession.Callback {

        override fun onGetLibraryRoot(
            session: MediaLibrarySession,
            browser: MediaSession.ControllerInfo,
            params: LibraryParams?
        ): ListenableFuture<LibraryResult<MediaItem>> {
            val rootItem = MediaItem.Builder()
                .setMediaId(MEDIA_ROOT_ID)
                .setMediaMetadata(
                    MediaMetadata.Builder()
                        .setTitle("The Claw Music")
                        .setIsBrowsable(true)
                        .setIsPlayable(false)
                        .setMediaType(MediaMetadata.MEDIA_TYPE_FOLDER_MIXED)
                        .build()
                )
                .build()
            return Futures.immediateFuture(LibraryResult.ofItem(rootItem, params))
        }

        override fun onGetChildren(
            session: MediaLibrarySession,
            browser: MediaSession.ControllerInfo,
            parentId: String,
            page: Int,
            pageSize: Int,
            params: LibraryParams?
        ): ListenableFuture<LibraryResult<ImmutableList<MediaItem>>> {
            return when (parentId) {
                MEDIA_ROOT_ID -> {
                    val children = ImmutableList.of(
                        MediaItem.Builder()
                            .setMediaId(MEDIA_RECENT_ID)
                            .setMediaMetadata(
                                MediaMetadata.Builder()
                                    .setTitle("Recent")
                                    .setIsBrowsable(true)
                                    .setIsPlayable(false)
                                    .setMediaType(MediaMetadata.MEDIA_TYPE_FOLDER_PLAYLISTS)
                                    .build()
                            )
                            .build()
                    )
                    Futures.immediateFuture(LibraryResult.ofItemList(children, params))
                }
                MEDIA_RECENT_ID -> {
                    // Return the currently playing item if any
                    val app = application as ClawApplication
                    val nowPlaying = app.musicPlayerManager.nowPlaying.value
                    val items = if (nowPlaying != null) {
                        ImmutableList.of(
                            MediaItem.Builder()
                                .setMediaId(nowPlaying.videoId)
                                .setMediaMetadata(
                                    MediaMetadata.Builder()
                                        .setTitle(nowPlaying.title)
                                        .setArtist(nowPlaying.artist)
                                        .setIsBrowsable(false)
                                        .setIsPlayable(true)
                                        .setMediaType(MediaMetadata.MEDIA_TYPE_MUSIC)
                                        .build()
                                )
                                .build()
                        )
                    } else {
                        ImmutableList.of()
                    }
                    Futures.immediateFuture(LibraryResult.ofItemList(items, params))
                }
                else -> {
                    Futures.immediateFuture(
                        LibraryResult.ofError(LibraryResult.RESULT_ERROR_BAD_VALUE)
                    )
                }
            }
        }

        override fun onPlaybackResumption(
            mediaSession: MediaSession,
            controller: MediaSession.ControllerInfo
        ): ListenableFuture<MediaSession.MediaItemsWithStartPosition> {
            // Return empty — we don't persist playback state
            return Futures.immediateFuture(
                MediaSession.MediaItemsWithStartPosition(
                    ImmutableList.of(),
                    0,
                    0
                )
            )
        }

        override fun onSetMediaItems(
            mediaSession: MediaSession,
            controller: MediaSession.ControllerInfo,
            mediaItems: MutableList<MediaItem>,
            startIndex: Int,
            startPositionMs: Long
        ): ListenableFuture<MediaSession.MediaItemsWithStartPosition> {
            // Handle voice search from Android Auto by sending to chat API
            val searchQuery = mediaItems.firstOrNull()?.mediaMetadata?.title?.toString()
            if (!searchQuery.isNullOrBlank()) {
                serviceScope.launch {
                    handleVoiceSearch(searchQuery)
                }
            }
            return super.onSetMediaItems(
                mediaSession, controller, mediaItems, startIndex, startPositionMs
            )
        }
    }

    private suspend fun handleVoiceSearch(query: String) {
        try {
            val app = application as ClawApplication
            val response = app.apiClient.chat("play $query")
            if (response.music != null) {
                app.musicPlayerManager.playFromServer(response.music)
            }
        } catch (e: Exception) {
            Log.e(TAG, "Voice search failed: $query", e)
        }
    }

    companion object {
        private const val TAG = "ClawMediaService"
        private const val MEDIA_ROOT_ID = "claw_media_root"
        private const val MEDIA_RECENT_ID = "claw_media_recent"
    }
}
