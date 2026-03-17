package com.claw.assistant.ui.screens

import android.annotation.SuppressLint
import android.graphics.Bitmap
import android.util.Log
import android.webkit.WebResourceRequest
import android.webkit.WebResourceResponse
import android.webkit.WebView
import android.webkit.WebViewClient
import androidx.activity.compose.BackHandler
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.toArgb
import androidx.compose.ui.viewinterop.AndroidView
import com.claw.assistant.ui.theme.ClawBackground
import java.net.HttpURLConnection
import java.net.URL

private const val TAG = "AdminScreen"

@SuppressLint("SetJavaScriptEnabled")
@Composable
fun AdminScreen(
    serverUrl: String,
    apiKey: String,
    modifier: Modifier = Modifier
) {
    val bgColorInt = remember { ClawBackground.toArgb() }
    var webViewRef by remember { mutableStateOf<WebView?>(null) }

    BackHandler(enabled = webViewRef?.canGoBack() == true) {
        webViewRef?.goBack()
    }

    Box(
        modifier = modifier
            .fillMaxSize()
            .background(ClawBackground)
    ) {
        AndroidView(
            factory = { context ->
                WebView(context).apply {
                    setBackgroundColor(bgColorInt)

                    settings.javaScriptEnabled = true
                    settings.domStorageEnabled = true
                    settings.loadWithOverviewMode = true
                    settings.useWideViewPort = true

                    webViewClient = object : WebViewClient() {
                        override fun shouldInterceptRequest(
                            view: WebView?,
                            request: WebResourceRequest?
                        ): WebResourceResponse? {
                            val url = request?.url?.toString() ?: return null
                            if (!url.startsWith(serverUrl)) return null

                            return try {
                                val conn = URL(url).openConnection() as HttpURLConnection
                                conn.setRequestProperty("X-API-Key", apiKey)
                                request.requestHeaders?.forEach { (key, value) ->
                                    conn.setRequestProperty(key, value)
                                }
                                conn.connectTimeout = 10_000
                                conn.readTimeout = 15_000

                                val responseCode = conn.responseCode
                                val responseMessage = conn.responseMessage ?: "OK"
                                val contentType = conn.contentType?.split(";")?.firstOrNull() ?: "text/html"
                                val encoding = conn.contentEncoding ?: "UTF-8"
                                val headers = conn.headerFields
                                    ?.filterKeys { it != null }
                                    ?.mapKeys { it.key!! }
                                    ?.mapValues { it.value.joinToString(", ") }
                                    ?: emptyMap()

                                val stream = if (responseCode in 200..399) {
                                    conn.inputStream
                                } else {
                                    conn.errorStream ?: conn.inputStream
                                }

                                WebResourceResponse(
                                    contentType,
                                    encoding,
                                    responseCode,
                                    responseMessage,
                                    headers,
                                    stream
                                )
                            } catch (e: Exception) {
                                Log.e(TAG, "Request interception failed for $url", e)
                                null
                            }
                        }

                        override fun onPageStarted(view: WebView?, url: String?, favicon: Bitmap?) {
                            super.onPageStarted(view, url, favicon)
                            Log.d(TAG, "Loading: $url")
                        }
                    }

                    webViewRef = this
                    loadUrl("$serverUrl/settings", mapOf("X-API-Key" to apiKey))
                }
            },
            update = { webView ->
                webViewRef = webView
            },
            modifier = Modifier.fillMaxSize()
        )
    }
}
