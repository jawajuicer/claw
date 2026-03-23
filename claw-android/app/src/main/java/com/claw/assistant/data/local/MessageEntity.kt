package com.claw.assistant.data.local

import androidx.room.Entity
import androidx.room.PrimaryKey

@Entity(tableName = "messages")
data class MessageEntity(
    @PrimaryKey val id: String,
    val content: String,
    val isUser: Boolean,
    val toolsUsed: String = "",  // comma-separated
    val timestamp: Long = System.currentTimeMillis(),
    val status: String = "sent",  // "sent", "failed", "pending"
    val isClaudeCode: Boolean = false
)
