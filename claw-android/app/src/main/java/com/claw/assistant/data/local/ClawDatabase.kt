package com.claw.assistant.data.local

import android.content.Context
import androidx.room.Database
import androidx.room.Room
import androidx.room.RoomDatabase

@Database(entities = [MessageEntity::class], version = 2, exportSchema = false)
abstract class ClawDatabase : RoomDatabase() {
    abstract fun messageDao(): MessageDao

    companion object {
        @Volatile
        private var INSTANCE: ClawDatabase? = null

        fun getInstance(context: Context): ClawDatabase {
            return INSTANCE ?: synchronized(this) {
                // Double-checked locking: re-check inside synchronized
                INSTANCE ?: Room.databaseBuilder(
                    context.applicationContext,
                    ClawDatabase::class.java,
                    "claw_database"
                )
                    .fallbackToDestructiveMigration()
                    .build()
                    .also { INSTANCE = it }
            }
        }
    }
}
