package util

import java.io.File
import java.security.MessageDigest

class Extensions {
}

class Hasher {
    fun hash(file: File): String {
        val bytes = file.toString().toByteArray()
        val md = MessageDigest.getInstance("SHA-256")
        val digest = md.digest(bytes)
        return digest.fold("") { str, it -> str + "%02x".format(it) }
    }
}