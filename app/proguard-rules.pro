# TFLite
-keep class org.tensorflow.** { *; }
-keepclassmembers class * {
    @org.tensorflow.lite.annotations.UsedByReflection *;
}

# Keep native method names
-keepclasseswithmembernames class * {
    native <methods>;
}

# Keep YuvPreprocessor (accessed from JNI)
-keep class com.xiaohaoo.yolo.util.YuvPreprocessor { *; }
