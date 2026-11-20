plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.compose)
}

android {
    namespace = "br.com.casaraoia"
    compileSdk = 37

    defaultConfig {
        applicationId = "br.com.casaraoia"
        minSdk = 26
        targetSdk = 37
        versionCode = 1
        versionName = "1.0"

        testInstrumentationRunner =
            "androidx.test.runner.AndroidJUnitRunner"
    }

    buildTypes {
        release {
            isMinifyEnabled = false

            proguardFiles(
                getDefaultProguardFile(
                    "proguard-android-optimize.txt"
                ),
                "proguard-rules.pro"
            )
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }

    buildFeatures {
        compose = true
    }

    androidResources {
        noCompress += "tflite"
    }

    packaging {
        resources {
            excludes += "/META-INF/{AL2.0,LGPL2.1}"
        }
    }
}

dependencies {
    // Android
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.lifecycle.runtime.ktx)
    implementation(libs.androidx.activity.compose)

    // Jetpack Compose
    implementation(
        platform(libs.androidx.compose.bom)
    )

    implementation(libs.androidx.compose.ui)
    implementation(libs.androidx.compose.ui.graphics)
    implementation(libs.androidx.compose.ui.tooling.preview)
    implementation(libs.androidx.compose.material3)

    // Ícones adicionais do Material Design
    implementation(
        "androidx.compose.material:material-icons-extended"
    )

    // LiteRT, responsável pela execução do modelo TFLite
    implementation(
        "com.google.ai.edge.litert:litert:1.4.2"
    )

    // CameraX
    val cameraXVersion = "1.6.2"

    implementation(
        "androidx.camera:camera-core:$cameraXVersion"
    )

    implementation(
        "androidx.camera:camera-camera2:$cameraXVersion"
    )

    implementation(
        "androidx.camera:camera-lifecycle:$cameraXVersion"
    )

    implementation(
        "androidx.camera:camera-view:$cameraXVersion"
    )

    // Tratamento da orientação EXIF das imagens
    implementation(
        "androidx.exifinterface:exifinterface:1.4.2"
    )

    // Testes unitários
    testImplementation(libs.junit)

    // Testes instrumentados
    androidTestImplementation(libs.androidx.junit)
    androidTestImplementation(libs.androidx.espresso.core)

    androidTestImplementation(
        platform(libs.androidx.compose.bom)
    )

    androidTestImplementation(
        libs.androidx.compose.ui.test.junit4
    )

    // Ferramentas de desenvolvimento do Compose
    debugImplementation(
        libs.androidx.compose.ui.tooling
    )

    debugImplementation(
        libs.androidx.compose.ui.test.manifest
    )
}