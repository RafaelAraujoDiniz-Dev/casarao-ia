package br.com.casaraoia

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.outlined.ArrowBack
import androidx.compose.material.icons.outlined.CameraAlt
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.FloatingActionButton
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.exifinterface.media.ExifInterface
import java.io.File
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

@Composable
fun CameraCaptureScreen(
    onImageCaptured: (Bitmap) -> Unit,
    onCancel: () -> Unit,
    onError: (String) -> Unit
) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current

    val mainExecutor = remember(context) {
        ContextCompat.getMainExecutor(context)
    }

    val cameraExecutor = remember {
        Executors.newSingleThreadExecutor()
    }

    val previewView = remember {
        PreviewView(context).apply {
            scaleType = PreviewView.ScaleType.FILL_CENTER
            implementationMode =
                PreviewView.ImplementationMode.COMPATIBLE
        }
    }

    val imageCapture = remember {
        ImageCapture.Builder()
            .setCaptureMode(
                ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY
            )
            .setJpegQuality(95)
            .build()
    }

    var cameraReady by remember {
        mutableStateOf(false)
    }

    var capturing by remember {
        mutableStateOf(false)
    }

    var cameraProvider by remember {
        mutableStateOf<ProcessCameraProvider?>(null)
    }

    LaunchedEffect(
        context,
        lifecycleOwner
    ) {
        val cameraProviderFuture =
            ProcessCameraProvider.getInstance(context)

        cameraProviderFuture.addListener(
            {
                try {
                    val provider = cameraProviderFuture.get()

                    cameraProvider = provider

                    val preview = Preview.Builder()
                        .build()
                        .also { cameraPreview ->
                            cameraPreview.surfaceProvider =
                                previewView.surfaceProvider
                        }

                    provider.unbindAll()

                    provider.bindToLifecycle(
                        lifecycleOwner,
                        CameraSelector.DEFAULT_BACK_CAMERA,
                        preview,
                        imageCapture
                    )

                    cameraReady = true
                } catch (exception: Exception) {
                    cameraReady = false

                    onError(
                        exception.message
                            ?: "Não foi possível iniciar a câmera."
                    )
                }
            },
            mainExecutor
        )
    }

    DisposableEffect(Unit) {
        onDispose {
            cameraProvider?.unbindAll()

            if (!cameraExecutor.isShutdown) {
                cameraExecutor.shutdown()
            }
        }
    }

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(Color.Black)
    ) {
        AndroidView(
            factory = {
                previewView
            },
            modifier = Modifier.fillMaxSize()
        )

        CameraTopBar(
            modifier = Modifier.align(Alignment.TopCenter),
            onCancel = {
                if (!capturing) {
                    cameraProvider?.unbindAll()
                    onCancel()
                }
            }
        )

        CameraGuide(
            modifier = Modifier.align(Alignment.Center)
        )

        CameraBottomControls(
            cameraReady = cameraReady,
            capturing = capturing,
            modifier = Modifier.align(Alignment.BottomCenter),
            onCapture = {
                if (cameraReady && !capturing) {
                    capturing = true

                    previewView.display?.rotation?.let { rotation ->
                        imageCapture.targetRotation = rotation
                    }

                    capturePhoto(
                        context = context,
                        imageCapture = imageCapture,
                        cameraExecutor = cameraExecutor,
                        onSuccess = { capturedBitmap ->
                            mainExecutor.execute {
                                capturing = false
                                cameraProvider?.unbindAll()
                                onImageCaptured(capturedBitmap)
                            }
                        },
                        onFailure = { message ->
                            mainExecutor.execute {
                                capturing = false
                                onError(message)
                            }
                        }
                    )
                }
            }
        )

        if (!cameraReady) {
            Surface(
                modifier = Modifier.align(Alignment.Center),
                shape = RoundedCornerShape(18.dp),
                color = Color.Black.copy(alpha = 0.72f)
            ) {
                Row(
                    modifier = Modifier.padding(
                        horizontal = 22.dp,
                        vertical = 18.dp
                    ),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement =
                        Arrangement.spacedBy(14.dp)
                ) {
                    CircularProgressIndicator(
                        modifier = Modifier.size(28.dp),
                        color = Color.White,
                        strokeWidth = 3.dp
                    )

                    Text(
                        text = "Preparando a câmera...",
                        color = Color.White,
                        style = MaterialTheme.typography.bodyLarge,
                        fontWeight = FontWeight.Medium
                    )
                }
            }
        }
    }
}

@Composable
private fun CameraTopBar(
    modifier: Modifier = Modifier,
    onCancel: () -> Unit
) {
    Row(
        modifier = modifier
            .fillMaxWidth()
            .background(
                color = Color.Black.copy(alpha = 0.50f)
            )
            .statusBarsPadding()
            .padding(
                horizontal = 12.dp,
                vertical = 10.dp
            ),
        verticalAlignment = Alignment.CenterVertically
    ) {
        IconButton(
            onClick = onCancel
        ) {
            Icon(
                imageVector = Icons.Outlined.ArrowBack,
                contentDescription = "Voltar",
                tint = Color.White,
                modifier = Modifier.size(28.dp)
            )
        }

        Spacer(
            modifier = Modifier.width(8.dp)
        )

        Column {
            Text(
                text = "Fotografar fachada",
                color = Color.White,
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.Bold
            )

            Text(
                text = "CASARÃO IA",
                color = Color.White.copy(alpha = 0.76f),
                style = MaterialTheme.typography.labelMedium
            )
        }
    }
}

@Composable
private fun CameraGuide(
    modifier: Modifier = Modifier
) {
    Card(
        modifier = modifier
            .fillMaxWidth()
            .padding(horizontal = 24.dp),
        shape = RoundedCornerShape(18.dp),
        colors = CardDefaults.cardColors(
            containerColor = Color.Black.copy(alpha = 0.42f)
        )
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(
                    horizontal = 20.dp,
                    vertical = 16.dp
                ),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Icon(
                imageVector = Icons.Outlined.CameraAlt,
                contentDescription = null,
                tint = Color.White,
                modifier = Modifier.size(28.dp)
            )

            Spacer(
                modifier = Modifier.height(8.dp)
            )

            Text(
                text = "Enquadre toda a fachada",
                color = Color.White,
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.Bold,
                textAlign = TextAlign.Center
            )

            Spacer(
                modifier = Modifier.height(4.dp)
            )

            Text(
                text = "Inclua portas, janelas e os principais " +
                        "detalhes arquitetônicos.",
                color = Color.White.copy(alpha = 0.88f),
                style = MaterialTheme.typography.bodyMedium,
                textAlign = TextAlign.Center
            )
        }
    }
}

@Composable
private fun CameraBottomControls(
    cameraReady: Boolean,
    capturing: Boolean,
    modifier: Modifier = Modifier,
    onCapture: () -> Unit
) {
    Column(
        modifier = modifier
            .fillMaxWidth()
            .background(
                color = Color.Black.copy(alpha = 0.58f)
            )
            .navigationBarsPadding()
            .padding(
                horizontal = 20.dp,
                vertical = 18.dp
            ),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Text(
            text = if (capturing) {
                "Processando fotografia..."
            } else {
                "Mantenha o aparelho firme"
            },
            color = Color.White,
            style = MaterialTheme.typography.bodyMedium,
            textAlign = TextAlign.Center
        )

        Spacer(
            modifier = Modifier.height(14.dp)
        )

        FloatingActionButton(
            onClick = onCapture,
            modifier = Modifier.size(76.dp),
            shape = CircleShape,
            containerColor = if (
                cameraReady && !capturing
            ) {
                Color.White
            } else {
                Color.LightGray
            },
            contentColor = Color(0xFF173A58)
        ) {
            if (capturing) {
                CircularProgressIndicator(
                    modifier = Modifier.size(32.dp),
                    color = Color(0xFF173A58),
                    strokeWidth = 3.dp
                )
            } else {
                Icon(
                    imageVector = Icons.Outlined.CameraAlt,
                    contentDescription = "Capturar fotografia",
                    modifier = Modifier.size(34.dp)
                )
            }
        }
    }
}

private fun capturePhoto(
    context: Context,
    imageCapture: ImageCapture,
    cameraExecutor: ExecutorService,
    onSuccess: (Bitmap) -> Unit,
    onFailure: (String) -> Unit
) {
    val photoFile = try {
        File.createTempFile(
            "casarao_",
            ".jpg",
            context.cacheDir
        )
    } catch (exception: Exception) {
        onFailure(
            exception.message
                ?: "Não foi possível criar o arquivo temporário."
        )

        return
    }

    val outputOptions =
        ImageCapture.OutputFileOptions.Builder(
            photoFile
        ).build()

    imageCapture.takePicture(
        outputOptions,
        cameraExecutor,
        object : ImageCapture.OnImageSavedCallback {

            override fun onImageSaved(
                outputFileResults:
                ImageCapture.OutputFileResults
            ) {
                try {
                    val originalBitmap =
                        BitmapFactory.decodeFile(
                            photoFile.absolutePath
                        )

                    if (originalBitmap == null) {
                        onFailure(
                            "Não foi possível carregar a fotografia capturada."
                        )

                        return
                    }

                    val rotationDegrees =
                        readExifRotation(photoFile)

                    val correctedBitmap =
                        rotateBitmap(
                            bitmap = originalBitmap,
                            rotationDegrees = rotationDegrees
                        )

                    onSuccess(correctedBitmap)
                } catch (exception: Exception) {
                    onFailure(
                        exception.message
                            ?: "Não foi possível processar a fotografia."
                    )
                } finally {
                    photoFile.delete()
                }
            }

            override fun onError(
                exception: ImageCaptureException
            ) {
                photoFile.delete()

                onFailure(
                    exception.message
                        ?: "Não foi possível capturar a fotografia."
                )
            }
        }
    )
}

private fun readExifRotation(
    photoFile: File
): Float {
    return try {
        val exifInterface =
            ExifInterface(photoFile.absolutePath)

        when (
            exifInterface.getAttributeInt(
                ExifInterface.TAG_ORIENTATION,
                ExifInterface.ORIENTATION_NORMAL
            )
        ) {
            ExifInterface.ORIENTATION_ROTATE_90 -> {
                90f
            }

            ExifInterface.ORIENTATION_ROTATE_180 -> {
                180f
            }

            ExifInterface.ORIENTATION_ROTATE_270 -> {
                270f
            }

            else -> {
                0f
            }
        }
    } catch (_: Exception) {
        0f
    }
}

private fun rotateBitmap(
    bitmap: Bitmap,
    rotationDegrees: Float
): Bitmap {
    if (rotationDegrees == 0f) {
        return bitmap
    }

    val matrix = Matrix().apply {
        postRotate(rotationDegrees)
    }

    val rotatedBitmap = Bitmap.createBitmap(
        bitmap,
        0,
        0,
        bitmap.width,
        bitmap.height,
        matrix,
        true
    )

    if (
        rotatedBitmap !== bitmap &&
        !bitmap.isRecycled
    ) {
        bitmap.recycle()
    }

    return rotatedBitmap
}