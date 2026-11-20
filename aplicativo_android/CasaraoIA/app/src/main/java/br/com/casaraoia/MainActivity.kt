package br.com.casaraoia

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageDecoder
import android.net.Uri
import android.os.Build
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.ExperimentalFoundationApi
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.relocation.BringIntoViewRequester
import androidx.compose.foundation.relocation.bringIntoViewRequester
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.outlined.CameraAlt
import androidx.compose.material.icons.outlined.LocationOn
import androidx.compose.material.icons.outlined.PhotoLibrary
import androidx.compose.material.icons.outlined.Refresh
import androidx.compose.material.icons.outlined.Verified
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.ContextCompat
import br.com.casaraoia.data.BuildingInfo
import br.com.casaraoia.data.BuildingRepository
import br.com.casaraoia.ml.CasaraoClassifier
import br.com.casaraoia.ml.ClassificationResult
import br.com.casaraoia.ml.ConfidenceStatus
import br.com.casaraoia.ui.theme.CasaraoIATheme
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.Locale

class MainActivity : ComponentActivity() {

 override fun onCreate(savedInstanceState: Bundle?) {
  super.onCreate(savedInstanceState)

  setContent {
   CasaraoIATheme {
    Scaffold(
     modifier = Modifier.fillMaxSize(),
     containerColor = MaterialTheme.colorScheme.background,
     contentWindowInsets = WindowInsets.safeDrawing
    ) { innerPadding ->
     CasaraoScreen(
      modifier = Modifier.padding(innerPadding)
     )
    }
   }
  }
 }
}

@OptIn(ExperimentalFoundationApi::class)
@Composable
private fun CasaraoScreen(
 modifier: Modifier = Modifier
) {
 val context = LocalContext.current
 val density = LocalDensity.current
 val scope = rememberCoroutineScope()
 val scrollState = rememberScrollState()

 /*
  * Deslocamento adicional após a fotografia ser posicionada.
  * Aumente para 130.dp caso o texto ainda fique muito baixo.
  */
 val resultScrollOffset = remember(density) {
  with(density) {
   110.dp.roundToPx()
  }
 }

 val imageBringIntoViewRequester = remember {
  BringIntoViewRequester()
 }

 val classifier = remember {
  CasaraoClassifier(context.applicationContext)
 }

 DisposableEffect(classifier) {
  onDispose {
   classifier.close()
  }
 }

 var selectedBitmap by remember {
  mutableStateOf<Bitmap?>(null)
 }

 var classificationResult by remember {
  mutableStateOf<ClassificationResult?>(null)
 }

 var processing by remember {
  mutableStateOf(false)
 }

 var errorMessage by remember {
  mutableStateOf<String?>(null)
 }

 var cameraOpened by remember {
  mutableStateOf(false)
 }

 fun classifyImage(bitmap: Bitmap) {
  scope.launch {
   processing = true
   classificationResult = null
   errorMessage = null
   selectedBitmap = bitmap

   try {
    val result = withContext(Dispatchers.Default) {
     classifier.classify(bitmap)
    }

    classificationResult = result
   } catch (exception: Exception) {
    errorMessage =
     exception.message ?: "Falha ao classificar a imagem."
   } finally {
    processing = false
   }
  }
 }

 val cameraPermissionLauncher =
  rememberLauncherForActivityResult(
   contract = ActivityResultContracts.RequestPermission(),
   onResult = { permissionGranted: Boolean ->
    if (permissionGranted) {
     cameraOpened = true
    } else {
     errorMessage =
      "A permissão da câmera é necessária para fotografar a fachada."
    }
   }
  )

 /*
  * A imagem é carregada diretamente neste callback.
  * Não existe mais dependência de loadBitmapFromUri().
  */
 val galleryLauncher =
  rememberLauncherForActivityResult(
   contract = ActivityResultContracts.PickVisualMedia(),
   onResult = { selectedUri: Uri? ->
    if (selectedUri != null) {
     scope.launch {
      errorMessage = null

      try {
       val loadedBitmap: Bitmap? =
        withContext(Dispatchers.IO) {
         if (
          Build.VERSION.SDK_INT >=
          Build.VERSION_CODES.P
         ) {
          val source =
           ImageDecoder.createSource(
            context.contentResolver,
            selectedUri
           )

          ImageDecoder.decodeBitmap(
           source
          ) { decoder, _, _ ->
           decoder.allocator =
            ImageDecoder.ALLOCATOR_SOFTWARE
          }
         } else {
          context.contentResolver
           .openInputStream(selectedUri)
           ?.use { inputStream ->
            BitmapFactory.decodeStream(
             inputStream
            )
           }
         }
        }

       if (loadedBitmap != null) {
        classifyImage(loadedBitmap)
       } else {
        errorMessage =
         "Não foi possível carregar a imagem selecionada."
       }
      } catch (exception: Exception) {
       errorMessage =
        exception.message
         ?: "Falha ao abrir a imagem selecionada."
      }
     }
    }
   }
  )

 if (cameraOpened) {
  CameraCaptureScreen(
   onImageCaptured = { capturedBitmap: Bitmap ->
    cameraOpened = false
    classifyImage(capturedBitmap)
   },
   onCancel = {
    cameraOpened = false
   },
   onError = { message: String ->
    cameraOpened = false
    errorMessage = message
   }
  )

  return
 }

 val initialScreen =
  selectedBitmap == null &&
          classificationResult == null &&
          !processing

 /*
  * Posiciona a imagem e sobe mais 110 dp para mostrar
  * uma parte maior do cartão de resultado.
  */
 LaunchedEffect(classificationResult) {
  if (classificationResult != null) {
   delay(250)

   imageBringIntoViewRequester.bringIntoView()

   delay(120)

   scrollState.animateScrollTo(
    value = (
            scrollState.value + resultScrollOffset
            ).coerceAtMost(scrollState.maxValue)
   )
  }
 }

 Box(
  modifier = modifier
   .fillMaxSize()
   .background(Color(0xFFF8F4EC))
 ) {
  if (initialScreen) {
   InitialScreenDecorations()
  }

  if (!initialScreen) {
   Image(
    painter = painterResource(
     id = R.drawable.azulejos_decorativos
    ),
    contentDescription = null,
    modifier = Modifier
     .align(Alignment.TopEnd)
     .padding(top = 305.dp)
     .size(150.dp)
     .alpha(0.07f),
    contentScale = ContentScale.Fit
   )

   Image(
    painter = painterResource(
     id = R.drawable.azulejos_decorativos
    ),
    contentDescription = null,
    modifier = Modifier
     .align(Alignment.BottomStart)
     .size(105.dp)
     .alpha(0.045f),
    contentScale = ContentScale.Fit
   )
  }

  Column(
   modifier = Modifier
    .fillMaxSize()
    .verticalScroll(scrollState),
   horizontalAlignment = Alignment.CenterHorizontally
  ) {
   HeroHeader()

   Column(
    modifier = Modifier
     .fillMaxWidth()
     .padding(
      horizontal = 10.dp,
      vertical = 16.dp
     ),
    horizontalAlignment = Alignment.CenterHorizontally
   ) {
    HomeActionButtons(
     enabled = !processing,
     onCamera = {
      val permissionGranted =
       ContextCompat.checkSelfPermission(
        context,
        Manifest.permission.CAMERA
       ) == PackageManager.PERMISSION_GRANTED

      if (permissionGranted) {
       cameraOpened = true
      } else {
       cameraPermissionLauncher.launch(
        Manifest.permission.CAMERA
       )
      }
     },
     onGallery = {
      galleryLauncher.launch(
       PickVisualMediaRequest(
        ActivityResultContracts
         .PickVisualMedia
         .ImageOnly
       )
      )
     }
    )

    Spacer(
     modifier = Modifier.height(14.dp)
    )

    if (initialScreen) {
     CaptureInstructionCard()

     Spacer(
      modifier = Modifier.height(220.dp)
     )
    }

    selectedBitmap?.let { bitmap ->
     SelectedImageCard(
      bitmap = bitmap,
      modifier = Modifier.bringIntoViewRequester(
       imageBringIntoViewRequester
      )
     )
    }

    if (processing) {
     ProcessingCard()
    }

    errorMessage?.let { message ->
     Spacer(
      modifier = Modifier.height(14.dp)
     )

     ErrorCard(
      message = message
     )
    }

    classificationResult?.let { result ->
     Spacer(
      modifier = Modifier.height(10.dp)
     )

     ResultCard(
      result = result,
      onTryAgain = {
       selectedBitmap = null
       classificationResult = null
       errorMessage = null
       processing = false

       scope.launch {
        scrollState.animateScrollTo(0)
       }
      }
     )
    }

    Spacer(
     modifier = Modifier.height(30.dp)
    )
   }
  }
 }
}

@Composable
private fun HeroHeader() {
 Box(
  modifier = Modifier
   .fillMaxWidth()
   .background(
    brush = Brush.verticalGradient(
     colors = listOf(
      Color(0xFF173A58),
      Color(0xFF285E7D)
     )
    )
   )
   .padding(
    horizontal = 10.dp,
    vertical = 18.dp
   )
 ) {
  Image(
   painter = painterResource(
    id = R.drawable.azulejos_decorativos
   ),
   contentDescription = null,
   modifier = Modifier
    .align(Alignment.BottomEnd)
    .size(145.dp)
    .alpha(0.13f),
   contentScale = ContentScale.Fit
  )

  Column(
   modifier = Modifier.fillMaxWidth(),
   horizontalAlignment = Alignment.CenterHorizontally
  ) {
   Surface(
    modifier = Modifier
     .fillMaxWidth()
     .height(90.dp),
    shape = RoundedCornerShape(18.dp),
    color = Color(0xFFFBF8F1),
    shadowElevation = 5.dp
   ) {
    Box(
     modifier = Modifier.fillMaxSize()
    ) {
     Image(
      painter = painterResource(
       id = R.drawable.casaroes_silhueta
      ),
      contentDescription = null,
      modifier = Modifier
       .align(Alignment.BottomEnd)
       .fillMaxWidth(0.48f)
       .height(85.dp)
       .alpha(0.20f),
      contentScale = ContentScale.Crop
     )

     Image(
      painter = painterResource(
       id = R.drawable.logo_casarao_ia_horizontal
      ),
      contentDescription = "Logotipo do CASARÃO IA",
      modifier = Modifier
       .align(Alignment.Center)
       .fillMaxWidth(0.64f)
       .height(65.dp),
      contentScale = ContentScale.Fit
     )
    }
   }

   Spacer(
    modifier = Modifier.height(14.dp)
   )

   Text(
    text = "Descubra edificações históricas de São Luís " +
            "por meio de uma fotografia.",
    modifier = Modifier.padding(horizontal = 12.dp),
    color = Color.White,
    style = MaterialTheme.typography.titleMedium,
    fontWeight = FontWeight.Medium,
    lineHeight = 23.sp,
    textAlign = TextAlign.Center
   )
  }
 }
}

@Composable
private fun InitialScreenDecorations() {
 Image(
  painter = painterResource(
   id = R.drawable.azulejos_decorativos
  ),
  contentDescription = null,
  modifier = Modifier
   .fillMaxSize()
   .wrapContentSize(Alignment.TopEnd)
   .padding(top = 285.dp)
   .size(235.dp)
   .alpha(0.30f)
 )

 Image(
  painter = painterResource(
   id = R.drawable.azulejos_decorativos
  ),
  contentDescription = null,
  modifier = Modifier
   .fillMaxSize()
   .wrapContentSize(Alignment.BottomEnd)
   .size(185.dp)
   .alpha(0.40f)
 )

 Image(
  painter = painterResource(
   id = R.drawable.casaroes_silhueta
  ),
  contentDescription = null,
  modifier = Modifier
   .fillMaxSize()
   .wrapContentSize(Alignment.BottomCenter)
   .fillMaxWidth()
   .height(205.dp)
   .alpha(0.78f),
  contentScale = ContentScale.FillWidth
 )
}

@Composable
private fun HomeActionButtons(
 enabled: Boolean,
 onCamera: () -> Unit,
 onGallery: () -> Unit
) {
 Column(
  modifier = Modifier.fillMaxWidth()
 ) {
  Button(
   onClick = onCamera,
   enabled = enabled,
   modifier = Modifier
    .fillMaxWidth()
    .height(42.dp),
   shape = RoundedCornerShape(14.dp),
   colors = ButtonDefaults.buttonColors(
    containerColor = Color(0xFF1D6688),
    contentColor = Color.White,
    disabledContainerColor =
     Color(0xFF1D6688).copy(alpha = 0.45f),
    disabledContentColor =
     Color.White.copy(alpha = 0.70f)
   ),
   elevation = ButtonDefaults.buttonElevation(
    defaultElevation = 3.dp,
    pressedElevation = 1.dp
   )
  ) {
   Icon(
    imageVector = Icons.Outlined.CameraAlt,
    contentDescription = "Fotografar fachada",
    modifier = Modifier.size(18.dp)
   )

   Spacer(
    modifier = Modifier.width(8.dp)
   )

   Text(
    text = "Fotografar fachada",
    style = MaterialTheme.typography.labelLarge,
    fontWeight = FontWeight.Bold
   )
  }

  Spacer(
   modifier = Modifier.height(10.dp)
  )

  OutlinedButton(
   onClick = onGallery,
   enabled = enabled,
   modifier = Modifier
    .fillMaxWidth()
    .height(42.dp),
   shape = RoundedCornerShape(14.dp),
   border = BorderStroke(
    width = 1.dp,
    color = if (enabled) {
     Color(0xFF173A58)
    } else {
     Color(0xFF173A58).copy(alpha = 0.40f)
    }
   ),
   colors = ButtonDefaults.outlinedButtonColors(
    contentColor = Color(0xFF173A58),
    disabledContentColor =
     Color(0xFF173A58).copy(alpha = 0.45f)
   )
  ) {
   Icon(
    imageVector = Icons.Outlined.PhotoLibrary,
    contentDescription = "Escolher imagem da galeria",
    modifier = Modifier.size(18.dp)
   )

   Spacer(
    modifier = Modifier.width(8.dp)
   )

   Text(
    text = "Escolher da galeria",
    style = MaterialTheme.typography.labelLarge,
    fontWeight = FontWeight.Medium
   )
  }
 }
}

@Composable
private fun CaptureInstructionCard() {
 Card(
  modifier = Modifier.fillMaxWidth(),
  shape = RoundedCornerShape(22.dp),
  colors = CardDefaults.cardColors(
   containerColor = Color(0xFFFFFBF4)
  ),
  elevation = CardDefaults.cardElevation(
   defaultElevation = 8.dp
  ),
  border = BorderStroke(
   width = 1.dp,
   color = Color(0xFFC79A3B).copy(alpha = 0.55f)
  )
 ) {
  Column(
   modifier = Modifier
    .fillMaxWidth()
    .padding(
     horizontal = 24.dp,
     vertical = 25.dp
    ),
   horizontalAlignment = Alignment.CenterHorizontally
  ) {
   Surface(
    modifier = Modifier.size(54.dp),
    shape = RoundedCornerShape(27.dp),
    color = Color(0xFFEDE3D2)
   ) {
    Box(
     contentAlignment = Alignment.Center
    ) {
     Icon(
      imageVector = Icons.Outlined.CameraAlt,
      contentDescription = null,
      tint = Color(0xFF173A58),
      modifier = Modifier.size(29.dp)
     )
    }
   }

   Spacer(
    modifier = Modifier.height(16.dp)
   )

   Text(
    text = "Como obter um bom resultado",
    style = MaterialTheme.typography.titleLarge,
    fontWeight = FontWeight.Bold,
    textAlign = TextAlign.Center
   )

   Spacer(
    modifier = Modifier.height(10.dp)
   )

   Text(
    text = "Centralize a fachada, inclua portas e janelas " +
            "e evite reflexos ou obstáculos.",
    style = MaterialTheme.typography.bodyLarge,
    lineHeight = 24.sp,
    textAlign = TextAlign.Center
   )
  }
 }
}

@Composable
private fun SelectedImageCard(
 bitmap: Bitmap,
 modifier: Modifier = Modifier
) {
 Card(
  modifier = modifier.fillMaxWidth(),
  shape = RoundedCornerShape(16.dp),
  elevation = CardDefaults.cardElevation(
   defaultElevation = 4.dp
  ),
  colors = CardDefaults.cardColors(
   containerColor = Color(0xFFE9E6DE)
  )
 ) {
  Box(
   modifier = Modifier
    .fillMaxWidth()
    .height(190.dp)
    .background(Color(0xFFE9E6DE)),
   contentAlignment = Alignment.Center
  ) {
   Image(
    bitmap = bitmap.asImageBitmap(),
    contentDescription =
     "Fotografia selecionada para análise",
    modifier = Modifier.fillMaxSize(),
    contentScale = ContentScale.Fit
   )
  }
 }
}

@Composable
private fun ProcessingCard() {
 Spacer(
  modifier = Modifier.height(14.dp)
 )

 Card(
  modifier = Modifier.fillMaxWidth(),
  shape = RoundedCornerShape(16.dp)
 ) {
  Row(
   modifier = Modifier.padding(18.dp),
   verticalAlignment = Alignment.CenterVertically,
   horizontalArrangement = Arrangement.spacedBy(14.dp)
  ) {
   CircularProgressIndicator(
    modifier = Modifier.size(34.dp),
    strokeWidth = 3.dp
   )

   Column {
    Text(
     text = "Analisando a fachada",
     fontWeight = FontWeight.Bold
    )

    Text(
     text = "Inferência local no aparelho."
    )
   }
  }
 }
}

@Composable
private fun ErrorCard(
 message: String
) {
 Card(
  modifier = Modifier.fillMaxWidth(),
  shape = RoundedCornerShape(16.dp),
  colors = CardDefaults.cardColors(
   containerColor =
    MaterialTheme.colorScheme.errorContainer
  )
 ) {
  Column(
   modifier = Modifier.padding(18.dp)
  ) {
   Text(
    text = "Não foi possível analisar",
    fontWeight = FontWeight.Bold
   )

   Spacer(
    modifier = Modifier.height(4.dp)
   )

   Text(
    text = message
   )
  }
 }
}

@Composable
private fun ResultCard(
 result: ClassificationResult,
 onTryAgain: () -> Unit
) {
 val confidence =
  formatPercentage(result.confidence)

 val secondConfidence =
  formatPercentage(result.secondConfidence)

 val building =
  BuildingRepository.findByLabel(result.label)

 val statusTitle: String
 val resultTitle: String
 val message: String
 val accentColor: Color
 val containerColor: Color

 when (result.status) {
  ConfidenceStatus.HIGH_CONFIDENCE -> {
   statusTitle = "Alta confiança"
   resultTitle = result.displayName
   message =
    "A edificação foi identificada com alta confiança."

   accentColor =
    MaterialTheme.colorScheme.primary

   containerColor =
    MaterialTheme.colorScheme
     .primaryContainer
     .copy(alpha = 0.45f)
  }

  ConfidenceStatus.PROBABLE_RESULT -> {
   statusTitle = "Possível correspondência"
   resultTitle = result.displayName
   message =
    "Há semelhança visual, mas confirme com outra fotografia."

   accentColor =
    MaterialTheme.colorScheme.tertiary

   containerColor =
    MaterialTheme.colorScheme
     .tertiaryContainer
     .copy(alpha = 0.55f)
  }

  ConfidenceStatus.INCONCLUSIVE -> {
   statusTitle = "Imagem não reconhecida"
   resultTitle = "Edificação não identificada"
   message =
    "A fotografia não corresponde com segurança às edificações cadastradas."

   accentColor =
    MaterialTheme.colorScheme.error

   containerColor =
    MaterialTheme.colorScheme
     .errorContainer
     .copy(alpha = 0.50f)
  }
 }

 Card(
  modifier = Modifier.fillMaxWidth(),
  shape = RoundedCornerShape(16.dp),
  colors = CardDefaults.cardColors(
   containerColor = containerColor
  ),
  elevation = CardDefaults.cardElevation(
   defaultElevation = 3.dp
  )
 ) {
  Box(
   modifier = Modifier.fillMaxWidth()
  ) {
   Image(
    painter = painterResource(
     id = R.drawable.azulejos_decorativos
    ),
    contentDescription = null,
    modifier = Modifier
     .align(Alignment.TopEnd)
     .size(115.dp)
     .alpha(0.05f),
    contentScale = ContentScale.Fit
   )

   Column(
    modifier = Modifier.padding(14.dp)
   ) {
    Row(
     verticalAlignment = Alignment.CenterVertically
    ) {
     Icon(
      imageVector = Icons.Outlined.Verified,
      contentDescription = null,
      tint = accentColor,
      modifier = Modifier.size(18.dp)
     )

     Spacer(
      modifier = Modifier.width(6.dp)
     )

     Text(
      text = statusTitle,
      color = accentColor,
      style = MaterialTheme.typography.labelLarge,
      fontWeight = FontWeight.Bold
     )
    }

    Spacer(
     modifier = Modifier.height(7.dp)
    )

    Text(
     text = resultTitle,
     style = MaterialTheme.typography.titleLarge,
     fontWeight = FontWeight.ExtraBold
    )

    Spacer(
     modifier = Modifier.height(4.dp)
    )

    Text(
     text = if (
      result.status ==
      ConfidenceStatus.INCONCLUSIVE
     ) {
      "Maior probabilidade: $confidence"
     } else {
      "Confiança: $confidence"
     },
     style = MaterialTheme.typography.bodyMedium,
     fontWeight = FontWeight.Medium
    )

    Spacer(
     modifier = Modifier.height(8.dp)
    )

    Text(
     text = message,
     style = MaterialTheme.typography.bodyMedium,
     lineHeight = 19.sp
    )

    if (
     result.status ==
     ConfidenceStatus.PROBABLE_RESULT
    ) {
     Spacer(
      modifier = Modifier.height(14.dp)
     )

     HorizontalDivider()

     Spacer(
      modifier = Modifier.height(10.dp)
     )

     Text(
      text = "Outra possibilidade",
      fontWeight = FontWeight.Bold
     )

     Spacer(
      modifier = Modifier.height(4.dp)
     )

     Text(
      text = "${result.secondDisplayName} · " +
              secondConfidence
     )
    }

    Spacer(
     modifier = Modifier.height(10.dp)
    )

    Text(
     text = "Processamento local: " +
             "${result.inferenceTimeMs} ms",
     style = MaterialTheme.typography.labelSmall
    )

    if (
     result.status ==
     ConfidenceStatus.HIGH_CONFIDENCE &&
     building != null
    ) {
     Spacer(
      modifier = Modifier.height(16.dp)
     )

     HorizontalDivider()

     Spacer(
      modifier = Modifier.height(16.dp)
     )

     HistoricalInformation(building)
    }

    if (
     result.status ==
     ConfidenceStatus.INCONCLUSIVE
    ) {
     Spacer(
      modifier = Modifier.height(16.dp)
     )

     HorizontalDivider()

     Spacer(
      modifier = Modifier.height(14.dp)
     )

     CaptureTips()
    }

    Spacer(
     modifier = Modifier.height(16.dp)
    )

    OutlinedButton(
     onClick = onTryAgain,
     modifier = Modifier.fillMaxWidth(),
     shape = RoundedCornerShape(12.dp)
    ) {
     Icon(
      imageVector = Icons.Outlined.Refresh,
      contentDescription = "Realizar nova análise",
      modifier = Modifier.size(18.dp)
     )

     Spacer(
      modifier = Modifier.width(8.dp)
     )

     Text(
      text = "Nova análise"
     )
    }
   }
  }
 }
}

@Composable
private fun HistoricalInformation(
 building: BuildingInfo
) {
 Text(
  text = "Sobre a edificação",
  style = MaterialTheme.typography.titleLarge,
  fontWeight = FontWeight.ExtraBold
 )

 Spacer(
  modifier = Modifier.height(4.dp)
 )

 Text(
  text = building.subtitle,
  color = MaterialTheme.colorScheme.primary,
  fontWeight = FontWeight.SemiBold
 )

 InformationSection(
  title = "Descrição",
  content = building.description
 )

 InformationSection(
  title = "Contexto histórico",
  content = building.historicalContext
 )

 InformationSection(
  title = "Características arquitetônicas",
  content = building.architecturalFeatures
 )

 Spacer(
  modifier = Modifier.height(18.dp)
 )

 Row(
  verticalAlignment = Alignment.Top
 ) {
  Icon(
   imageVector = Icons.Outlined.LocationOn,
   contentDescription = null,
   tint = MaterialTheme.colorScheme.primary
  )

  Spacer(
   modifier = Modifier.width(8.dp)
  )

  Column {
   Text(
    text = "Localização",
    fontWeight = FontWeight.Bold
   )

   Spacer(
    modifier = Modifier.height(3.dp)
   )

   Text(
    text = building.location
   )
  }
 }

 Spacer(
  modifier = Modifier.height(12.dp)
 )

 Text(
  text = "Fonte: ${building.sourceName}",
  style = MaterialTheme.typography.bodySmall,
  color = MaterialTheme.colorScheme.onSurface
   .copy(alpha = 0.72f),
  lineHeight = 16.sp
 )
}

@Composable
private fun InformationSection(
 title: String,
 content: String
) {
 Spacer(
  modifier = Modifier.height(18.dp)
 )

 Text(
  text = title,
  fontWeight = FontWeight.Bold
 )

 Spacer(
  modifier = Modifier.height(5.dp)
 )

 Text(
  text = content,
  lineHeight = 20.sp
 )
}

@Composable
private fun CaptureTips() {
 Text(
  text = "Como melhorar a fotografia",
  fontWeight = FontWeight.Bold
 )

 Spacer(
  modifier = Modifier.height(8.dp)
 )

 val tips = listOf(
  "Centralize a fachada.",
  "Inclua portas e janelas.",
  "Evite reflexos e obstáculos.",
  "Não fotografe objetos isolados."
 )

 tips.forEach { tip ->
  Text(
   text = "• $tip",
   modifier = Modifier.padding(vertical = 4.dp)
  )
 }
}

private fun formatPercentage(
 value: Float
): String {
 return String.format(
  Locale.getDefault(),
  "%.2f%%",
  value * 100
 )
}