from pathlib import Path
from datetime import datetime
import csv
import hashlib
import re
import unicodedata

import cv2


# ============================================================
# CONFIGURAÇÕES
# ============================================================

RAIZ_PROJETO = Path(r"F:\casarao_ia")

PASTA_ENTRADA = RAIZ_PROJETO / "01_dados_brutos"
PASTA_SAIDA = RAIZ_PROJETO / "02_quadros_extraidos"
PASTA_RELATORIOS = RAIZ_PROJETO / "06_metadados"


# Extrair um quadro a cada segundo.
INTERVALO_SEGUNDOS = 1.0

EXTENSOES_VIDEO = {
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
    ".m4v",
}

# False: mantém quadros existentes.
# True: substitui quadros existentes.
SOBRESCREVER = False

QUALIDADE_JPEG = 95


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def validar_configuracoes():
    """Valida as configurações antes do início da extração."""

    if INTERVALO_SEGUNDOS <= 0:
        raise ValueError(
            "INTERVALO_SEGUNDOS deve ser maior que zero. "
            f"Valor informado: {INTERVALO_SEGUNDOS}"
        )

    if not 1 <= QUALIDADE_JPEG <= 100:
        raise ValueError(
            "QUALIDADE_JPEG deve estar entre 1 e 100. "
            f"Valor informado: {QUALIDADE_JPEG}"
        )

    if not PASTA_ENTRADA.exists():
        raise FileNotFoundError(
            f"A pasta de entrada não foi encontrada: {PASTA_ENTRADA}"
        )


def normalizar_nome(texto):
    """
    Converte um texto para um formato seguro para nomes de arquivos.

    Exemplo:
    'Vídeo Fachada 01' -> 'video_fachada_01'
    """

    texto = unicodedata.normalize("NFKD", texto)
    texto = texto.encode("ascii", "ignore").decode("ascii")
    texto = texto.lower().strip()
    texto = re.sub(r"[^a-z0-9]+", "_", texto)

    return texto.strip("_")


def gerar_identificador_video(caminho_video):
    """
    Gera um identificador curto e reproduzível para o vídeo.

    O identificador considera:
    - caminho relativo;
    - nome do arquivo;
    - extensão.

    Isso evita colisões entre arquivos com o mesmo nome-base.
    """

    caminho_relativo = caminho_video.relative_to(PASTA_ENTRADA)
    texto_identificador = caminho_relativo.as_posix().lower()

    hash_curto = hashlib.sha256(
        texto_identificador.encode("utf-8")
    ).hexdigest()[:10]

    extensao = normalizar_nome(
        caminho_video.suffix.lower().replace(".", "")
    )

    nome_base = normalizar_nome(caminho_video.stem)

    return f"{nome_base}_{extensao}_{hash_curto}"


def localizar_videos():
    """Localiza os vídeos existentes na estrutura esperada."""

    videos = []

    for arquivo in PASTA_ENTRADA.rglob("*"):
        if not arquivo.is_file():
            continue

        if arquivo.suffix.lower() not in EXTENSOES_VIDEO:
            continue

        try:
            obter_classe_e_sessao(arquivo)
        except ValueError as erro:
            print(f"[AVISO] Vídeo ignorado: {arquivo}")
            print(f"        Motivo: {erro}")
            print()
            continue

        videos.append(arquivo)

    return sorted(videos)


def obter_classe_e_sessao(caminho_video):
    """
    Identifica a classe e a sessão com base na estrutura:

    01_dados_brutos/
        classe/
            sessao_01/
                videos/
                    arquivo.mp4
    """

    caminho_relativo = caminho_video.relative_to(PASTA_ENTRADA)
    partes = caminho_relativo.parts

    if len(partes) != 4:
        raise ValueError(
            "Estrutura inválida. Esperado: "
            "classe/sessao/videos/arquivo.extensao. "
            f"Encontrado: {caminho_relativo}"
        )

    classe, sessao, pasta_tipo, nome_arquivo = partes

    if pasta_tipo.lower() != "videos":
        raise ValueError(
            f"O arquivo está na pasta '{pasta_tipo}', "
            "mas deveria estar na pasta 'videos'."
        )

    if not classe:
        raise ValueError("A classe da edificação não foi identificada.")

    if not sessao.startswith("sessao_"):
        raise ValueError(
            f"Nome de sessão inválido: '{sessao}'. "
            "Utilize nomes como sessao_01, sessao_02 etc."
        )

    if not nome_arquivo:
        raise ValueError("O nome do arquivo de vídeo está vazio.")

    return classe, sessao


def obter_duracao_estimada(captura, fps, total_quadros):
    """
    Calcula a duração estimada do vídeo.

    Retorna zero quando os metadados não forem suficientes.
    """

    if fps > 0 and total_quadros > 0:
        return total_quadros / fps

    return 0.0


def criar_resultado_base(video, classe="", sessao=""):
    """Cria o dicionário padrão utilizado no relatório."""

    return {
        "video": str(video),
        "classe": classe,
        "sessao": sessao,
        "status": "",
        "fps": 0,
        "total_quadros_video": 0,
        "duracao_segundos": 0,
        "intervalo_segundos": INTERVALO_SEGUNDOS,
        "quadros_planejados": 0,
        "quadros_salvos": 0,
        "quadros_existentes": 0,
        "falhas_gravacao": 0,
        "observacao": "",
    }


# ============================================================
# EXTRAÇÃO
# ============================================================

def extrair_quadros(video, intervalo_segundos):
    """Extrai quadros de um vídeo em intervalos regulares."""

    if intervalo_segundos <= 0:
        raise ValueError(
            "O intervalo de extração deve ser maior que zero. "
            f"Valor recebido: {intervalo_segundos}"
        )

    classe, sessao = obter_classe_e_sessao(video)

    resultado = criar_resultado_base(
        video=video,
        classe=classe,
        sessao=sessao,
    )

    pasta_destino = PASTA_SAIDA / classe / sessao
    pasta_destino.mkdir(parents=True, exist_ok=True)

    captura = cv2.VideoCapture(str(video))

    if not captura.isOpened():
        resultado["status"] = "erro_abertura"
        resultado["observacao"] = "Não foi possível abrir o vídeo."
        return resultado

    try:
        fps = float(captura.get(cv2.CAP_PROP_FPS))
        total_quadros = int(
            captura.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        resultado["fps"] = round(fps, 3) if fps > 0 else fps
        resultado["total_quadros_video"] = max(0, total_quadros)

        if fps <= 0:
            resultado["status"] = "fps_invalido"
            resultado["observacao"] = (
                "O vídeo não informou uma taxa de quadros válida."
            )
            return resultado

        duracao_estimada = obter_duracao_estimada(
            captura=captura,
            fps=fps,
            total_quadros=total_quadros,
        )

        resultado["duracao_segundos"] = round(
            duracao_estimada,
            3,
        )

        intervalo_quadros = round(fps * intervalo_segundos)

        if intervalo_quadros <= 0:
            raise ValueError(
                "O intervalo calculado em quadros é inválido: "
                f"{intervalo_quadros}"
            )

        if total_quadros > 0:
            resultado["quadros_planejados"] = (
                (total_quadros - 1) // intervalo_quadros
            ) + 1

        identificador_video = gerar_identificador_video(video)

        quadro_atual = 0
        quantidade_salva = 0
        quantidade_existente = 0
        falhas_gravacao = 0
        mensagens_falha = []

        parametros_jpeg = [
            cv2.IMWRITE_JPEG_QUALITY,
            QUALIDADE_JPEG,
        ]

        while True:
            sucesso, imagem = captura.read()

            if not sucesso:
                break

            if quadro_atual % intervalo_quadros == 0:
                instante_segundos = quadro_atual / fps
                nome_imagem = (
                    f"{classe}_{sessao}_{identificador_video}"
                    f"_t{instante_segundos:07.2f}s.jpg"
                )
                caminho_imagem = pasta_destino / nome_imagem

                if caminho_imagem.exists() and not SOBRESCREVER:
                    quantidade_existente += 1
                else:
                    arquivo_salvo = cv2.imwrite(
                        str(caminho_imagem),
                        imagem,
                        parametros_jpeg,
                    )

                    if arquivo_salvo:
                        quantidade_salva += 1
                    else:
                        falhas_gravacao += 1
                        if len(mensagens_falha) < 3:
                            mensagens_falha.append(
                                f"Falha ao salvar {caminho_imagem}"
                            )

            quadro_atual += 1

        resultado["quadros_salvos"] = quantidade_salva
        resultado["quadros_existentes"] = quantidade_existente
        resultado["falhas_gravacao"] = falhas_gravacao

        if falhas_gravacao:
            resultado["status"] = "concluido_com_falhas"
            resultado["observacao"] = "; ".join(mensagens_falha)
        else:
            resultado["status"] = "concluido"

        return resultado
    finally:
        captura.release()


# ============================================================
# RELATÓRIO
# ============================================================

def salvar_relatorio(resultados):
    """Salva o relatório da extração em formato CSV."""

    PASTA_RELATORIOS.mkdir(parents=True, exist_ok=True)
    data_execucao = datetime.now().strftime("%Y%m%d_%H%M%S")
    arquivo_relatorio = (
        PASTA_RELATORIOS
        / f"relatorio_extracao_{data_execucao}.csv"
    )

    campos = [
        "video",
        "classe",
        "sessao",
        "status",
        "fps",
        "total_quadros_video",
        "duracao_segundos",
        "intervalo_segundos",
        "quadros_planejados",
        "quadros_salvos",
        "quadros_existentes",
        "falhas_gravacao",
        "observacao",
    ]

    with arquivo_relatorio.open(
        mode="w",
        newline="",
        encoding="utf-8-sig",
    ) as arquivo_csv:
        escritor = csv.DictWriter(
            arquivo_csv,
            fieldnames=campos,
            delimiter=";",
        )
        escritor.writeheader()
        escritor.writerows(resultados)

    return arquivo_relatorio


# ============================================================
# EXECUÇÃO
# ============================================================

def main():
    print("=" * 65)
    print("CASARÃO IA - EXTRAÇÃO DE QUADROS")
    print("=" * 65)
    print(f"Pasta de entrada: {PASTA_ENTRADA}")
    print(f"Pasta de saída:   {PASTA_SAIDA}")
    print(f"Intervalo:        {INTERVALO_SEGUNDOS} segundo(s)")
    print()

    validar_configuracoes()
    videos = localizar_videos()

    if not videos:
        print("Nenhum vídeo válido foi encontrado.")
        return

    print(f"Vídeos encontrados: {len(videos)}")
    print()

    resultados = []
    total_quadros = 0

    for numero, video in enumerate(videos, start=1):
        print(f"[{numero}/{len(videos)}] Processando:")
        print(f"    {video}")

        try:
            resultado = extrair_quadros(
                video=video,
                intervalo_segundos=INTERVALO_SEGUNDOS,
            )
        except Exception as erro:
            resultado = criar_resultado_base(video)
            resultado["status"] = "erro"
            resultado["observacao"] = str(erro)

        resultados.append(resultado)
        total_quadros += resultado["quadros_salvos"]

        print(f"    Status: {resultado['status']}")
        print(f"    Quadros salvos: {resultado['quadros_salvos']}")
        print(f"    Já existentes: {resultado['quadros_existentes']}")
        if resultado["observacao"]:
            print(f"    Observação: {resultado['observacao']}")
        print()

    relatorio = salvar_relatorio(resultados)

    print("=" * 65)
    print("EXTRAÇÃO FINALIZADA")
    print("=" * 65)
    print(f"Vídeos processados: {len(videos)}")
    print(f"Quadros salvos:     {total_quadros}")
    print(f"Relatório:          {relatorio}")


if __name__ == "__main__":
    main()