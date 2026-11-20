from pathlib import Path
from datetime import datetime
import csv
import hashlib
import shutil
import sys


# ============================================================
# CONFIGURAÇÕES
# ============================================================

RAIZ_PROJETO = Path(r"F:\casarao_ia")

PASTA_ORIGEM = (
    RAIZ_PROJETO
    / "03_imagens_convertidas"
)

PASTA_DESCARTADAS = (
    RAIZ_PROJETO
    / "03_imagens_descartadas"
)

PASTA_RELATORIOS = (
    RAIZ_PROJETO
    / "06_metadados"
)

RELATORIO_SIMILARIDADE = (
    PASTA_RELATORIOS
    / "relatorio_similaridade_20261118_114622.csv"
)

# False: apenas simula.
# True: movimenta ou remove da origem quando já existe
# uma cópia idêntica na pasta de descartadas.
EXECUTAR_MOVIMENTACAO = True

# Somente os pares com distância perceptual igual a zero
# serão considerados.
DISTANCIA_PARA_DESCARTE = 0

# False: considera todos os pares com distância zero.
# True: considera somente pares pertencentes à mesma sessão.
SOMENTE_MESMA_SESSAO = False

EXTENSOES_VALIDAS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
}


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def converter_distancia(valor):
    """
    Converte o valor de distância do CSV para inteiro.
    """

    texto = str(valor).strip().replace(",", ".")

    if not texto:
        raise ValueError(
            "O valor de distância está vazio."
        )

    return int(float(texto))


def calcular_sha256(caminho):
    """
    Calcula o hash SHA-256 de um arquivo.
    """

    hash_arquivo = hashlib.sha256()

    with caminho.open("rb") as arquivo:
        for bloco in iter(
            lambda: arquivo.read(1024 * 1024),
            b"",
        ):
            hash_arquivo.update(bloco)

    return hash_arquivo.hexdigest()


def caminhos_equivalentes(caminho_a, caminho_b):
    """
    Verifica se dois caminhos representam o mesmo arquivo,
    sem depender de diferenças entre letras maiúsculas e
    minúsculas no Windows.
    """

    return (
        str(caminho_a).casefold()
        == str(caminho_b).casefold()
    )


def criar_destino(arquivo_origem):
    """
    Preserva a estrutura relativa da imagem.

    Exemplo:

    Origem:
    03_imagens_convertidas/
        palacio_dos_leoes/
            sessao_01/
                imagem.jpg

    Destino:
    03_imagens_descartadas/
        palacio_dos_leoes/
            sessao_01/
                imagem.jpg
    """

    caminho_relativo = (
        arquivo_origem.relative_to(
            PASTA_ORIGEM
        )
    )

    return (
        PASTA_DESCARTADAS
        / caminho_relativo
    )


def localizar_destino_pelo_caminho_original(
    caminho_original,
):
    """
    Obtém o destino esperado mesmo quando o arquivo de origem
    já não existe.

    O caminho ainda precisa estar localizado logicamente dentro
    de 03_imagens_convertidas.
    """

    caminho_relativo = (
        caminho_original.relative_to(
            PASTA_ORIGEM
        )
    )

    return (
        PASTA_DESCARTADAS
        / caminho_relativo
    )


def criar_destino_alternativo(
    destino,
    hash_origem,
):
    """
    Cria um nome alternativo quando já existe um arquivo com
    o mesmo nome no destino, mas com conteúdo diferente.
    """

    identificador = hash_origem[:10]

    return destino.with_name(
        f"{destino.stem}_"
        f"{identificador}"
        f"{destino.suffix}"
    )


def contar_imagens(pasta):
    """
    Conta as imagens existentes em uma pasta.
    """

    if not pasta.exists():
        return 0

    return sum(
        1
        for arquivo in pasta.rglob("*")
        if (
            arquivo.is_file()
            and arquivo.suffix.lower()
            in EXTENSOES_VALIDAS
        )
    )


# ============================================================
# LEITURA DO RELATÓRIO
# ============================================================

def ler_arquivos_b_para_descarte():
    """
    Lê o relatório de similaridade e retorna os caminhos únicos
    da coluna arquivo_b quando distancia_hash for igual a zero.

    Caminhos repetidos são consolidados automaticamente.
    """

    arquivos_descartar = set()
    erros = []

    with RELATORIO_SIMILARIDADE.open(
        mode="r",
        encoding="utf-8-sig",
        newline="",
    ) as arquivo_csv:

        leitor = csv.DictReader(
            arquivo_csv,
            delimiter=";",
        )

        colunas_necessarias = {
            "arquivo_b",
            "distancia_hash",
        }

        if SOMENTE_MESMA_SESSAO:
            colunas_necessarias.update(
                {
                    "sessao_a",
                    "sessao_b",
                }
            )

        colunas_encontradas = set(
            leitor.fieldnames or []
        )

        colunas_ausentes = (
            colunas_necessarias
            - colunas_encontradas
        )

        if colunas_ausentes:
            raise ValueError(
                "Colunas ausentes no relatório: "
                + ", ".join(
                    sorted(colunas_ausentes)
                )
            )

        for numero_linha, linha in enumerate(
            leitor,
            start=2,
        ):
            try:
                distancia = converter_distancia(
                    linha.get(
                        "distancia_hash",
                        "",
                    )
                )

            except Exception as erro:
                erros.append(
                    f"Linha {numero_linha}: "
                    f"distância inválida: {erro}"
                )
                continue

            if (
                distancia
                != DISTANCIA_PARA_DESCARTE
            ):
                continue

            if SOMENTE_MESMA_SESSAO:
                sessao_a = (
                    linha.get("sessao_a")
                    or ""
                ).strip()

                sessao_b = (
                    linha.get("sessao_b")
                    or ""
                ).strip()

                if sessao_a != sessao_b:
                    continue

            caminho_texto = (
                linha.get("arquivo_b")
                or ""
            ).strip()

            if not caminho_texto:
                erros.append(
                    f"Linha {numero_linha}: "
                    "arquivo_b vazio."
                )
                continue

            caminho = Path(caminho_texto)

            try:
                caminho.relative_to(
                    PASTA_ORIGEM
                )

            except ValueError:
                erros.append(
                    f"Linha {numero_linha}: "
                    "arquivo fora da pasta "
                    f"esperada: {caminho}"
                )
                continue

            if (
                caminho.suffix.lower()
                not in EXTENSOES_VALIDAS
            ):
                erros.append(
                    f"Linha {numero_linha}: "
                    "extensão não permitida: "
                    f"{caminho}"
                )
                continue

            arquivos_descartar.add(
                caminho
            )

    return (
        sorted(
            arquivos_descartar,
            key=lambda caminho: (
                str(caminho).casefold()
            ),
        ),
        erros,
    )


# ============================================================
# RELATÓRIO DA MOVIMENTAÇÃO
# ============================================================

def salvar_relatorio(
    registros,
    modo,
):
    """
    Salva o relatório da execução em CSV.
    """

    PASTA_RELATORIOS.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    arquivo_relatorio = (
        PASTA_RELATORIOS
        / (
            "relatorio_duplicatas_"
            f"{modo}_{timestamp}.csv"
        )
    )

    campos = [
        "arquivo_origem",
        "arquivo_destino",
        "classe",
        "sessao",
        "distancia_hash",
        "acao",
        "status",
        "hash_origem",
        "hash_destino",
        "observacao",
    ]

    with arquivo_relatorio.open(
        mode="w",
        encoding="utf-8-sig",
        newline="",
    ) as arquivo_csv:

        escritor = csv.DictWriter(
            arquivo_csv,
            fieldnames=campos,
            delimiter=";",
        )

        escritor.writeheader()
        escritor.writerows(registros)

    return arquivo_relatorio


# ============================================================
# PROCESSAMENTO
# ============================================================

def processar_arquivo(
    origem,
    executar,
):
    """
    Processa um arquivo indicado no relatório.

    Possíveis situações:

    1. Origem existe e destino não existe:
       move o arquivo.

    2. Origem existe e destino possui cópia idêntica:
       remove somente a origem, pois o backup já existe.

    3. Origem existe e destino possui conteúdo diferente:
       move a origem usando um nome alternativo.

    4. Origem não existe e destino existe:
       considera que o arquivo já foi processado.

    5. Origem e destino não existem:
       registra erro.
    """

    destino = localizar_destino_pelo_caminho_original(
        origem
    )

    caminho_relativo = (
        origem.relative_to(
            PASTA_ORIGEM
        )
    )

    partes = caminho_relativo.parts

    classe = (
        partes[0]
        if len(partes) >= 1
        else ""
    )

    sessao = (
        partes[1]
        if len(partes) >= 2
        else ""
    )

    registro = {
        "arquivo_origem": str(origem),
        "arquivo_destino": str(destino),
        "classe": classe,
        "sessao": sessao,
        "distancia_hash": (
            DISTANCIA_PARA_DESCARTE
        ),
        "acao": "mover_arquivo_b",
        "status": "",
        "hash_origem": "",
        "hash_destino": "",
        "observacao": "",
    }

    # --------------------------------------------------------
    # CASO 1: origem não existe
    # --------------------------------------------------------

    if not origem.exists():
        if destino.exists():
            registro["status"] = (
                "ja_processado"
            )

            registro["hash_destino"] = (
                calcular_sha256(destino)
            )

            registro["observacao"] = (
                "O arquivo já havia sido "
                "movido em uma execução anterior."
            )

            return registro

        registro["status"] = (
            "arquivo_nao_encontrado"
        )

        registro["observacao"] = (
            "O arquivo não foi encontrado "
            "nem na origem nem no destino."
        )

        return registro

    # --------------------------------------------------------
    # Origem existe: calcular seu hash
    # --------------------------------------------------------

    hash_origem = calcular_sha256(
        origem
    )

    registro["hash_origem"] = (
        hash_origem
    )

    # --------------------------------------------------------
    # CASO 2: destino não existe
    # --------------------------------------------------------

    if not destino.exists():
        if executar:
            destino.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            shutil.move(
                str(origem),
                str(destino),
            )

            registro["status"] = (
                "movido"
            )

            registro["observacao"] = (
                "Arquivo transferido para "
                "a pasta de descartadas."
            )

        else:
            registro["status"] = (
                "simulado_mover"
            )

            registro["observacao"] = (
                "Na execução definitiva, "
                "o arquivo será movido."
            )

        return registro

    # --------------------------------------------------------
    # Destino existe: comparar conteúdo
    # --------------------------------------------------------

    hash_destino = calcular_sha256(
        destino
    )

    registro["hash_destino"] = (
        hash_destino
    )

    # --------------------------------------------------------
    # CASO 3: arquivos idênticos
    # --------------------------------------------------------

    if hash_origem == hash_destino:
        if executar:
            origem.unlink()

            registro["status"] = (
                "removido_origem_com_backup"
            )

            registro["observacao"] = (
                "Uma cópia idêntica já estava "
                "preservada na pasta de "
                "descartadas. A cópia da origem "
                "foi removida."
            )

        else:
            registro["status"] = (
                "simulado_remover_origem"
            )

            registro["observacao"] = (
                "Na execução definitiva, "
                "a origem será removida porque "
                "já existe uma cópia idêntica "
                "na pasta de descartadas."
            )

        return registro

    # --------------------------------------------------------
    # CASO 4: mesmo nome, conteúdo diferente
    # --------------------------------------------------------

    destino_alternativo = (
        criar_destino_alternativo(
            destino,
            hash_origem,
        )
    )

    registro["arquivo_destino"] = str(
        destino_alternativo
    )

    if destino_alternativo.exists():
        hash_alternativo = (
            calcular_sha256(
                destino_alternativo
            )
        )

        if hash_alternativo == hash_origem:
            if executar:
                origem.unlink()

                registro["status"] = (
                    "removido_origem_com_"
                    "backup_alternativo"
                )

                registro["observacao"] = (
                    "Uma cópia idêntica já "
                    "existia com nome alternativo. "
                    "A origem foi removida."
                )

            else:
                registro["status"] = (
                    "simulado_remover_origem_"
                    "backup_alternativo"
                )

                registro["observacao"] = (
                    "Na execução definitiva, "
                    "a origem será removida, pois "
                    "já existe uma cópia idêntica "
                    "com nome alternativo."
                )

            return registro

        registro["status"] = (
            "conflito_destino_alternativo"
        )

        registro["observacao"] = (
            "O destino alternativo já existe "
            "com conteúdo diferente. "
            "Nenhuma alteração foi realizada."
        )

        return registro

    if executar:
        destino_alternativo.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        shutil.move(
            str(origem),
            str(destino_alternativo),
        )

        registro["status"] = (
            "movido_nome_alternativo"
        )

        registro["observacao"] = (
            "Já existia um arquivo com o "
            "mesmo nome e conteúdo diferente. "
            "A imagem foi movida usando o "
            "hash no nome."
        )

    else:
        registro["status"] = (
            "simulado_nome_alternativo"
        )

        registro["observacao"] = (
            "Na execução definitiva, "
            "a imagem será movida usando "
            "um nome alternativo."
        )

    return registro


# ============================================================
# EXECUÇÃO PRINCIPAL
# ============================================================

def main():
    print("=" * 76)
    print(
        "CASARÃO IA - MOVIMENTAÇÃO "
        "DE IMAGENS REDUNDANTES"
    )
    print("=" * 76)

    print(
        f"Origem:      {PASTA_ORIGEM}"
    )

    print(
        f"Descartadas: {PASTA_DESCARTADAS}"
    )

    print(
        f"Relatório:   {RELATORIO_SIMILARIDADE}"
    )

    print(
        "Modo:        "
        + (
            "MOVIMENTAÇÃO"
            if EXECUTAR_MOVIMENTACAO
            else "SIMULAÇÃO"
        )
    )

    print(
        "Filtro:      distância "
        f"{DISTANCIA_PARA_DESCARTE}"
    )

    print(
        "Mesma sessão: "
        f"{SOMENTE_MESMA_SESSAO}"
    )

    print()

    # --------------------------------------------------------
    # Validações iniciais
    # --------------------------------------------------------

    if not PASTA_ORIGEM.exists():
        print(
            "ERRO: pasta de origem "
            "não encontrada:"
        )
        print(PASTA_ORIGEM)
        return 1

    if not RELATORIO_SIMILARIDADE.exists():
        print(
            "ERRO: relatório de "
            "similaridade não encontrado:"
        )
        print(RELATORIO_SIMILARIDADE)
        return 1

    try:
        arquivos, erros = (
            ler_arquivos_b_para_descarte()
        )

    except Exception as erro:
        print(
            "ERRO AO LER O RELATÓRIO:"
        )

        print(
            f"{type(erro).__name__}: "
            f"{erro}"
        )

        return 1

    if erros:
        print(
            "ERROS ENCONTRADOS "
            "NO RELATÓRIO:"
        )

        for erro in erros:
            print(f"- {erro}")

        print()
        print(
            "Nenhum arquivo foi alterado."
        )

        return 1

    if not arquivos:
        print(
            "Nenhum arquivo B com a "
            "distância definida foi encontrado."
        )

        return 0

    # --------------------------------------------------------
    # Totais iniciais
    # --------------------------------------------------------

    total_origem_antes = contar_imagens(
        PASTA_ORIGEM
    )

    total_descartadas_antes = (
        contar_imagens(
            PASTA_DESCARTADAS
        )
    )

    print(
        "Arquivos B únicos identificados: "
        f"{len(arquivos)}"
    )

    print(
        "Imagens na origem antes:         "
        f"{total_origem_antes}"
    )

    print(
        "Imagens descartadas antes:       "
        f"{total_descartadas_antes}"
    )

    print()

    # --------------------------------------------------------
    # Processamento
    # --------------------------------------------------------

    registros = []

    for numero, origem in enumerate(
        arquivos,
        start=1,
    ):
        try:
            relativo = origem.relative_to(
                PASTA_ORIGEM
            )

        except ValueError:
            relativo = origem

        print(
            f"[{numero}/{len(arquivos)}] "
            f"{relativo}"
        )

        try:
            registro = processar_arquivo(
                origem=origem,
                executar=(
                    EXECUTAR_MOVIMENTACAO
                ),
            )

        except Exception as erro:
            registro = {
                "arquivo_origem": str(origem),
                "arquivo_destino": "",
                "classe": "",
                "sessao": "",
                "distancia_hash": (
                    DISTANCIA_PARA_DESCARTE
                ),
                "acao": "erro",
                "status": "erro_processamento",
                "hash_origem": "",
                "hash_destino": "",
                "observacao": (
                    f"{type(erro).__name__}: "
                    f"{erro}"
                ),
            }

        registros.append(registro)

        print(
            f"    Status: "
            f"{registro['status']}"
        )

        if registro["observacao"]:
            print(
                f"    Observação: "
                f"{registro['observacao']}"
            )

    # --------------------------------------------------------
    # Resumo pelos status
    # --------------------------------------------------------

    status_movidos = {
        "movido",
        "movido_nome_alternativo",
        "removido_origem_com_backup",
        (
            "removido_origem_com_"
            "backup_alternativo"
        ),
    }

    status_simulados = {
        "simulado_mover",
        "simulado_remover_origem",
        "simulado_nome_alternativo",
        (
            "simulado_remover_origem_"
            "backup_alternativo"
        ),
    }

    movimentados = sum(
        registro["status"]
        in status_movidos
        for registro in registros
    )

    simulados = sum(
        registro["status"]
        in status_simulados
        for registro in registros
    )

    ja_processados = sum(
        registro["status"]
        == "ja_processado"
        for registro in registros
    )

    nao_encontrados = sum(
        registro["status"]
        == "arquivo_nao_encontrado"
        for registro in registros
    )

    conflitos = sum(
        registro["status"]
        == "conflito_destino_alternativo"
        for registro in registros
    )

    erros_processamento = sum(
        registro["status"]
        == "erro_processamento"
        for registro in registros
    )

    # --------------------------------------------------------
    # Relatório
    # --------------------------------------------------------

    modo = (
        "movimentacao"
        if EXECUTAR_MOVIMENTACAO
        else "simulacao"
    )

    relatorio_saida = salvar_relatorio(
        registros,
        modo,
    )

    total_origem_depois = contar_imagens(
        PASTA_ORIGEM
    )

    total_descartadas_depois = (
        contar_imagens(
            PASTA_DESCARTADAS
        )
    )

    # --------------------------------------------------------
    # Resultado final
    # --------------------------------------------------------

    print()
    print("=" * 76)
    print("OPERAÇÃO FINALIZADA")
    print("=" * 76)

    print(
        f"Arquivos identificados:   "
        f"{len(arquivos)}"
    )

    print(
        f"Movidos/removidos:        "
        f"{movimentados}"
    )

    print(
        f"Simulados:                 "
        f"{simulados}"
    )

    print(
        f"Já processados:            "
        f"{ja_processados}"
    )

    print(
        f"Não encontrados:           "
        f"{nao_encontrados}"
    )

    print(
        f"Conflitos:                 "
        f"{conflitos}"
    )

    print(
        f"Erros de processamento:    "
        f"{erros_processamento}"
    )

    print(
        f"Origem antes:              "
        f"{total_origem_antes}"
    )

    print(
        f"Origem depois:             "
        f"{total_origem_depois}"
    )

    print(
        f"Descartadas antes:         "
        f"{total_descartadas_antes}"
    )

    print(
        f"Descartadas depois:        "
        f"{total_descartadas_depois}"
    )

    print(
        f"Relatório gerado:          "
        f"{relatorio_saida}"
    )

    # --------------------------------------------------------
    # Validação final
    # --------------------------------------------------------

    if not EXECUTAR_MOVIMENTACAO:
        print()
        print(
            "SIMULAÇÃO CONCLUÍDA: "
            "nenhum arquivo foi alterado."
        )

        print(
            "Confira os resultados e altere "
            "EXECUTAR_MOVIMENTACAO = True."
        )

        return 0

    esperado_origem = (
        total_origem_antes
        - movimentados
    )

    if (
        total_origem_depois
        != esperado_origem
    ):
        print()
        print(
            "ATENÇÃO: o total da origem "
            "não corresponde ao esperado."
        )

        print(
            f"Esperado: {esperado_origem}"
        )

        print(
            f"Encontrado: "
            f"{total_origem_depois}"
        )

        return 1

    if (
        conflitos > 0
        or erros_processamento > 0
        or nao_encontrados > 0
    ):
        print()
        print(
            "ATENÇÃO: a operação terminou, "
            "mas existem pendências no relatório."
        )

        return 1

    print()
    print(
        "Movimentação validada "
        "com sucesso."
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())