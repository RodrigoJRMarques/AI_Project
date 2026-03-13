# Trabalho Prático Final – Métodos de Procura, OCR e LLM

> **Unidade Curricular** – IADE / Universidade Europeia · 2026  
> **Valor** – 8 valores  
> **Entrega** – 17 de Abril de 2026  

Programa Python que encontra caminhos entre cidades portuguesas usando quatro
algoritmos de procura, com login via OCR de matrícula e descrição de atrações
turísticas fornecida por um LLM local.

---

## Funcionalidades

| Feature | Detalhe |
|---|---|
| **Login via OCR** | Lê a matrícula do veículo a partir de uma imagem |
| **Custo Uniforme** | Garante o caminho de menor custo |
| **Profundidade Limitada** | DFS com limite de profundidade configurável |
| **Procura Sôfrega** | Guia-se pela heurística (distância em linha reta a Faro) |
| **A\*** | f(n) = g(n) + h(n) — ótimo e eficiente |
| **Iterações** | Tabela com cada nó expandido e os seus valores g/h/f |
| **Atrações LLM** | Descrição das 3 principais atrações de cada cidade (Ollama) |

---

## Estrutura do Projeto

```
AI_Project/
├── main.py          # Ponto de entrada (interface CLI)
├── graph.py         # Grafo de cidades + heurística
├── algorithms.py    # UCS · DLS · Greedy · A*
├── ocr.py           # Leitura de matrícula via OCR
├── llm_client.py    # Cliente Ollama (LLM local)
└── requirements.txt
```

---

## Instalação

### 1. Pré-requisitos
- Python 3.10+

### 2. Instalar dependências Python
```bash
pip install -r requirements.txt
```

### 3. (Opcional) LLM local — Ollama
1. Instalar Ollama: <https://ollama.com>
2. Transferir um modelo:
   ```bash
   ollama pull llama3.2
   ```
3. Garantir que o servidor está ativo (`ollama serve`)

> Se o Ollama não estiver disponível, o programa usa informações pré-definidas
> sobre as atrações de cada cidade.

### 4. (Opcional) OCR com pytesseract
Se preferir pytesseract em vez de easyocr, instale o Tesseract-OCR:
- **Windows**: <https://github.com/UB-Mannheim/tesseract/wiki>
- **Linux/macOS**: `sudo apt install tesseract-ocr` / `brew install tesseract`

---

## Utilização

```bash
python main.py
```

Fluxo:
1. **Login** – fornece imagem de matrícula ou introduz manualmente
2. **Origem** – escolhe cidade de partida
3. **Destino** – escolhe cidade de chegada
4. **Algoritmo** – seleciona UCS, DLS, Greedy ou A*
5. **Resultados** – tabela de iterações + caminho final + atrações (LLM)

---

## Grafo de Cidades

### Tabela 1 – Distâncias rodoviárias (km)

| Cidade | Ligações |
|---|---|
| Aveiro | Porto (68) · Viseu (95) · Coimbra (68) · Leiria (115) |
| Braga | Viana do Castelo (48) · Vila Real (106) · Porto (53) |
| Bragança | Vila Real (137) · Guarda (202) |
| Beja | Évora (78) · Faro (152) · Setúbal (142) |
| Castelo Branco | Coimbra (159) · Guarda (106) · Portalegre (80) · Évora (203) |
| Coimbra | Viseu (96) · Leiria (67) |
| Évora | Lisboa (150) · Santarém (117) · Portalegre (131) · Setúbal (103) |
| Faro | Setúbal (249) · Lisboa (299) |
| Guarda | Vila Real (157) · Viseu (85) |
| Leiria | Lisboa (129) · Santarém (70) |
| Lisboa | Santarém (78) · Setúbal (50) |
| Porto | Viana do Castelo (71) · Vila Real (116) · Viseu (133) |
| Vila Real | Viseu (110) |

*Todas as ligações são bidirecionais.*

### Tabela 2 – Heurística: distância em linha reta a Faro (km)

| Cidade | h | Cidade | h |
|---|---|---|---|
| Aveiro | 366 | Portalegre | 228 |
| Braga | 454 | Porto | 418 |
| Bragança | 487 | Santarém | 231 |
| Beja | 99 | Setúbal | 168 |
| Castelo Branco | 280 | Viana do Castelo | 473 |
| Coimbra | 319 | Vila Real | 429 |
| Évora | 157 | Viseu | 363 |
| Guarda | 352 | Faro | 0 |
| Leiria | 278 | Lisboa | 195 |

---

## Algoritmos

### Custo Uniforme (UCS)
Expande sempre o nó com menor custo acumulado **g(n)**. Garante o caminho
ótimo independentemente da heurística.

### Profundidade Limitada (DLS)
DFS com limite de profundidade configurável. Não garante a solução ótima mas
é útil para explorar grafos grandes sem se perder em ramos infinitos.

### Procura Sôfrega (Greedy)
Expande o nó com menor valor heurístico **h(n)**. Rápido mas não garante
a solução ótima.

### A\*
Combina custo real e heurística: **f(n) = g(n) + h(n)**. Com a heurística
admissível fornecida (distância em linha reta ≤ distância real), garante a
solução ótima.

---

## Autores


## Elementos no grupo

Francisco Vitorino 25288\
Rodrigo Marques 25971\
Tiago Carvalgo 22598
