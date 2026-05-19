# Sistema Inteligente de Triagem Educacional

Trabalho N2 — Inteligência Artificial  
Prof. Claudinei Dias (Ney) — Católica de Santa Catarina

**Integrantes:** EDER ZEREK DUARTE, LUIS FELIPE FACHINI, JOSÉ LUCAS ANDRADE FONSECA, MAX BUZZARELLO MAUL, SIDNEY CARDOSO DE OLIVEIRA JUNIOR


---

## O que é

Sistema que analisa o feedback de alunos sobre tópicos de um curso e sugere a melhor sequência de estudo. O projeto segue a linha do Exemplo 5 da especificação (Curadoria de Conteúdo Educacional), utilizando SA no lugar do GA.

As três camadas são encadeadas: a saída de uma é a entrada da próxima.

---

## Dataset

**Coursera Reviews** (Kaggle) — avaliações reais de alunos com notas de 1 a 5 estrelas.

Download: https://www.kaggle.com/datasets/imuhammad/course-reviews-on-coursera

Após baixar, colocar os dois arquivos em `backend/data/` (não estão no repositório por causa do tamanho):
- `reviews.csv`
- `reviews_by_course.csv`

---

## As três camadas

**Camada I — Naive Bayes** (`layer1_nlp.py`)

Classifica o texto do feedback como positivo, negativo ou neutro. Faz pré-processamento completo (tokenização, stop words, stemming) e usa TF-IDF com MultinomialNB. Treinado com ~14k amostras balanceadas do dataset, acurácia de 66,7%. A saída principal é a probabilidade de o texto ser positivo (0 a 1).

**Camada II — Fuzzy Mamdani** (`layer2_fuzzy.py`)

Recebe a probabilidade positiva da Camada I e a nota do aluno (1–5) e calcula um score de engajamento de 0 a 10. Tem 9 regras cobrindo todo o espaço de entrada — por exemplo: sentimento positivo + nota alta = engajamento alto; sentimento negativo + nota alta = engajamento médio (o Fuzzy trata essa contradição).

**Camada III — Simulated Annealing** (`layer3_sa.py`)

Recebe os scores de engajamento de todos os tópicos e encontra a sequência de estudo ideal. A função de custo penaliza saltos bruscos de dificuldade e começo com tópico de baixo engajamento. Vizinhança por swap entre dois tópicos aleatórios.

---

## Como rodar

**Pré-requisitos:** Python 3.10+ e Node.js 18+

**Backend**
```bash
cd backend
pip install -r requirements.txt
python -m uvicorn main:app --port 8001 --reload
```
O treinamento do modelo acontece na inicialização (~30s). Quando aparecer `Application startup complete` está pronto.

**Frontend** (em outro terminal)
```bash
cd frontend
npm install
npm run dev
```

Acesse `http://localhost:5173`

> **Windows:** se o `npm` travar com erro de permissão, rode antes:
> `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned -Force`
