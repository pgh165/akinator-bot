# 🎯 Akinator Bot

> **LLM 기반 아키네이터 챗봇** — 마음속으로 생각한 대상을 AI가 예/아니오 질문으로 맞춥니다.

[![Django](https://img.shields.io/badge/Django-5.x-092E20?style=flat-square&logo=django&logoColor=white)](https://www.djangoproject.com/)
[![Ollama](https://img.shields.io/badge/Ollama-LLM-black?style=flat-square)](https://ollama.com/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=flat-square&logo=docker&logoColor=white)](https://docs.docker.com/compose/)

---

## 소개

사용자가 머릿속으로 하나의 대상(인물, 동물, 사물, 브랜드 등)을 떠올리면, 로컬에서 구동되는 **Ollama LLM**이 "예/아니오/잘 모름" 질문을 반복하며 그 대상을 추론해 맞추는 웹 게임입니다.

### 주요 특징

- **완전 로컬 실행** — 외부 API 없이 Docker Compose + Ollama로 GPU 환경에서 동작
- **Best-of-N 질문 선택** — 매 턴 N개의 질문 후보를 병렬 생성하고, 후보군을 가장 효과적으로 절반 나누는 질문을 자동 선택
- **임베딩 기반 중복 질문 감지** — `bge-m3` 임베딩으로 의미적으로 같은 질문 반복을 방지
- **카테고리별 질문 로드맵** — 인물(실존/가상 분기), 동물, 사물 각각에 최적화된 질문 우선순위
- **후보군 동적 추정** — 매 턴 LLM이 남은 후보 Top-N을 추정하여 탐색 공간 시각화
- **20턴 제한 + 강제 추측** — 무한 질문 루프 방지

---

## 기술 스택

| 구분 | 기술 |
|------|------|
| Backend | Django 5.x, SQLite |
| LLM | Ollama (`llama3.1:8b`) |
| Embedding | Ollama `bge-m3` |
| Frontend | HTML + Fetch API |
| Infra | Docker Compose, NVIDIA GPU |

---

## 프로젝트 구조

```
akinator-bot/
├── docker-compose.yml          # web + ollama 서비스
├── Dockerfile
├── requirements.txt
├── manage.py
├── akinator_project/           # Django 프로젝트 설정
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
└── game/                       # 메인 앱
    ├── models.py               # GameSession, QA
    ├── views.py                # API 엔드포인트
    ├── akinator_logic.py       # 프롬프트 + 턴 진행 + Best-of-N
    ├── ollama_client.py        # Ollama API 호출
    └── templates/game/
        └── index.html          # 게임 UI
```

---

## 빠른 시작

### 사전 요구사항

- Docker & Docker Compose
- NVIDIA GPU + [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

### 실행

```bash
# 1. 클론
git clone https://github.com/pgh165/akinator-bot.git
cd akinator-bot

# 2. 환경변수 설정
cp .env.example .env
# .env의 DJANGO_SECRET_KEY를 랜덤 문자열로 변경

# 3. 빌드 & 실행
docker compose up -d --build

# 4. 모델 다운로드 (최초 1회)
docker compose exec ollama ollama pull llama3.1:8b
docker compose exec ollama ollama pull bge-m3

# 5. DB 마이그레이션
docker compose exec web python manage.py migrate

# 6. 브라우저에서 접속
# http://localhost:8001
```

---

## 게임 흐름

```
카테고리 선택 (인물/동물/사물/브랜드)
        │
        ▼
   ┌─ LLM 첫 질문 생성 ◄──────────────────┐
   │        │                               │
   │        ▼                               │
   │  사용자 답변 (예/아니오/잘 모름)        │
   │        │                               │
   │        ▼                               │
   │  후보군 갱신 → Best-of-N 질문 생성 ────┘
   │        │
   │        ▼  (확신 또는 20턴 초과)
   │
   └─► 최종 추측 출력
```

---

## API

### `POST /api/start/`

새 게임을 시작합니다.

```json
// 요청
{ "category": "인물" }

// 응답
{ "session_id": "uuid", "action": "ask", "question": "실존하는 인물인가요?" }
```

### `POST /api/answer/`

답변을 제출하고 다음 질문 또는 추측을 받습니다.

```json
// 요청
{ "session_id": "uuid", "answer": "예" }

// 응답 (질문 계속)
{ "action": "ask", "question": "현재 생존해 있나요?" }

// 응답 (추측)
{ "action": "guess", "answer": "아인슈타인" }
```

---

## 환경변수

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `DJANGO_SECRET_KEY` | — | Django 보안 키 (필수 변경) |
| `DJANGO_DEBUG` | `True` | 디버그 모드 |
| `DJANGO_ALLOWED_HOSTS` | `localhost,127.0.0.1` | 허용 호스트 |
| `OLLAMA_HOST` | `http://ollama:11434` | Ollama 컨테이너 주소 |
| `OLLAMA_MODEL` | `llama3.1:8b` | 사용할 LLM 모델 |
| `OLLAMA_EMBED_MODEL` | `bge-m3` | 임베딩 모델 |

---

## 핵심 설계

### Stateless LLM + DB 기반 세션

Ollama 호출은 매번 독립적이므로, 매 턴 전체 대화 이력을 프롬프트에 포함합니다. 세션 상태와 Q&A 이력은 Django DB(SQLite)에 저장됩니다.

### Best-of-N 질문 전략

매 턴 3개의 질문 후보를 병렬 생성한 뒤, LLM이 후보군 기준으로 변별력이 가장 높은(50%에 가장 가까운) 질문을 자동 선택합니다.

### 금지 표현 필터링

"특정", "관련 있" 등 정보 이득이 낮은 모호한 표현과, "또는", "중 한" 등 복수 후보 추측을 자동 감지하여 재생성합니다.

### JSON 강제 출력

Ollama의 `format: "json"` 옵션으로 응답을 구조화하며, 파싱 실패 시 fallback 질문으로 자동 복구합니다.
