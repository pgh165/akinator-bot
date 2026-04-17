"""아키네이터 턴 진행 로직 + 프롬프트 구성.

Ollama 호출은 ollama_client.py에 격리되어 있고, 여기서는 프롬프트 빌드 +
응답 파싱 방어 로직만 담당한다.
"""
import json
import logging

from .models import GameSession
from .ollama_client import OllamaError, call_ollama

log = logging.getLogger(__name__)

# 최대 턴. 초과 시 강제 추측 모드 진입.
MAX_TURNS = 20

# JSON 파싱 실패 또는 응답 형식 오류 시 복구용 기본 질문.
FALLBACK_QUESTION = "혹시 그 대상은 실존 인물인가요?"

# 질문에 섞이면 정보 이득이 떨어지는 금지 표현. 생성된 질문에서 발견되면 재생성.
BANNED_ASK_PHRASES = ("주로", "특정", "관련 있", "관련있", "관련 된", "관련된", "관련이 있")

# 추측(guess) answer에서 발견되면 복수/모호로 간주하고 재생성.
BANNED_GUESS_PHRASES = (",", " 또는 ", " / ", "중 한", " 중 ", "같은", "비슷한", "등의")

# 재생성 최대 시도 횟수. 이만큼 해도 위반이면 마지막 결과를 그대로 사용.
MAX_REGENERATE = 2

SYSTEM_PROMPT = """당신은 '아키네이터'입니다. 사용자가 머릿속으로 한 가지 대상을 생각하고, \
당신은 예/아니오 질문만으로 그 대상을 맞춰야 합니다.

규칙:
1. 매 턴 정확히 하나의 질문을 하거나, 충분히 좁혀졌을 때만 추측합니다.
2. 사용자의 답은 "예", "아니오", "잘 모름" 세 가지뿐입니다. 이 셋 중 하나로 답할 수 있는 단일 질문만 하세요.
3. "A 또는 B?", "A, B, C 중 무엇?", "~은 무엇인가요?" 같은 선택형/주관식 질문은 절대 금지.
4. 넓은 범주 → 구체적 특징 순으로 좁혀갑니다.
5. 이전 턴에서 이미 나온 질문, 또는 같은 속성을 다른 말로 재확인하는 질문은 절대 하지 마세요. 직전 대화 기록을 반드시 검토하세요.
6. **정보 이득이 큰 질문만** 하세요. 남은 후보를 대략 절반으로 가를 수 있는 질문이 이상적입니다. 아래 모호한 표현은 금지:
   - "주로 ~" ("주로 예술 분야인가요?" 등 → 그냥 "예술가인가요?"로)
   - "특정 ~" ("특정 시대의 인물인가요?", "특정 국가 출신인가요?" 등 → 동어반복)
   - "~과 관련이 있나요?", "~와 관련된 일을 하나요?" (약한 연관)
   - 이미 '예'로 답한 사실을 돌려 묻기 (예: "역사적 인물=예" 이후 "과거에 살았던 인물인가요?" 금지)
   - 거의 모든 대상에 '예'가 나오는 질문 (예: "이름이 있나요?")
7. 추측(guess)의 answer는 반드시 **단일 고유명사 하나**여야 합니다. 아래는 모두 금지:
   - "아인슈타인 또는 뉴턴" (여러 후보)
   - "아인슈타인, 뉴턴, 갈릴레이 중 한 명" (열거)
   - "유명한 과학자 중 한 명" (일반화)
   - "아인슈타인 같은 인물" (모호)
   확신이 서지 않으면 guess 하지 말고 ask로 더 좁히세요.
8. 설명이나 서론 없이, 반드시 JSON 한 줄로만 출력합니다.

출력 형식:
- 질문: {"action": "ask", "question": "..."}
- 추측: {"action": "guess", "answer": "..."}

좋은 예:
{"action": "ask", "question": "이 인물은 과학자인가요?"}
{"action": "guess", "answer": "아인슈타인"}

나쁜 예 (절대 따라하지 말 것):
{"action": "ask", "question": "직업이 과학자인가요, 아니면 예술가인가요?"}  ← 선택형
{"action": "ask", "question": "이 인물은 주로 예술 분야와 관련이 있나요?"}  ← 모호어 "주로/관련"
{"action": "ask", "question": "이 인물은 특정 시대의 인물인가요?"}  ← "특정" 동어반복
{"action": "guess", "answer": "아인슈타인 또는 뉴턴"}  ← 복수 후보

다른 키, 다른 필드, 추가 텍스트, 마크다운 코드블록은 절대 포함하지 마세요."""


def _build_user_prompt(session: GameSession, force_guess: bool = False) -> str:
    """대화 이력 + 현재 지시사항을 문자열로 직렬화."""
    qas = list(session.qas.all())
    lines = [f"카테고리: {session.category}"]

    if qas:
        lines.append("")
        lines.append("지금까지의 질문-답변(절대 중복 금지):")
        for qa in qas:
            lines.append(f"  Q{qa.turn}: {qa.question}")
            lines.append(f"  A{qa.turn}: {qa.answer}")
        lines.append("")
        lines.append(
            "⚠ 위 Q1~Q{} 중 어느 하나와 같거나 의미가 겹치는 질문을 다시 하면 실패입니다. "
            "반드시 새로운 속성을 묻는 질문을 하세요.".format(len(qas))
        )
    else:
        lines.append("")
        lines.append("아직 아무 질문도 하지 않았습니다. 첫 질문을 해주세요.")

    lines.append("")
    if force_guess:
        lines.append(
            f"[강제 추측 모드] 이미 {MAX_TURNS}턴이 소진되었습니다. "
            "반드시 action=\"guess\"로, 지금까지의 답변을 바탕으로 한 최선의 추측을 하나만 내세요."
        )
    else:
        lines.append(
            "다음 행동을 결정하세요: 더 좁히고 싶으면 action=\"ask\"로 질문, "
            "충분히 확신이 들면 action=\"guess\"로 추측."
        )

    return "\n".join(lines)


def _parse_response(raw: str) -> dict:
    """LLM의 JSON 응답을 파싱. 실패 시 FALLBACK_QUESTION으로 복구."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        log.warning("Ollama 응답 JSON 파싱 실패: %r", raw[:200])
        return {"action": "ask", "question": FALLBACK_QUESTION}

    if not isinstance(data, dict):
        log.warning("Ollama 응답이 객체가 아님: %r", data)
        return {"action": "ask", "question": FALLBACK_QUESTION}

    action = data.get("action")
    if action == "ask":
        question = data.get("question")
        if isinstance(question, str) and question.strip():
            return {"action": "ask", "question": question.strip()}
    elif action == "guess":
        answer = data.get("answer")
        if isinstance(answer, str) and answer.strip():
            return {"action": "guess", "answer": answer.strip()}

    log.warning("Ollama 응답 형식 오류: %r", data)
    return {"action": "ask", "question": FALLBACK_QUESTION}


def _find_violation(parsed: dict) -> str | None:
    """위반 표현이 있으면 해당 표현을 반환, 없으면 None."""
    if parsed["action"] == "ask":
        for p in BANNED_ASK_PHRASES:
            if p in parsed["question"]:
                return p
    elif parsed["action"] == "guess":
        for p in BANNED_GUESS_PHRASES:
            if p in parsed["answer"]:
                return p
    return None


def next_turn(session: GameSession) -> dict:
    """세션 상태 → LLM 호출 → 파싱된 액션 반환.

    위반 표현 감지 시 최대 MAX_REGENERATE번 재생성. 재시도에는 이전 위반 내용을
    프롬프트에 명시해 같은 실수를 피하도록 유도.

    반환값:
        {"action": "ask",   "question": "..."}  또는
        {"action": "guess", "answer":   "..."}
    """
    turn_count = session.qas.count()
    force_guess = turn_count >= MAX_TURNS

    base_prompt = _build_user_prompt(session, force_guess=force_guess)

    rejected: list[str] = []
    parsed: dict = {"action": "ask", "question": FALLBACK_QUESTION}

    for attempt in range(MAX_REGENERATE + 1):
        prompt = base_prompt
        if rejected:
            prompt += (
                "\n\n[직전 시도가 거부되었습니다. 같은 실수를 반복하지 마세요]\n"
                + "\n".join(rejected)
            )

        try:
            raw = call_ollama(prompt=prompt, system=SYSTEM_PROMPT, temperature=0.3)
        except OllamaError as e:
            log.error("Ollama 호출 실패, fallback 사용: %s", e)
            return {"action": "ask", "question": FALLBACK_QUESTION}

        parsed = _parse_response(raw)
        violation = _find_violation(parsed)

        if violation is None:
            return parsed

        key = "question" if parsed["action"] == "ask" else "answer"
        rejected.append(
            f'- 거부됨 ({parsed["action"]}): "{parsed[key]}" '
            f'— 금지 표현 "{violation}" 포함'
        )
        log.info("재생성 %d/%d: %s", attempt + 1, MAX_REGENERATE, rejected[-1])

    log.warning("재생성 %d회 모두 위반, 마지막 결과 통과: %s", MAX_REGENERATE, parsed)
    return parsed
