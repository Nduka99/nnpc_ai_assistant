"""The JSON API: the same answer as the command line, over HTTP (Lime Green's app,
carried over for NNPC).

Every answer is `view(answer)`, the same data the CLI prints: the v1 API returns it
with the id of its audit record and its place in a conversation. An unreachable model
or a missing index is an operational error: HTTP 503 with one fixed message, never an
answer. The cause is logged on the server and never sent to the client, since it can
name internal addresses (OWASP API8:2023).

A conversation (Lime Green ADR 0035) is a sequence of single-turn answers. A request
without `conversation_id` starts one; the reply gives its id and the turn's number,
and the next request sends the id back. The server rebuilds the history from what it
showed; a request carrying any other field, such as a history, is refused (Lime Green
D60).

Load (K28): at most as many answers are worked on at once as the generator has slots;
a few more wait in line, in order of arrival, and the stream says each one's place.
Beyond that, or when one client asks too often, the request is refused with 429 before
any turn is reserved. A reader can mark an answer helpful or not right, from the
answer's own conversation; feedback is kept for history and never served.
"""

import base64
import contextvars
import hashlib
import logging
import queue
import re
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Annotated, TypedDict
from uuid import UUID

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.sse import EventSourceResponse, ServerSentEvent
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, StringConstraints
from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from petroknowledge import assistant, config, llm, telemetry
from petroknowledge.assistant import ConversationEnded, Turn, Unavailable
from petroknowledge.llm import ModelServerError
from petroknowledge.records import Answer
from petroknowledge.view import AnswerView, view

# No interactive docs: an API's inventory is not served to every visitor (OWASP
# API9:2023).
app = FastAPI(title="PetroKnowledge", docs_url=None, redoc_url=None, openapi_url=None)
telemetry.instrument(app)
logger = logging.getLogger(__name__)
# The web app's build (ADR 0010). Without one no directory is served (below): Starlette
# fails every request to a directory that does not exist. A new build needs a restart.
BUILT = config.WEB_DIST if config.WEB_DIST.is_dir() else None
INLINE_SCRIPT = re.compile(r"<script(?![^>]*\ssrc=)[^>]*>(.*?)</script>", re.DOTALL)


def inline_script_hashes(folder: Path | None) -> str:
    """The CSP hash sources of the inline scripts in a build's pages (CSP Level 3).
    Nuxt writes its runtime config and import map inline; a hash allows exactly those
    scripts, where 'unsafe-inline' would allow any."""
    if folder is None:
        return ""
    found = set()
    for page in folder.glob("*.html"):
        for script in INLINE_SCRIPT.findall(page.read_text(encoding="utf-8")):
            digest = base64.b64encode(hashlib.sha256(script.encode()).digest())
            found.add(f" 'sha256-{digest.decode('ascii')}'")
    return "".join(sorted(found))


# Sent with every response (OWASP Secure Headers Project): the page runs only its own
# scripts and styles, cannot be framed, and sends no referrer to the sources it links.
SECURITY_HEADERS = {
    "Content-Security-Policy": "default-src 'self'; "
    f"script-src 'self'{inline_script_hashes(BUILT)}; object-src 'none'; "
    "base-uri 'none'; frame-ancestors 'none'; form-action 'self'",
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "Cross-Origin-Opener-Policy": "same-origin",
}
UNAVAILABLE = "The assistant can't answer right now. Try again in a moment."
ENDED = "This conversation has ended. Please start a new conversation."
BUSY = "The assistant is busy. Try again in a minute."
NOT_FOUND = "There is no such answer in this conversation."


class Busy(Exception):
    """The line of waiting questions is full."""


class Place:
    """One answer's place in line: started once it holds a slot, left once it gives
    the slot back or gives up waiting."""

    def __init__(self, position: int) -> None:
        self.position = position  # its place in line on arrival; 0: it starts at once
        self.started = False
        self.left = False


class Slots:
    """At most `size` answers worked on at once; up to `queue_length` more wait in
    order of arrival, and any beyond are refused."""

    def __init__(self, size: int, queue_length: int) -> None:
        self.size = size
        self.queue_length = queue_length
        self.active = 0
        self.waiting: list[Place] = []
        self.changed = threading.Condition()

    def join(self) -> Place:
        """A place at the end of the line; Busy when the line is full."""
        with self.changed:
            free = self.active < self.size and not self.waiting
            if not free and len(self.waiting) >= self.queue_length:
                raise Busy
            place = Place(0 if free else len(self.waiting) + 1)
            self.waiting.append(place)
            return place

    def start(self, place: Place) -> bool:
        """Wait for the place's turn and take a slot; False if it gave up first."""
        with self.changed:
            self.changed.wait_for(
                lambda: (
                    place.left or (self.waiting[0] is place and self.active < self.size)
                )
            )
            if place.left:
                return False
            self.waiting.pop(0)
            self.active += 1
            place.started = True
            self.changed.notify_all()
            return True

    def leave(self, place: Place) -> None:
        """Give back the place's slot, or its place in line if it never started."""
        with self.changed:
            if place.started and not place.left:
                self.active -= 1
            elif not place.left:
                self.waiting.remove(place)
            place.left = True
            self.changed.notify_all()


class Rate:
    """At most `limit` questions a minute from one client."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.asked: dict[str, list[float]] = {}
        self.lock = threading.Lock()

    def allow(self, client: str, now: float) -> bool:
        with self.lock:
            recent = [when for when in self.asked.get(client, []) if now - when < 60]
            allowed = len(recent) < self.limit
            if allowed:
                recent.append(now)
            self.asked[client] = recent
            return allowed


SLOTS = Slots(config.ANSWERS_AT_ONCE, config.QUEUE_LENGTH)
RATE = Rate(config.QUESTIONS_PER_MINUTE)


class AnswerRequest(BaseModel):
    """A question for the v1 API, checked after surrounding spaces are removed, and
    the conversation it continues, if any. Any other field is refused."""

    model_config = ConfigDict(extra="forbid")

    question: Annotated[
        str,
        StringConstraints(
            strip_whitespace=True, min_length=1, max_length=config.MAX_QUESTION_CHARS
        ),
    ]
    conversation_id: UUID | None = None


class FeedbackRequest(BaseModel):
    """A reader's judgement of one answer, from the conversation it belongs to."""

    model_config = ConfigDict(extra="forbid")

    conversation_id: UUID
    helpful: bool
    comment: Annotated[
        str,
        StringConstraints(strip_whitespace=True, max_length=config.MAX_FEEDBACK_CHARS),
    ] = ""


class RecordedAnswer(TypedDict):
    id: int  # the answer's audit record
    conversation_id: str
    turn: int  # 1 for the first question of a conversation
    understood_as: list[str]  # the search questions; shown on a follow-up
    answer: AnswerView


class Readiness(TypedDict):
    ready: bool
    checks: dict[str, bool]


class SecurityHeaders:
    """Adds SECURITY_HEADERS to every HTTP response. A plain ASGI middleware, so the
    answer stream passes through unbuffered and keeps its trace context."""

    def __init__(self, inner: ASGIApp) -> None:
        self.inner = inner

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        async def send_with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                for name, value in SECURITY_HEADERS.items():
                    headers[name] = value
            await send(message)

        await self.inner(scope, receive, send_with_headers)


app.add_middleware(SecurityHeaders)


def opened_turn(request: AnswerRequest) -> Turn:
    """The request's turn, reserved before any answer work starts, so an ended
    conversation is a plain 404 even on the stream (whose status is sent with its
    first event)."""
    asked = str(request.conversation_id) if request.conversation_id else None
    try:
        return assistant.open_turn(asked)
    except ConversationEnded as ended:
        assistant.record_failure(request.question, "ended", str(ended), asked)
        raise HTTPException(status_code=404, detail=ENDED) from ended
    except Unavailable as problem:
        logger.warning("answer failed: %s", problem)
        assistant.record_failure(request.question, "unavailable", str(problem), asked)
        raise HTTPException(status_code=503, detail=UNAVAILABLE) from problem


def client_of(http: Request) -> str:
    """Who is asking: the visitor's address the tunnel passes on, else the peer's."""
    forwarded = http.headers.get("cf-connecting-ip")
    return forwarded or (http.client.host if http.client else "")


def opened(request: AnswerRequest, http: Request) -> tuple[Place, Turn]:
    """A place in line and the request's turn. A client asking too often, or a full
    line, is refused (429) before any turn is reserved; a turn that cannot be opened
    gives the place back."""
    if not RATE.allow(client_of(http), time.monotonic()):
        raise HTTPException(status_code=429, detail=BUSY, headers={"Retry-After": "60"})
    try:
        place = SLOTS.join()
    except Busy as busy:
        headers = {"Retry-After": "30"}
        raise HTTPException(status_code=429, detail=BUSY, headers=headers) from busy
    try:
        return place, opened_turn(request)
    except HTTPException:
        SLOTS.leave(place)
        raise


def failed(request: AnswerRequest, turn: Turn, problem: Exception) -> None:
    """Log an answer that could not be given, and keep its question (history)."""
    logger.warning("answer failed: %s", problem)
    assistant.record_failure(
        request.question, "unavailable", str(problem), turn.conversation_id,
        turn.number,
    )  # fmt: skip


def recorded(result: Answer, answer_id: int, turn: Turn) -> RecordedAnswer:
    """What the v1 API returns for one answered turn."""
    return {
        "id": answer_id,
        "conversation_id": turn.conversation_id,
        "turn": turn.number,
        "understood_as": list(result.understood),
        "answer": view(result),
    }


@app.post("/api/v1/answers")
def create_answer(
    request: AnswerRequest,
    opening: Annotated[tuple[Place, Turn], Depends(opened)],
) -> RecordedAnswer:
    """Answer from the served Postgres index, in turn, and record it for audit."""
    place, turn = opening
    SLOTS.start(place)
    try:
        result, answer_id = assistant.ask_and_record(request.question, turn=turn)
    except (Unavailable, ModelServerError) as problem:
        failed(request, turn, problem)
        raise HTTPException(status_code=503, detail=UNAVAILABLE) from problem
    finally:
        SLOTS.leave(place)
    return recorded(result, answer_id, turn)


@app.post("/api/v1/answers/stream", response_class=EventSourceResponse)
def stream_answer(
    request: AnswerRequest,
    opening: Annotated[tuple[Place, Turn], Depends(opened)],
) -> Iterator[ServerSentEvent]:
    """The same answer as server-sent events: `queued` with its place in line if it
    must wait, `stage` as each stage starts, then `answer` once it is verified, or
    `error`. Unverified text is never sent. A reader who leaves while waiting gives
    the place back."""
    place, turn = opening
    events: queue.Queue[ServerSentEvent | Exception] = queue.Queue()

    def report(stage: str) -> None:
        events.put(ServerSentEvent(event="stage", data={"stage": stage}))

    def run() -> None:
        if not SLOTS.start(place):  # given up while waiting: nobody is listening
            events.put(ServerSentEvent(event="error", data={"detail": BUSY}))
            return
        try:
            result, answer_id = assistant.ask_and_record(
                request.question, report, turn=turn
            )
        except (Unavailable, ModelServerError) as problem:
            failed(request, turn, problem)
            events.put(ServerSentEvent(event="error", data={"detail": UNAVAILABLE}))
        except Exception as error:  # a bug: raised below, where the server logs it
            events.put(error)
        else:
            data = recorded(result, answer_id, turn)
            events.put(ServerSentEvent(event="answer", data=data))
        finally:
            SLOTS.leave(place)

    try:
        if place.position:
            yield ServerSentEvent(event="queued", data={"position": place.position})
        # The answer runs in its own thread so each stage is sent as it starts. A new
        # thread starts with an empty context, so the request's is copied in: its
        # spans then belong to the request's trace.
        context = contextvars.copy_context()
        threading.Thread(target=context.run, args=(run,), daemon=True).start()
        while True:
            event = events.get()
            if isinstance(event, Exception):
                raise event
            yield event
            if event.event != "stage":
                return
    finally:
        # A reader gone while waiting gives the place back; once started, the
        # answer's own thread gives the slot back when it ends.
        if not place.started:
            SLOTS.leave(place)


@app.post("/api/v1/answers/{answer_id}/feedback", status_code=204)
def give_feedback(answer_id: int, feedback: FeedbackRequest) -> Response:
    """Keep a reader's judgement of an answer, from its own conversation (K28)."""
    try:
        kept = assistant.record_feedback(
            answer_id, str(feedback.conversation_id), feedback.helpful, feedback.comment
        )
    except Unavailable as problem:
        logger.warning("feedback failed: %s", problem)
        raise HTTPException(status_code=503, detail=UNAVAILABLE) from problem
    if not kept:
        raise HTTPException(status_code=404, detail=NOT_FOUND)
    return Response(status_code=204)


@app.get("/healthz")
def healthz() -> dict[str, str]:
    """Liveness: the process answers HTTP. It checks nothing else, so a slow model
    or database never gets a working process restarted."""
    return {"status": "ok"}


@app.get("/readyz")
def readyz(response: Response) -> Readiness:
    """Readiness: whether an answer can be served now. HTTP 503 until the database
    has a matching index to serve and every model server has loaded its model."""
    checks = {
        "database": assistant.database_ready(),
        "embedding": llm.healthy(config.EMBEDDING_URL),
        "reranking": llm.healthy(config.RERANK_URL),
        "generation": llm.healthy(config.CHAT_URL),
        "judging": llm.healthy(config.JUDGE_URL),
    }
    ready = all(checks.values())
    if not ready:
        response.status_code = 503
    return {"ready": ready, "checks": checks}


# The web app last, so every route above is matched first; without a build its paths
# are 404.
web = StaticFiles(directory=BUILT, html=True)
app.mount("/", web, name="web")
