<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from "vue";
import { askStream, type StreamEvent } from "../lib/api";
import { loadThread, saveThread, type SavedThread } from "../lib/saved";
import { announcement, settled, type Outcome, type TurnState } from "../lib/thread";
import AskingGuide from "./AskingGuide.vue";
import LineIcon from "./LineIcon.vue";
import QuestionComposer from "./QuestionComposer.vue";
import ThreadTurn from "./ThreadTurn.vue";

// One conversation. It is kept in this tab's session storage, so a reload keeps it;
// "New conversation" starts again, and the server keeps the history. Tests may pass
// their own `ask`. It has no Vue default: the production build called a function
// default as a factory instead of using it.
const props = defineProps<{ ask?: typeof askStream }>();
const ask = props.ask ?? askStream;

// Starter questions (K33): questions written for NNPC managers that the assistant
// answered soundly, kept in their tested wording.
const STARTERS = [
  "Give me the headline figures from our latest audited results: profit, revenue, dividend, and daily oil and gas production.",
  "Who is listed as NGIC's managing director, and what is that company responsible for?",
  "Under the PIA, who regulates upstream, and who regulates midstream and downstream?",
  "What gas products does NGML offer?",
  "Who joined NNPC Shipping in the tanker venture announced in February 2025?",
  "Who holds the government's shares in NNPC Limited under the PIA?",
] as const;
// The same words as the command line's (view.INDEPENDENT, K02).
const INDEPENDENT =
  "Independent research prototype, not affiliated with or endorsed by NNPC Limited. Answers quote public documents; check the linked source before relying on one.";

const saved = loadThread();
const turns = ref<TurnState[]>(saved.turns);
const conversationId = ref<string | null>(saved.conversationId);
const busy = ref(false);
// The conversation "New conversation" cleared, kept until the next question so
// one click can bring it back (Nielsen 3; no time limit, WCAG 2.2.1).
const cleared = ref<SavedThread | null>(null);
const composer = ref<InstanceType<typeof QuestionComposer> | null>(null);
const composerBar = ref<HTMLElement | null>(null);
// How to ask (K33): open over the conversation, from the question box's toggle or a
// refusal's Tips for asking; closed by its toggle, Close, Esc or asking.
const guideOpen = ref(false);
const guide = ref<InstanceType<typeof AskingGuide> | null>(null);
// What the status line says about the thread itself (cleared, restored) until the
// next question; otherwise the latest turn's outcome (WCAG 4.1.3).
const threadNews = ref("");
const spoken = computed(() => threadNews.value || announcement(turns.value.at(-1)));
let stopping: AbortController | null = null;

watch(
  [turns, conversationId],
  () => saveThread({ turns: turns.value, conversationId: conversationId.value }),
  { deep: true },
);

// Keyboard focus and scrolled-to targets stay clear of the sticky question box
// (WCAG 2.4.11): the page's bottom scroll padding follows the box's height.
const resized = new ResizeObserver((entries) => {
  for (const entry of entries) {
    const height = entry.target.getBoundingClientRect().height;
    document.documentElement.style.setProperty("--composer-height", `${height}px`);
  }
});
// Esc closes the guide if it is open; otherwise it stops an answer on its way,
// wherever focus is, not only in the question box.
function onEscape(event: KeyboardEvent): void {
  if (event.key !== "Escape") return;
  if (guideOpen.value) closeGuide();
  else if (busy.value) onStop();
}

/** Open the guide and move focus to its heading, so it is read from the start. */
async function openGuide(): Promise<void> {
  guideOpen.value = true;
  await nextTick();
  guide.value?.focus();
}

/** Close the guide; focus returns to its toggle, where the reader can reopen it. */
function closeGuide(): void {
  guideOpen.value = false;
  composer.value?.focusGuide();
}

function toggleGuide(): void {
  if (guideOpen.value) closeGuide();
  else void openGuide();
}
onMounted(() => {
  resized.observe(composerBar.value as HTMLElement);
  document.addEventListener("keydown", onEscape);
});
onBeforeUnmount(() => {
  resized.disconnect();
  document.removeEventListener("keydown", onEscape);
});

function update(key: number, change: (turn: TurnState) => TurnState): void {
  turns.value = turns.value.map((turn) => (turn.key === key ? change(turn) : turn));
}

/** Bring a turn's question to the top of the view, so its answer reads from the
 * start (peak-end). On arrival this happens only if the reader has not scrolled
 * away to read something else. */
async function showTurn(key: number, onlyIfFollowing: boolean): Promise<void> {
  await nextTick();
  const element = document.getElementById(`turn-${key}`);
  if (element === null) return;
  if (onlyIfFollowing && element.getBoundingClientRect().top > window.innerHeight) {
    return;
  }
  element.scrollIntoView({ block: "start" });
}

async function onAsk(question: string): Promise<void> {
  if (busy.value) return;
  cleared.value = null;
  threadNews.value = "";
  // Asking closes the guide, so the answer is not hidden under it.
  guideOpen.value = false;
  const key = (turns.value.at(-1)?.key ?? 0) + 1;
  turns.value = [...turns.value, { key, question, stages: [] }];
  busy.value = true;
  stopping = new AbortController();
  void showTurn(key, false);
  await ask(
    question,
    conversationId.value,
    (event: StreamEvent) => {
      if (event.kind === "queued") {
        update(key, (turn) => ({ ...turn, queued: event.position }));
        return;
      }
      if (event.kind === "stage") {
        update(key, (turn) => ({ ...turn, stages: [...turn.stages, event.stage] }));
        return;
      }
      if (event.kind === "answer")
        conversationId.value = event.recorded.conversation_id;
      const outcome: Outcome = event;
      update(key, (turn) => ({ ...turn, outcome }));
      void showTurn(key, true);
    },
    stopping.signal,
  );
  stopping = null;
  busy.value = false;
}

function onStop(): void {
  stopping?.abort();
}

/** An example asks itself; focus goes to the question box, where Esc and the
 * next question are. */
function onExample(question: string): void {
  composer.value?.focus();
  void onAsk(question);
}

/** Ask a failed or unfinished question again, in place of its old turn. Focus
 * goes to the question box, since the pressed button leaves with the turn. */
function onRetry(key: number, question: string): void {
  if (busy.value) return;
  turns.value = turns.value.filter((turn) => turn.key !== key);
  composer.value?.focus();
  void onAsk(question);
}

function onNewConversation(): void {
  stopping?.abort();
  cleared.value = { turns: turns.value, conversationId: conversationId.value };
  turns.value = [];
  conversationId.value = null;
  threadNews.value = "Started a new conversation. Undo is available.";
  composer.value?.focus();
}

function onUndo(): void {
  const restored = cleared.value as SavedThread;
  turns.value = settled(restored.turns);
  conversationId.value = restored.conversationId;
  cleared.value = null;
  threadNews.value = "Conversation restored.";
  composer.value?.focus();
}
</script>

<template>
  <div class="shell">
    <header class="header">
      <div>
        <h1>PetroKnowledge</h1>
        <p class="tagline">
          <strong>Independent of NNPC Limited.</strong> Answers from its public pages
          and documents, with the quote behind every statement.
        </p>
      </div>
      <button
        v-if="turns.length > 0"
        type="button"
        class="button"
        @click="onNewConversation"
      >
        <LineIcon name="new" />
        New conversation
      </button>
    </header>
    <main class="thread" aria-label="Conversation">
      <section v-if="turns.length === 0" class="empty">
        <p v-if="cleared" class="undo">
          Started a new conversation.
          <button type="button" class="button" @click="onUndo">
            <LineIcon name="undo" />
            Undo
          </button>
        </p>
        <h2>Ask a question to start</h2>
        <p>
          Ask about NNPC's businesses, results, people or published documents. Follow-up
          questions can refer to the earlier ones, such as "when was he appointed?".
          Every statement comes with the quote it rests on and the source's date; when
          the sources do not say, the assistant says so. Questions are recorded to
          improve the assistant, so leave out personal details.
        </p>
        <p>
          Tips for getting good answers are under How to ask, beside the question box.
        </p>
        <ul class="examples" aria-label="Example questions">
          <li v-for="example in STARTERS" :key="example">
            <button type="button" class="example" @click="onExample(example)">
              {{ example }}
            </button>
          </li>
        </ul>
      </section>
      <template v-else>
        <ThreadTurn
          v-for="turn in turns"
          :key="turn.key"
          :turn="turn"
          @new-conversation="onNewConversation"
          @retry="onRetry"
          @guide="openGuide"
        />
      </template>
      <p class="visually-hidden" role="status">{{ spoken }}</p>
    </main>
    <div ref="composerBar" class="composer-bar">
      <AskingGuide
        v-if="guideOpen"
        ref="guide"
        :starters="STARTERS"
        @close="closeGuide"
        @ask="onExample"
      />
      <QuestionComposer
        ref="composer"
        :busy="busy"
        :guide-open="guideOpen"
        @ask="onAsk"
        @stop="onStop"
        @guide="toggleGuide"
      />
    </div>
    <footer class="notice-footer">
      <p>
        {{ INDEPENDENT }} Not investment, legal or safety advice. Questions and answers
        are recorded to improve the assistant, so leave out personal details.
      </p>
    </footer>
  </div>
</template>
