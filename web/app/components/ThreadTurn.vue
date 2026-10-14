<script setup lang="ts">
import { computed } from "vue";
import { STAGES, type Stage } from "../lib/api";
import type { TurnState } from "../lib/thread";
import AnswerBody from "./AnswerBody.vue";
import AnswerFeedback from "./AnswerFeedback.vue";
import LineIcon from "./LineIcon.vue";

// One turn of the conversation, read like a short document: the question is its
// heading, then the work in progress or the outcome.
const props = defineProps<{ turn: TurnState }>();
const emit = defineEmits<{
  newConversation: [];
  retry: [key: number, question: string];
  guide: [];
}>();

const STAGE_LABELS: Record<Stage, string> = {
  understanding: "Understanding",
  searching: "Searching",
  answering: "Answering",
  checking: "Checking quotes",
};

const STAGE_STATUS: Record<Stage, string> = {
  understanding: "Reading the question…",
  searching: "Searching NNPC's public pages and documents…",
  answering: "Writing an answer from the passages found…",
  checking: "Checking every quote against its source…",
};

const STATE_TEXT = { waiting: "not started", current: "in progress", done: "done" };

// Before the first stage: waiting in line for the assistant, or being sent.
const opening = computed(() =>
  props.turn.queued
    ? `Other questions are being answered. Yours is number ${props.turn.queued} in line.`
    : "Sending the question…",
);

// The four stages, each marked done once the next has started.
const current = computed(() => props.turn.stages.at(-1));
const progress = computed(() =>
  STAGES.map((stage) => {
    const reached = props.turn.stages.includes(stage);
    const state = !reached ? "waiting" : stage === current.value ? "current" : "done";
    return { stage, state, label: STAGE_LABELS[stage], spoken: STATE_TEXT[state] };
  }),
);
</script>

<template>
  <article :id="`turn-${turn.key}`" class="turn">
    <h2 class="question">{{ turn.question }}</h2>
    <div v-if="turn.outcome === undefined" class="progress">
      <ol class="stages">
        <li
          v-for="step in progress"
          :key="step.stage"
          :class="['stage', `stage-${step.state}`]"
        >
          <span class="stage-mark" aria-hidden="true">
            <LineIcon v-if="step.state === 'done'" name="check" />
          </span>
          {{ step.label }}
          <span class="visually-hidden">({{ step.spoken }})</span>
        </li>
      </ol>
      <p class="stage-status">
        {{ current ? STAGE_STATUS[current] : opening }}
        <span class="stage-expect">Answers usually take 15–30 seconds.</span>
      </p>
    </div>
    <template v-else-if="turn.outcome.kind === 'answer'">
      <AnswerBody :recorded="turn.outcome.recorded" @guide="emit('guide')" />
      <AnswerFeedback
        :answer-id="turn.outcome.recorded.id"
        :conversation-id="turn.outcome.recorded.conversation_id"
      />
    </template>
    <div v-else-if="turn.outcome.kind === 'ended'" class="problem" role="alert">
      <p class="problem-title">{{ turn.outcome.detail }}</p>
      <button type="button" class="button" @click="emit('newConversation')">
        Start a new conversation
      </button>
    </div>
    <div
      v-else
      :class="['problem', { 'problem-error': turn.outcome.kind === 'error' }]"
      :role="turn.outcome.kind === 'error' ? 'alert' : undefined"
    >
      <p class="problem-title">
        <LineIcon v-if="turn.outcome.kind === 'error'" name="alert" />
        {{
          turn.outcome.kind === "error"
            ? turn.outcome.detail
            : "This answer did not finish."
        }}
      </p>
      <button
        type="button"
        class="button"
        @click="emit('retry', turn.key, turn.question)"
      >
        <LineIcon name="retry" />
        Try again
      </button>
    </div>
  </article>
</template>
