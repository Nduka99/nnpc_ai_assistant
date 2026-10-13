<script setup lang="ts">
import { computed, ref } from "vue";
import LineIcon from "./LineIcon.vue";

// The question box: Enter asks, Shift+Enter starts a new line. It stays ready for
// typing while an answer is on its way; only asking waits, and the button then
// stops the answer instead (Nielsen 3, user control).
const props = defineProps<{ busy: boolean }>();
const emit = defineEmits<{ ask: [question: string]; stop: [] }>();

// The API's limit (`config.MAX_QUESTION_CHARS`); the count shows near it.
const MAX_QUESTION_CHARS = 1000;
const SHOW_COUNT_FROM = 900;

const text = ref("");
const input = ref<HTMLTextAreaElement | null>(null);
const question = computed(() => text.value.trim());
const canAsk = computed(
  () =>
    !props.busy && question.value.length > 0 && text.value.length <= MAX_QUESTION_CHARS,
);

function ask(): void {
  if (!canAsk.value) {
    input.value?.focus();
    return;
  }
  emit("ask", question.value);
  text.value = "";
}

// One button that asks, or stops while an answer is on its way: the same element
// keeps keyboard focus when its job changes (WCAG 2.4.3), and it is never disabled,
// so focus is never dropped either. It is a plain button, never a submit button: a
// stop ends synchronously and re-renders the button before the click's default
// action, which would then have submitted the draft.
function onButton(): void {
  if (props.busy) emit("stop");
  else ask();
}

function onKeydown(event: KeyboardEvent): void {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    ask();
  }
}

defineExpose({ focus: () => input.value?.focus() });
</script>

<template>
  <form class="composer">
    <label class="visually-hidden" for="question">Your question</label>
    <textarea
      id="question"
      ref="input"
      v-model="text"
      rows="2"
      placeholder="Ask about NNPC: its businesses, results, people or documents"
      :maxlength="MAX_QUESTION_CHARS"
      @keydown="onKeydown"
    />
    <div class="composer-row">
      <span v-if="text.length >= SHOW_COUNT_FROM" class="count">
        {{ text.length }} / {{ MAX_QUESTION_CHARS }}
      </span>
      <span v-else class="hint">{{
        busy
          ? "Answering… Esc or Stop ends it; your text stays here"
          : "Enter to ask · Shift+Enter for a new line"
      }}</span>
      <button
        type="button"
        :class="['button', { 'button-primary': !busy }]"
        @click="onButton"
      >
        <LineIcon :name="busy ? 'stop' : 'send'" />
        {{ busy ? "Stop" : "Ask" }}
      </button>
    </div>
  </form>
</template>
