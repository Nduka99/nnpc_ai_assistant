<script setup lang="ts">
import { nextTick, ref, useId } from "vue";
import { sendFeedback } from "../lib/api";
import LineIcon from "./LineIcon.vue";

// A reader's judgement of one answer: right, or not right with an optional
// comment. It is kept with the answer for review and shown to nobody else. The
// buttons are never disabled, so keyboard focus stays where the reader left it.
const props = defineProps<{ answerId: number; conversationId: string }>();

const MAX_COMMENT = 1000;
const id = useId();
const telling = ref(false);
const comment = ref("");
const status = ref<"open" | "sending" | "sent" | "failed">("open");
const commentBox = ref<HTMLTextAreaElement | null>(null);
const form = ref<HTMLDivElement | null>(null);
const thanks = ref<HTMLParagraphElement | null>(null);

// The whole form comes into view, Send included, clear of the question box below
// (the page's scroll padding), with the comment box focused (WCAG 2.4.11).
async function tell(): Promise<void> {
  telling.value = true;
  await nextTick();
  commentBox.value?.focus({ preventScroll: true });
  form.value?.scrollIntoView({ block: "nearest" });
}

async function send(helpful: boolean): Promise<void> {
  if (status.value === "sending") return;
  status.value = "sending";
  const kept = await sendFeedback(
    props.answerId,
    props.conversationId,
    helpful,
    helpful ? "" : comment.value.trim(),
  );
  status.value = kept ? "sent" : "failed";
  if (kept) {
    await nextTick();
    thanks.value?.focus();
  }
}
</script>

<template>
  <section class="feedback" :aria-labelledby="`${id}-question`">
    <p v-if="status === 'sent'" ref="thanks" class="feedback-done" tabindex="-1">
      <LineIcon name="check" />
      Thanks. Your feedback was recorded.
    </p>
    <template v-else>
      <div class="feedback-row">
        <span :id="`${id}-question`" class="feedback-question"
          >Was this answer right?</span
        >
        <button type="button" class="button" @click="send(true)">Yes</button>
        <button type="button" class="button" :aria-expanded="telling" @click="tell">
          Not right
        </button>
      </div>
      <div v-if="telling" ref="form" class="feedback-form">
        <label :for="`${id}-comment`" class="feedback-label"
          >What was wrong? (optional)</label
        >
        <textarea
          :id="`${id}-comment`"
          ref="commentBox"
          v-model="comment"
          rows="3"
          :maxlength="MAX_COMMENT"
        />
        <button type="button" class="button" @click="send(false)">Send</button>
      </div>
      <p v-if="status === 'failed'" class="feedback-failed" role="alert">
        <LineIcon name="alert" />
        Feedback could not be sent. Try again.
      </p>
    </template>
  </section>
</template>
