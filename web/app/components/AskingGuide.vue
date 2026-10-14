<script setup lang="ts">
import { ref } from "vue";
import LineIcon from "./LineIcon.vue";

// How to ask (K33): five habits measured to fix failed questions on development data,
// and starter questions the assistant answered soundly. It opens over the
// conversation without leaving it (non-modal), and the reader can keep it open
// while writing.
defineProps<{ starters: readonly string[] }>();
const emit = defineEmits<{ close: []; ask: [question: string] }>();

const RULES = [
  {
    rule: "Name what you mean.",
    example:
      "Say the unit, company or report: “NGIC's managing director”, not “who runs the gas pipes”.",
  },
  {
    rule: "Ask for a person by their role.",
    example:
      "Use the title or the unit's initials: “Who is NGPIS's Chief Gas and Power Investment Officer?”",
  },
  {
    rule: "One thing at a time works best.",
    example:
      "Ask for a long list group by group (“Which units are in the Downstream group?”), and compare two things in two questions.",
  },
  {
    rule: "In a longer conversation, name the subject again.",
    example:
      "After a few questions, or when you change topic, say “NNPC HMO's growth phase”, not “its”.",
  },
  {
    rule: "Ask what the documents say, not what to do.",
    example:
      "“Does NNPC HMO list occupational health?” gets an answer; “Should we sign their contract?” gets an advice referral.",
  },
] as const;

const heading = ref<HTMLHeadingElement | null>(null);
defineExpose({ focus: () => heading.value?.focus() });
</script>

<template>
  <section id="asking-guide" class="guide" aria-labelledby="guide-title">
    <div class="guide-head">
      <h2 id="guide-title" ref="heading" tabindex="-1">How to ask</h2>
      <button type="button" class="button guide-close" @click="emit('close')">
        <LineIcon name="close" />
        Close
      </button>
    </div>
    <p class="guide-lead">
      Answers come only from the public pages and documents collected here, each
      statement with the quote it rests on. These habits get the most out of it.
    </p>
    <ul class="guide-rules">
      <li v-for="item in RULES" :key="item.rule">
        <strong>{{ item.rule }}</strong>
        {{ item.example }}
      </li>
    </ul>
    <h3 class="guide-subhead">Try one</h3>
    <ul class="examples" aria-label="Starter questions">
      <li v-for="starter in starters" :key="starter">
        <button type="button" class="example" @click="emit('ask', starter)">
          {{ starter }}
        </button>
      </li>
    </ul>
  </section>
</template>
