<script setup lang="ts">
import { computed, ref } from "vue";
import type { RecordedAnswer } from "../lib/api";
import { citationText } from "../lib/cite";
import { hasOwnMark, pageOf, readableDate } from "../lib/format";
import { safeLink } from "../lib/links";
import LineIcon from "./LineIcon.vue";
import NoticeText from "./NoticeText.vue";

// One verified answer, led by its evidence: the claims are the strongest text, and
// each source leads with the quote code found word for word (or in a list compiled
// from its page, which it says), then where it is (page, dates) and which document. A link off the sources' hosts is shown as plain text;
// every source link opens in a new tab, so the thread stays where it is.
const props = defineProps<{ recorded: RecordedAnswer }>();
// A refusal offers the How to ask guide as its next step (Nielsen's heuristic 9).
const emit = defineEmits<{ guide: [] }>();

const answer = computed(() => props.recorded.answer);
const turn = computed(() => props.recorded.turn);
// A follow-up shows how it was understood; a refusal shows what was searched for.
// A refusal says what was searched for, unless that only repeats the question.
const understoodLabel = computed(() => {
  const searched = understood.value.join(" · ").toLowerCase();
  if (answer.value.status !== "answered") {
    return searched === answer.value.question.toLowerCase() ? "" : "Searched for";
  }
  return turn.value > 1 ? "Understood as" : "";
});
const copied = ref("");
const understood = computed(() => props.recorded.understood_as);
const sources = computed(() =>
  answer.value.sources.map((source) => ({
    ...source,
    href: safeLink(source.link),
    page: pageOf(source.link),
    published: readableDate(source.published),
    retrieved: readableDate(source.captured),
    // A quote that opens with its own quotation mark needs no added one.
    ownMark: hasOwnMark(source.quote),
  })),
);
const closest = computed(() =>
  answer.value.closest_pages.map((page) => ({ ...page, href: safeLink(page.url) })),
);

interface Place {
  page: number | null;
  href: string | null;
  label: string;
}

// Search can hit several places in one document: each document is listed once, with
// its pages in order, so the reader compares documents, not lines (Tesler's law).
const pointers = computed(() => {
  const documents = new Map<string, { title: string; kind: string; places: Place[] }>();
  for (const pointer of answer.value.pointers) {
    const key = `${pointer.title}|${pointer.kind}`;
    const found = documents.get(key) ?? { ...pointer, places: [] };
    documents.set(key, found);
    const page = pageOf(pointer.link);
    if (found.places.some((place) => place.page === page)) continue;
    const label = page === null ? "open" : `p.\u00a0${page}`;
    found.places.push({ page, href: safeLink(pointer.link), label });
  }
  const listed = [...documents.values()];
  for (const found of listed) {
    found.places.sort((a, b) => Number(a.page) - Number(b.page));
  }
  return listed;
});

/** Copy the claims with their sources, for a note or report; the result is said
 * aloud through the status line beside the button. */
async function copyAnswer(): Promise<void> {
  copied.value = "";
  try {
    await navigator.clipboard.writeText(citationText(answer.value));
    copied.value = "Copied with its sources.";
  } catch {
    copied.value = "Copying failed; select the text instead.";
  }
}

function anchor(number: number): string {
  return `t${turn.value}-s${number}`;
}
</script>

<template>
  <section
    v-if="answer.status === 'safety_referral'"
    class="referral"
    :aria-labelledby="`t${turn}-referral`"
  >
    <h3 :id="`t${turn}-referral`" class="referral-heading">
      <LineIcon name="info" />
      Advice referral
    </h3>
    <NoticeText :text="answer.notice" />
  </section>
  <div v-else class="answer">
    <p v-if="understoodLabel && understood.length > 0" class="understood">
      <span class="understood-label">{{ understoodLabel }}</span>
      {{ understood.join(" · ") }}
    </p>
    <ul v-if="answer.claims.length > 0" class="claims">
      <li v-for="(claim, index) in answer.claims" :key="index">
        {{ claim.text }}
        <span class="refs"
          ><a
            v-for="number in claim.sources"
            :key="number"
            class="ref"
            :href="`#${anchor(number)}`"
            :aria-label="`Source ${number}`"
            >{{ number }}</a
          ></span
        >
      </li>
    </ul>
    <NoticeText v-if="answer.notice" :text="answer.notice" />
    <p v-if="answer.status === 'insufficient_evidence'" class="next-step">
      Asking another way can help.
      <button type="button" class="button" @click="emit('guide')">
        <LineIcon name="guide" />
        Tips for asking
      </button>
    </p>
    <section v-if="sources.length > 0" class="sources" aria-label="Sources">
      <h3 class="section-heading">Sources</h3>
      <ol>
        <li v-for="source in sources" :id="anchor(source.number)" :key="source.number">
          <span class="source-number" aria-hidden="true">{{ source.number }}</span>
          <div class="source-body">
            <blockquote :class="['quote', { 'quote-marked': source.ownMark }]">
              {{ source.quote }}
            </blockquote>
            <p class="source-meta">
              <span class="verified"
                ><LineIcon name="check" />{{
                  source.compiled
                    ? "Compiled from the page's list"
                    : "Found word for word"
                }}</span
              >
              <span v-if="source.page">p.&nbsp;{{ source.page }}</span>
              <span v-if="source.published">Published {{ source.published }}</span>
              <span>Retrieved {{ source.retrieved }}</span>
            </p>
            <p class="source-title">
              <a
                v-if="source.href"
                :href="source.href"
                target="_blank"
                rel="noopener noreferrer"
                >{{ source.title
                }}<span class="visually-hidden"> (opens in a new tab)</span
                ><LineIcon name="open"
              /></a>
              <template v-else>{{ source.title }}</template>
              <span v-if="source.heading" class="source-heading">
                › {{ source.heading }}</span
              >
            </p>
          </div>
        </li>
      </ol>
    </section>
    <section v-if="closest.length > 0" class="pages" aria-label="Closest pages">
      <h3 class="section-heading">Closest pages</h3>
      <ul>
        <li v-for="page in closest" :key="page.url">
          <a
            v-if="page.href"
            :href="page.href"
            target="_blank"
            rel="noopener noreferrer"
            >{{ page.title
            }}<span class="visually-hidden"> (opens in a new tab)</span></a
          >
          <template v-else>{{ page.title }}</template>
        </li>
      </ul>
    </section>
    <section
      v-if="pointers.length > 0"
      class="pages"
      aria-label="Pages to check yourself"
    >
      <h3 class="section-heading">Pages to check yourself</h3>
      <p class="pages-note">
        Search found these scanned pages or pictures. Their words can't be checked word
        for word, so they are not quoted.
      </p>
      <ul>
        <li v-for="pointer in pointers" :key="`${pointer.title}|${pointer.kind}`">
          {{ pointer.title }}
          <span class="pointer-place">({{ pointer.kind }}):&#32;</span>
          <template v-for="(place, index) in pointer.places" :key="place.label">
            <a
              v-if="place.href"
              :href="place.href"
              target="_blank"
              rel="noopener noreferrer"
              >{{ place.label
              }}<span class="visually-hidden">
                of {{ pointer.title }} (opens in a new tab)</span
              ></a
            ><span v-else>{{ place.label }}</span
            ><template v-if="index < pointer.places.length - 1">, </template>
          </template>
        </li>
      </ul>
    </section>
    <div v-if="answer.claims.length > 0" class="answer-actions">
      <button type="button" class="button" @click="copyAnswer">
        <LineIcon name="copy" />
        Copy answer with sources
      </button>
      <span class="copied" aria-live="polite">{{ copied }}</span>
    </div>
    <!-- With every answer, as the command line prints it. -->
    <p class="answer-notice">
      Independent of NNPC Limited. Check the linked source before relying on an answer.
    </p>
  </div>
</template>
