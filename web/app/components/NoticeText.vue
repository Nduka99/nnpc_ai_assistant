<script setup lang="ts">
import { computed } from "vue";

// Fixed application text: lines starting "- " are steps, shown as a list.
const props = defineProps<{ text: string }>();

const blocks = computed(() => {
  const found: (string | string[])[] = [];
  for (const line of props.text.split("\n")) {
    const last = found.at(-1);
    if (line.startsWith("- ") && Array.isArray(last)) last.push(line.slice(2));
    else if (line.startsWith("- ")) found.push([line.slice(2)]);
    else found.push(line);
  }
  return found;
});
</script>

<template>
  <div class="notice">
    <template v-for="(block, index) in blocks" :key="index">
      <ul v-if="Array.isArray(block)">
        <li v-for="(step, item) in block" :key="item">{{ step }}</li>
      </ul>
      <p v-else>{{ block }}</p>
    </template>
  </div>
</template>
