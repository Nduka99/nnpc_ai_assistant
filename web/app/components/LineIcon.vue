<script setup lang="ts">
// Drawn icons, one stroke style (24-unit grid, 1.75 stroke, round joins), sized by
// the surrounding text and coloured by `currentColor`. They are decorative: every
// control that shows one also has a text label.
const PATHS = {
  send: ["M12 19V5", "m5 12 7-7 7 7"],
  new: ["M12 5v14", "M5 12h14"],
  check: ["m5 12.5 4.5 4.5L19 7.5"],
  alert: [
    "M10.3 3.9 2.6 17.2A2 2 0 0 0 4.3 20h15.4a2 2 0 0 0 1.7-2.8L13.7 3.9a2 2 0 0 0-3.4 0Z",
    "M12 9v4.5",
    "M12 17h.01",
  ],
  open: [
    "M14 4h6v6",
    "M20 4 11 13",
    "M18 14v5a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V7a1 1 0 0 1 1-1h5",
  ],
  info: ["M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18Z", "M12 11v5", "M12 7.5h.01"],
  stop: ["M8 7h8a1 1 0 0 1 1 1v8a1 1 0 0 1-1 1H8a1 1 0 0 1-1-1V8a1 1 0 0 1 1-1Z"],
  retry: ["M4.5 12a7.5 7.5 0 1 0 2.2-5.3", "M4.5 4.5v4h4"],
  undo: ["M9 14 4 9l5-5", "M4 9h10.5a5.5 5.5 0 0 1 0 11H11"],
  copy: [
    "M9 9h10a1 1 0 0 1 1 1v10a1 1 0 0 1-1 1H9a1 1 0 0 1-1-1V10a1 1 0 0 1 1-1Z",
    "M5 15H4a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1h10a1 1 0 0 1 1 1v1",
  ],
} as const;

defineProps<{ name: keyof typeof PATHS }>();
</script>

<template>
  <svg
    class="icon"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    stroke-width="1.75"
    stroke-linecap="round"
    stroke-linejoin="round"
    aria-hidden="true"
    focusable="false"
  >
    <path v-for="path in PATHS[name]" :key="path" :d="path" />
  </svg>
</template>
