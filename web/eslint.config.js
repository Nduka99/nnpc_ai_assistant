import js from "@eslint/js";
import pluginVue from "eslint-plugin-vue";
import pluginVueA11y from "eslint-plugin-vuejs-accessibility";
import globals from "globals";
import tseslint from "typescript-eslint";

// Prettier formats; ESLint checks correctness only, so Vue's formatting rules
// ("strongly recommended") are left out. Templates are also checked for
// accessibility before they render (Rule 6; axe checks the rendered page).
export default tseslint.config(
  {
    ignores: [
      ".nuxt",
      ".output",
      "dist",
      "coverage",
      "playwright-report",
      "test-results",
      "app/lib/api-types.ts",
    ],
  },
  js.configs.recommended,
  ...tseslint.configs.strict,
  ...pluginVue.configs["flat/essential"],
  ...pluginVueA11y.configs["flat/recommended"],
  {
    // A label names its control by `for`/`id` or by wrapping it; either is valid
    // (WCAG technique H44). The rule's default demands both.
    rules: {
      "vuejs-accessibility/label-has-for": [
        "error",
        { required: { some: ["nesting", "id"] } },
      ],
    },
  },
  {
    files: ["**/*.vue"],
    languageOptions: { parserOptions: { parser: tseslint.parser } },
  },
  { languageOptions: { globals: { ...globals.browser, ...globals.node } } },
);
