import js from "@eslint/js";
import tseslint from "typescript-eslint";
import reactHooks from "eslint-plugin-react-hooks";
import reactRefresh from "eslint-plugin-react-refresh";

export default tseslint.config(
  { ignores: ["dist", "node_modules", "*.config.js", "*.config.ts", "**/*.d.ts"] },
  {
    extends: [js.configs.recommended, ...tseslint.configs.recommended],
    files: ["**/*.{ts,tsx}"],
    plugins: {
      "react-hooks": reactHooks,
      "react-refresh": reactRefresh,
    },
    rules: {
      // === REACT HOOKS RULES (catches useEffect ordering issues) ===
      // This is the key rule that catches hooks called conditionally or after early returns
      "react-hooks/rules-of-hooks": "error",
      // Warns when useEffect dependencies are missing or incorrect
      "react-hooks/exhaustive-deps": "warn",

      // === REACT REFRESH (for Vite HMR) ===
      "react-refresh/only-export-components": [
        "warn",
        { allowConstantExport: true },
      ],

      // === TYPESCRIPT RULES ===
      // Allow unused vars if prefixed with underscore (common pattern)
      "@typescript-eslint/no-unused-vars": [
        "warn",
        {
          argsIgnorePattern: "^_",
          varsIgnorePattern: "^_",
          caughtErrorsIgnorePattern: "^_",
        },
      ],
      // Allow explicit any in some cases (warn instead of error)
      "@typescript-eslint/no-explicit-any": "warn",
      // Allow non-null assertions (common in React refs)
      "@typescript-eslint/no-non-null-assertion": "off",

      // === GENERAL BEST PRACTICES ===
      // Disallow console.log in production (warn to allow during dev)
      "no-console": ["warn", { allow: ["warn", "error", "info"] }],
      // Require === and !== instead of == and !=
      eqeqeq: ["error", "always", { null: "ignore" }],
      // Disallow unnecessary boolean casts
      "no-extra-boolean-cast": "error",
      // Warn on debugger statements
      "no-debugger": "warn",
    },
  }
);
