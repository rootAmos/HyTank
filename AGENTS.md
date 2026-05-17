# Repository Coding Rules

Prioritize simplicity ruthlessly. Prefer the smallest direct implementation that is correct, readable, and easy to rerun.

- Do not add helper functions, classes, wrappers, or abstractions unless they remove real duplication or are required by an external API.
- Before adding a function, explain what uses it and why it must exist. If it is not used several times in the repo, keep the logic inline.
- For open-ended requests, do a strict second pass over all new functions. Keep only functions called several times, required entry points, or functions that separate a large independent concern.
- Add a short comment directly under each remaining new function explaining why it exists.
- When making an important simplification, add the kernel of the approach here so future edits follow the same pattern.
- Fewer lines of code are better when the behavior remains clear.
- Prefer vectorized array operations over Python loops.
- Keep example scripts linear and editable. Use top-level constants for run settings instead of expanding command-line interfaces.
- Pass data explicitly. Avoid hidden mutable module state.
- Avoid global variables for project state or case-specific settings. Keep values local, pass them explicitly, or store them in the input file that defines the case. Module-level constants are acceptable only for true constants such as unit conversions or immutable defaults.
- When applying rate limits to states such as `u`, `q`, or `v`, constrain the derivative of that state directly. Do not manually compute deltas across nodes.
- For mission examples, normalize external schedule data once, then construct the model directly at the call site. Do not wrap a constructor with a one-use conversion helper.
- Leave unrelated files and generated artifacts alone unless the task explicitly requires updating them.
