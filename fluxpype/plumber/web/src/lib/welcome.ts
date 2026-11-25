// Plum's opening guide, shown in the Ask panel before the first question of a chat. The text lives in
// fluxpype/plumber/ask/welcome.md, next to Plum's own guide, and is bundled here at build time: no model call, and
// one file to edit (ask.py also gives it to Plum, so its answer to "how can I use you?" agrees).
import raw from "../../../ask/welcome.md?raw";

const text = raw.replace(/<!--[\s\S]*?-->/g, "").trim();
const [body, questions = ""] = text.split(/^## Try asking[ \t]*$/m);

/** The guide's prose and lists, as AskAnswer renders them. */
export const WELCOME = body.trim();

/** The "Try asking" questions, each asked exactly as written. */
export const WELCOME_QUESTIONS = questions
  .split("\n")
  .map((l) => /^\s*[-*]\s+(.+?)\s*$/.exec(l)?.[1])
  .filter((q): q is string => !!q);
