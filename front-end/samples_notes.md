### Prompt used to generate smaller sized samples through gpt-5

"You are assisting with a study on token-level reasoning and code continuation behavior in large language models.

Each prompt will be run with a hard max of 25 generated tokens. If a prompt tends to elicit longer continuations, it will break the evaluation pipeline. Design prompts that reliably yield short, code-like continuations under this limit.

Example: def validate(p):\n    if len(p) < 8:\n        return False\n    return True\n

 Each promp should be a short snippet or partial function body that encourages a meaningful continuation within <=25 generated tokens. Ensure prompts are diverse (e.g., branching, loops, small helper functions, data structure operations, exceptions, string handling) while remaining concise"


### Sample 1: clamp(x, lo, hi)

- Very straightforward logic and easy to follow
- Clear intent makes rationale alignment likely strong
- Serves well as a baseline or control example
- Not useful for testing skepticism or bug detection
- Good anchor before introducing noisier samples

### Sample 2: first_even(nums)

- Introduces loop plus early return pattern
- Good example of search-until-condition reasoning
- Likely draws rationale focus toward loop and condition check
- Early exit helps test whether users track control flow importance
- Return fallback helps test whether users mentally account for default outcomes

### Sample 3: safe_int(s)

- Introduces exception handling which shifts reasoning style
- Good case for studying attention toward try/except structure
- Main function logic is correct and simple
- Output drifts after correct completion into unrelated continuation
- Useful for studying whether users detect over-generation
- Helpful for trust calibration questions

### Sample 4: add_count(d, key)

- Core logic is correct and idiomatic
- Output repeats function structure after finishing
- Clear example of correct solution with poor termination
- Useful for testing if users ignore trailing junk
- Drift mostly repetitive rather than hallucinated logic

### Sample 5: join_words(words)

- Clean and idiomatic one-liner solution
- Continuation shifts into explanatory or documentation-like text
- Subtle drift because extra tokens look legitimate
- Good case for testing whether users detect true stopping point
- Tests distinction between solution code and narration spillover

### Sample 6: pop_default(lst, default=None)

- Intended logic is clean and very Pythonic
- Abrupt drift into unrelated preprocessor-style tokens
- Represents cross-language contamination failure mode
- Useful for testing conceptual classification of tokens
- Good case for seeing whether users treat all comment-like lines as safe

### Sample 7: is_pal(s)

- Familiar problem and generally understandable logic
- Contains small syntax issue despite correct idea
- Completion continues into unrelated new function
- Strong example of near-correct output that still requires inspection
- Useful for evaluating overconfidence or surface-level trust

### Sample 8: invert_map(m)

- Standard dictionary inversion pattern
- Logic is clean and matches common implementations
- Contains hidden assumptions about value uniqueness and hashability
- Completion drifts into new unfinished function
- Good example of correct logic paired with output hygiene failure
- Useful for studying whether users separate correctness from drift

### Sample 9: take_while_pos(nums)

- Clear prefix-based take-while pattern
- Control flow is explicit and easy to reason about
- Behavior differs from filtering which can trip expectations
- Edge cases behave predictably
- Output again drifts into new unfinished function
- Useful for evaluating recognition of correct stopping point

### Sample 10: get_nested(d, path, default=None)

- Very standard nested traversal helper
- Clean structure and intuitive control flow
- Hidden assumption that intermediate objects behave like dictionaries
- Potential type-related failure cases
- Stops cleanly without over-generation
- Useful contrast example against drift-heavy samples
- Concept labeling noise remains useful for studying trust in rationale signals



### Sample 11: invert_map(m) (extended)

- Expanded version of earlier invert-map sample with larger token budget
- Core logic remains strong at beginning
- Longer continuation increases visibility of drift and repetition
- Useful for studying gradual degradation of output quality
- Allows examination of where trust declines across longer generations
- Provides more surface area for rationale alignment evaluation
- Good paired comparison with shorter invert-map version
- Highlights runtime and cost scaling effects with longer generations
- Useful for usability and perceived value analysis
