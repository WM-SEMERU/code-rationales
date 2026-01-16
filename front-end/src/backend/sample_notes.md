# Survey Sample Notes (Backend JSON Generation)

Note: Right now the backend JSON generation's capabilities are very limited due to current generation settings so most prompts are very short/simplistic

---

## example6.json

**Command:**
```bash
time curl -X POST http://127.0.0.1:5000/prompt \
  -H "Content-Type: application/json" \
  -d '{"prompt":"def validate(password): if len(password) < 8: print(\"Too short\")"}'
```
**Why it may be good for the survey:**
- Simple and realistic password validation logic
- Uses a clear conditional so it’s easy to judge if the rationales match the behavior
- Short enough that participants can understand quickly.

---

## example9.json

**Command:**
```bash
time curl -sS -X POST http://127.0.0.1:5000/prompt \
  -H "Content-Type: application/json" \
  -d '{"prompt":"# dedupe keep order\ndef dedupe(a): seen=set(); out=[];"}'
```
**Why it may be good for the survey:**
- Has both natural language + code (comment + function stub).
- Starts a real algorithm pattern (seen=set(); out=[]), so rationale links should be meaningful.
- Short enough that participants can understand quickly.

## example10.json

**Command:**
```bash
time curl -sS -X POST http://127.0.0.1:5000/prompt \
  -H "Content-Type: application/json" \
  -d '{"prompt":"# return True if x is even\ndef is_even(x): return x % 2 == "}'
```
**Why it may be good for the survey:**
- NL to code setup.
- Uses common operators and a simple condition, so it’s easy to judge whether the rationales actually match the logic
- Short enough that participants can understand quickly.

Note: The generation still drifts into weird non-Python tokens near the end (~~(x +)

