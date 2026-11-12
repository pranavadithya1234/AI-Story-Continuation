# AI Story Continuation — Test Results Report

**Date:** 2026-11-12 10:06:43

**Model:** DistilGPT2 (local PyTorch inference)

---

## 1. Valid Story Prompts (10+ prompts)

| # | Prompt (truncated) | Tokens | Temp | Latency (ms) | Generated Continuation (first 120 chars) |
|---|---|---|---|---|---|
| 1 | One day, a student discovered a mysterious box in the colleg… | 80 | 0.8 | 1164 | . He called it ‪#“ #_DewardLithgowShrine - this is one of his favorite places to hang out with students and all over aga… |
| 2 | A robot entered the classroom for the first time | 80 | 0.8 | 246 | in a month. The robots have been used by children and adults to teach teachers on how to play their own toys, according:… |
| 3 | It was a dark and rainy night when the phone rang | 80 | 0.8 | 597 | . “The police arrived, but they were unable to get them back for more than two hours until paramedics could arrive at th… |
| 4 | The astronaut looked out the window and saw something imposs… | 80 | 0.8 | 544 | : a giant, black-roofed wing. It appeared to be an artificial eye (I'll call it that), but was also one of two objects i… |
| 5 | Deep in the forest, an old woman kept a garden of strange fl… | 80 | 0.8 | 613 | around her. It was not long before she decided to come back and start gathering all these plants for study or research o… |
| 6 | The letter arrived without a stamp, without an address, with… | 80 | 0.8 | 608 | . The letters came from the office of Governor-General or to Mr. President; in fact, that was only on account of their n… |
| 7 | On the last day of school, the teacher revealed a secret | 80 | 0.8 | 597 | letter he had received from his mother and an aunt. The girl was told that she would be able to attend with her sister i… |
| 8 | A cat wandered into the laboratory and knocked over a beaker | 80 | 0.8 | 599 | . He ran away, but when he saw it that his eyes were being closed on him, I said to myself: ‘Well what’s wrong? The whol… |
| 9 | The ship had been drifting for three months when they spotte… | 80 | 0.8 | 603 | in the northern part of France. The crew said it would be possible to return from a routine landing on Wednesday evening… |
| 10 | Under the floorboards, there was a map drawn in invisible in… | 80 | 0.8 | 597 | . Then we saw that someone had been sitting on it and said “The room—? As if this wasn't so strange as he thought I migh… |
| 11 | The clock struck thirteen and the world changed forever | 80 | 0.8 | 538 | . And now, with this new technology for measuring time travel in real-time we're just beginning to get used up on it! If… |
| 12 | A child found a door in the back of the wardrobe that had ne… | 80 | 0.8 | 521 | , which was just another reason why he went to work. The young man told police on Friday night: 'I left my car and I wal… |

## 2. Edge Case Tests

| Test Case | Status | Details |
|---|---|---|
| Empty prompt | PASS | Prompt is required. |
| Whitespace-only prompt | PASS | Prompt is required. |
| Long prompt (5000 chars) | Handled (truncated by tokenizer) | Generated 1915 chars in 574 ms |

## 3. Continuation Length Comparison

| Setting | Prompt | max_new_tokens | Temperature | Output Length (chars) | Latency (ms) |
|---|---|---|---|---|---|
| Short continuation | One day, a student discovered a mysterious box in … | 50 | 0.8 | 255 | 380 |
| Medium continuation | A robot entered the classroom for the first time… | 100 | 0.8 | 297 | 654 |
| Long continuation | A robot entered the classroom for the first time… | 150 | 0.8 | 508 | 748 |

## 4. Temperature Comparison (Same Prompt)

### Prompt: *"It was a dark and rainy night"*

**Low temperature (0.3) — less varied** (temperature=0.3):

> , but I think it's been pretty good," he said. "I'm just glad that the weather is getting better."
The man who died in his home on Saturday morning has not yet returned to work for The Associated Press. He had worked as an air traffic controller at one of several airports around the country since 20

*Output length: 303 chars | Latency: 517 ms*

**High temperature (1.2) — more varied** (temperature=1.2):

> , so I didn't think for as long as we could possibly remember what happened to me on the way back out in July. What if this wasn"t some other event? Or maybe something more serious than that?"
The following is how you can see one last time through:

If someone says "my dad's doing it because his wif

*Output length: 354 chars | Latency: 621 ms*

**Observation:** Lower temperature produces more predictable, focused text. Higher temperature introduces more creative and sometimes unexpected word choices, but can also produce less coherent output.

## 5. Reproducibility Test (Same Prompt, Same Settings, Two Runs)

**Run 1:** . It was a giant robot that appeared to be flying from space, but it could not move or fly off its own accord as if he were there in time for his next mission — before an alien spacecraft landed on Ea

**Run 2:** .
"It looks like you're going to die!" he said, holding his hands up in pain before a sudden crash landed on him at about 7:30 p..he was pronounced dead shortly after 8 p.m., when another man ran into

**Outputs identical?** NO

**Explanation:** When `do_sample=True`, the model samples from the probability distribution at each step, so different runs produce different continuations even with identical inputs.

## 6. Latency Summary (Local PyTorch)

| Metric | Value |
|---|---|
| Number of requests | 17 |
| Average latency | 597 ms |
| Min latency | 246 ms |
| Max latency | 1164 ms |

> **Note:** The first request includes model loading time.  Latency varies with hardware, system load, prompt length and max_new_tokens setting.  Cloud (SageMaker) latency will additionally include network round-trip and container startup time.

## 7. API Contract Verification

### Request JSON

```json
{
  "prompt": "One day, a student discovered a mysterious box...",
  "max_new_tokens": 50,
  "temperature": 0.8,
  "top_p": 0.9
}
```

### Response JSON

```json
{
  "generated_text": "(continuation text here)"
}
```

### Error Response JSON

```json
{
  "error": "prompt is required"
}
```
