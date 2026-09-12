# Brand Analysis: AppleSupport (REVISED)

## Phase 1: Data Exploration & Brand Selection

### Dataset Overview
- **Source:** Kaggle `thoughtvector/customer-support-on-twitter`
- **Total rows:** 2,811,774 tweets
- **Columns:** tweet_id, author_id, inbound, created_at, text, response_tweet_id, in_response_to_tweet_id

### Brand Selection Process

Analyzed top brands by response volume:

| Brand | Total Tweets | English Responses | Unique Customers |
|-------|-------------|-------------------|------------------|
| AmazonHelp | 169,840 | 81,054 | N/A (Japanese-heavy) |
| AppleSupport | 106,860 | 33,296 | 23,668 |
| Uber_Support | 56,270 | 16,003 | 12,780 |
| SpotifyCares | 43,265 | 19,917 | 8,723 |
| Delta | 42,253 | 21,600 | 8,034 |

**Selected: AppleSupport** — Largest English dataset with 33K responses across 23K unique customers. Issues span iOS updates, device hardware, app bugs, and account access — providing good variety for multi-intent classification.

---

## Resolution Rate Analysis (REVISED)

### Methodology
- Sampled 200 customer queries directed at AppleSupport
- Traced conversation threads
- Analyzed resolution patterns with content-aware heuristics

### Key Findings

| Metric | Value |
|--------|-------|
| Brand response rate | 61.0% |
| Single-turn threads (brand responded, customer stopped) | 83 |
| Of those: DM deflections (no real troubleshooting) | 59 (71.1%) |
| Of those: Real single-turn resolutions | 24 (28.9%) |
| Multi-turn conversations | 22.3% |

### Critical Finding: DM Deflection Problem

**71.1% of "single-turn resolutions" are actually DM deflections**, not real resolutions. Brand responses like "DM us" or "send us a DM" don't constitute actual troubleshooting. This means:

- Simple "brand responded = resolved" heuristic is UNRELIABLE
- Must check brand response CONTENT before counting as resolution
- Retrieval pool needs filtering to exclude DM deflections

### Revised Resolution Heuristics

A thread is "resolved" ONLY if:
1. Customer expresses gratitude AND does not report ongoing issue
2. Customer confirms fix ("working", "worked", "resolved", "got them back")
3. Brand provided ACTUAL TROUBLESHOOTING (not just "DM us") AND customer stopped responding
4. Thread has 2+ turns and brand provided solution

**Exclusions:**
- Brand response is ONLY "DM us" / "send us a DM" = NOT resolved
- Customer expresses continued frustration = NOT resolved

---

## Intent Categories (REVISED)

### Changes from Initial Proposal
1. **Removed `positive_feedback`** — Not a customer problem, just conversation closure. Moved to resolution heuristic logic only.
2. **Removed `feature_question`** — Too thin (only 4 messages in sample). Merged version questions into `ios_update_issues`.
3. **Added boundary notes** — Each intent now has clear "classify as X, NOT Y" guidance.

### Final 5 Intent Categories

#### 1. ios_update_issues
Problems with iOS updates including installation errors, failed downloads, version compatibility questions, and post-update bugs.

**Examples:**
- "Yes 'there was an error downloading the software'"
- "They are not on the same iOS one is 11.0.2 and the other is 11.1.2"
- "the worse ios deployment to date period!"
- "Was apps, after I did the iOS DL. I rebooted, twice, they finally updated."

**Boundary:** If customer asks "should I upgrade?", classify here (not feature_question).

---

#### 2. battery_drain_after_update
Battery draining faster than expected, rapid power loss, charging issues, or battery percentage jumping unexpectedly.

**Examples:**
- "11.1.2, I've turned off the background usage and automatic updates. Still draining the battery"
- "Just took phone off the charger 10 minutes ago. Battery down 6 percent already and phone is hot."

**Boundary:** If message mentions BOTH battery drain AND crashing/freezing, classify as `device_crash_freeze` (crash is the more severe symptom).

---

#### 3. device_crash_freeze
Phone crashing, freezing, shutting down unexpectedly, becoming unresponsive, or requiring forced restarts.

**Examples:**
- "It happens randomly. Not in any app specifically. I think I've had to hard boot my iPhone at least 10x today because it kept freezing up"
- "Nope. Didn't work. Computer completely froze while trying to log into Google Drive"
- "11.0.3 - it has slowed down and the screen doesn't rotate sometimes.. sometimes it hangs"

**Boundary:** Include 'sluggish' and 'slow' here even though they could be battery-related. The user experience is "device not working properly."

---

#### 4. app_malfunction
Specific Apple apps (Messages, Safari, Music, iTunes, etc.) not working correctly, Bluetooth/connectivity issues, or feature-specific bugs.

**Examples:**
- "when I tap it nothing happens. restarting does nothing. i went to itunes and it said I hadn't bought the song when I did"
- "I have deleted and re-installed the app twice with no luck. I found the Amazon app is not working either."
- "Nope. Cant hear. Cant call. Using messenger Calls atm."
- "I'm using a 6s Plus. Attached img of error. phone cncts via BT for calls, USB for audio."

**Boundary:** Catch-all for feature-specific bugs. If it's about a specific app or feature not working, but the device itself is fine, it's `app_malfunction`.

---

#### 5. account_access
Apple ID issues, iCloud problems, login difficulties, password resets, backup/restore issues, or account recovery.

**Examples:**
- "2/2 and it has my email address there and I am asked for my password which I put in and that is when I get the error message."
- "Done twice, but issue remains. I've signed out of iCloud in settings as well, signin there works, but not in the app Store App."
- "No haven't backed up since May and its the July onwards photos I want😭"

**Boundary:** Include backup/restore issues here even though they could involve apps. The root cause is account/iCloud, not the app itself.

---

## Embedding Clustering Results (REVISED)

KMeans clustering (k=7) on sentence-transformer embeddings (`all-MiniLM-L6-v2`):

| Cluster | Size | Assessment | Intent Mapping |
|---------|------|------------|----------------|
| 0 | 36 | MIXED BAG — non-English, vague questions, fragments | No clear intent (noise) |
| 1 | 52 | Mostly ios_update_issues + some device_crash_freeze | BOUNDARY ISSUE |
| 2 | 60 | App/software errors + some positive_feedback | app_malfunction |
| 3 | 30 | Device-specific + transfer issues | app_malfunction (sub-type) |
| 4 | 21 | Battery drain, shutdown, freezing | battery_drain_after_update + device_crash_freeze |
| 5 | 77 | Thank you messages, confirmations | RESOLUTION SIGNAL (not an intent) |
| 6 | 24 | Version numbers + short responses | ios_update_issues (metadata) |

### Honest Assessment
- **2 clean clusters:** Clusters 4 and 5 clearly map to categories
- **3 muddy clusters:** Clusters 0, 1, 3 need manual review
- **1 noise cluster:** Cluster 5 is resolution signals, not an intent
- **Conclusion:** 5 intents is more honest than 7 — the data doesn't cleanly support 7 distinct categories

---

## Blind Sort Validation

Sorted 30 random messages into intent buckets, then compared:

| Intent | Count | Notes |
|--------|-------|-------|
| app_malfunction | 9 | Largest real issue category |
| positive_feedback | 8 | REMOVED — not an intent |
| device_crash_freeze | 4 | Clean category |
| feature_question | 4 | REMOVED — merged into others |
| ios_update_issues | 3 | Clean category |
| account_access | 2 | Clean category |
| battery_drain_after_update | 0 | Not in this sample (but exists in larger data) |

**Key finding:** `positive_feedback` and `feature_question` don't hold up as real intents.

---

## Sampling Method

- **Full dataset:** 2,811,774 tweets
- **AppleSupport subset:** 106,860 brand tweets
- **Customer queries to AppleSupport:** 36,658
- **Final sample:** 300 messages (stratified random sample with seed=42)
- **Validation sample:** 30 messages for blind sorting
- **Resolution analysis sample:** 200 threads

---

## Files Created

- `data/twcs.csv` — Full dataset (3M+ rows)
- `data/apple_support_sample.csv` — 300 sampled AppleSupport customer messages
- `src/intent_definitions.json` — Machine-readable intent specification (5 intents)
- `analysis/brand_analysis.md` — This document (revised)
